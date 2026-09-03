import multiprocessing as mp

import cv2
from PIL import Image

from .config import (
    CONFIRMED_REFINE_DIST,
    DEPTH_MAX,
    DEPTH_MIN,
    EXPLORE_DESC,
    EXPLORE_REACHED_DIST,
    LIVE_PREVIEW_PATH,
    MAX_LOST_RELOCATES,
    MAX_STEPS_NO_PROGRESS,
    MAX_UNCONFIRMED_RELOCATES,
    MOVE_FWD,
    RELOCATE_EVERY,
    RGB_HFOV_DEG,
    SEARCH_PLAN,
    SEARCH_PLAN_HEADING_DEG,
    SIM_TURN_ANGLE,
    STOP,
    SUBGOAL_REACHED_DIST,
    SUCCESS_DIST,
    TURN_LEFT,
    TURN_MANEUVER_TURNS,
    TURN_RIGHT,
)
from .geometry import horiz_dist, is_backtracking, parse_guided_turn, turns_to_heading
from .hud import draw_target_marker, render_hud_frame
from .perception import cosmos3_worker, detect_current_frame
from .state import AgentState


def run_closed_loop(episode_index=None):
    print("Initializing IPC Cosmos Worker...")
    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    worker = ctx.Process(target=cosmos3_worker, args=(task_queue, result_queue), daemon=True)
    worker.start()

    print("Loading Habitat simulator...")
    import habitat
    from habitat.config.default import get_config
    from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

    config = get_config("benchmark/nav/vln_r2r.yaml", overrides=[
        "habitat.dataset.data_path=data/vln_ce/raw_data/r2r/val_seen/val_seen.json.gz",
        "habitat.dataset.scenes_dir=data/scene_data/",
        "habitat.environment.max_episode_steps=10000",  # our loop controls termination
        f"habitat.task.measurements.success.success_distance={SUCCESS_DIST}",  # keep in sync with SUCCESS_DIST
        # Widened from Habitat's 90deg default so more of the scene fits in frame per
        # step -- must stay in sync with RGB_HFOV_DEG, which geometry.unproject_pixel()
        # also reads; both sensors, since a detected pixel's depth is read at the same
        # index in both images.
        f"habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov={RGB_HFOV_DEG}",
        f"habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.hfov={RGB_HFOV_DEG}",
    ])
    env = habitat.Env(config=config)
    follower = ShortestPathFollower(env.sim, goal_radius=SUBGOAL_REACHED_DIST, return_one_hot=False)

    if episode_index is not None:
        if 0 <= episode_index < len(env.episodes):
            target_ep = env.episodes[episode_index]
            env.episode_iterator = iter([target_ep])
        else:
            print(f"Warning: Episode index '{episode_index}' out of bounds (dataset size: {len(env.episodes)}). Falling back to random episode.")

    obs = env.reset()
    instruction_text = env.current_episode.instruction.instruction_text
    ep_id = env.current_episode.episode_id

    rgb_img0 = obs["rgb"]
    task_queue.put(("decompose", (instruction_text, Image.fromarray(rgb_img0))))
    subgoal_queue = result_queue.get()
    print(f"[Episode {ep_id}] Decomposed into {len(subgoal_queue)} subgoals")
    for i, sg in enumerate(subgoal_queue):
        print(f"  Subgoal {i+1}: {sg.get('description', '?')}")
    print(f"\nLive view: watch -n 0.5 'feh {LIVE_PREVIEW_PATH}'\n")

    from tqdm import tqdm
    frames = []
    step = 0
    current_subgoal_idx = 0

    state = AgentState.SEARCHING
    plan_index = 0             # index into SEARCH_PLAN currently being turned-to / detected
    plan_subturns_left = SEARCH_PLAN[0][1]   # physical turns still needed to reach SEARCH_PLAN[plan_index]
    search_memory = []         # per-heading dicts collected during the current sweep
    pending_actions = []       # queued raw actions (reorient turns + one blind step) for the
                                # rare "no heading had usable depth anywhere" fallback below

    # Running estimate of the agent's absolute facing direction, in degrees (arbitrary
    # zero, TURN_LEFT +SIM_TURN_ANGLE / TURN_RIGHT -SIM_TURN_ANGLE) -- kept in sync with
    # every turn action actually sent to env.step() below. Lets EXPLORING compare
    # directions across separate search sweeps, which each start from wherever the
    # agent happens to be facing (not a shared zero).
    agent_heading_deg = 0
    sweep_start_heading_deg = 0   # agent_heading_deg captured at the start of the current sweep
    # Absolute heading (degrees) of the last EXPLORING (unconfirmed best-guess) direction
    # committed to for the CURRENT subgoal -- None once a new subgoal starts or none chosen
    # yet. Used to keep the agent from picking the reverse of its last exploring move and
    # oscillating between the same two spots (see is_backtracking()).
    last_explore_heading_deg = None

    cached_target_world_pt = None
    cached_target_pixel = None
    pursuing_confirmed = False   # True: cached target is a literally-FOUND subgoal. False: best-guess heading.
    refined_close = False        # True if close-up 3D target refinement has already been performed
    steps_since_relocate = RELOCATE_EVERY
    consecutive_lost_relocates = 0
    consecutive_unconfirmed_relocates = 0   # relocates spent chasing the SAME unconfirmed
                                              # best-guess point -- see MAX_UNCONFIRMED_RELOCATES

    best_dist_to_goal = float("inf")
    steps_since_progress = 0

    # What Cosmos most recently reported seeing, for the "Sees:" HUD line --
    # updated at every detect_current_frame() call, held over on frames where
    # no fresh detection happens (mid-turn, or driving between relocates).
    last_seen_found = False
    last_seen_label = "(nothing yet)"
    last_seen_confidence = 0.0
    # Cosmos's own live guess at what room/area the agent is currently in, from the
    # same detect_landmarks() call -- for the HUD's "Current location:" field.
    last_current_location = "(unknown)"

    def get_current_subgoal():
        if current_subgoal_idx >= len(subgoal_queue):
            return None
        return subgoal_queue[current_subgoal_idx]

    def get_next_target_location():
        """The subgoal AFTER the current one, for reason_best_heading()'s extra
        context -- a single scan's per-heading cues are often too weak to judge
        from the current objective alone (e.g. several headings all show "an
        open doorway"); knowing what comes next helps disambiguate. None if the
        current subgoal is the last one (or the queue is already exhausted)."""
        next_idx = current_subgoal_idx + 1
        if next_idx < len(subgoal_queue):
            nxt = subgoal_queue[next_idx]
            return nxt.get("target_location") or nxt.get("description")
        return None

    def get_subgoal_items():
        """For the HUD's 'Subgoals:' list -- the full plan exactly as
        decomposed at step 0, each one flagged achieved/not so the panel can
        color it green/red. Order and count never change after decomposition;
        only which prefix of the list is marked achieved does."""
        return [(sg.get("description", "?"), i < current_subgoal_idx)
                for i, sg in enumerate(subgoal_queue)]

    def start_new_search():
        nonlocal state, plan_index, plan_subturns_left, search_memory, sweep_start_heading_deg
        state = AgentState.SEARCHING
        plan_index = 0
        plan_subturns_left = SEARCH_PLAN[0][1]
        search_memory = []
        sweep_start_heading_deg = agent_heading_deg

    def reset_pursuit():
        nonlocal cached_target_world_pt, cached_target_pixel, pursuing_confirmed
        nonlocal steps_since_relocate, consecutive_lost_relocates, consecutive_unconfirmed_relocates, refined_close
        cached_target_world_pt = None
        cached_target_pixel = None
        pursuing_confirmed = False
        refined_close = False
        steps_since_relocate = RELOCATE_EVERY
        consecutive_lost_relocates = 0
        consecutive_unconfirmed_relocates = 0
        start_new_search()

    def advance_subgoal():
        """Subgoal complete: advance the queue, drop anti-backtrack memory, reset
        into a fresh search -- and if the just-completed subgoal's guided_direction
        said left/right, queue that as a deterministic turn maneuver BEFORE the
        next search begins (via the same pending_actions queue already used by the
        SEARCHING branch's no-usable-depth fallback below; reset_pursuit() has just
        put us into a fresh SEARCHING state with an empty queue, so this is safe).
        Turning is pure geometry once the target is grounded -- not something to
        ask the VLM to visually judge from a single frame."""
        nonlocal current_subgoal_idx, last_explore_heading_deg, pending_actions
        completed = subgoal_queue[current_subgoal_idx]
        current_subgoal_idx += 1
        last_explore_heading_deg = None   # fresh subgoal -- drop the anti-backtrack memory
        reset_pursuit()
        turn_dir = parse_guided_turn(completed.get("guided_direction"))
        if turn_dir is not None:
            turn_action = TURN_LEFT if turn_dir == "left" else TURN_RIGHT
            pending_actions = [turn_action] * TURN_MANEUVER_TURNS

    def enter_navigating(world_pt, pixel, confirmed):
        nonlocal cached_target_world_pt, cached_target_pixel, pursuing_confirmed
        nonlocal steps_since_relocate, consecutive_lost_relocates, consecutive_unconfirmed_relocates, state, refined_close
        cached_target_world_pt = world_pt
        cached_target_pixel = pixel
        pursuing_confirmed = confirmed
        refined_close = False
        steps_since_relocate = 0
        consecutive_lost_relocates = 0
        consecutive_unconfirmed_relocates = 0
        state = AgentState.NAVIGATING

    def record_seen(result):
        """Track what Cosmos most recently reported seeing/its own location, for the HUD."""
        nonlocal last_seen_found, last_seen_label, last_seen_confidence, last_current_location
        last_seen_found = result["found"]
        if result["found"]:
            last_seen_label = target_desc
            last_seen_confidence = 1.0
        else:
            last_seen_label = result["guess_label"] or "nothing recognizable"
            last_seen_confidence = result["guess_confidence"]
        cl = result.get("current_location")
        if cl and cl.strip() and cl.strip().lower() not in ["unknown", "(unknown)", "none", ""]:
            last_current_location = cl.strip()

    def commit_to_found(result):
        """Shared handling for 'the current subgoal's landmark is literally visible',
        whether that happened while SEARCHING: enter NAVIGATING and pick this
        same step's action immediately (advance now if already in reach)."""
        nonlocal action
        enter_navigating(result["found_world_pt"], result["found_pixel"], confirmed=True)
        agent_pos = env.sim.get_agent_state().position
        dist = horiz_dist(agent_pos, cached_target_world_pt)
        if dist < SUBGOAL_REACHED_DIST:
            advance_subgoal()
        else:
            action = follower.get_next_action(cached_target_world_pt)
            if action is None or action == 0:
                action = MOVE_FWD

    # ---------------------------------------------------------------------
    # The two questions this state machine answers
    # ---------------------------------------------------------------------
    # Q1: What does the agent do when it can't see/detect the current
    #     subgoal's landmark?
    #   -> AgentState.SEARCHING. It walks SEARCH_PLAN, one fixed ordered
    #      sweep of 8 headings (near-current-heading first, then the rest
    #      of a full 360deg), calling detect_current_frame() once per
    #      heading. The instant any heading's detect literally confirms the
    #      target, commit_to_found() fires -- see Q2. If the WHOLE plan is
    #      exhausted with nothing found, the accumulated search_memory
    #      (per-heading visible landmarks / guess confidence / collision)
    #      is handed to Cosmos3Reasoner.reason_best_heading(), which picks
    #      the most promising heading; the agent drives there as an
    #      UNCONFIRMED NAVIGATING target (pursuing_confirmed=False) and
    #      falls back into a fresh SEARCHING sweep (start_new_search(), via
    #      reset_pursuit()) once that point is reached or lost. Reaching an
    #      unconfirmed point never advances the subgoal queue.
    #
    # Q2: How does the agent know it just finished a subgoal?
    #   -> Exactly two conditions, both required (checked in commit_to_found
    #      and in the NAVIGATING branch's relocate/distance logic):
    #        (a) detect_current_frame(), on the CURRENT frame (never a
    #            stale/cached claim), has literally confirmed the current
    #            subgoal's landmark visible with usable depth and
    #            depth-unprojected it to a grounded 3D world point; AND
    #        (b) the agent's actual simulator position has closed to within
    #            SUBGOAL_REACHED_DIST of that grounded point.
    #      Only then does current_subgoal_idx advance. There is no separate
    #      "ask the model if we arrived" check -- depth-grounded distance is
    #      the sole criterion, matching standard ObjectNav/ImageNav
    #      "close enough to the estimated target location" success criteria.
    # ---------------------------------------------------------------------
    target_desc = subgoal_queue[0].get("description", "Unknown") if subgoal_queue else ""
    target_location = (subgoal_queue[0].get("target_location") or target_desc) if subgoal_queue else EXPLORE_DESC
    guided_direction = (subgoal_queue[0].get("guided_direction") or "None") if subgoal_queue else "None"
    exhausted = (current_subgoal_idx >= len(subgoal_queue))

    with tqdm(desc="Navigation Steps", unit="step") as pbar:
        while True:
            rgb_img = obs["rgb"]
            depth_img = obs["depth"]
            img_pil = Image.fromarray(rgb_img)
            display_img = cv2.cvtColor(rgb_img.copy(), cv2.COLOR_RGB2BGR)

            # ---- R2R ground-truth success check (independent of our subgoal logic) ----
            metrics = env.get_metrics()
            dist_to_goal = metrics.get("distance_to_goal", 9999)
            if dist_to_goal < SUCCESS_DIST:
                tqdm.write(f"Step {step} -> Within {dist_to_goal:.2f}m of R2R goal, issuing STOP...")
                obs = env.step(STOP)
                final_metrics = env.get_metrics()
                tqdm.write(f"Step {step} -> Success={final_metrics.get('success', 0.0)}, "
                            f"dist={final_metrics.get('distance_to_goal', 9999):.2f}m")

                # --- Render and save the final STOP frame ---
                final_display_img = cv2.cvtColor(obs["rgb"].copy(), cv2.COLOR_RGB2BGR)
                view_color = (0, 255, 0) if last_seen_found else (0, 200, 255)
                canvas = render_hud_frame(
                    final_display_img, step, "STOPPED", (0, 255, 0), "STOP", (0, 255, 0),
                    "REACHED", get_subgoal_items(), last_seen_label, view_color,
                    last_seen_confidence, final_metrics.get("distance_to_goal", 9999),
                    last_current_location, target_location,
                )
                cv2.imwrite(LIVE_PREVIEW_PATH, canvas)
                frames.append(canvas)
                break

            subgoal = get_current_subgoal()
            exhausted = subgoal is None
            # target_desc: human-readable action text, for HUD/logs only.
            # target_location: the actual object/place string sent to Cosmos's detector and
            # matched against its "visible" landmark names -- must NOT be the action
            # sentence ("Turn right at the staircase"), or detect_landmarks() can never
            # literally match it and the subgoal can never be confirmed found.
            # guided_direction: explicit directional/spatial cue from the instruction for
            # this subgoal (e.g. "turn right"), or "None" if it has none -- passed to
            # detect_landmarks() as context, not applied by any client-side logic here.
            if exhausted:
                target_desc = EXPLORE_DESC
                target_location = EXPLORE_DESC
                guided_direction = "None"
            else:
                target_desc = subgoal.get("description", "Unknown")
                target_location = subgoal.get("target_location") or target_desc
                guided_direction = subgoal.get("guided_direction") or "None"
            action = None

            # ---- Stuck/no-progress safety net ----
            # Last-resort backstop: if R2R distance-to-goal hasn't improved in a
            # while (e.g. the agent is wedged against geometry the depth sensor
            # can't see through), abort instead of running for thousands of steps.
            if dist_to_goal < best_dist_to_goal - 0.05:
                best_dist_to_goal = dist_to_goal
                steps_since_progress = 0
            else:
                steps_since_progress += 1
            if steps_since_progress >= MAX_STEPS_NO_PROGRESS:
                tqdm.write(f"Step {step} -> No progress in {MAX_STEPS_NO_PROGRESS} steps "
                            f"(best dist={best_dist_to_goal:.2f}m), aborting episode.")
                break

            # =========================================================
            # NAVIGATING: drive toward cached_target_world_pt every step.
            # It's either the CURRENT subgoal's literally-FOUND landmark
            # (pursuing_confirmed=True) or a best-guess exploration point
            # from the last search sweep (pursuing_confirmed=False). Subgoals
            # are pursued strictly in order -- no skipping ahead.
            # =========================================================
            if state == AgentState.NAVIGATING and cached_target_world_pt is not None:
                agent_pos = env.sim.get_agent_state().position
                dist = horiz_dist(agent_pos, cached_target_world_pt)

                # 1. Check arrival conditions first
                if pursuing_confirmed and dist < SUBGOAL_REACHED_DIST:
                    tqdm.write(f"Step {step} -> Subgoal '{target_desc}' REACHED "
                                f"(dist={dist:.2f}m). Advancing queue.")
                    advance_subgoal()
                elif not pursuing_confirmed and dist < EXPLORE_REACHED_DIST:
                    # Reached the committed exploration waypoint (doorway / open space).
                    # Stop here and start a fresh 360 scan from this new vantage point.
                    tqdm.write(f"Step {step} -> Reached exploration waypoint (dist={dist:.2f}m). "
                                f"Starting fresh scan from new area.")
                    reset_pursuit()
                else:
                    # 2. Relocation / Re-probing:
                    # For CONFIRMED targets:
                    #   - Re-detect when entering close range (dist <= CONFIRMED_REFINE_DIST, e.g. 1.5m)
                    #     to refine the exact 3D coordinates from up close.
                    #   - Otherwise check periodically every RELOCATE_EVERY steps.
                    # For EXPLORING targets:
                    #   - Check every RELOCATE_EVERY steps ONLY to check if the real target landmark appeared.
                    #   - Do NOT overwrite the waypoint with new guesses, avoiding zigzagging!
                    if pursuing_confirmed:
                        should_relocate = (dist <= CONFIRMED_REFINE_DIST and not refined_close) or (steps_since_relocate >= RELOCATE_EVERY)
                    else:
                        should_relocate = steps_since_relocate >= RELOCATE_EVERY

                    if should_relocate:
                        steps_since_relocate = 0
                        result = detect_current_frame(
                            task_queue, result_queue, target_location, exhausted,
                            img_pil, rgb_img, depth_img, env, guided_direction=guided_direction,
                        )
                        record_seen(result)

                        if result["found"]:
                            cached_target_world_pt = result["found_world_pt"]
                            cached_target_pixel = result["found_pixel"]
                            pursuing_confirmed = True
                            consecutive_lost_relocates = 0
                            if dist <= CONFIRMED_REFINE_DIST:
                                refined_close = True
                                tqdm.write(f"Step {step} -> Refined target '{target_location}' close-up at dist={dist:.2f}m.")
                        elif pursuing_confirmed:
                            # Confirmed target temporarily occluded -- continue driving toward cached target
                            consecutive_lost_relocates += 1
                            if consecutive_lost_relocates >= MAX_LOST_RELOCATES:
                                tqdm.write(f"Step {step} -> Lost '{target_location}' for "
                                            f"{consecutive_lost_relocates} relocates, dropping target.")
                                reset_pursuit()
                        else:
                            # Exploring: landmark is still not directly visible.
                            # Keep committed to the current exploration waypoint without jittering.
                            consecutive_unconfirmed_relocates += 1
                            if consecutive_unconfirmed_relocates >= MAX_UNCONFIRMED_RELOCATES:
                                tqdm.write(f"Step {step} -> Exploration waypoint for '{target_location}' unreachable after "
                                            f"{consecutive_unconfirmed_relocates} checks, re-searching.")
                                reset_pursuit()
                    else:
                        steps_since_relocate += 1

                    if cached_target_world_pt is not None:
                        action = follower.get_next_action(cached_target_world_pt)
                        if action is None or action == 0:
                            action = MOVE_FWD
                        if cached_target_pixel:
                            draw_target_marker(display_img, cached_target_pixel, target_desc, pursuing_confirmed)

            # =========================================================
            # SEARCHING: at most ONE full 360 sweep -- see the Q1 answer
            # above. Walks SEARCH_PLAN, one detect per heading (near the
            # current heading first, then the rest of the circle). Stops
            # immediately the moment the current subgoal's landmark is
            # literally found. Otherwise records this heading's visible
            # landmarks + guess confidence + collision flag to memory.
            # After the full plan with nothing found, hand that memory to
            # Cosmos and let IT reason about which heading to commit to,
            # then drive there via the follower (no manual turn-then-step
            # -- the agent can move forward as many steps as it takes).
            # =========================================================
            elif state == AgentState.SEARCHING:
                if pending_actions:
                    action = pending_actions.pop(0)
                    if not pending_actions:
                        start_new_search()
                elif plan_subturns_left > 0:
                    action = SEARCH_PLAN[plan_index][0]
                    plan_subturns_left -= 1
                else:
                    result = detect_current_frame(
                        task_queue, result_queue, target_location, exhausted,
                        img_pil, rgb_img, depth_img, env, guided_direction=guided_direction,
                    )
                    record_seen(result)

                    if result["found"]:
                        tqdm.write(f"Step {step} -> Found '{target_location}' while searching "
                                    f"(plan step {plan_index+1}/{len(SEARCH_PLAN)}). Switching to NAVIGATING.")
                        commit_to_found(result)
                    else:
                        search_memory.append({
                            "heading": plan_index,
                            "abs_heading_deg": (sweep_start_heading_deg + SEARCH_PLAN_HEADING_DEG[plan_index]) % 360,
                            "visible_landmarks": result["visible_names"],
                            "current_location": result.get("current_location", ""),
                            "observation": result.get("guess_label", ""),
                            "guess_confidence": result["guess_confidence"],
                            "collision": result["collision"],
                            "world_pt": result["guess_world_pt"],
                            "pixel": result["guess_pixel"],
                        })
                        depth_note = "" if result["guess_world_pt"] is not None else " [no usable depth]"
                        tqdm.write(f"Step {step} -> Searching (plan step {plan_index+1}/{len(SEARCH_PLAN)}), "
                                    f"loc='{result.get('current_location', last_current_location)}', "
                                    f"obs='{result.get('guess_label', '')}', "
                                    f"confidence={result['guess_confidence']:.2f} "
                                    f"collision={result['collision']} for '{target_location[:24]}'{depth_note}")

                        if plan_index < len(SEARCH_PLAN) - 1:
                            plan_index += 1
                            next_turn, next_turns = SEARCH_PLAN[plan_index]
                            action = next_turn
                            plan_subturns_left = next_turns - 1
                        else:
                            # Full 360 sweep done, nothing literally visible anywhere.
                            # Ask Cosmos to reason over the accumulated per-heading
                            # memory and pick which direction to commit to.
                            usable = [e for e in search_memory if e["world_pt"] is not None]
                            if usable:
                                task_queue.put(("reason", (
                                    [{"heading": e["heading"], "abs_heading_deg": e["abs_heading_deg"],
                                      "observation": e.get("observation", ""), "visible_landmarks": e["visible_landmarks"],
                                      "guess_confidence": e["guess_confidence"], "collision": e["collision"]}
                                     for e in search_memory],
                                    target_location,
                                    get_next_target_location(),
                                    last_current_location,
                                )))
                                reasoning = result_queue.get()
                                best_heading = reasoning.get("best_heading", 0)
                                reason_text = reasoning.get("reason", "")

                                non_collision = [e for e in usable if not e["collision"]]
                                ranked_pool = non_collision or usable
                                # Exclude directions that would backtrack toward wherever the
                                # agent just came from while exploring this same subgoal --
                                # unless EVERY option would (e.g. a narrow dead-end corridor),
                                # in which case allow it rather than getting stuck with no move.
                                non_backtrack_pool = [
                                    e for e in ranked_pool
                                    if not is_backtracking(e["abs_heading_deg"], last_explore_heading_deg)
                                ]
                                ranked_pool = non_backtrack_pool or ranked_pool

                                chosen = next((e for e in ranked_pool if e["heading"] == best_heading), None)
                                if chosen is None:
                                    chosen = max(ranked_pool, key=lambda e: e["guess_confidence"])
                                    tqdm.write(f"Step {step} -> Cosmos picked heading {best_heading} "
                                                f"(invalid/blocked/no depth/backtracking) -- falling back to "
                                                f"heading {chosen['heading']} by confidence.")
                                else:
                                    tqdm.write(f"Step {step} -> Cosmos reasoning chose heading "
                                                f"{chosen['heading']}: {reason_text}")

                                last_explore_heading_deg = chosen["abs_heading_deg"]
                                enter_navigating(chosen["world_pt"], chosen["pixel"], confirmed=False)
                                action = follower.get_next_action(cached_target_world_pt)
                                if action is None or action == 0:
                                    action = MOVE_FWD
                            else:
                                # No heading had usable depth at all (rare -- e.g. wedged
                                # against geometry in every direction). No grounded world
                                # point exists for any heading, so there's nothing to hand
                                # the follower -- but Cosmos can still reason over what it
                                # SAW (visible landmarks, collision flags) at each heading,
                                # same call as the grounded case above. Reorient to face
                                # its pick and take one blind step, then restart the sweep.
                                task_queue.put(("reason", (
                                    [{"heading": e["heading"], "abs_heading_deg": e["abs_heading_deg"],
                                      "observation": e.get("observation", ""), "visible_landmarks": e["visible_landmarks"],
                                      "guess_confidence": e["guess_confidence"], "collision": e["collision"]}
                                     for e in search_memory],
                                    target_location,
                                    get_next_target_location(),
                                    last_current_location,
                                )))
                                reasoning = result_queue.get()
                                best_heading = reasoning.get("best_heading", 0)
                                reason_text = reasoning.get("reason", "")

                                def heading_abs_deg(h):
                                    return (sweep_start_heading_deg + SEARCH_PLAN_HEADING_DEG[h]) % 360

                                # Prefer a heading that's neither collision-flagged nor a
                                # backtrack of the last exploring move; relax one constraint
                                # at a time rather than getting stuck with no valid choice.
                                all_headings = list(range(len(SEARCH_PLAN)))
                                non_collision_headings = [h for h in all_headings if not search_memory[h]["collision"]]
                                safe_headings = [
                                    h for h in non_collision_headings
                                    if not is_backtracking(heading_abs_deg(h), last_explore_heading_deg)
                                ] or non_collision_headings or all_headings

                                if not (0 <= best_heading < len(SEARCH_PLAN)) or best_heading not in safe_headings:
                                    best_heading = safe_headings[0]

                                last_explore_heading_deg = heading_abs_deg(best_heading)
                                turn_action, turns = turns_to_heading(plan_index, best_heading)
                                tqdm.write(f"Step {step} -> Sweep complete, no usable depth anywhere; "
                                            f"Cosmos reasoning points toward heading {best_heading}: "
                                            f"{reason_text or '(no reason given)'}")
                                pending_actions = ([turn_action] * turns if turn_action else []) + [MOVE_FWD]

            if action is None:
                pbar.update(1)
                step += 1
                continue

            # ---- HUD ----
            state_color = {
                AgentState.NAVIGATING: (0, 255, 0) if pursuing_confirmed else (0, 200, 255),
                AgentState.SEARCHING: (255, 200, 0),
            }.get(state, (255, 255, 255))

            if state == AgentState.NAVIGATING:
                current_sg_state = "NAVIGATING (Target Locked)" if pursuing_confirmed else "EXPLORING (Towards Waypoint)"
            elif state == AgentState.SEARCHING:
                current_sg_state = "SEARCHING (360° Scan)"
            else:
                current_sg_state = state.name

            view_color = (0, 255, 0) if last_seen_found else (0, 200, 255)
            canvas = render_hud_frame(
                display_img, step, state.name, state_color, str(action), (255, 255, 255),
                current_sg_state, get_subgoal_items(), last_seen_label, view_color,
                last_seen_confidence, dist_to_goal,
                last_current_location, target_location,
            )
            cv2.imwrite(LIVE_PREVIEW_PATH, canvas)
            frames.append(canvas)

            # Keep the running absolute-heading estimate in sync with every turn actually
            # taken (SEARCH_PLAN sweep turns, follower-issued turns while NAVIGATING, and
            # the blind reorientation turns above) -- this is what lets EXPLORING compare
            # directions chosen across separate search sweeps, which each start from
            # wherever the agent happens to be facing, not a shared zero.
            if action == TURN_LEFT:
                agent_heading_deg = (agent_heading_deg + SIM_TURN_ANGLE) % 360
            elif action == TURN_RIGHT:
                agent_heading_deg = (agent_heading_deg - SIM_TURN_ANGLE) % 360

            obs = env.step(action)

            if env.episode_over:
                tqdm.write("Episode ended by simulator.")
                break

            pbar.update(1)
            step += 1

    metrics = env.get_metrics()
    env.close()

    if current_subgoal_idx >= len(subgoal_queue):
        print(f"\nAll {len(subgoal_queue)} subgoals completed -> queue exhausted")
    else:
        print(f"\nStopped at subgoal {current_subgoal_idx}/{len(subgoal_queue)} "
              f"(episode ended by simulator or success)")

    print("\n--- GROUND TRUTH METRICS ---")
    print(f"Distance to Goal: {metrics['distance_to_goal']:.2f} meters")
    print(f"Success:          {'YES' if metrics['success'] == 1.0 else 'NO'}")
    print(f"SPL (Efficiency): {metrics['spl']:.2f}")
    print("----------------------------\n")

    out_path = f"/home/dungtn21/InternNav/vln_subgoal_pipeline/closed_loop_ep{ep_id}.mp4"
    print(f"Saving video to {out_path}...")
    if frames:
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, 5.0, (width, height))
        for frame in frames:
            out.write(frame)
        out.release()
    print("Done!")

    task_queue.put(None)
    worker.join()
