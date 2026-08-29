import os
import sys
import re
import cv2
import multiprocessing as mp
import numpy as np
from PIL import Image
from enum import Enum, auto

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SUBGOAL_REACHED_DIST = 0.5        # meters, when to advance the subgoal queue
DEPTH_MIN = 0.05                  # ignore junk/too-close depth pixels
DEPTH_MAX = 8.0                   # ignore unreliable far depth pixels
SUCCESS_DIST = 3.0                # meters to R2R ground-truth goal to attempt STOP -- standard
                                   # R2R VLN-CE success radius (Anderson et al.); must match the
                                   # habitat.task.measurements.success.success_distance override
                                   # passed to get_config() below, or Habitat's own "success" metric
                                   # (which defaults success_distance to 0.2m) will never agree with
                                   # our STOP decision even when we stop at the "correct" distance.

RELOCATE_EVERY = 3                # only re-probe every N steps while pursuing a target
MAX_LOST_RELOCATES = 4            # consecutive failed re-locates before dropping a CONFIRMED target
MAX_STEPS_NO_PROGRESS = 300       # abort if R2R distance-to-goal hasn't improved in this many steps

SIM_TURN_ANGLE = 15               # habitat sim's turn_angle (see habitat/config/.../vln_r2r.yaml)
SCAN_TURN_ANGLE = 45              # degrees the agent rotates between observations while exploring
TURNS_PER_SCAN_STEP = SCAN_TURN_ANGLE // SIM_TURN_ANGLE   # physical sim turns per logical scan step
MAX_SEARCH_TURNS = 360 // SCAN_TURN_ANGLE                 # logical headings = ONE full 360 sweep

TURN_LEFT = 2   # habitat/config/habitat/task/vln_r2r.yaml actions: [stop, move_forward, turn_left, turn_right]
TURN_RIGHT = 3
MOVE_FWD = 1
STOP = 0

LIVE_PREVIEW_PATH = "/home/dungtn21/InternNav/vln_subgoal_pipeline/live_view.jpg"

EXPLORE_DESC = "the most open, unobstructed path forward toward the destination"

# One full 360deg sweep, near-current-heading-first: observe where we already
# are, peek 45deg left, swing 90deg right through center to -45deg, then keep
# turning right through the remaining 5 headings to close the circle. Every
# entry is (turn_action_or_None, physical_turns_from_the_PREVIOUS_plan_step's
# heading), and gets exactly one detect -- no heading is ever re-checked twice.
SEARCH_PLAN = [
    (None, 0),                             # heading    0  (current heading, as-is)
    (TURN_LEFT, TURNS_PER_SCAN_STEP),      # heading  +45
    (TURN_RIGHT, TURNS_PER_SCAN_STEP * 2), # heading  -45 (swing through center)
    (TURN_RIGHT, TURNS_PER_SCAN_STEP),     # heading  -90
    (TURN_RIGHT, TURNS_PER_SCAN_STEP),     # heading -135
    (TURN_RIGHT, TURNS_PER_SCAN_STEP),     # heading  180
    (TURN_RIGHT, TURNS_PER_SCAN_STEP),     # heading  135
    (TURN_RIGHT, TURNS_PER_SCAN_STEP),     # heading   90
]
assert len(SEARCH_PLAN) == MAX_SEARCH_TURNS

# Degrees offset from the sweep-start heading for each SEARCH_PLAN index --
# mirrors the per-entry comments above. Used only by the rare "no heading had
# usable depth anywhere" fallback, to reorient toward Cosmos's reasoned
# heading when there's no grounded 3D point to hand the follower.
SEARCH_PLAN_HEADING_DEG = [0, 45, -45, -90, -135, 180, 135, 90]


def turns_to_heading(from_heading_idx, to_heading_idx):
    """(turn_action, turn_count) of SIM_TURN_ANGLE physical turns to reorient
    from one SEARCH_PLAN heading index to another, shortest direction."""
    delta = SEARCH_PLAN_HEADING_DEG[to_heading_idx] - SEARCH_PLAN_HEADING_DEG[from_heading_idx]
    delta = (delta + 180) % 360 - 180   # normalize to (-180, 180]
    turns = round(abs(delta) / SIM_TURN_ANGLE)
    if turns == 0:
        return None, 0
    return (TURN_LEFT if delta > 0 else TURN_RIGHT), turns


ANTI_BACKTRACK_TOLERANCE_DEG = 45   # a candidate exploring direction within this many degrees
                                     # of the exact reverse of the last chosen exploring direction
                                     # counts as "backtracking" and is excluded where possible


def is_backtracking(candidate_deg, last_explore_deg):
    """True if candidate_deg is close to the exact opposite of last_explore_deg --
    i.e. would send the agent back the way it just came, the oscillation pattern
    that leaves it stuck ping-ponging between the same two spots while exploring."""
    if last_explore_deg is None:
        return False
    reverse_deg = (last_explore_deg + 180) % 360
    diff = abs(candidate_deg - reverse_deg) % 360
    diff = min(diff, 360 - diff)
    return diff < ANTI_BACKTRACK_TOLERANCE_DEG


class AgentState(Enum):
    NAVIGATING = auto()   # cached world point (confirmed subgoal or best-guess), drive to it every step
    SEARCHING = auto()    # working through SEARCH_PLAN, one detect per heading, until found or exhausted


def cosmos3_worker(task_queue, result_queue):
    from vln_subgoal_pipeline.models.cosmos3_reasoner import Cosmos3Reasoner
    reasoner = Cosmos3Reasoner()
    while True:
        task = task_queue.get()
        if task is None:
            break
        mode, payload = task
        if mode == "decompose":
            instruction, image = payload
            result_queue.put(reasoner.decompose(instruction=instruction, image=image))
        elif mode == "detect":
            candidates, image = payload
            result_queue.put(reasoner.detect_landmarks(target_landmarks=candidates, image=image))
        elif mode == "reason":
            memory, target_desc = payload
            result_queue.put(reasoner.reason_best_heading(
                memory=memory, target_desc=target_desc, scan_turn_angle=SCAN_TURN_ANGLE
            ))


def unproject_pixel(x, y, depth_val, sensor_state):
    import quaternion
    WIDTH, HEIGHT = 256, 256
    HFOV = np.pi / 2.0
    fx = (WIDTH / 2.0) / np.tan(HFOV / 2.0)
    fy = fx
    cx, cy = WIDTH / 2.0, HEIGHT / 2.0

    Z = depth_val * 10.0
    X_cam = (x - cx) * Z / fx
    Y_cam = -(y - cy) * Z / fy
    Z_cam = -Z
    local_pt = np.array([X_cam, Y_cam, Z_cam])

    pos = np.array(sensor_state.position)
    rot_matrix = quaternion.as_rotation_matrix(sensor_state.rotation)
    return pos + rot_matrix @ local_pt


def horiz_dist(agent_pos, target_pt):
    """
    Floor-plane (X/Z) distance only -- Habitat is Y-up, and a grounded landmark
    pixel (e.g. a tabletop) usually sits well above floor height. A full 3D
    distance would keep the vertical offset baked in forever, so the agent
    could stand right against the table and still never register "reached".
    "Reached the landmark" means horizontally close to it, not the same height.
    """
    return float(np.linalg.norm([agent_pos[0] - target_pt[0], agent_pos[2] - target_pt[2]]))


def parse_pixel_target(landmark_str, img_shape):
    match = re.search(r"\[(\d+),\s*(\d+)\]", landmark_str)
    if not match:
        return None
    y_norm, x_norm = int(match.group(1)), int(match.group(2))
    # Model-reported norms are nominally [0, 1000] but aren't guaranteed to stay
    # in range (e.g. exactly 1000, or a stray out-of-range value) -- clamp so the
    # pixel indices always stay valid for a img_shape[0] x img_shape[1] image.
    y_pixel = min(max(int((y_norm / 1000.0) * img_shape[0]), 0), img_shape[0] - 1)
    x_pixel = min(max(int((x_norm / 1000.0) * img_shape[1]), 0), img_shape[1] - 1)
    return x_pixel, y_pixel


def valid_depth(depth_img, x_pixel, y_pixel):
    d_val = depth_img[y_pixel, x_pixel, 0]
    if DEPTH_MIN < d_val < DEPTH_MAX / 10.0:
        return d_val
    return None


COLLISION_NEAR = DEPTH_MIN   # normalized depth (~0.5m): a wall/obstacle is right there
COLLISION_CLEAR = 0.15       # normalized depth (~1.5m): comfortably open to walk into


def heading_clearance(depth_img):
    """
    0-1 score for how physically open the path straight ahead is -- from the
    depth image's central forward band, the region the agent will actually
    walk through if it commits to this heading and steps forward. This is
    independent of where the model's pointed-to pixel lands (that pixel might
    be off to one side, e.g. a doorway across the room, while a wall sits
    directly in front). 0 = blocked immediately ahead (or no usable depth at
    all in that band), 1 = comfortably clear.
    """
    h, w = depth_img.shape[:2]
    band = depth_img[h // 2:, w // 3: 2 * w // 3, 0]
    valid = band[band > 0]
    if valid.size == 0:
        return 0.0
    near = float(np.min(valid))
    return float(np.clip((near - COLLISION_NEAR) / (COLLISION_CLEAR - COLLISION_NEAR), 0.0, 1.0))


def detect_current_frame(task_queue, result_queue, target_desc, exhausted,
                          img_pil, rgb_img, depth_img, env):
    """
    One detect_landmarks() call for ONLY the current subgoal's landmark -- no
    cross-subgoal skip-ahead. A generic landmark description (e.g. "desk") can
    easily match a different, unrelated instance of that object elsewhere in
    the scene; checking every remaining subgoal at once and jumping to
    whichever one matched made that false positive skip past every subgoal in
    between. Subgoals are now pursued strictly in order: current one only,
    advance by exactly one only once IT is actually reached.

    Returns a dict:
      found: True if the current subgoal's landmark is literally visible with
        usable depth
      found_world_pt / found_pixel: its unprojected point, or None
      guess_world_pt / guess_pixel: best-guess exploration point (always
        attempted; valid only if the guessed pixel has usable depth)
      guess_label: short phrase naming what Cosmos reports being at that
        guess point (e.g. "open doorway on the left") -- what to show on
        the HUD so it's visible what the model is actually looking at
      guess_confidence: 0-1, Cosmos's own estimate combined with this
        heading's physical clearance (a confident guess pointed past a wall
        directly in the walking path scores low overall)
      visible_names: raw landmark strings Cosmos reported seeing (kept for
        the exploration memory log)
      collision: True if the forward path for this heading is blocked
        within COLLISION_NEAR
      clearance: 0-1 continuous version of the same signal
    """
    candidates = [] if exhausted else [target_desc]
    task_queue.put(("detect", (candidates, img_pil)))
    result = result_queue.get()

    visible_names = [item["landmark"] for item in result.get("visible", [])]

    found = False
    found_world_pt = None
    found_pixel = None
    for item in result.get("visible", []):
        if item["landmark"] != target_desc:
            continue
        px = parse_pixel_target(item["pixel"], rgb_img.shape)
        if px is None:
            continue
        x_pixel, y_pixel = px
        d_val = valid_depth(depth_img, x_pixel, y_pixel)
        if d_val is None:
            continue
        sensor_state = env.sim.get_agent_state().sensor_states["rgb"]
        found = True
        found_world_pt = unproject_pixel(x_pixel, y_pixel, d_val, sensor_state)
        found_pixel = px
        break

    guess_pixel = None
    guess_world_pt = None
    guess_pixel_str = result.get("guess_pixel")
    if guess_pixel_str:
        px = parse_pixel_target(guess_pixel_str, rgb_img.shape)
        if px is not None:
            x_pixel, y_pixel = px
            d_val = valid_depth(depth_img, x_pixel, y_pixel)
            if d_val is not None:
                sensor_state = env.sim.get_agent_state().sensor_states["rgb"]
                guess_world_pt = unproject_pixel(x_pixel, y_pixel, d_val, sensor_state)
                guess_pixel = px

    clearance = heading_clearance(depth_img)
    collision = clearance <= 0.0
    raw_confidence = float(result.get("guess_confidence", 0.0))

    return {
        "found": found,
        "found_world_pt": found_world_pt,
        "found_pixel": found_pixel,
        "guess_world_pt": guess_world_pt,
        "guess_pixel": guess_pixel,
        "guess_label": result.get("guess_label", ""),
        "guess_confidence": raw_confidence * clearance,
        "visible_names": visible_names,
        "collision": collision,
        "clearance": clearance,
    }


def draw_target_marker(display_img, pixel, label, confirmed):
    """Rectangle for a literally-found landmark, circle for a best-guess exploration point."""
    px, py = pixel
    if confirmed:
        half = 20
        cv2.rectangle(display_img, (px - half, py - half), (px + half, py + half), (0, 255, 0), 2)
        cv2.putText(display_img, label[:24], (px - half, py - half - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    else:
        color = (0, 200, 255)
        cv2.circle(display_img, (px, py), 10, color, 2)
        cv2.putText(display_img, label[:24], (px, py - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)



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
    steps_since_relocate = RELOCATE_EVERY
    consecutive_lost_relocates = 0

    best_dist_to_goal = float("inf")
    steps_since_progress = 0

    # What Cosmos most recently reported seeing, for the "Sees:" HUD line --
    # updated at every detect_current_frame() call, held over on frames where
    # no fresh detection happens (mid-turn, or driving between relocates).
    last_seen_found = False
    last_seen_label = "(nothing yet)"
    last_seen_confidence = 0.0

    def get_current_subgoal():
        if current_subgoal_idx >= len(subgoal_queue):
            return None
        return subgoal_queue[current_subgoal_idx]

    def start_new_search():
        nonlocal state, plan_index, plan_subturns_left, search_memory, sweep_start_heading_deg
        state = AgentState.SEARCHING
        plan_index = 0
        plan_subturns_left = SEARCH_PLAN[0][1]
        search_memory = []
        sweep_start_heading_deg = agent_heading_deg

    def reset_pursuit():
        nonlocal cached_target_world_pt, cached_target_pixel, pursuing_confirmed
        nonlocal steps_since_relocate, consecutive_lost_relocates
        cached_target_world_pt = None
        cached_target_pixel = None
        pursuing_confirmed = False
        steps_since_relocate = RELOCATE_EVERY
        consecutive_lost_relocates = 0
        start_new_search()

    def enter_navigating(world_pt, pixel, confirmed):
        nonlocal cached_target_world_pt, cached_target_pixel, pursuing_confirmed
        nonlocal steps_since_relocate, consecutive_lost_relocates, state
        cached_target_world_pt = world_pt
        cached_target_pixel = pixel
        pursuing_confirmed = confirmed
        steps_since_relocate = 0
        consecutive_lost_relocates = 0
        state = AgentState.NAVIGATING

    def record_seen(result):
        """Track what Cosmos most recently reported seeing, for the HUD's 'Sees:' line."""
        nonlocal last_seen_found, last_seen_label, last_seen_confidence
        last_seen_found = result["found"]
        if result["found"]:
            last_seen_label = target_desc
            last_seen_confidence = 1.0
        else:
            last_seen_label = result["guess_label"] or "nothing recognizable"
            last_seen_confidence = result["guess_confidence"]

    def commit_to_found(result):
        """Shared handling for 'the current subgoal's landmark is literally visible',
        whether that happened while SEARCHING: enter NAVIGATING and pick this
        same step's action immediately (advance now if already in reach)."""
        nonlocal action, current_subgoal_idx, last_explore_heading_deg
        enter_navigating(result["found_world_pt"], result["found_pixel"], confirmed=True)
        agent_pos = env.sim.get_agent_state().position
        dist = horiz_dist(agent_pos, cached_target_world_pt)
        if dist < SUBGOAL_REACHED_DIST:
            current_subgoal_idx += 1
            last_explore_heading_deg = None   # fresh subgoal -- drop the anti-backtrack memory
            reset_pursuit()
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
                final_rgb = obs["rgb"]
                final_display_img = cv2.cvtColor(final_rgb.copy(), cv2.COLOR_RGB2BGR)
                final_canvas = np.zeros((256, 512, 3), dtype=np.uint8)
                final_canvas[:, :256] = final_display_img
                
                # Draw STOP text labels on right panel
                cv2.putText(final_canvas, f"Step: {step}", (262, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
                cv2.putText(final_canvas, f"State: STOPPED", (262, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)
                cv2.putText(final_canvas, f"Action: STOP", (262, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)
                
                sg_label = f"SG {current_subgoal_idx+1}/{len(subgoal_queue)}" if not exhausted else "EXPLORING"
                cv2.putText(final_canvas, f"Subgoal: {sg_label}", (262, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
                
                desc_str = f"Desc: {target_desc}"
                if len(desc_str) > 30:
                    cv2.putText(final_canvas, desc_str[:30], (262, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
                    cv2.putText(final_canvas, desc_str[30:60], (262, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
                else:
                    cv2.putText(final_canvas, desc_str, (262, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
                
                seen_color = (0, 255, 0) if last_seen_found else (0, 200, 255)
                cv2.putText(final_canvas, "Sees:", (262, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.42, seen_color, 1)
                
                seen_lbl_str = last_seen_label
                if len(seen_lbl_str) > 30:
                    cv2.putText(final_canvas, seen_lbl_str[:30], (262, 146), cv2.FONT_HERSHEY_SIMPLEX, 0.38, seen_color, 1)
                    cv2.putText(final_canvas, seen_lbl_str[30:60], (262, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.38, seen_color, 1)
                else:
                    cv2.putText(final_canvas, seen_lbl_str, (262, 146), cv2.FONT_HERSHEY_SIMPLEX, 0.38, seen_color, 1)
                
                cv2.putText(final_canvas, f"Conf: {last_seen_confidence:.2f}", (262, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.4, seen_color, 1)
                cv2.putText(final_canvas, f"R2R dist: {final_metrics.get('distance_to_goal', 9999):.1f}m", (262, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                
                cv2.imwrite(LIVE_PREVIEW_PATH, final_canvas)
                frames.append(final_canvas)
                break

            subgoal = get_current_subgoal()
            exhausted = subgoal is None
            # target_desc: human-readable action text, for HUD/logs only.
            # target_landmark: the actual object string sent to Cosmos's detector and
            # matched against its "visible" landmark names -- must NOT be the action
            # sentence ("Turn right at the staircase"), or detect_landmarks() can never
            # literally match it and the subgoal can never be confirmed found.
            if exhausted:
                target_desc = EXPLORE_DESC
                target_landmark = EXPLORE_DESC
            else:
                target_desc = subgoal.get("description", "Unknown")
                target_landmark = subgoal.get("target_landmark") or target_desc
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
                should_relocate = steps_since_relocate >= RELOCATE_EVERY
                if should_relocate:
                    steps_since_relocate = 0
                    result = detect_current_frame(
                        task_queue, result_queue, target_landmark, exhausted,
                        img_pil, rgb_img, depth_img, env
                    )
                    record_seen(result)

                    if result["found"]:
                        cached_target_world_pt = result["found_world_pt"]
                        cached_target_pixel = result["found_pixel"]
                        pursuing_confirmed = True
                        consecutive_lost_relocates = 0
                    elif pursuing_confirmed:
                        # Confirmed target didn't reconfirm this relocate -- don't downgrade
                        # to some unrelated best-guess point on a single miss. Keep driving
                        # toward the stale cached target and tolerate transient occlusion up
                        # to MAX_LOST_RELOCATES before giving up on it.
                        consecutive_lost_relocates += 1
                        if consecutive_lost_relocates >= MAX_LOST_RELOCATES:
                            tqdm.write(f"Step {step} -> Lost '{target_landmark}' for "
                                        f"{consecutive_lost_relocates} relocates, dropping target.")
                            reset_pursuit()
                    elif result["guess_world_pt"] is not None:
                        # Already exploring (not confirmed) -- refresh to the newest guess.
                        cached_target_world_pt = result["guess_world_pt"]
                        cached_target_pixel = result["guess_pixel"]
                        pursuing_confirmed = False
                        consecutive_lost_relocates = 0
                    else:
                        # Was chasing a best-guess point and the refresh came back unusable
                        # (no depth anywhere) -- that guess is stale, start a fresh search.
                        reset_pursuit()
                else:
                    steps_since_relocate += 1

                if cached_target_world_pt is not None:
                    agent_pos = env.sim.get_agent_state().position
                    dist = horiz_dist(agent_pos, cached_target_world_pt)
                    if dist < SUBGOAL_REACHED_DIST:
                        if pursuing_confirmed:
                            tqdm.write(f"Step {step} -> Subgoal '{target_desc}' REACHED "
                                        f"(dist={dist:.2f}m). Advancing queue.")
                            current_subgoal_idx += 1
                            last_explore_heading_deg = None   # fresh subgoal -- drop the anti-backtrack memory
                        else:
                            tqdm.write(f"Step {step} -> Reached exploration point "
                                        f"(dist={dist:.2f}m). Rescanning.")
                        reset_pursuit()
                    else:
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
                        task_queue, result_queue, target_landmark, exhausted,
                        img_pil, rgb_img, depth_img, env
                    )
                    record_seen(result)

                    if result["found"]:
                        tqdm.write(f"Step {step} -> Found '{target_landmark}' while searching "
                                    f"(plan step {plan_index+1}/{len(SEARCH_PLAN)}). Switching to NAVIGATING.")
                        commit_to_found(result)
                    else:
                        search_memory.append({
                            "heading": plan_index,
                            "abs_heading_deg": (sweep_start_heading_deg + SEARCH_PLAN_HEADING_DEG[plan_index]) % 360,
                            "visible_landmarks": result["visible_names"],
                            "guess_confidence": result["guess_confidence"],
                            "collision": result["collision"],
                            "world_pt": result["guess_world_pt"],
                            "pixel": result["guess_pixel"],
                        })
                        depth_note = "" if result["guess_world_pt"] is not None else " [no usable depth]"
                        tqdm.write(f"Step {step} -> Searching (plan step {plan_index+1}/{len(SEARCH_PLAN)}), "
                                    f"confidence={result['guess_confidence']:.2f} "
                                    f"collision={result['collision']} for '{target_landmark[:24]}'{depth_note}")

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
                                    [{"heading": e["heading"], "visible_landmarks": e["visible_landmarks"],
                                      "guess_confidence": e["guess_confidence"], "collision": e["collision"]}
                                     for e in search_memory],
                                    target_landmark,
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
                                    [{"heading": e["heading"], "visible_landmarks": e["visible_landmarks"],
                                      "guess_confidence": e["guess_confidence"], "collision": e["collision"]}
                                     for e in search_memory],
                                    target_landmark,
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

            # ---- HUD (Side-by-side Panel on Right) ----
            state_color = {
                AgentState.NAVIGATING: (0, 255, 0) if pursuing_confirmed else (0, 200, 255),
                AgentState.SEARCHING: (255, 200, 0),
            }.get(state, (255, 255, 255))
            
            # Create a 256x512 canvas: left half is simulator image, right half is black HUD panel
            canvas = np.zeros((256, 512, 3), dtype=np.uint8)
            canvas[:, :256] = display_img
            
            # Draw text labels on the right sidebar (x start = 262)
            cv2.putText(canvas, f"Step: {step}", (262, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
            cv2.putText(canvas, f"State: {state.name}", (262, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.42, state_color, 1)
            cv2.putText(canvas, f"Action: {action}", (262, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
            
            sg_label = f"SG {current_subgoal_idx+1}/{len(subgoal_queue)}" if not exhausted else "EXPLORING"
            cv2.putText(canvas, f"Subgoal: {sg_label}", (262, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
            
            # Break down description into two lines to fit nicely in 256px width
            desc_str = f"Desc: {target_desc}"
            if len(desc_str) > 30:
                cv2.putText(canvas, desc_str[:30], (262, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
                cv2.putText(canvas, desc_str[30:60], (262, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
            else:
                cv2.putText(canvas, desc_str, (262, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
                
            seen_color = (0, 255, 0) if last_seen_found else (0, 200, 255)
            cv2.putText(canvas, "Sees:", (262, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.42, seen_color, 1)
            
            seen_lbl_str = last_seen_label
            if len(seen_lbl_str) > 30:
                cv2.putText(canvas, seen_lbl_str[:30], (262, 146), cv2.FONT_HERSHEY_SIMPLEX, 0.38, seen_color, 1)
                cv2.putText(canvas, seen_lbl_str[30:60], (262, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.38, seen_color, 1)
            else:
                cv2.putText(canvas, seen_lbl_str, (262, 146), cv2.FONT_HERSHEY_SIMPLEX, 0.38, seen_color, 1)
                
            cv2.putText(canvas, f"Conf: {last_seen_confidence:.2f}", (262, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.4, seen_color, 1)
            cv2.putText(canvas, f"R2R dist: {dist_to_goal:.1f}m", (262, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Cosmos3 Habitat closed-loop navigation (strict subgoal order + search-plan exploration).")
    parser.add_argument("--episode_idx", type=int, default=None, help="Index of the episode in the dataset (0 to dataset_size - 1)")
    args = parser.parse_args()
    run_closed_loop(episode_index=args.episode_idx)
