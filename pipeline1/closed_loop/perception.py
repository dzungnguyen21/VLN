from .config import SCAN_TURN_ANGLE
from .geometry import heading_clearance, parse_pixel_target, unproject_pixel, valid_depth


def cosmos3_worker(task_queue, result_queue):
    from pipeline1.models.cosmos3_reasoner import Cosmos3Reasoner
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
            candidates, image, guided_direction = payload
            result_queue.put(reasoner.detect_landmarks(
                target_landmarks=candidates, image=image, guided_direction=guided_direction,
            ))
        elif mode == "reason":
            memory, target_desc = payload
            result_queue.put(reasoner.reason_best_heading(
                memory=memory, target_desc=target_desc, scan_turn_angle=SCAN_TURN_ANGLE
            ))


def detect_current_frame(task_queue, result_queue, target_desc, exhausted,
                          img_pil, rgb_img, depth_img, env, guided_direction=None):
    """
    One detect_landmarks() call for ONLY the current subgoal's landmark -- no
    cross-subgoal skip-ahead. A generic landmark description (e.g. "desk") can
    easily match a different, unrelated instance of that object elsewhere in
    the scene; checking every remaining subgoal at once and jumping to
    whichever one matched made that false positive skip past every subgoal in
    between. Subgoals are now pursued strictly in order: current one only,
    advance by exactly one only once IT is actually reached.

    guided_direction: the current subgoal's directional/spatial cue from the
    instruction (e.g. "turn right"), or None/"None" if it has none. Passed
    through as context so Cosmos itself can judge whether a candidate match
    is really the intended instance, not just a same-type lookalike elsewhere
    -- see DETECT_PROMPT_TEMPLATE.

    Returns a dict:
      current_location: Cosmos's own live guess at what room/area the agent
        is currently in, based on this frame alone (e.g. "bedroom") -- for
        the HUD's "Current location:" field
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
    task_queue.put(("detect", (candidates, img_pil, guided_direction)))
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
        "current_location": result.get("current_location", ""),
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
