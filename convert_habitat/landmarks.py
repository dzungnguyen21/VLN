"""Task B/C: object-pointing + scene understanding. Builds rows matching
Cosmos3Reasoner.DETECT_PROMPT_TEMPLATE's exact response JSON, from the SAME kept keyframes
Task D already produced (episode.py's build_episode) — no duplicate render, and every
image_path here points at exactly the JPEG episode.save_frame_images already wrote.
"""
from libs import *
from config import (HABITAT_EXCLUDE_CATEGORIES, HABITAT_GUESS_MAX_LOOKAHEAD,
                    HABITAT_MAX_INSTANCE_FRACTION, HABITAT_MIN_INSTANCE_PIXELS,
                    HABITAT_N_CANDIDATES_MAX, HABITAT_N_NEGATIVES, HABITAT_N_POSITIVES,
                    HABITAT_OUT_H, HABITAT_OUT_W)
from geometry import instance_pixel_stats, project_to_pixel, region_for_point, to_norm_yx
from episode import CHUNK_SIZE, rgb_image_path, write_jsonl


def build_scan_vocabulary(scene):
    """Bare category-name strings (e.g. 'cushion'), never instance ids — matches
    DETECT_PROMPT_TEMPLATE's candidates convention (a landmark string with no instance
    disambiguation)."""
    return {obj.category.name() for obj in scene.objects} - HABITAT_EXCLUDE_CATEGORIES


def frame_positives(scene, semantic_frame):
    """category -> (pixel_count, row, col) of the largest-footprint qualifying instance of
    that category visible in this frame. Instances under HABITAT_MIN_INSTANCE_PIXELS are
    dropped (avoids a teeny barely-in-frame technically-true positive); instances over
    HABITAT_MAX_INSTANCE_FRACTION of the frame are also dropped (verified directly: MP3D's
    mesh reconstruction around mirrors/glass can bleed a nearby object's semantic id into an
    implausibly large, spatially incoherent footprint — training on that would teach the
    model to point at a reconstruction artifact, not the real object)."""
    n_objects = len(scene.objects)
    max_pixels = HABITAT_MAX_INSTANCE_FRACTION * semantic_frame.size
    best = {}
    for instance_id in np.unique(semantic_frame).tolist():
        if instance_id < 0 or instance_id >= n_objects:
            continue
        stats = instance_pixel_stats(semantic_frame, instance_id)
        if stats is None:
            continue
        count, row, col = stats
        if count < HABITAT_MIN_INSTANCE_PIXELS or count > max_pixels:
            continue
        category = scene.objects[instance_id].category.name()
        if category in HABITAT_EXCLUDE_CATEGORIES:
            continue
        if category not in best or count > best[category][0]:
            best[category] = (count, row, col)
    return best


def sample_candidates(scene, vocabulary, semantic_frame, rng):
    """(candidates[], visible[]) for one frame. Negatives are sampled uniformly from the
    scan vocabulary MINUS every category actually visible this frame (not just the capped
    positives kept below) — otherwise a visible-but-uncapped category could be offered as a
    negative and directly contradict the label. `visible` is built only from categories that
    survive the final HABITAT_N_CANDIDATES_MAX cap, so it never references a candidate that
    isn't actually in the returned candidate list."""
    positives_all = frame_positives(scene, semantic_frame)
    visible_categories = set(positives_all.keys())

    positive_pool = list(positives_all.keys())
    rng.shuffle(positive_pool)
    chosen_positives = positive_pool[:HABITAT_N_POSITIVES]

    negative_pool = list(vocabulary - visible_categories)
    rng.shuffle(negative_pool)
    chosen_negatives = negative_pool[:HABITAT_N_NEGATIVES]

    candidates = chosen_positives + chosen_negatives
    rng.shuffle(candidates)
    candidates = candidates[:HABITAT_N_CANDIDATES_MAX]

    visible = []
    for category in candidates:
        if category in positives_all:
            count, row, col = positives_all[category]
            visible.append({
                "landmark": category,
                "pixel_norm": to_norm_yx((col, row), HABITAT_OUT_W, HABITAT_OUT_H),
            })
    return candidates, visible


def region_label(region):
    return region.category.name() if region is not None else None


def build_guess_label(frames, frame_index, goal_frame, scene):
    """Deterministic template from the region-category transition between the current
    frame's region and the guess frame's region — no model call."""
    if goal_frame == len(frames) - 1:
        return "toward the destination"

    current_region = region_for_point(scene, frames[frame_index]["floor_position"])
    goal_region = region_for_point(scene, frames[goal_frame]["floor_position"])
    if current_region is None or goal_region is None:
        return "toward open floor space ahead"
    if current_region is goal_region:
        return f"keep moving toward the destination within this {region_label(current_region)}"
    return f"toward the {region_label(goal_region)}"


def build_guess(frames, subgoal_frames, frame_index, scene, intrinsics):
    """guess_pixel_norm/guess_label/guess_confidence — the model's best-guess exploration
    point, using Task D's own subgoal frames as ground truth of "where the oracle path goes
    next". Unlike Task D's tight near-term goal (MIN_GOAL_K/MAX_GOAL_FRAMES bounded), this
    walks arbitrarily far ahead through later subgoals until one actually projects on-image,
    matching DETECT_PROMPT_TEMPLATE's framing of a coarser, longer-horizon guide."""
    if frame_index == len(frames) - 1:
        # The last frame has no subgoal ahead of it BY CONSTRUCTION (subgoal_frames never
        # exceeds len(frames)-1) — that's not "couldn't find a path", it's "already arrived".
        # Verified directly: without this, the true final frame of an episode got the same
        # "unable to see a path forward" / 0.0 confidence as a genuine failure, which are
        # semantically opposite situations.
        return None, "arrived at the destination", 0.0

    later_subgoals = [sg for sg in subgoal_frames if sg > frame_index]
    for skip_count, goal_frame in enumerate(later_subgoals):
        pixel = project_to_pixel(frames[frame_index]["pose"],
                                 frames[goal_frame]["floor_position"], intrinsics)
        if pixel is None:
            continue
        lookahead = goal_frame - frame_index - 1
        confidence = max(0.3, min(1.0, 1.0 - lookahead / HABITAT_GUESS_MAX_LOOKAHEAD)) \
            * (0.85 ** skip_count)
        guess_pixel_norm = to_norm_yx(pixel, HABITAT_OUT_W, HABITAT_OUT_H)
        guess_label = build_guess_label(frames, frame_index, goal_frame, scene)
        return guess_pixel_norm, guess_label, round(float(confidence), 3)

    return None, "unable to see a path forward", 0.0


def detect_path(scene_dir, episode_index):
    return (Path(scene_dir) / "detect" / f"chunk-{episode_index // CHUNK_SIZE:03d}"
            / f"episode_{episode_index:06d}.jsonl")


def build_detect_rows(episode, scene, vocabulary, scene_dir, episode_index, scan_id,
                      instruction, seed):
    """One JSONL row per kept frame with at least one candidate — frames with zero eligible
    positives are KEPT (candidates = pure negatives, visible=[]), since "correctly report
    nothing is visible" is exactly the case worth training. A frame is skipped only if even
    the scan-wide vocabulary can't fill any candidates at all (near-empty/mono-category scan)."""
    frames = episode["frames"]
    subgoal_frames = episode["subgoal_frames"]
    intrinsics = episode["intrinsics"]
    rng = random.Random((seed, episode_index))

    rows = []
    for frame_index, frame in enumerate(frames):
        candidates, visible = sample_candidates(scene, vocabulary, frame["semantic"], rng)
        if not candidates:
            continue

        guess_pixel_norm, guess_label, guess_confidence = build_guess(
            frames, subgoal_frames, frame_index, scene, intrinsics)

        rows.append({
            "episode_index": episode_index,
            "frame_index": frame_index,
            "image_path": str(rgb_image_path(scene_dir, episode_index, frame_index)
                              .relative_to(Path(scene_dir))),
            "scan_id": scan_id,
            "instruction": instruction,
            "guided_direction": "None",
            "candidates": candidates,
            "visible": visible,
            "guess_pixel_norm": guess_pixel_norm,
            "guess_label": guess_label,
            "guess_confidence": guess_confidence,
            "current_location": region_label(region_for_point(scene, frame["floor_position"])) or "",
        })
    return rows


def write_detect_rows(rows, scene_dir, episode_index):
    path = detect_path(scene_dir, episode_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(path, rows)
