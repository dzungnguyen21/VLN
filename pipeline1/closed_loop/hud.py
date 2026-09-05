import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 16:9 Aspect Ratio Layout Constants
# ---------------------------------------------------------------------------
CANVAS_HEIGHT = 720
CANVAS_WIDTH = 1280       # 16:9 aspect ratio (1280x720)
PANEL_WIDTH_RATIO = 0.30  # 30% of total canvas width for HUD text display

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
FONT_THICKNESS = 1
LINE_HEIGHT = 24
ROW_GAP = 10
PANEL_PADDING = 16


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


def _wrap_text(text, max_width_px, font_scale=FONT_SCALE, thickness=FONT_THICKNESS):
    """Greedy word-wrap so a value always fully fits within max_width_px."""
    words = str(text).split()
    if not words:
        return [""]
    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        (w, _), _ = cv2.getTextSize(candidate, FONT, font_scale, thickness)
        if w > max_width_px:
            lines.append(current)
            current = word
        else:
            current = candidate
    lines.append(current)
    return lines


def render_hud_frame(bgr_display_img, step, state_name, state_color, action_label, action_color,
                      subgoal_achieved, subgoal_items, view_label, view_color,
                      confidence, dist_to_goal, current_location="(unknown)", target_location=""):
    """
    Build the HUD canvas with 16:9 aspect ratio:
    - 70% width on the left for the camera/simulator frame.
    - 30% width on the right for the HUD text display panel.
    - Text automatically wraps to subsequent lines if it exceeds the available width space.

    subgoal_items: the FULL subgoal queue exactly as decomposed at step 0, as a
    list of (description, achieved) tuples -- each line stays green forever
    once achieved, red until then. This is the whole plan and its progress at
    a glance, not just "current"/"next".
    """
    img_h, img_w = bgr_display_img.shape[:2]

    # Enforce 16:9 aspect ratio
    canvas_h = max(img_h, CANVAS_HEIGHT)
    canvas_w = int(round(canvas_h * 16.0 / 9.0))
    panel_w = int(round(canvas_w * PANEL_WIDTH_RATIO))  # exactly 30% width
    img_area_w = canvas_w - panel_w                     # remaining 70% width

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # Scale and center camera image inside the left 70% area
    scale = min(img_area_w / img_w, canvas_h / img_h)
    scaled_w = max(1, int(round(img_w * scale)))
    scaled_h = max(1, int(round(img_h * scale)))
    scaled_img = cv2.resize(bgr_display_img, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

    pad_x = (img_area_w - scaled_w) // 2
    pad_y = (canvas_h - scaled_h) // 2
    canvas[pad_y:pad_y + scaled_h, pad_x:pad_x + scaled_w] = scaled_img

    # Vertical separator line between image area and HUD panel
    cv2.line(canvas, (img_area_w, 0), (img_area_w, canvas_h), (45, 45, 45), 2)

    # Right HUD text panel area
    fields = [
        ("Step:", str(step), (255, 255, 255)),
        ("State:", state_name, state_color),
        ("Action:", str(action_label), action_color),
        ("Current subgoal state:", str(subgoal_achieved),
         (0, 255, 0) if subgoal_achieved else (0, 100, 255)),
        ("Current location:", str(current_location or "(unknown)"), (200, 200, 200)),
        ("Target location:", str(target_location or ""), (200, 200, 200)),
        ("View:", view_label, view_color),
        ("Conf:", f"{confidence:.2f}", view_color),
        ("R2R dist:", f"{dist_to_goal:.1f}m", (255, 255, 255)),
    ]

    panel_start_x = img_area_w + PANEL_PADDING
    max_text_w = panel_w - (2 * PANEL_PADDING)
    y = 36

    for label, value, color in fields:
        prefix = f"{label} "
        (prefix_w, _), _ = cv2.getTextSize(prefix, FONT, FONT_SCALE, FONT_THICKNESS)

        # Check if value can be wrapped next to prefix or needs subsequent lines
        available_val_w = max_text_w - prefix_w
        if available_val_w > 60:
            lines = _wrap_text(value, available_val_w, FONT_SCALE, FONT_THICKNESS)
            cv2.putText(canvas, f"{prefix}{lines[0]}", (panel_start_x, y),
                        FONT, FONT_SCALE, color, FONT_THICKNESS, cv2.LINE_AA)
            for extra_line in lines[1:]:
                y += LINE_HEIGHT
                cv2.putText(canvas, extra_line, (panel_start_x + prefix_w, y),
                            FONT, FONT_SCALE, color, FONT_THICKNESS, cv2.LINE_AA)
        else:
            # If label itself is wide, print label on first line and wrap value on next lines
            cv2.putText(canvas, label, (panel_start_x, y),
                        FONT, FONT_SCALE, color, FONT_THICKNESS, cv2.LINE_AA)
            lines = _wrap_text(value, max_text_w, FONT_SCALE, FONT_THICKNESS)
            for line in lines:
                y += LINE_HEIGHT
                cv2.putText(canvas, line, (panel_start_x + 12, y),
                            FONT, FONT_SCALE, color, FONT_THICKNESS, cv2.LINE_AA)

        y += LINE_HEIGHT + ROW_GAP

    # Subgoal queue: the full plan exactly as decomposed at step 0, one line
    # per subgoal, numbered in order. Green once achieved, red until then --
    # never any other color, and never reordered/removed.
    cv2.putText(canvas, "Subgoals:", (panel_start_x, y), FONT, FONT_SCALE, (255, 255, 255),
                FONT_THICKNESS, cv2.LINE_AA)
    y += LINE_HEIGHT + ROW_GAP
    for i, (desc, achieved) in enumerate(subgoal_items):
        color = (0, 255, 0) if achieved else (0, 0, 255)
        prefix = f"{i + 1}. "
        (prefix_w, _), _ = cv2.getTextSize(prefix, FONT, FONT_SCALE, FONT_THICKNESS)
        lines = _wrap_text(desc, max_text_w - prefix_w, FONT_SCALE, FONT_THICKNESS)
        cv2.putText(canvas, f"{prefix}{lines[0]}", (panel_start_x, y),
                    FONT, FONT_SCALE, color, FONT_THICKNESS, cv2.LINE_AA)
        for extra_line in lines[1:]:
            y += LINE_HEIGHT
            cv2.putText(canvas, extra_line, (panel_start_x + prefix_w, y),
                        FONT, FONT_SCALE, color, FONT_THICKNESS, cv2.LINE_AA)
        y += LINE_HEIGHT + ROW_GAP

    return canvas
