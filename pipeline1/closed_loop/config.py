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
MAX_UNCONFIRMED_RELOCATES = 4     # consecutive relocates spent chasing the SAME unconfirmed best-guess
                                   # point before giving up on it and re-searching -- without this, a
                                   # guess that keeps refreshing to a near-identical but never-quite-
                                   # reachable point (e.g. something seen through a window/doorway the
                                   # navmesh can't get within SUBGOAL_REACHED_DIST of) pursues forever,
                                   # since "still returns a guess" is not the same as "making progress".
MAX_STEPS_NO_PROGRESS = 300       # abort if R2R distance-to-goal hasn't improved in this many steps

SIM_TURN_ANGLE = 15               # habitat sim's turn_angle (see habitat/config/.../vln_r2r.yaml)
SCAN_TURN_ANGLE = 45              # degrees the agent rotates between observations while exploring
TURNS_PER_SCAN_STEP = SCAN_TURN_ANGLE // SIM_TURN_ANGLE   # physical sim turns per logical scan step
MAX_SEARCH_TURNS = 360 // SCAN_TURN_ANGLE                 # logical headings = ONE full 360 sweep

TURN_LEFT = 2   # habitat/config/habitat/task/vln_r2r.yaml actions: [stop, move_forward, turn_left, turn_right]
TURN_RIGHT = 3
MOVE_FWD = 1
STOP = 0

LIVE_PREVIEW_PATH = "/home/dungtn21/InternNav/pipeline1/live_view.jpg"
VIDEO_OUTPUT_DIR = "/home/dungtn21/InternNav/pipeline1"

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

ANTI_BACKTRACK_TOLERANCE_DEG = 45   # a candidate exploring direction within this many degrees
                                     # of the exact reverse of the last chosen exploring direction
                                     # counts as "backtracking" and is excluded where possible

COLLISION_NEAR = DEPTH_MIN   # normalized depth (~0.5m): a wall/obstacle is right there
COLLISION_CLEAR = 0.15       # normalized depth (~1.5m): comfortably open to walk into
