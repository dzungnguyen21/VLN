"""The only module that touches habitat/habitat_sim directly.

Builds one habitat.Env for a whole R2R split (habitat-lab already handles multi-scene episode
datasets and reloads the scene mesh transparently on env.reset() when the next episode's
scene_id differs — verified directly against this checkout, so there's no need for a separate
env per scan or for the untested `habitat.dataset.content_scenes` filter the original design
sketch considered) and drives one episode at a time via ShortestPathFollower, waypoint by
waypoint, yielding one step dict per sim step so callers never hold a whole episode's raw
rgb/depth/semantic frames in memory at once.
"""
from libs import *
from config import (HABITAT_GOAL_RADIUS_FINAL, HABITAT_GOAL_RADIUS_LEG, HABITAT_HFOV,
                    HABITAT_MAX_DEPTH_M, HABITAT_MAX_EPISODE_STEPS, HABITAT_MAX_STEPS_PER_LEG,
                    HABITAT_OUT_H, HABITAT_OUT_W, HABITAT_SCENES_DIR, HABITAT_SNAP_TOLERANCE_M)
from r2r_data import split_path


class RolloutError(ValueError):
    """One episode could not be walked cleanly — caller rejects the episode and moves on."""


def _sensor_overrides(sensor_name):
    prefix = f"habitat.simulator.agents.main_agent.sim_sensors.{sensor_name}"
    overrides = [
        f"{prefix}.width={HABITAT_OUT_W}",
        f"{prefix}.height={HABITAT_OUT_H}",
        # hfov's schema field is Integer (HabitatSimRGBSensorConfig etc.) — passing a float
        # literal like "90.0" fails OmegaConf's strict validation.
        f"{prefix}.hfov={int(round(HABITAT_HFOV))}",
    ]
    if sensor_name == "depth_sensor":
        overrides.append(f"{prefix}.max_depth={HABITAT_MAX_DEPTH_M}")
    return overrides


def build_env(split):
    """One habitat.Env for the whole split, rgb+depth+semantic sensors at the configured
    resolution/hfov. Caller is responsible for env.close()."""
    import habitat
    from habitat.config.default import get_config

    overrides = [
        # Quoted: Hydra's override grammar treats bare '=' specially, and an absolute path
        # can legitimately contain one (e.g. certain network-mounted paths) — quoting is a
        # general defensive move, not specific to any one filesystem layout.
        f"habitat.dataset.data_path='{split_path(split)}'",
        f"habitat.dataset.scenes_dir='{HABITAT_SCENES_DIR}'",
        "habitat/simulator/sensor_setups@habitat.simulator.agents.main_agent=rgbds_agent",
        f"habitat.environment.max_episode_steps={HABITAT_MAX_EPISODE_STEPS}",
        *_sensor_overrides("rgb_sensor"),
        *_sensor_overrides("depth_sensor"),
        *_sensor_overrides("semantic_sensor"),
    ]
    config = get_config("benchmark/nav/vln_r2r.yaml", overrides=overrides)
    return habitat.Env(config=config)


def episode_lookup(env):
    """episode_id (str) -> habitat-lab's own parsed Episode object (goals/instruction as real
    dataclasses, not the plain dicts r2r_data.py parses from the raw split file)."""
    return {str(episode.episode_id): episode for episode in env.episodes}


def select_episode(env, lookup, episode_id):
    episode = lookup.get(episode_id)
    if episode is None:
        raise RolloutError(f"episode_id {episode_id!r} not found in the loaded split")
    env.episode_iterator = iter([episode])
    return episode


def _current_step(env, obs):
    """position/rotation are the RGB SENSOR's own pose (used to build camera_to_world) —
    elevated above the floor by the rig height. agent_position is the agent's own position,
    which sits exactly at floor height (the agent is navmesh-constrained), used as the 3D
    'floor point' when this frame is later selected as someone else's subgoal target — using
    the sensor's elevated position there instead would place every goal ~1.25m above the
    floor it's meant to mark."""
    state = env.sim.get_agent_state()
    sensor_state = state.sensor_states["rgb"]
    return {
        "rgb": obs["rgb"],
        "depth": obs["depth"][..., 0],
        "semantic": obs["semantic"][..., 0],
        "position": np.array(sensor_state.position, dtype=np.float64),
        "rotation": sensor_state.rotation,
        "agent_position": np.array(state.position, dtype=np.float64),
    }


def _snap(env, point):
    snapped = env.sim.pathfinder.snap_point(point)
    return np.array([snapped.x, snapped.y, snapped.z], dtype=np.float64)


def walk_episode(env, episode):
    """Generator yielding one step dict {rgb, depth, semantic, position, rotation} per sim
    step, in order: the reset frame first, then every step taken while chasing each waypoint
    of reference_path[1:] + [goal]. Raises RolloutError (never silently drops a bad episode)
    on: a waypoint too far off the navmesh, a leg exceeding its step budget, the follower
    returning STOP before actually reaching the target (ShortestPathFollower.get_next_action
    returns STOP both on success and, silently, on an internal GreedyFollowerError — verified
    directly — so an early STOP is only trusted after checking the actual distance), or the
    follower failing to find any path at all.
    """
    from habitat.sims.habitat_simulator.actions import HabitatSimActions
    from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

    obs = env.reset()
    yield _current_step(env, obs)

    waypoints = list(episode.reference_path[1:]) + [episode.goals[0].position]
    n_waypoints = len(waypoints)
    final_goal_radius = float(episode.goals[0].radius) if episode.goals[0].radius \
        else HABITAT_GOAL_RADIUS_FINAL

    for leg_index, raw_target in enumerate(waypoints):
        is_final = leg_index == n_waypoints - 1
        goal_radius = final_goal_radius if is_final else HABITAT_GOAL_RADIUS_LEG

        snapped_pt = _snap(env, raw_target)
        snap_dist = float(np.linalg.norm(snapped_pt - np.asarray(raw_target, dtype=np.float64)))
        if snap_dist > HABITAT_SNAP_TOLERANCE_M:
            raise RolloutError(f"waypoint {leg_index} snaps {snap_dist:.2f}m off the navmesh "
                               f"(> {HABITAT_SNAP_TOLERANCE_M}m)")

        follower = ShortestPathFollower(env.sim, goal_radius=goal_radius, return_one_hot=False)
        steps_taken = 0
        while True:
            if env.episode_over:
                raise RolloutError("episode exceeded HABITAT_MAX_EPISODE_STEPS "
                                   f"during leg {leg_index}")
            if steps_taken >= HABITAT_MAX_STEPS_PER_LEG:
                raise RolloutError(f"leg {leg_index} exceeded HABITAT_MAX_STEPS_PER_LEG "
                                   f"({HABITAT_MAX_STEPS_PER_LEG})")

            action = follower.get_next_action(snapped_pt)
            if action is None:
                raise RolloutError(f"leg {leg_index}: follower found no path to the target")
            if action == HabitatSimActions.stop:
                break

            obs = env.step(action)
            steps_taken += 1
            yield _current_step(env, obs)

        reached = np.array(env.sim.get_agent_state().position, dtype=np.float64)
        reached_dist = float(np.linalg.norm(reached - snapped_pt))
        if reached_dist > goal_radius:
            raise RolloutError(f"leg {leg_index}: follower returned STOP early "
                               f"(dist={reached_dist:.2f}m > goal_radius={goal_radius:.2f}m)")
