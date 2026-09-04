"""CLI: python main.py build [--force]

Split/scan/episode-count selection is via .env / real env vars (HABITAT_SPLIT, HABITAT_SCENES,
HABITAT_LIMIT_EPISODES — see config.py), matching convert/'s own convention (e.g.
`TARGET_FPS=3 python main.py build`) rather than a separate CLI surface, since those values
are read once at config.py's import time. Only --force (redo work already on disk) is a true
"how to run this batch" flag, mirroring convert/main.py's --force.
"""
from libs import *
from config import HABITAT_DATASET_NAME, HABITAT_SEED, HABITAT_SPLIT, OUT_DIR
import episode as episode_mod
import landmarks
import r2r_data
from rollout import build_env, episode_lookup, select_episode

SEPARATOR = "-" * 16


def build_all(jobs, all_jobs, output_dir, force):
    print(f"\nStart Building {SEPARATOR}")
    print(f"building up to {len(jobs)} episode(s) from split '{HABITAT_SPLIT}' into {output_dir}")

    env = build_env(HABITAT_SPLIT)
    try:
        lookup = episode_lookup(env)

        built = skipped = failed = 0
        total = len(jobs)

        for index, job in enumerate(jobs, start=1):
            episode_index = job["episode_index"]
            label = f"[{index}/{total}] episode_id={job['episode_id']} -> episode {episode_index}"

            if episode_mod.parquet_path(output_dir, episode_index).is_file() and not force:
                skipped += 1
                continue
            if not job["instruction"]:
                skipped += 1
                continue

            try:
                episode_obj = select_episode(env, lookup, job["episode_id"])
                episode = episode_mod.build_episode(env, episode_obj)
                episode_mod.write_episode(episode, output_dir, episode_index, job["task_index"])

                scene = env.sim.semantic_scene
                vocabulary = landmarks.build_scan_vocabulary(scene)
                rows = landmarks.build_detect_rows(
                    episode, scene, vocabulary, output_dir, episode_index,
                    job["scan_id"], job["instruction"], HABITAT_SEED,
                )
                if rows:
                    landmarks.write_detect_rows(rows, output_dir, episode_index)

            except ValueError as exc:
                # RolloutError (bad episode: unwalkable, follower stuck, ...) and the plain
                # ValueError build_episode raises for "too few keyframes" both land here.
                failed += 1
                print(f"{label}: FAILED — {exc}")
                continue

            built += 1
            print(f"{label}: {len(episode['frames'])} frames, {episode['n_goal']} goal, "
                  f"{len(rows)} detect row(s)")

    finally:
        env.close()

    if not list(output_dir.glob("data/chunk-*/episode_*.parquet")):
        print("nothing on disk to describe — meta/ not written")
        return 1

    episode_mod.fix_index_column(output_dir)
    episode_mod.rebuild_meta(output_dir, r2r_data.episode_info_by_index(all_jobs))

    print(f"built {built}, skipped {skipped}, failed {failed} — meta/ written for {output_dir}")
    print(f"Finish Building {SEPARATOR}")

    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build",))
    parser.add_argument("--force", action="store_true", help="redo work already finished")
    args = parser.parse_args()

    all_jobs = r2r_data.load_jobs(HABITAT_SPLIT)
    if not all_jobs:
        print(f"nothing to do: split '{HABITAT_SPLIT}' produced no episodes "
              "(check HABITAT_SCENES / HABITAT_R2R_DATA_ROOT)")
        return 0

    output_dir = OUT_DIR / HABITAT_DATASET_NAME / HABITAT_SPLIT
    return build_all(all_jobs, all_jobs, output_dir, args.force)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ValueError as exc:
        print(f"error: {exc}")
        sys.exit(1)
