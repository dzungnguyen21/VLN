from libs import *
from config import *
from bag import TRAJ_FILENAME, Bag
from episode import (
    build_episode,
    check_goal_reach,
    parquet_path,
    rebuild_meta,
    require_rig,
    write_episode,
)
from track import track_bag


BAG_NAME_PATTERN = re.compile(r"^zed_record_run_(\d+)_(\d+)$")
SEPARATOR = "-" * 16


def load_instructions():
    entries = json.loads(INSTRUCTION_LOG.read_text(encoding="utf-8"))

    return {
        str(entry.get("id", "")).strip(): str(entry.get("instruction") or "").strip()
        for entry in entries
    }


def load_jobs():
    instructions = load_instructions()
    jobs = []

    for folder in sorted(RECORD_DIR.glob("zed_record_run_*")):
        match = BAG_NAME_PATTERN.match(folder.name)
        if match is None:
            continue

        run_number = int(match.group(1))
        take_number = int(match.group(2))

        run_id = f"run_{run_number:02d}"
        task_index = run_number - 1
        repeat_index = take_number - 1

        if not 0 <= repeat_index < TAKES_PER_INSTRUCTION:
            continue

        if run_id not in instructions:
            continue

        jobs.append({
            "run_id": run_id,
            "name": folder.name,
            "instruction": instructions[run_id],
            "task_index": task_index,
            "repeat_index": repeat_index,
            "episode_index": task_index * TAKES_PER_INSTRUCTION + repeat_index,
            "record_dir": folder,
            "traj_path": RESULTS_DIR / folder.name / TRAJ_FILENAME,
        })

    return jobs


def filter_jobs(jobs, run_filter):
    wanted = {
        run if run.startswith("run_") else f"run_{int(run):02d}"
        for run in run_filter
    }

    return [job for job in jobs if job["run_id"] in wanted]

def episode_info_by_index(jobs):
    return {
        job["episode_index"]: {
            "instruction": job["instruction"],
            "task_index": job["task_index"],
            "repeat_index": job["repeat_index"],
            "run_id": job["run_id"],
        }
        for job in jobs
        if job["instruction"]
    }

def track_all(jobs, force):
    print(f"\nStart Tracking {SEPARATOR}")

    trackable_jobs = [
        job
        for job in jobs
        if job["record_dir"].is_dir()
        and any(job["record_dir"].glob("*.db3"))
    ]

    tracked = 0
    skipped = 0
    failed = 0
    total = len(trackable_jobs)

    for index, job in enumerate(trackable_jobs, start=1):
        label = f"[{index}/{total}] {job['name']}"

        if job["traj_path"].is_file() and not force:
            skipped += 1
            continue

        try:
            result = track_bag(job["record_dir"], job["traj_path"].parent)
        except Exception as exc:
            failed += 1
            print(f"{label}: FAILED — {str(exc).strip()}")
            continue

        tracked += 1
        print(f"{label}: {result['n_poses']} poses ({result['fps']:.1f} fps), path "
              f"{result['path_length']:.2f} units, {result['n_failures']} failures")

    print(f"tracked {tracked}, skipped {skipped}, failed {failed}")
    print(f"Finish Tracking {SEPARATOR}")

    return 1 if failed else 0


def build_all(jobs, all_jobs, force):
    print(f"\nStart Building {SEPARATOR}")

    require_rig()

    output_dir = OUT_DIR / DATASET_NAME / SCENE
    buildable_jobs = [
        job
        for job in jobs
        if job["traj_path"].is_file()
    ]

    if not buildable_jobs:
        return 1

    print(f"building {len(buildable_jobs)} episode(s) into {output_dir}")

    built = 0
    skipped = 0
    failed = 0
    total = len(buildable_jobs)
    collected_warnings = []

    for index, job in enumerate(buildable_jobs, start=1):
        episode_index = job["episode_index"]
        label = f"[{index}/{total}] {job['name']} -> episode {episode_index}"

        if parquet_path(output_dir, episode_index).is_file() and not force:
            skipped += 1
            continue

        if not job["instruction"]:
            skipped += 1
            continue

        try:
            override = SCALE_OVERRIDES.get(job["name"], {})

            with Bag.open(job["record_dir"]) as bag:
                episode = build_episode(bag, job["traj_path"], traj_scale=override.get("traj_scale"), path_m=override.get("path_length"))
                write_episode(bag, episode, output_dir, episode_index, job["task_index"])

        except Exception as exc:
            failed += 1
            print(f"{label}: FAILED — {str(exc).strip()}")
            continue

        built += 1

        print(f"{label}: {len(episode['stamps'])} frames, {episode['n_goal']} goal, "
              f"{episode['speed']:.2f} m/s ({episode['reason']})")

        collected_warnings.extend(
            f"{job['name']}: {warning}"
            for warning in episode["warnings"]
        )

    if not list(output_dir.glob("data/chunk-*/episode_*.parquet")):
        print("nothing on disk to describe — meta/ not written")
        return 1

    rebuild_meta(
        output_dir,
        episode_info_by_index(all_jobs),
    )

    print(
        f"built {built}, skipped {skipped}, failed {failed} — "
        f"meta/ written for {output_dir}"
    )
    print(f"Finish Building {SEPARATOR}")

    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("track", "build", "all"),
    )
    parser.add_argument(
        "--runs",
        default=None,
        help="comma-separated run ids, e.g. 1,2,7",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="redo work already finished",
    )
    args = parser.parse_args()

    all_jobs = load_jobs()
    jobs = all_jobs

    if args.runs:
        selected_runs = [
            run
            for run in args.runs.replace(" ", ",").split(",")
            if run
        ]
        jobs = filter_jobs(all_jobs, selected_runs)

    if not jobs:
        print("nothing to do: no bag in the instruction log matched")
        return 0

    exit_code = 0

    if args.command in ("track", "all"):
        exit_code = max(
            exit_code,
            track_all(jobs, args.force),
        )

    if args.command in ("build", "all"):
        exit_code = max(
            exit_code,
            build_all(jobs, all_jobs, args.force),
        )

    return exit_code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ValueError as exc:
        print(f"error: {exc}")
        sys.exit(1)
