import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from vln_subgoal_pipeline.test.closed_loop.runner import run_closed_loop

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Cosmos3 Habitat closed-loop navigation (strict subgoal order + search-plan exploration).")
    parser.add_argument("--episode_idx", type=int, default=None, help="Index of the episode in the dataset (0 to dataset_size - 1)")
    args = parser.parse_args()
    run_closed_loop(episode_index=args.episode_idx)
