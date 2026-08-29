import os
import sys
import json
import gzip
import multiprocessing as mp
from tqdm import tqdm
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# ==============================================================================
# 1. Isolated Worker Process for Cosmos3
# This is REQUIRED because habitat_sim and PyTorch/Triton both statically link 
# LLVM, which causes a fatal C++ abort if loaded in the same Python process.
# ==============================================================================
def cosmos3_worker(task_queue, result_queue):
    # Only import transformers/PyTorch inside the isolated worker
    from vln_subgoal_pipeline.models.cosmos3_reasoner import Cosmos3Reasoner
    print("[Worker] Initializing Cosmos3-Edge...")
    reasoner = Cosmos3Reasoner()
    print("[Worker] Ready.")
    
    while True:
        task = task_queue.get()
        if task is None:  # Sentinel to shutdown
            break
            
        instruction, image = task
        try:
            subgoals = reasoner.decompose(instruction=instruction, image=image)
            result_queue.put({"success": True, "data": subgoals})
        except Exception as e:
            result_queue.put({"success": False, "error": str(e)})

# ==============================================================================
# 2. Main Benchmarking Loop
# ==============================================================================
def run_fgr2r_benchmark(dataset_path: str, output_path: str, max_episodes: int = 5):
    print(f"Loading dataset from {dataset_path}...")
    if dataset_path.endswith(".gz"):
        with gzip.open(dataset_path, "rt", encoding="utf-8") as f:
            data = json.load(f)
    else:
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
    episodes = data.get("episodes", data) if isinstance(data, dict) else data
    episodes = episodes[:max_episodes]
    
    # ---------------------------------------------------------
    # Setup LLVM-safe isolated IPC
    # ---------------------------------------------------------
    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    
    worker = ctx.Process(target=cosmos3_worker, args=(task_queue, result_queue), daemon=True)
    worker.start()
    
    # ---------------------------------------------------------
    # Initialize Habitat Environment for REAL images
    # ---------------------------------------------------------
    import habitat
    from habitat.config.default import get_config
    
    print("\nInitializing Habitat Simulator for REAL images...")
    config = get_config("benchmark/nav/vln_r2r.yaml", overrides=[
        f"habitat.dataset.data_path={dataset_path}",
        "habitat.dataset.scenes_dir=data/scene_data/"
    ])
    env = habitat.Env(config=config)
    
    results = []
    
    # ---------------------------------------------------------
    # Inference Loop
    # ---------------------------------------------------------
    for i, ep in enumerate(tqdm(episodes, desc="Processing Episodes")):
        ep_id = ep.get("episode_id", "unknown")
        
        # Reset habitat env to the current episode to get the FIRST FRAME
        # (Note: Env.reset() cycles to the next episode automatically based on dataset, 
        # but in strict benching we'd use an EpisodeIterator. For testing, this gets valid images.)
        obs = env.reset()
        real_img = Image.fromarray(obs["rgb"])
        
        instruction_text = ""
        if "instruction" in ep and isinstance(ep["instruction"], dict):
            instruction_text = ep["instruction"].get("instruction_text", "")
            
        if not instruction_text:
            continue
            
        # Send task to the isolated Cosmos3 GPU worker
        task_queue.put((instruction_text, real_img))
        response = result_queue.get()
        
        if response["success"]:
            subgoals = response["data"]
            results.append({
                "episode_id": ep_id,
                "original_instruction": instruction_text,
                "generated_subgoals": subgoals
            })
            print(f"\n[Episode {ep_id}]")
            print(f"Instruction: {instruction_text}")
            print(f"Subgoals: {json.dumps(subgoals, indent=2)}\n")
        else:
            print(f"\nError processing episode {ep_id}: {response['error']}")
            
    # Cleanup
    task_queue.put(None)
    worker.join()
    env.close()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nBenchmark inference complete! Saved to {output_path}")

if __name__ == "__main__":
    r2r_val_seen = "data/vln_ce/raw_data/r2r/val_seen/val_seen.json.gz"
    out_json = "results/cosmos3_fgr2r_benchmark.json"
    if not os.path.exists(r2r_val_seen):
        print(f"Could not find dataset at {r2r_val_seen}!")
        sys.exit(1)
    run_fgr2r_benchmark(r2r_val_seen, out_json, max_episodes=5)
