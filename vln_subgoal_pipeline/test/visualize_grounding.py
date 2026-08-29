import os
import sys
import json
import re
import cv2
import multiprocessing as mp
import numpy as np
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

def cosmos3_worker(task_queue, result_queue):
    from vln_subgoal_pipeline.models.cosmos3_reasoner import Cosmos3Reasoner
    reasoner = Cosmos3Reasoner()
    while True:
        task = task_queue.get()
        if task is None: break
        subgoals = reasoner.decompose(instruction=task[0], image=task[1])
        result_queue.put(subgoals)

def draw_visual_grounding():
    print("Initializing IPC Cosmos Worker...")
    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    worker = ctx.Process(target=cosmos3_worker, args=(task_queue, result_queue), daemon=True)
    worker.start()
    
    print("Loading Habitat for real image generation...")
    import habitat
    from habitat.config.default import get_config
    
    dataset_path = "vln_subgoal_pipeline/FGR2R/FGR2R_val_seen.json.gz"
    import gzip
    with gzip.open(dataset_path, "rt", encoding="utf-8") as f:
        data = json.load(f)
        
    ep = data[3]
    ep_id = ep.get("path_id", 3)
    instruction_text = ep["instructions"][0]
    
    config = get_config("benchmark/nav/vln_r2r.yaml", overrides=[
        "habitat.dataset.data_path=data/vln_ce/raw_data/r2r/val_seen/val_seen.json.gz",
        "habitat.dataset.scenes_dir=data/scene_data/"
    ])
    env = habitat.Env(config=config)
    
    for _ in range(4):
        obs = env.reset()
        
    rgb_img = obs["rgb"]
    img = Image.fromarray(rgb_img)
    env.close()
    
    print(f"Instruction: {instruction_text}")
    print("Running Cosmos3-Edge...")
    task_queue.put((instruction_text, img))
    subgoals = result_queue.get()
    
    print("Raw output subgoals:")
    print(json.dumps(subgoals, indent=2))
    
    draw_img = cv2.cvtColor(rgb_img.copy(), cv2.COLOR_RGB2BGR)
    height, width = draw_img.shape[:2]
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]
    
    for i, sg in enumerate(subgoals):
        landmark = sg.get("target_landmark", "")
        match = re.search(r"\[(\d+),\s*(\d+)\]", landmark)
        if match:
            # NV Cosmos uses [y, x] in [0, 1000]
            val1 = int(match.group(1))
            val2 = int(match.group(2))
            
            y_pixel = int((val1 / 1000.0) * height)
            x_pixel = int((val2 / 1000.0) * width)
            
            color = colors[i % len(colors)]
            cv2.circle(draw_img, (x_pixel, y_pixel), 6, color, -1)
            cv2.putText(draw_img, str(sg["id"]), (x_pixel + 8, y_pixel - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
    out_path = f"/home/dungtn21/.gemini/antigravity/brain/2caa08d5-870a-4966-bf56-0d6f33c0e83c/grounding_ep{ep_id}.jpg"
    cv2.imwrite(out_path, draw_img)
    print(f"Saved visualization to {out_path}")
    
    task_queue.put(None)
    worker.join()

if __name__ == "__main__":
    draw_visual_grounding()
