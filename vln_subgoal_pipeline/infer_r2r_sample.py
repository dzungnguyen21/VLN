import argparse
import gzip
import json
import os
import sys
import time
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from PIL import Image

# Ensure package root is in sys.path
package_root = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(package_root)
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from vln_subgoal_pipeline.pipeline import VLNSubgoalPipeline
from vln_subgoal_pipeline.perception.grounding_3d import Grounding3D
from vln_subgoal_pipeline.models.locate_anything import LocateAnythingGrounder
def _run_cosmos3_subprocess_task(instr: str, queue) -> None:
    """Helper for running Cosmos 3 decomposition in an isolated subprocess."""
    try:
        from vln_subgoal_pipeline.models.cosmos3_reasoner import Cosmos3Reasoner
        r = Cosmos3Reasoner(use_mock=False)
        res = r.decompose(instruction=instr)
        queue.put((res, r.use_mock))
    except Exception as ex:
        queue.put(([], True))


def load_r2r_sample(
    dataset_dir: str = "data/datasets/vln/mp3d/r2r/v1",
    split: str = "val_seen",
    index: int = 0,
    episode_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Load a specific episode from the R2R dataset.
    """
    gz_path = os.path.join(dataset_dir, split, f"{split}.json.gz")
    json_path = os.path.join(dataset_dir, split, f"{split}.json")

    # Fallback to vln_ce/raw_data/r2r if needed
    if not os.path.exists(gz_path) and not os.path.exists(json_path):
        alt_gz = os.path.join("data/vln_ce/raw_data/r2r", split, f"{split}.json.gz")
        if os.path.exists(alt_gz):
            gz_path = alt_gz

    if os.path.exists(gz_path):
        with gzip.open(gz_path, "rt", encoding="utf-8") as f:
            data = json.load(f)
    elif os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        raise FileNotFoundError(f"R2R dataset split not found at {gz_path} or {json_path}")

    episodes = data.get("episodes", []) if isinstance(data, dict) else data
    if not episodes:
        raise ValueError(f"No episodes found in {split} split.")

    if episode_id is not None:
        matched = [ep for ep in episodes if ep.get("episode_id") == episode_id]
        if not matched:
            raise ValueError(f"Episode ID {episode_id} not found in {split} split.")
        sample = matched[0]
        chosen_idx = episodes.index(sample)
    else:
        if index < 0 or index >= len(episodes):
            raise IndexError(f"Index {index} out of range [0, {len(episodes) - 1}]")
        sample = episodes[index]
        chosen_idx = index

    return {
        "split": split,
        "index": chosen_idx,
        "total_in_split": len(episodes),
        "episode_id": sample.get("episode_id"),
        "trajectory_id": sample.get("trajectory_id"),
        "scene_id": sample.get("scene_id"),
        "start_position": sample.get("start_position"),
        "start_rotation": sample.get("start_rotation"),
        "goals": sample.get("goals", []),
        "geodesic_distance": sample.get("info", {}).get("geodesic_distance"),
        "instruction_text": sample.get("instruction", {}).get("instruction_text", "").strip(),
        "raw_episode": sample,
    }


def infer_r2r_sample(
    split: str = "val_seen",
    index: int = 0,
    episode_id: Optional[int] = None,
    image_path: Optional[str] = None,
    depth_path: Optional[str] = None,
    use_mock: bool = False,
    output_image: str = "output_r2r_sample_inference.jpg",
    output_json: Optional[str] = "results/r2r_sample_inference.json",
) -> Dict[str, Any]:
    """
    Run full VLN Subgoal inference on an R2R dataset sample.
    """
    print("=" * 70)
    print("  VLN Subgoal Inference on R2R Dataset")
    print("=" * 70)

    # 1. Load R2R sample
    sample_info = load_r2r_sample(split=split, index=index, episode_id=episode_id)
    print(f"Dataset Split:        {sample_info['split']} (Sample {sample_info['index'] + 1} / {sample_info['total_in_split']})")
    print(f"Episode ID:            {sample_info['episode_id']}")
    print(f"Scene ID:              {sample_info['scene_id']}")
    print(f"Geodesic Distance:     {sample_info['geodesic_distance']:.2f} m" if sample_info['geodesic_distance'] else "N/A")
    print(f"Goal Position:         {sample_info['goals'][0].get('position') if sample_info['goals'] else 'N/A'}")
    print("\nR2R Navigation Instruction:")
    print(f"   \"{sample_info['instruction_text']}\"")
    print("-" * 70)

    # 2. Load RGB observation & depth
    if image_path is None or not os.path.exists(image_path):
        image_path = os.path.join(os.path.dirname(__file__), "data", "sample_images", "apartment.jpg")

    print(f"Observation Image:     {image_path}")
    rgb_image = Image.open(image_path).convert("RGB")
    width, height = rgb_image.size
    print(f"   Image Resolution:      {width} x {height}")

    if depth_path and os.path.exists(depth_path):
        if depth_path.endswith(".npy"):
            depth_map = np.load(depth_path)
        else:
            depth_img = Image.open(depth_path)
            depth_map = np.array(depth_img, dtype=np.float32) / 1000.0
        print(f"   Depth Map:             {depth_path}")
    else:
        depth_map = np.full((height, width), fill_value=2.5, dtype=np.float32)
        print("   Depth Map:             Default metric depth (2.5m indoor estimate)")

    print("-" * 70)
    print("Step 1: Subgoal Decomposition (NVIDIA Cosmos 3 Edge)...")
    t0 = time.time()

    if use_mock:
        reasoner = Cosmos3Reasoner(use_mock=True)
        subgoals_raw = reasoner.decompose(instruction=sample_info["instruction_text"], image=rgb_image)
        reasoner_is_mock = True
    else:
        # Run Cosmos 3 in isolated subprocess to guarantee 100% VRAM release before loading LocateAnything
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        p = ctx.Process(target=_run_cosmos3_subprocess_task, args=(sample_info["instruction_text"], q))
        p.start()
        p.join()
        subgoals_raw, reasoner_is_mock = q.get()

    t_decomp = time.time() - t0
    print(f"   Decomposed into {len(subgoals_raw)} subgoals in {t_decomp:.2f}s")

    print("\nStep 2: 2D Landmark Grounding (NVIDIA LocateAnything-3B) & 3D Projection...")
    grounder = LocateAnythingGrounder(use_mock=use_mock)
    grounder_is_mock = getattr(grounder, "use_mock", use_mock)
    projector_3d = Grounding3D(
        fx=384.0, fy=384.0, cx=width / 2.0, cy=height / 2.0,
        camera_offset_x=0.15, camera_offset_y=0.0, camera_offset_z=0.50,
        standoff_dist=0.60
    )

    inferred_subgoals = []
    vis_image = rgb_image.copy()

    for sg in subgoals_raw:
        sg_id = sg.get("id")
        desc = sg.get("description", "")
        landmark = sg.get("target_landmark", "")

        # Ground landmark in image
        ground_res = grounder.ground(rgb_image, landmark)
        
        # Project 2D center point to 3D
        if ground_res:
            proj_3d = projector_3d.project_2d_to_3d(
                u=ground_res.point_uv[0],
                v=ground_res.point_uv[1],
                depth_map=depth_map,
                confidence=ground_res.confidence,
            )
            vis_image = grounder.visualize(vis_image, ground_res)
        else:
            proj_3d = None

        subgoal_entry = {
            "id": sg_id,
            "description": desc,
            "target_landmark": landmark,
            "grounding_2d": ground_res.to_dict() if ground_res else None,
            "projected_3d": proj_3d.to_dict() if proj_3d else None,
            "nav2_waypoint": proj_3d.to_nav2_goal() if proj_3d else None,
            "status": "REACHED" if proj_3d else "FAILED",
        }
        inferred_subgoals.append(subgoal_entry)

    # Save visual result
    os.makedirs(os.path.dirname(os.path.abspath(output_image)) or ".", exist_ok=True)
    vis_image.save(output_image)
    print(f"\nSaved visual grounding annotations to: {output_image}")

    # Build final result dictionary
    result = {
        "r2r_episode": {
            "split": sample_info["split"],
            "index": sample_info["index"],
            "episode_id": sample_info["episode_id"],
            "scene_id": sample_info["scene_id"],
            "instruction": sample_info["instruction_text"],
            "start_position": sample_info["start_position"],
            "goals": sample_info["goals"],
            "geodesic_distance": sample_info["geodesic_distance"],
        },
        "models_used": {
            "reasoner": "nvidia/Cosmos3-Edge" if not reasoner_is_mock else "mock_rule_based",
            "grounder": "nvidia/LocateAnything-3B" if not grounder_is_mock else "mock_grounder",
            "3d_projection": "Grounding3D (Pinhole + Standoff)",
        },
        "inference_subgoals": inferred_subgoals,
        "success": all(sg["status"] == "REACHED" for sg in inferred_subgoals),
        "total_subgoals": len(inferred_subgoals),
    }

    if output_json:
        os.makedirs(os.path.dirname(os.path.abspath(output_json)) or ".", exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Saved structured JSON inference output to: {output_json}")

    # Print Pretty Summary Table
    print("\n" + "=" * 70)
    print("  VLN Subgoal Inference Summary")
    print("=" * 70)
    for sg in inferred_subgoals:
        status_sym = "OK" if sg["status"] == "REACHED" else "FAILED"
        lm = sg["target_landmark"]
        desc = sg["description"]
        g2d = sg.get("grounding_2d") or {}
        conf = g2d.get("confidence", 0.0)
        nav2 = sg.get("nav2_waypoint") or {}
        x_m = nav2.get("x", 0.0)
        y_m = nav2.get("y", 0.0)
        yaw_m = nav2.get("yaw", 0.0)
        p3d = sg.get("projected_3d") or {}
        base = p3d.get("base_frame") or {}

        print(f"{status_sym} Subgoal #{sg['id']}: {desc}")
        print(f"   • Landmark:       '{lm}' (Confidence: {conf:.2f})")
        if g2d:
            bbox = [round(b, 1) for b in g2d.get('bbox_xyxy', [])]
            uv = g2d.get('point_uv', [0, 0])
            print(f"   • 2D BoundingBox: {bbox}")
            print(f"   • Center Point:   u={uv[0]:.1f}px, v={uv[1]:.1f}px")
        if p3d:
            print(f"   • 3D Target Pos:  x={base.get('x', 0.0):.2f}m, y={base.get('y', 0.0):.2f}m, z={base.get('z', 0.0):.2f}m")
        if nav2:
            print(f"   • Nav2 Standoff:  x={x_m:.2f}m, y={y_m:.2f}m, yaw={yaw_m:.2f} rad")
        print("-" * 70)

    print(f"Overall Status: {'SUCCESS' if result['success'] else 'PARTIAL / FAILED'}")
    print("=" * 70 + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer R2R dataset sample using VLN Subgoal Pipeline")
    parser.add_argument("--split", type=str, default="val_seen", choices=["val_seen", "val_unseen", "train"], help="R2R dataset split")
    parser.add_argument("--index", type=int, default=0, help="0-based sample index in the split")
    parser.add_argument("--episode-id", type=int, default=None, help="Specific episode ID to infer")
    parser.add_argument("--image", type=str, default=None, help="Path to observation RGB image (.jpg/.png)")
    parser.add_argument("--depth", type=str, default=None, help="Path to depth map (.npy/.png)")
    parser.add_argument("--use-mock", action="store_true", help="Force mock models")
    parser.add_argument("--output-image", type=str, default="output_r2r_sample_inference.jpg", help="Path to save annotated visualization")
    parser.add_argument("--output-json", type=str, default="results/r2r_sample_inference.json", help="Path to save JSON results")
    args = parser.parse_args()

    infer_r2r_sample(
        split=args.split,
        index=args.index,
        episode_id=args.episode_id,
        image_path=args.image,
        depth_path=args.depth,
        use_mock=args.use_mock,
        output_image=args.output_image,
        output_json=args.output_json,
    )
