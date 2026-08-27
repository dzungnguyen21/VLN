import argparse
import sys
import os
import time
from typing import Optional, Dict, Any, Tuple
import numpy as np
from PIL import Image

# Ensure package root is in sys.path
package_root = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(package_root)
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from vln_subgoal_pipeline.pipeline import VLNSubgoalPipeline
from vln_subgoal_pipeline.training.vln_dataset_formatter import VLNDatasetFormatter


def run_real_image_test(
    image_path: str,
    depth_path: Optional[str] = None,
    instruction: Optional[str] = None,
    target: Optional[str] = None,
    output_path: str = "output_grounding.jpg",
    use_mock: bool = False,
):
    print("=" * 65)
    print("  Real Image Grounding & Navigation Test")
    print("=" * 65)

    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return

    # 1. Load real RGB image
    print(f"Loading real RGB image: {image_path}")
    rgb_image = Image.open(image_path).convert("RGB")
    width, height = rgb_image.size
    print(f"   Resolution: {width}x{height}")

    # 2. Load or construct depth
    if depth_path and os.path.exists(depth_path):
        if depth_path.endswith(".npy"):
            depth_map = np.load(depth_path)
        else:
            depth_img = Image.open(depth_path)
            depth_map = np.array(depth_img, dtype=np.float32) / 1000.0  # mm to meters
        print(f"   Loaded custom depth map from {depth_path}")
    else:
        # Default metric depth plane (approx 2.5m for indoor testing)
        depth_map = np.full((height, width), fill_value=2.5, dtype=np.float32)
        print("   Using default metric depth (2.5m plane) for 3D projection.")

    # 3. Single target grounding mode
    if target:
        print(f"\nTarget Landmark: '{target}'")
        pipeline = VLNSubgoalPipeline(use_mock_models=use_mock, use_mock_nav2=True)
        res = pipeline.grounder.ground(rgb_image, target)
        if res:
            print(f"  Detected '{res.target_name}' with confidence {res.confidence:.2f}")
            print(f"  2D Bounding Box: {res.bbox_xyxy}")
            print(f"  Center Point (u, v): ({res.point_uv[0]:.1f}, {res.point_uv[1]:.1f})")

            # Project to 3D
            proj_3d = pipeline.projector_3d.project_2d_to_3d(
                u=res.point_uv[0],
                v=res.point_uv[1],
                depth_map=depth_map,
                confidence=res.confidence,
            )
            if proj_3d:
                print(f"  3D Target Coordinates (Robot Base Frame): x={proj_3d.x_base:.2f}m, y={proj_3d.y_base:.2f}m, z={proj_3d.z_base:.2f}m")
                print(f"  3D Nav Target (Map Frame): x={proj_3d.x_map:.2f}m, y={proj_3d.y_map:.2f}m, yaw={proj_3d.yaw_map:.2f} rad")
                print(f"  Nav2 Standoff Waypoint: x={proj_3d.standoff_x_map:.2f}m, y={proj_3d.standoff_y_map:.2f}m")

            pipeline.grounder.visualize(rgb_image, res, output_path=output_path)
            print(f"  Saved annotated visualization to: {output_path}")
        else:
            print(f"  Landmark '{target}' could not be grounded.")
        return

    # 4. Full Instruction + Subgoal Decomposition with Real Image
    instruction = instruction or "Go to the refrigerator near the kitchen counter and stop."
    print(f"\n Instruction: \"{instruction}\"")

    pipeline = VLNSubgoalPipeline(use_mock_models=use_mock, use_mock_nav2=True)

    def camera_stream():
        return rgb_image, depth_map

    def on_progress(event):
        subgoal = event.get("subgoal", {})
        target_3d = event.get("projected_pose", {}).get("map_frame", {})
        print(f" Subgoal #{subgoal.get('id')}: '{subgoal.get('target_landmark')}' -> 3D Map Target: (x={target_3d.get('x', 0):.2f}, y={target_3d.get('y', 0):.2f})")

    results = pipeline.run(
        instruction=instruction,
        get_rgbd_observation_fn=camera_stream,
        initial_image=rgb_image,
        progress_callback=on_progress,
    )

    # Save visualization of first detected landmark
    if pipeline.controller.execution_log:
        for ev in pipeline.controller.execution_log:
            if ev.step_name == "locate_anything":
                target_name = ev.details.get("target")
                res = pipeline.grounder.ground(rgb_image, target_name)
                if res:
                    pipeline.grounder.visualize(rgb_image, res, output_path=output_path)
                    print(f"\n Saved annotated visualization of '{target_name}' to: {output_path}")
                    break

    print("\n" + "=" * 65)
    print("  Navigation Task Summary")
    print("=" * 65)
    print(f"Status: {'SUCCESS' if results['success'] else 'FAILED'}")
    print(f"Total Subgoals: {len(results['subgoals'])}")
    for sg in results['subgoals']:
        status_icon = "✓" if sg['status'] == "REACHED" else "✗"
        print(f"  [{status_icon}] #{sg['id']} [{sg['status']}]: {sg['description']} (Landmark: {sg['target_landmark']})")
    print("=" * 65)


def run_demo():
    print("=" * 60)
    print("  VLN Pipeline with Subgoal Handling - Demo Runner")
    print("=" * 60)

    # Default sample image
    sample_img_path = os.path.join(os.path.dirname(__file__), "data", "sample_images", "apartment.jpg")
    if os.path.exists(sample_img_path):
        print(f"Found sample real image at: {sample_img_path}")
        run_real_image_test(
            image_path=sample_img_path,
            instruction="Walk past the dining chairs and stop at the refrigerator.",
            output_path="output_grounding.jpg",
        )
        return

    # Fallback synthetic demo
    pipeline = VLNSubgoalPipeline(use_mock_models=False, use_mock_nav2=True)
    instruction = (
        "Leave the living room, walk down the main hallway past the bookshelf, "
        "and stop in the kitchen near the refrigerator."
    )
    print(f"\n[Input Instruction]\n\"{instruction}\"\n")

    def mock_camera_stream():
        rgb = Image.new("RGB", (640, 360), color=(120, 160, 210))
        y, x = np.mgrid[0:360, 0:640]
        depth = 2.0 + (y / 360.0) * 1.0
        return rgb, depth.astype(np.float32)

    def on_progress(event):
        subgoal = event.get("subgoal", {})
        target_3d = event.get("projected_pose", {}).get("map_frame", {})
        print(f"  Subgoal #{subgoal.get('id')}: '{subgoal.get('target_landmark')}' -> 3D Map Target: (x={target_3d.get('x', 0):.2f}, y={target_3d.get('y', 0):.2f}, yaw={target_3d.get('yaw', 0):.2f})")

    results = pipeline.run(
        instruction=instruction,
        get_rgbd_observation_fn=mock_camera_stream,
        progress_callback=on_progress,
    )
    print("\nStatus:", "SUCCESS" if results["success"] else "FAILED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLN Subgoal Pipeline Runner")
    parser.add_argument("--demo", action="store_true", help="Run end-to-end demo execution")
    parser.add_argument("--image", type=str, default=None, help="Path to real RGB image file (.jpg, .png)")
    parser.add_argument("--depth", type=str, default=None, help="Path to real depth file (.npy, .png)")
    parser.add_argument("--instruction", type=str, default=None, help="Custom navigation instruction")
    parser.add_argument("--target", type=str, default=None, help="Specific target landmark to locate (e.g. 'refrigerator')")
    parser.add_argument("--output", type=str, default="output_grounding.jpg", help="Path to save annotated visual result")
    parser.add_argument("--use-mock", action="store_true", help="Force mock models")
    parser.add_argument("--server", action="store_true", help="Start FastAPI model server")
    parser.add_argument("--format-data", action="store_true", help="Generate sample training data for fine-tuning")
    parser.add_argument("--port", type=int, default=8000, help="Port for server")
    parser.add_argument("--r2r", action="store_true", help="Infer sample from R2R dataset")
    parser.add_argument("--r2r-split", type=str, default="val_seen", choices=["val_seen", "val_unseen", "train"], help="R2R dataset split")
    parser.add_argument("--r2r-index", type=int, default=0, help="R2R sample index (0-based)")
    parser.add_argument("--r2r-episode-id", type=int, default=None, help="R2R specific episode ID")
    parser.add_argument("--benchmark-r2r", action="store_true", help="Benchmark pipeline on R2R with SR/NE/OS/SPL")
    parser.add_argument("--benchmark-max-episodes", type=int, default=10, help="Number of episodes for R2R benchmark")
    parser.add_argument("--benchmark-output-json", type=str, default="results/vln_subgoal_pipeline/r2r_benchmark_result.json", help="Path to save benchmark metrics JSON")
    args = parser.parse_args()

    if args.server:
        from vln_subgoal_pipeline.server.api_server import start_server
        start_server(port=args.port)
    elif args.format_data:
        formatter = VLNDatasetFormatter()
        out_path = os.path.join(os.path.dirname(__file__), "data", "synthetic_vln_train.jsonl")
        formatter.create_synthetic_vln_dataset(out_path)
    elif args.r2r or args.r2r_episode_id is not None:
        from vln_subgoal_pipeline.infer_r2r_sample import infer_r2r_sample
        infer_r2r_sample(
            split=args.r2r_split,
            index=args.r2r_index,
            episode_id=args.r2r_episode_id,
            image_path=args.image,
            depth_path=args.depth,
            use_mock=args.use_mock,
            output_image=args.output,
        )
    elif args.benchmark_r2r:
        from vln_subgoal_pipeline.benchmark_r2r import PipelineR2RBenchmarker

        benchmarker = PipelineR2RBenchmarker(
            habitat_config_path="scripts/eval/configs/vln_r2r_lowmem.yaml",
            split=args.r2r_split,
            use_mock_models=args.use_mock,
            max_steps_per_episode=500,
            max_steps_per_subgoal=80,
        )
        try:
            result = benchmarker.run(max_episodes=args.benchmark_max_episodes)
        finally:
            benchmarker.close()

        os.makedirs(os.path.dirname(os.path.abspath(args.benchmark_output_json)) or ".", exist_ok=True)
        with open(args.benchmark_output_json, "w", encoding="utf-8") as f:
            import json

            json.dump(result, f, indent=2)

        print("\nFinal R2R Metrics")
        print(f"  SR:  {result['metrics']['SR']:.4f}")
        print(f"  NE:  {result['metrics']['NE']:.4f}")
        print(f"  OS:  {result['metrics']['OS']:.4f}")
        print(f"  SPL: {result['metrics']['SPL']:.4f}")
        print(f"Saved result json: {args.benchmark_output_json}")
    elif args.image:
        run_real_image_test(
            image_path=args.image,
            depth_path=args.depth,
            instruction=args.instruction,
            target=args.target,
            output_path=args.output,
            use_mock=args.use_mock,
        )
    else:
        run_demo()
