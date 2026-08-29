import argparse
import sys
import os
import json
import re
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from vln_subgoal_pipeline.models.cosmos3_reasoner import Cosmos3Reasoner

class UnifiedVLNPipeline:
    def __init__(self):
        print("Initializing Unified VLN Pipeline (Cosmos3-Edge Only)...")
        # Initialize Cosmos3 as BOTH the Reasoner and the Grounder
        self.reasoner = Cosmos3Reasoner()
        
    def process_instruction(self, instruction: str, image_path: str):
        print(f"\nProcessing Instruction: '{instruction}'")
        print(f"Loading visual observation from: {image_path}")
        
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image: {e}")
            return
            
        width, height = image.size
        print(f"Image dimensions: {width}x{height}")
        
        # 1. Generate Subgoals AND 2D Grounding Coordinates in one pass
        print("Running Cosmos3 Reasoner & Grounder...")
        subgoals = self.reasoner.decompose(instruction=instruction, image=image)
        
        print("\n--- Pipeline Output ---")
        for sg in subgoals:
            print(f"\nSubgoal {sg['id']}: {sg['description']}")
            
            # Parse the coordinates from the target_landmark string
            landmark_text = sg.get("target_landmark", "")
            print(f"  Raw Target: {landmark_text}")
            
            match = re.search(r"\[(\d+),\s*(\d+)\]", landmark_text)
            if match:
                y_norm, x_norm = int(match.group(1)), int(match.group(2))
                # Convert normalized [0, 1000] to actual image pixels
                y_pixel = int((y_norm / 1000.0) * height)
                x_pixel = int((x_norm / 1000.0) * width)
                print(f"  Grounding Coordinates: [y={y_pixel}, x={x_pixel}]")
                # TODO: Pass (x_pixel, y_pixel) to depth_utils to get 3D waypoint for Nav2!
            else:
                print("  Grounding Coordinates: Not found by model.")
                
        print("\nPipeline execution complete.")

def main():
    parser = argparse.ArgumentParser(description="Unified Cosmos3 VLN Pipeline")
    parser.add_argument("--instruction", type=str, required=True, help="Long-horizon instruction")
    parser.add_argument("--image", type=str, required=True, help="Path to current robot RGB frame")
    args = parser.parse_args()
    
    pipeline = UnifiedVLNPipeline()
    pipeline.process_instruction(args.instruction, args.image)

if __name__ == "__main__":
    main()
