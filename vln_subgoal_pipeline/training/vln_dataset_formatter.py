import json
import re
import os
import logging
from typing import List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VLNDatasetFormatter")


class VLNDatasetFormatter:
    """
    Converts raw long-horizon navigation datasets (e.g. InternData-N1, R2R, RxR)
    into formatted (Instruction, Subgoal List) supervision pairs for fine-tuning Cosmos 3 Reasoner.
    """

    SYSTEM_PROMPT = """You are an expert embodied navigation assistant and reasoner.
Given a long-horizon navigation instruction and the current visual observation of the environment, your task is to decompose the long instruction into a chronological sequence of fine-grained, navigable subgoals.

Each subgoal must:
1. Be a clear intermediate step towards the final destination.
2. Specify the prominent visual landmark or object to ground and navigate towards.
3. Be ordered logically from start to completion.

You MUST respond strictly with a valid JSON array of objects with the following schema:
[
  {
    "id": 1,
    "description": "<actionable step description>",
    "target_landmark": "<specific visual landmark or object for 2D/3D grounding>"
  },
  ...
]"""

    def format_sample(
        self,
        instruction: str,
        subgoals: List[Dict[str, Any]],
        image_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Format a single training sample into chat template JSON."""
        messages = [
            {
                "role": "user",
                "content": (
                    f"{self.SYSTEM_PROMPT}\n\n"
                    f"Instruction: \"{instruction}\"\n"
                    f"Decompose into JSON subgoals:"
                ),
            },
            {
                "role": "assistant",
                "content": json.dumps(subgoals, indent=2),
            },
        ]
        sample = {"messages": messages}
        if image_path:
            sample["image_path"] = image_path
        return sample

    def convert_r2r_data(self, r2r_json_path: str, output_jsonl_path: str) -> int:
        """
        Extract and convert standard R2R (Room-to-Room) instructions into training pairs.
        """
        if not os.path.exists(r2r_json_path):
            logger.warning(f"File not found: {r2r_json_path}")
            return 0

        with open(r2r_json_path, "r") as f:
            data = json.load(f)

        formatted_samples = []
        for item in data:
            instructions = item.get("instructions", [])
            for inst in instructions:
                # Rule-based auto-labeling for bootstrapping training data
                clauses = [c.strip() for c in re.split(r"\b(?:and then|then|after that|and|\.|\,)\b", inst, flags=re.IGNORECASE) if len(c.strip()) > 3]
                subgoals = []
                for idx, c in enumerate(clauses):
                    words = c.split()
                    landmark = " ".join(words[-2:]) if len(words) >= 2 else c
                    subgoals.append({
                        "id": idx + 1,
                        "description": c.capitalize(),
                        "target_landmark": landmark,
                    })

                sample = self.format_sample(instruction=inst, subgoals=subgoals)
                formatted_samples.append(sample)

        os.makedirs(os.path.dirname(os.path.abspath(output_jsonl_path)), exist_ok=True)
        with open(output_jsonl_path, "w") as f:
            for s in formatted_samples:
                f.write(json.dumps(s) + "\n")

        logger.info(f"Saved {len(formatted_samples)} formatted training samples to {output_jsonl_path}")
        return len(formatted_samples)

    def create_synthetic_vln_dataset(self, output_jsonl_path: str) -> int:
        """Create sample dataset for verification and training pipeline testing."""
        examples = [
            (
                "Leave the bedroom, turn right into the hallway, walk past the mirror, and enter the kitchen near the refrigerator.",
                [
                    {"id": 1, "description": "Exit the bedroom door into hallway", "target_landmark": "bedroom door"},
                    {"id": 2, "description": "Turn right and walk down hallway", "target_landmark": "hallway corridor"},
                    {"id": 3, "description": "Walk past the decorative mirror", "target_landmark": "wall mirror"},
                    {"id": 4, "description": "Enter the kitchen and stop near the refrigerator", "target_landmark": "refrigerator"},
                ]
            ),
            (
                "Walk straight across the living room, go between the sofa and coffee table, and stop by the sliding glass door.",
                [
                    {"id": 1, "description": "Walk straight across the living room area", "target_landmark": "living room center"},
                    {"id": 2, "description": "Navigate between sofa and coffee table", "target_landmark": "coffee table"},
                    {"id": 3, "description": "Approach and stop by the sliding glass door", "target_landmark": "sliding glass door"},
                ]
            ),
            (
                "Head towards the dining room, walk around the wooden dining table, and stop at the sideboard cabinet.",
                [
                    {"id": 1, "description": "Head towards the dining room entrance", "target_landmark": "dining room doorway"},
                    {"id": 2, "description": "Walk around the wooden dining table", "target_landmark": "dining table"},
                    {"id": 3, "description": "Approach the sideboard cabinet", "target_landmark": "sideboard cabinet"},
                ]
            ),
        ]

        formatted = [self.format_sample(inst, subs) for inst, subs in examples]
        os.makedirs(os.path.dirname(os.path.abspath(output_jsonl_path)), exist_ok=True)
        with open(output_jsonl_path, "w") as f:
            for s in formatted:
                f.write(json.dumps(s) + "\n")

        logger.info(f"Generated {len(formatted)} synthetic training samples at {output_jsonl_path}")
        return len(formatted)
