import re
import json
import logging
from typing import List, Dict, Any, Optional
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Cosmos3Reasoner")


class Cosmos3Reasoner:
    """
    Subgoal Reasoner powered by NVIDIA Cosmos 3 Edge.
    Decomposes long-horizon VLN instructions into sequential navigable subgoals with distinct landmarks.
    """

    SUBGOAL_SYSTEM_PROMPT = """You are an expert embodied navigation assistant and reasoner.
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
]
Do not output any introductory or concluding text outside the JSON array."""

    def __init__(
        self,
        model_id: str = "nvidia/Cosmos3-Edge",
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        max_new_tokens: int = 512,
        use_mock: bool = False,
    ):
        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.use_mock = use_mock
        self.model = None
        self.processor = None

        if not self.use_mock:
            self._load_model()

    def _load_model(self):
        try:
            import torch
            import os
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            from transformers import AutoModelForImageTextToText, AutoProcessor

            logger.info(f"Loading Cosmos 3 Edge model from: {self.model_id}...")
            dtype = getattr(torch, self.torch_dtype) if hasattr(torch, self.torch_dtype) else torch.bfloat16

            load_kwargs = dict(
                attn_implementation="sdpa",
                trust_remote_code=True,
            )
            if self.device == "cuda" and torch.cuda.is_available():
                try:
                    import bitsandbytes  # noqa: F401
                    load_kwargs["load_in_8bit"] = True
                    load_kwargs["device_map"] = "auto"
                    logger.info("Loading Cosmos 3 in 8-bit quantization to save VRAM...")
                except ImportError:
                    load_kwargs["torch_dtype"] = dtype
                    load_kwargs["device_map"] = "auto"
                    logger.info("bitsandbytes not available, loading in bfloat16...")
            else:
                load_kwargs["torch_dtype"] = dtype

            self.model = AutoModelForImageTextToText.from_pretrained(self.model_id, **load_kwargs)
            self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            self._gpu_device = "cuda" if (self.device == "cuda" and torch.cuda.is_available()) else "cpu"
            logger.info("Cosmos 3 Edge model loaded successfully.")
        except Exception as e:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.warning(
                f"Could not load live Cosmos 3 model ({e}). "
                f"Falling back to high-fidelity rule-based decomposition mode."
            )
            self.use_mock = True

    def _move_model_to_gpu(self):
        pass

    def _move_model_to_cpu(self):
        pass


    def decompose(
        self,
        instruction: str,
        image: Optional[Image.Image] = None,
    ) -> List[Dict[str, Any]]:
        """
        Decompose a long-horizon navigation instruction into subgoals.

        Args:
            instruction: Long-horizon language command (e.g. 'Walk past the dining table into the kitchen and stop by the fridge')
            image: Optional current visual frame from robot RGB camera

        Returns:
            List of subgoal dictionaries containing 'id', 'description', 'target_landmark'.
        """
        if self.use_mock or self.model is None:
            return self._mock_decompose(instruction)

        import torch

        # Prepare messages
        content = []
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({
            "type": "text",
            "text": f"{self.SUBGOAL_SYSTEM_PROMPT}\n\nInstruction: \"{instruction}\"\nDecompose into JSON subgoals:"
        })

        messages = [{"role": "user", "content": content}]

        try:
            # Move to GPU for inference, release VRAM immediately after
            self._move_model_to_gpu()
            try:
                chat_text = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                )
                device = next(self.model.parameters()).device
                if image is not None:
                    inputs = self.processor(text=chat_text, images=image, return_tensors="pt").to(device)
                else:
                    inputs = self.processor(text=chat_text, return_tensors="pt").to(device)

                with torch.no_grad():
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                    )
            finally:
                self._move_model_to_cpu()

            # Strip prompt tokens
            generated_ids = output_ids[:, inputs.input_ids.shape[1] :]
            response_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            return self._parse_json_subgoals(response_text, instruction)
        except Exception as e:
            logger.error(f"Inference error with Cosmos 3: {e}. Using fallback parser.")
            return self._mock_decompose(instruction)

    def _parse_json_subgoals(self, response_text: str, fallback_instruction: str) -> List[Dict[str, Any]]:
        """Extract and sanitize JSON array from model output."""
        try:
            # 1. Look for ```json ... ``` markdown block
            match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", response_text, re.DOTALL)
            if match:
                return json.loads(match.group(1))

            # 2. Look for bare JSON array [ ... ]
            match_raw = re.search(r"(\[.*\])", response_text, re.DOTALL)
            if match_raw:
                return json.loads(match_raw.group(1))

            # 3. Direct parse attempt
            return json.loads(response_text.strip())
        except Exception:
            logger.warning("Failed to parse raw JSON from response, using linguistic chunker.")
            return self._mock_decompose(fallback_instruction)

    def _mock_decompose(self, instruction: str) -> List[Dict[str, Any]]:
        """
        Deterministic linguistic rule-based decomposition for testing / offline fallback.
        Splits on conjunctions ('and', 'then', 'after that', 'before', periods, commas).
        """
        # Split on sentence boundaries and common VLN transition words
        delimiters = r"\b(?:and then|then|after that|after which|and|followed by|\.|\,)\b"
        clauses = [c.strip() for c in re.split(delimiters, instruction, flags=re.IGNORECASE) if len(c.strip()) > 3]

        if not clauses:
            clauses = [instruction.strip()]

        subgoals = []
        for idx, clause in enumerate(clauses):
            # Extract landmark heuristic: pick last noun phrase or important keywords
            words = clause.split()
            # Simple heuristic landmark extraction
            landmark = " ".join(words[-2:]) if len(words) >= 2 else clause
            # Remove leading prepositions from landmark
            landmark = re.sub(r"^(to the|to|at the|at|near the|near|towards the|towards|the|a|an)\s+", "", landmark, flags=re.IGNORECASE)

            subgoals.append({
                "id": idx + 1,
                "description": clause.capitalize(),
                "target_landmark": landmark.strip() if landmark.strip() else clause,
            })

        return subgoals
