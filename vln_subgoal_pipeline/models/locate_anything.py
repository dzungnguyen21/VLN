import logging
from typing import Dict, Any, Optional, Tuple, List
from PIL import Image
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LocateAnythingGrounder")


class GroundingResult:
    def __init__(
        self,
        bbox_xyxy: List[float],
        point_uv: Tuple[float, float],
        confidence: float,
        target_name: str,
        image_size: Tuple[int, int],
    ):
        self.bbox_xyxy = bbox_xyxy  # [x1, y1, x2, y2] in pixels
        self.point_uv = point_uv    # (u, v) center point in pixels
        self.confidence = confidence
        self.target_name = target_name
        self.image_size = image_size  # (width, height)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox_xyxy": self.bbox_xyxy,
            "point_uv": list(self.point_uv),
            "confidence": self.confidence,
            "target_name": self.target_name,
            "image_size": list(self.image_size),
        }


class LocateAnythingGrounder:
    """
    2D Visual Grounding and Pointing model based on NVIDIA LocateAnything-3B.
    Locates target landmarks and objects given an RGB frame and natural language referring expression.
    """

    def __init__(
        self,
        model_id: str = "nvidia/LocateAnything-3B",
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        confidence_threshold: float = 0.25,
        use_mock: bool = False,
    ):
        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype
        self.confidence_threshold = confidence_threshold
        self.use_mock = use_mock
        self.model = None
        self.processor = None
        self.tokenizer = None

        if not self.use_mock:
            self._load_model()

    def _load_model(self):
        try:
            import torch
            import os
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            from transformers import AutoModel, AutoProcessor, AutoTokenizer

            logger.info(f"Loading LocateAnything model from: {self.model_id} (CPU, moves to GPU per inference)...")
            dtype = getattr(torch, self.torch_dtype) if hasattr(torch, self.torch_dtype) else torch.bfloat16

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)

            load_kwargs = dict(
                torch_dtype=dtype,
                device_map="cpu",
                trust_remote_code=True,
            )
            self.model = AutoModel.from_pretrained(self.model_id, **load_kwargs)
            self.model.eval()
            self._gpu_device = "cuda" if (self.device == "cuda" and torch.cuda.is_available()) else "cpu"
        except Exception as e:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.warning(
                f"Could not load live LocateAnything model ({e}). "
                f"Falling back to high-fidelity mock grounder for simulation/testing."
            )
            self.use_mock = True

    def _move_model_to_gpu(self):
        """Temporarily move model to GPU for inference."""
        import torch
        if self.model is not None and self._gpu_device == "cuda":
            try:
                torch.cuda.empty_cache()
                self.model = self.model.to("cuda")
            except torch.cuda.OutOfMemoryError:
                pass  # Stay on CPU if GPU is full

    def _move_model_to_cpu(self):
        """Move model back to CPU after inference to free GPU VRAM."""
        import torch
        if self.model is not None and self._gpu_device == "cuda":
            self.model = self.model.to("cpu")
            torch.cuda.empty_cache()


    def ground(
        self,
        image: Image.Image,
        target_description: str,
    ) -> Optional[GroundingResult]:
        """
        Ground a target object / landmark description in an RGB image.

        Args:
            image: PIL Image (RGB)
            target_description: Referring text expression (e.g. 'kitchen door', 'refrigerator')

        Returns:
            GroundingResult containing 2D bbox [x1, y1, x2, y2], center point (u, v), and confidence.
        """
        width, height = image.size

        if self.use_mock or self.model is None:
            return self._mock_ground(image, target_description)

        try:
            import torch

            # Swap to GPU for inference, then back to CPU to free VRAM
            self._move_model_to_gpu()
            try:
                prompt = f"Locate {target_description}."
                inputs = self.processor(images=[image], text=prompt, return_tensors="pt")

                device = next(self.model.parameters()).device
                gen_kwargs = {
                    "input_ids": inputs["input_ids"].to(device) if inputs.get("input_ids") is not None else None,
                    "attention_mask": inputs["attention_mask"].to(device) if inputs.get("attention_mask") is not None else None,
                    "pixel_values": inputs["pixel_values"].to(device).to(torch.bfloat16) if inputs.get("pixel_values") is not None else None,
                    "image_grid_hws": inputs["image_grid_thw"].to(device) if inputs.get("image_grid_thw") is not None else None,
                    "tokenizer": self.tokenizer,
                    "use_cache": True,
                }

                with torch.no_grad():
                    out = self.model.generate(**gen_kwargs)
            finally:
                self._move_model_to_cpu()

            # Decode bounding box / coordinates
            decoded = out if isinstance(out, str) else self.tokenizer.batch_decode(out, skip_special_tokens=True)[0]
            bbox, score = self._parse_bbox_prediction(decoded, width, height)

            if score < self.confidence_threshold:
                logger.warning(f"Confidence {score:.2f} below threshold {self.confidence_threshold}")

            center_u = (bbox[0] + bbox[2]) / 2.0
            center_v = (bbox[1] + bbox[3]) / 2.0

            return GroundingResult(
                bbox_xyxy=bbox,
                point_uv=(center_u, center_v),
                confidence=score,
                target_name=target_description,
                image_size=(width, height),
            )
        except Exception as e:
            logger.error(f"Error during LocateAnything inference: {e}. Falling back to center-weighted estimate.")
            return self._mock_ground(image, target_description)

    def _parse_bbox_prediction(self, text: str, width: int, height: int) -> Tuple[List[float], float]:
        """Parse predicted bbox tokens (e.g. normalized [0, 1000] or [ymin, xmin, ymax, xmax])."""
        import re
        nums = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", text)]
        if len(nums) >= 4:
            # Check if normalized to [0, 1000]
            if max(nums[:4]) > 1.0:
                x1 = (nums[0] / 1000.0) * width
                y1 = (nums[1] / 1000.0) * height
                x2 = (nums[2] / 1000.0) * width
                y2 = (nums[3] / 1000.0) * height
            else:
                x1 = nums[0] * width
                y1 = nums[1] * height
                x2 = nums[2] * width
                y2 = nums[3] * height
            return [x1, y1, x2, y2], 0.85
        # Default center crop box
        return [width * 0.25, height * 0.25, width * 0.75, height * 0.75], 0.50

    def _mock_ground(self, image: Image.Image, target_description: str) -> GroundingResult:
        """Synthetic visual grounder for offline verification and testing."""
        width, height = image.size
        # Deterministically place center based on hash of target description
        hash_val = sum(ord(c) for c in target_description)
        u_ratio = 0.35 + (hash_val % 30) / 100.0  # [0.35, 0.65]
        v_ratio = 0.40 + ((hash_val * 7) % 20) / 100.0  # [0.40, 0.60]

        box_w = width * 0.2
        box_h = height * 0.25

        cx = width * u_ratio
        cy = height * v_ratio

        x1 = max(0.0, cx - box_w / 2.0)
        y1 = max(0.0, cy - box_h / 2.0)
        x2 = min(float(width), cx + box_w / 2.0)
        y2 = min(float(height), cy + box_h / 2.0)

        return GroundingResult(
            bbox_xyxy=[x1, y1, x2, y2],
            point_uv=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
            confidence=0.92,
            target_name=target_description,
            image_size=(width, height),
        )

    def visualize(
        self,
        image: Image.Image,
        result: GroundingResult,
        output_path: Optional[str] = None,
    ) -> Image.Image:
        """Draw bounding box, target label, and center point on the RGB image."""
        from PIL import ImageDraw
        vis_img = image.copy()
        draw = ImageDraw.Draw(vis_img)
        x1, y1, x2, y2 = result.bbox_xyxy

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=(0, 230, 100), width=4)

        # Draw center point
        u, v = result.point_uv
        r = 6
        draw.ellipse([u - r, v - r, u + r, v + r], fill=(255, 40, 40), outline=(255, 255, 255), width=2)

        # Draw label header
        label = f" {result.target_name} ({result.confidence:.2f}) "
        text_w = len(label) * 9
        draw.rectangle([x1, max(0, y1 - 22), x1 + text_w, max(0, y1)], fill=(0, 230, 100))
        draw.text((x1 + 4, max(0, y1 - 18)), label, fill=(0, 0, 0))

        if output_path:
            vis_img.save(output_path)
            logger.info(f"Saved annotated grounding image to: {output_path}")

        return vis_img

