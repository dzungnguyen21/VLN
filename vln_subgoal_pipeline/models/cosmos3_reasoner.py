import re
import json
import logging
from typing import List, Dict, Any, Optional
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Cosmos3Reasoner")
logging.getLogger("httpx").setLevel(logging.WARNING)


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
Do not output any introductory or concluding text outside the JSON array. Do not show your reasoning process."""

    DETECT_PROMPT_TEMPLATE = """You are looking at a single camera frame during robot navigation.
Candidate target landmarks, in priority order (LAST is highest priority): {landmarks}

Respond with ONLY this JSON:
{{"visible": [{{"landmark": "<exact string from the candidate list>", "pixel": [y, x]}}], "guess_pixel": [y, x], "guess_label": "<short phrase>", "guess_confidence": <float 0.0-1.0>}}

- "visible": every candidate landmark that is CLEARLY visible in this frame, each with its center
  pixel normalized to [0, 1000]. Use the landmark string EXACTLY as given. Empty list if none of
  the candidates are visible.
- "guess_pixel" / "guess_label" / "guess_confidence" are REQUIRED in every response, even when
  "visible" is empty: your best guess at the single most promising point in this frame to move
  toward in search of the candidates -- an open doorway, the mouth of a hallway, unobstructed
  floor space, etc. "guess_label" is a short phrase (3-6 words) naming WHAT is actually at that
  pixel (e.g. "open doorway on the left", "hallway leading forward", "a couch blocking the path").
  "guess_confidence" is how confident you are that moving toward that pixel leads toward the
  candidates (0.0 = no idea, 1.0 = very confident). Base all of this on visible cues: doorways,
  hallways, the general room layout, or partial glimpses of similar objects.

No explanation, no reasoning, no extra text."""

    REASONING_PROMPT_TEMPLATE = """You are an embodied navigation agent. You just completed a full \
360-degree scan without finding your target and need to pick a direction to explore next.

Current objective: '{target_desc}'

Here is what was observed at each heading during the scan (heading 0 is where the scan started; \
each next heading is {scan_turn_angle} degrees further, turning left):
{memory_summary}

Reason about which heading is most likely to eventually lead to the objective -- consider visible \
landmarks, hallway/doorway cues, and general room layout. Avoid headings marked collision=true \
(a wall/obstacle blocks that direction immediately). Respond with ONLY this JSON:
{{"best_heading": <int>, "reason": "<one short sentence>"}}"""

    def __init__(
        self,
        model_id: str = "nvidia/Cosmos3-Edge",
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_new_tokens: int = 2048,
        use_mock: bool = False,
        enable_thinking: bool = False,
    ):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.use_mock = use_mock
        self.enable_thinking = enable_thinking
        self.model = None
        self.processor = None
        self._supports_enable_thinking_kwarg = None

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
            dtype = getattr(torch, self.dtype) if hasattr(torch, self.dtype) else torch.bfloat16

            load_kwargs = dict(attn_implementation="sdpa", trust_remote_code=True)
            if self.device == "cuda" and torch.cuda.is_available():
                load_kwargs["dtype"] = dtype
                load_kwargs["device_map"] = "cuda"
            else:
                load_kwargs["dtype"] = dtype

            self.model = AutoModelForImageTextToText.from_pretrained(self.model_id, **load_kwargs)
            self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            self._gpu_device = "cuda" if (self.device == "cuda" and torch.cuda.is_available()) else "cpu"
            logger.info("Cosmos 3 Edge model loaded successfully.")
        except Exception as e:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.error(f"Could not load live Cosmos 3 model: {e}")
            raise

    def _move_model_to_gpu(self):
        pass

    def _move_model_to_cpu(self):
        pass

    def _apply_chat_template(self, messages):
        if self._supports_enable_thinking_kwarg is not False:
            try:
                text = self.processor.apply_chat_template(
                    messages, add_generation_prompt=True, enable_thinking=self.enable_thinking,
                )
                self._supports_enable_thinking_kwarg = True
                return text
            except TypeError:
                self._supports_enable_thinking_kwarg = False
                logger.warning("Chat template does not accept 'enable_thinking'; using default template.")
        return self.processor.apply_chat_template(messages, add_generation_prompt=True)

    @staticmethod
    def _strip_thinking(text: str) -> str:
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        cleaned = re.sub(r"^.*?</think>", "", cleaned, flags=re.DOTALL)
        return cleaned.strip()

    def decompose(self, instruction: str, image: Optional[Image.Image] = None) -> List[Dict[str, Any]]:
        if self.model is None:
            raise RuntimeError("Cosmos3Reasoner model is not loaded.")
        import torch

        content = []
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({
            "type": "text",
            "text": f"{self.SUBGOAL_SYSTEM_PROMPT}\n\nInstruction: \"{instruction}\"\nDecompose into JSON subgoals:"
        })
        messages = [{"role": "user", "content": content}]

        try:
            self._move_model_to_gpu()
            last_response_text = ""
            for attempt in range(3):
                chat_text = self._apply_chat_template(messages)
                device = next(self.model.parameters()).device
                if image is not None:
                    inputs = self.processor(text=chat_text, images=image, return_tensors="pt").to(device)
                else:
                    inputs = self.processor(text=chat_text, return_tensors="pt").to(device)

                with torch.no_grad():
                    output_ids = self.model.generate(
                        **inputs, max_new_tokens=self.max_new_tokens, do_sample=True, temperature=0.4,
                    )

                generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
                response_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                last_response_text = response_text

                if not response_text.strip():
                    logger.warning(f"Empty output (attempt {attempt+1}/3). Retrying...")
                    continue

                cleaned = self._strip_thinking(response_text)
                if not cleaned:
                    logger.warning(f"Output was entirely reasoning, no final answer (attempt {attempt+1}/3). Retrying...")
                    continue

                try:
                    return self._parse_json_subgoals(cleaned, instruction)
                except ValueError:
                    logger.warning(f"Parse failed on attempt {attempt+1}/3, retrying...")
                    continue

            # All 3 attempts produced malformed/truncated JSON (e.g. the model ran out
            # of its length budget mid-object). Rather than discarding everything for a
            # single hardcoded placeholder subgoal (which has no target_landmark and can
            # never be found -- see detect_current_frame), salvage whichever COMPLETE
            # {id, description, target_landmark} objects are present in the last attempt.
            salvaged = self._salvage_json_subgoals(last_response_text)
            if salvaged:
                logger.warning(f"Failed after 3 attempts to get well-formed JSON; salvaged "
                                f"{len(salvaged)} complete subgoal(s) from the last attempt.")
                return salvaged
            logger.error(f"Failed after 3 attempts and nothing salvageable. Last output: {last_response_text[:500]}...")
            return [{"id": 1, "description": "Explore the area (Error)", "target_landmark": ""}]
        except Exception as e:
            logger.error(f"Inference error with Cosmos 3: {e}")
            return [{"id": 1, "description": "Explore the area (Error)", "target_landmark": ""}]
        finally:
            self._move_model_to_cpu()

    def detect_landmarks(self, target_landmarks: List[str], image: Image.Image) -> Dict[str, Any]:
        """
        Single call per frame checking ALL candidate landmarks at once (priority-by-visibility
        is resolved by the caller from the "visible" list, not here).

        Returns: {"visible": [{"landmark": str, "pixel": "[y, x]"}], "guess_pixel": "[y, x]" or
        None, "guess_label": str, "guess_confidence": float 0.0-1.0}
        - "visible" lists every candidate literally seen in this frame, with its pixel.
        - "guess_pixel"/"guess_label"/"guess_confidence" are populated even when nothing is
          visible: the model's best guess at the most promising point to move toward in search
          of the candidates, a short phrase naming what's actually there (e.g. "open doorway"),
          and how confident it is in that guess. guess_pixel is None only if the model failed to
          return a usable response at all.
        """
        if self.model is None:
            raise RuntimeError("Cosmos3Reasoner model is not loaded.")
        import torch

        landmarks_text = (
            "; ".join(target_landmarks) if target_landmarks
            else "(none specific right now -- just find the most promising open path forward)"
        )
        content = [
            {"type": "image", "image": image},
            {"type": "text", "text": self.DETECT_PROMPT_TEMPLATE.format(landmarks=landmarks_text)},
        ]
        messages = [{"role": "user", "content": content}]

        default_result = {"visible": [], "guess_pixel": None, "guess_label": "", "guess_confidence": 0.0}

        try:
            self._move_model_to_gpu()
            chat_text = self._apply_chat_template(messages)
            device = next(self.model.parameters()).device
            inputs = self.processor(text=chat_text, images=image, return_tensors="pt").to(device)

            with torch.no_grad():
                output_ids = self.model.generate(**inputs, max_new_tokens=384, do_sample=False)

            generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
            response_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            cleaned = self._strip_thinking(response_text)

            if not cleaned:
                return default_result

            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if not match:
                return default_result

            try:
                parsed = json.loads(match.group(0))
                visible = []
                for item in parsed.get("visible", []) or []:
                    landmark = item.get("landmark")
                    px = item.get("pixel")
                    if landmark and isinstance(px, list) and len(px) == 2:
                        visible.append({"landmark": str(landmark), "pixel": f"[{int(px[0])}, {int(px[1])}]"})
                guess_pixel = None
                gp = parsed.get("guess_pixel")
                if isinstance(gp, list) and len(gp) == 2:
                    guess_pixel = f"[{int(gp[0])}, {int(gp[1])}]"
                guess_label = str(parsed.get("guess_label", ""))
                guess_confidence = float(parsed.get("guess_confidence", 0.0))
            except Exception as parse_err:
                # Small/edge models frequently drop a comma in the nested "visible"
                # array (e.g. `"landmark": "x" "pixel": [..]`). Rather than losing
                # the whole observation -- which starves every heading of a usable
                # guess_pixel and leaves the agent spinning with nothing to act on --
                # salvage whatever fields are recoverable via regex.
                logger.debug(f"JSON parsing failed for detect_landmarks response: '{cleaned}'. Error: {parse_err}")
                visible = [
                    {"landmark": m.group(1), "pixel": f"[{int(m.group(2))}, {int(m.group(3))}]"}
                    for m in re.finditer(
                        r'"landmark"\s*:\s*"([^"]*)"[^{}\[\]]*?"pixel"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]',
                        cleaned, re.DOTALL,
                    )
                    if m.group(1)
                ]
                gp_match = re.search(r'"guess_pixel"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]', cleaned)
                guess_pixel = f"[{int(gp_match.group(1))}, {int(gp_match.group(2))}]" if gp_match else None
                gl_match = re.search(r'"guess_label"\s*:\s*"([^"]*)"', cleaned)
                guess_label = gl_match.group(1) if gl_match else ""
                gc_match = re.search(r'"guess_confidence"\s*:\s*(\d+\.?\d*)', cleaned)
                guess_confidence = float(gc_match.group(1)) if gc_match else 0.0

            guess_confidence = max(0.0, min(1.0, guess_confidence))
            return {
                "visible": visible, "guess_pixel": guess_pixel,
                "guess_label": guess_label, "guess_confidence": guess_confidence,
            }

        except Exception as e:
            logger.error(f"Inference error in detect_landmarks(): {e}")
            return default_result
        finally:
            self._move_model_to_cpu()

    def reason_best_heading(
        self, memory: List[Dict[str, Any]], target_desc: str, scan_turn_angle: int = 45
    ) -> Dict[str, Any]:
        """
        Text-only reasoning call over the per-heading memory collected during a completed
        360-degree scan (each entry: heading index, visible_landmarks, guess_confidence,
        collision). Asks Cosmos to choose which heading to commit to next.

        Returns: {"best_heading": int, "reason": str}. Caller is responsible for validating
        that best_heading is in range and not marked collision=true before using it.
        """
        if self.model is None:
            raise RuntimeError("Cosmos3Reasoner model is not loaded.")
        import torch

        lines = []
        for entry in memory:
            landmarks = ", ".join(entry.get("visible_landmarks") or []) or "none"
            lines.append(
                f"Heading {entry['heading']}: visible landmarks=[{landmarks}], "
                f"confidence={entry.get('guess_confidence', 0.0):.2f}, "
                f"collision={entry.get('collision', False)}"
            )
        memory_summary = "\n".join(lines)
        prompt_text = self.REASONING_PROMPT_TEMPLATE.format(
            target_desc=target_desc, scan_turn_angle=scan_turn_angle, memory_summary=memory_summary,
        )
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]

        default_result = {"best_heading": 0, "reason": ""}

        try:
            self._move_model_to_gpu()
            chat_text = self._apply_chat_template(messages)
            device = next(self.model.parameters()).device
            inputs = self.processor(text=chat_text, return_tensors="pt").to(device)

            with torch.no_grad():
                output_ids = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)

            generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
            response_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            cleaned = self._strip_thinking(response_text)

            if not cleaned:
                return default_result

            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if not match:
                return default_result

            try:
                parsed = json.loads(match.group(0))
                best_heading = int(parsed.get("best_heading", 0))
                reason = str(parsed.get("reason", ""))
            except Exception as parse_err:
                logger.debug(f"JSON parsing failed for reason_best_heading response: '{cleaned}'. Error: {parse_err}")
                bh_match = re.search(r'"best_heading"\s*:\s*(\d+)', cleaned)
                if not bh_match:
                    return default_result
                best_heading = int(bh_match.group(1))
                reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', cleaned)
                reason = reason_match.group(1) if reason_match else ""

            return {"best_heading": best_heading, "reason": reason}

        except Exception as e:
            logger.error(f"Inference error in reason_best_heading(): {e}")
            return default_result
        finally:
            self._move_model_to_cpu()

    def _parse_json_subgoals(self, response_text: str, fallback_instruction: str) -> List[Dict[str, Any]]:
        try:
            match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", response_text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            match_raw = re.search(r"(\[.*\])", response_text, re.DOTALL)
            if match_raw:
                return json.loads(match_raw.group(1))
            return json.loads(response_text.strip())
        except Exception as e:
            # Expected/recoverable: decompose() retries on ValueError and logs its own
            # WARNING for this, so this isn't itself a real error -- debug-log only, to
            # match the same convention used in detect_landmarks/reason_best_heading.
            logger.debug(f"Failed to parse JSON from cleaned response: '{response_text[:300]}'. Error: {e}")
            raise ValueError(f"Invalid JSON from model: {response_text}") from e

    @staticmethod
    def _salvage_json_subgoals(response_text: str) -> List[Dict[str, Any]]:
        """
        Last-resort recovery when every retry attempt in decompose() still produced
        malformed/truncated JSON (e.g. the model ran out of its length budget mid-object,
        leaving a dangling unterminated string/object at the end). Pulls out whichever
        COMPLETE {id, description, target_landmark} objects are present, in order, and
        discards the truncated tail. Returns [] if nothing complete is recoverable.
        """
        return [
            {"id": int(m.group("id")), "description": m.group("description"),
             "target_landmark": m.group("landmark")}
            for m in re.finditer(
                r'\{\s*"id"\s*:\s*(?P<id>\d+)\s*,\s*"description"\s*:\s*"(?P<description>(?:[^"\\]|\\.)*)"\s*,\s*'
                r'"target_landmark"\s*:\s*"(?P<landmark>(?:[^"\\]|\\.)*)"\s*\}',
                response_text, re.DOTALL,
            )
        ]