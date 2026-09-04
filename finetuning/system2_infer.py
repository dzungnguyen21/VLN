from libs import *
import re

from config import (SYSTEM2_ACTION_NAMES, SYSTEM2_ACTION_PROMPT_TEMPLATE, SYSTEM2_MAX_NEW_TOKENS,
                    SYSTEM2_PROMPT_TEMPLATE)

# The literal deployed prompt, imported rather than copied, so the "detect" task's training
# data can never silently drift from what Cosmos3Reasoner.detect_landmarks() actually sends
# at inference time. libs.py puts the repo root on sys.path for this.
from vln_subgoal_pipeline.models.cosmos3_reasoner import Cosmos3Reasoner


def _user_turn(image, text):
    return [{"role": "user", "content": [{"type": "image", "image": image},
                                         {"type": "text", "text": text}]}]


def build_prompt_messages(image, instruction):
    """Pixel-goal task prompt. Same message shape system2_train.py trains on — every prompt in
    the multi-task mix is built here, in one place, so inference can never silently drift from
    what the model was actually fine-tuned against."""
    return _user_turn(image, SYSTEM2_PROMPT_TEMPLATE.format(instruction=instruction))


def build_action_prompt_messages(image, instruction, goal_u, goal_v):
    """Action task prompt: the goal pixel is GIVEN, the model names the discrete action."""
    return _user_turn(image, SYSTEM2_ACTION_PROMPT_TEMPLATE.format(
        instruction=instruction, u=int(goal_u), v=int(goal_v)))


def build_detect_prompt_messages(image, candidates, guided_direction):
    """Detect task prompt — object-pointing + scene understanding. Mirrors
    Cosmos3Reasoner.detect_landmarks()'s own prompt assembly exactly (same landmarks-join and
    guided_direction fallback), since DETECT_PROMPT_TEMPLATE itself is imported, not copied."""
    landmarks_text = ("; ".join(candidates) if candidates
                      else "(none specific right now -- just find the most promising open "
                            "path forward)")
    direction_text = guided_direction if guided_direction and guided_direction != "None" else "None"
    return _user_turn(image, Cosmos3Reasoner.DETECT_PROMPT_TEMPLATE.format(
        landmarks=landmarks_text, guided_direction=direction_text))


def parse_pixel_goal(generated_text):
    """"123,45" -> (123, 45), or None if the model didn't emit two numbers.

    Matches this project's own SYSTEM2_TARGET_TEMPLATE convention (u, v in that order). If this
    checkpoint is ever wired into InternNav's own internvla_n1_policy.py in place of its Qwen2.5-VL
    System 2, check its parsing (`coord = re.findall(r'\\d+', ...); pixel_goal = [coord[1], coord[0]]`)
    against this — that code assumes a specific digit order from ITS OWN training data, which is
    not guaranteed to match SYSTEM2_TARGET_TEMPLATE here.
    """
    numbers = re.findall(r"-?\d+", generated_text)
    if len(numbers) < 2:
        return None
    return int(numbers[0]), int(numbers[1])


def parse_action(generated_text):
    """"FORWARD" -> 1, i.e. an index into config.SYSTEM2_ACTION_NAMES (== data.py's
    STOP/FORWARD/LEFT/RIGHT), or None if the model named no known action.

    Substring match rather than equality: the model may pad its answer ("Action: FORWARD"),
    and the four names share no prefix, so the first one to appear is unambiguous.
    """
    upper = generated_text.upper()
    hits = [(upper.index(name), index) for index, name in enumerate(SYSTEM2_ACTION_NAMES)
            if name in upper]
    return min(hits)[1] if hits else None


class System2PixelGoalPolicy:
    """Inference-time wrapper around the fine-tuned Reasoner Tower. `predict` is the pixel-goal
    task — (image, instruction) -> (u, v), or None when the model doesn't commit to a waypoint —
    and is what System 1 (system1_model.PixelGoalController) is driven by at deployment time.

    `predict_action` is the multi-task checkpoint's second head: given a goal pixel it names the
    discrete action directly, so a run trained with the "action" task in SYSTEM2_TASKS can be
    closed-looped on its own, without System 1, as a baseline to compare that controller against.
    """

    def __init__(self, model, processor, device, max_new_tokens=SYSTEM2_MAX_NEW_TOKENS):
        self.model = model.eval()
        self.processor = processor
        self.device = device
        self.max_new_tokens = max_new_tokens

    @torch.no_grad()
    def generate(self, messages):
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(self.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        new_tokens = generated_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]

    def predict(self, image, instruction):
        return parse_pixel_goal(self.generate(build_prompt_messages(image, instruction)))

    def predict_action(self, image, instruction, goal_u, goal_v):
        return parse_action(
            self.generate(build_action_prompt_messages(image, instruction, goal_u, goal_v))
        )
