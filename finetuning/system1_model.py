from libs import *
from config import (SYSTEM1_BACKBONE, SYSTEM1_GOAL_EMBED_DIM, SYSTEM1_HIDDEN_DIM,
                    SYSTEM1_IMAGE_EMBED_DIM, SYSTEM1_IMAGE_SIZE, SYSTEM1_N_ACTIONS)

import torch.nn as nn


def build_image_encoder(backbone_name=SYSTEM1_BACKBONE, embed_dim=SYSTEM1_IMAGE_EMBED_DIM):
    """Pretrained torchvision CNN with its classifier replaced by a linear projection to
    embed_dim. Deliberately small — System 1 runs every control tick, System 2 (Cosmos) does not."""
    import torchvision.models as tv_models

    backbone = getattr(tv_models, backbone_name)(weights="DEFAULT")
    backbone.fc = nn.Linear(backbone.fc.in_features, embed_dim)
    return backbone


class PixelGoalController(nn.Module):
    """(rgb frame, goal pixel (u, v), goal_valid) -> STOP/FORWARD/LEFT/RIGHT logits.

    This is the "System 1" half of the pair: System 2 (Cosmos, system2_infer.py) decides WHERE
    to go and speaks pixels; this decides WHAT TO DO ABOUT IT and speaks discrete actions, learned
    directly from convert/episode.py's action labels rather than any geometric heuristic.

    goal_uv is normalized to [0, 1] by image_size before the goal MLP. An invalid goal (no
    subgoal in range — convert/episode.py's (-1, -1) placeholder) is zeroed and flagged via
    goal_valid instead of passed through as a literal -1, so the network isn't left to discover
    on its own that -1 means "ignore this input".
    """

    def __init__(self, image_size=SYSTEM1_IMAGE_SIZE, image_embed_dim=SYSTEM1_IMAGE_EMBED_DIM,
                goal_embed_dim=SYSTEM1_GOAL_EMBED_DIM, hidden_dim=SYSTEM1_HIDDEN_DIM,
                n_actions=SYSTEM1_N_ACTIONS):
        super().__init__()
        self.image_size = image_size
        self.image_encoder = build_image_encoder(embed_dim=image_embed_dim)
        self.goal_encoder = nn.Sequential(
            nn.Linear(3, goal_embed_dim),  # (u, v, valid)
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(image_embed_dim + goal_embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, image, goal_uv, goal_valid):
        goal_normalized = (goal_uv / self.image_size) * goal_valid.unsqueeze(-1)
        goal_input = torch.cat([goal_normalized, goal_valid.float().unsqueeze(-1)], dim=-1)

        image_features = self.image_encoder(image)
        goal_features = self.goal_encoder(goal_input)
        return self.head(torch.cat([image_features, goal_features], dim=-1))
