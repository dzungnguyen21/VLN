from internnav.configs.agent import AgentCfg
from internnav.configs.evaluator import EnvCfg, EvalCfg

eval_cfg = EvalCfg(
    agent=AgentCfg(
        model_name='internvla_n1',
        model_settings={
            "mode": "system2",  # inference mode: dual_system or system2
            "model_path": "checkpoints/InternVLA-N1-System2",  # path to model checkpoint
            "num_history": 2,
            "resize_w": 224,  # image resize width
            "resize_h": 224,  # image resize height
            "max_new_tokens": 128,  # cap generation to reduce VRAM
            "vis_debug": False,
            "vis_debug_path": "./logs/habitat/vis_debug",
        },
    ),
    env=EnvCfg(
        env_type='habitat',
        env_settings={
            # habitat sim specifications - agent, sensors, tasks, measures etc. are defined in the habitat config file
            'config_path': 'scripts/eval/configs/vln_r2r_lowmem.yaml',
        },
    ),
    eval_type='habitat_vln',
    eval_settings={
        "output_path": "./logs/habitat/r2r_s2_8gb",
        "save_video": False,
        "epoch": 0,
        "max_steps_per_episode": 500,
        # distributed settings
        "port": "2333",
        "dist_url": "env://",
    },
)
