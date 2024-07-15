import random
import json
from types import SimpleNamespace
import os
from typing import Callable

import gymnasium as gym
import numpy as np
import torch


def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episode: int,
    run_name: str,
    Model: torch.nn.Module,
    video_path: str,
    device: torch.device = torch.device("cpu"),
    epsilon: float = 0.05,
    capture_video: bool = True,
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, 0, capture_video, run_name,video_path)])
    model = Model(envs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episode:
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = model(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()
        next_obs, _, _, _, infos = envs.step(actions)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]
        obs = next_obs

    return episodic_returns


if __name__ == "__main__":

    from dqn_atari import QNetwork, make_env

    with open('config.json') as f:
        args_dict = json.load(f)

    args = SimpleNamespace(**args_dict)

    run_name = None

    for folder_name in os.listdir(f"runs/{args.wandb_project_name}"):
        run_name = folder_name

    assert run_name is not None, "No model found"

    print(f"Using model from {run_name}")

    model_path = f"runs/{args.wandb_project_name}/{run_name}/{args.exp_name}.pth"
    # model_path = ".pth"
    episodic_returns = evaluate(
        model_path,
        make_env,
        args.env_id,
        eval_episode=10,
        run_name=f"{run_name}-eval",
        Model=QNetwork,
        video_path=args.video_path,
        device="cuda" if torch.cuda.is_available() and args.cuda else "cpu",
        epsilon=0,
        capture_video=False
    )

    print(f"mean episodic return={np.mean(episodic_returns)}")