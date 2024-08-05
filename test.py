from q_network import QNetwork
import torch
from utils import make_env
from utils import parse_args
from types import SimpleNamespace
import time
import json
import gymnasium as gym

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

if __name__ == "__main__":
    cfg = parse_args().config

    try:
        with open(cfg) as f:
            args_dict = json.load(f)
    except FileNotFoundError:
        print("Config file not found")
        exit(1)

    print("Loaded config: ", cfg)
    print(args_dict)

    args = SimpleNamespace(**args_dict)

    run_name = f"{args.wandb_project_name}/{args.exp_name}__{args.plasticity_injection}__{int(time.time())}"

    envs = gym.vector.SyncVectorEnv(
        [make_env(f"{args.wandb_project_name}/{args.exp_name}", args.seed + i, i, args.capture_video, run_name, args.video_path) for i in range(args.num_envs)]
    )
    net = QNetwork(envs)
    net.plasticity_bias.load_state_dict(net.head.state_dict())
    net.plasticity_bias_correction.load_state_dict(net.head.state_dict())

    net_comparison = QNetwork(envs)
    net_comparison.load_state_dict(net.state_dict())

    optimizer_net = optim.Adam(net.parameters(), lr=args.learning_rate)
    optimizer_comparison = optim.Adam(net_comparison.parameters(), lr=args.learning_rate)

    data = torch.randn(1, 4, 84, 84)

    assert torch.equal(net(data), net_comparison(data)), "Error: The network is not equal to the comparison network before injection"

    net.inject_plasticity()

    assert torch.equal(net(data), net_comparison(data)), "Error: The network is not equal to the comparison network after injection"

    out_net = net(data)
    out_comparison = net_comparison(data)

    test_target = torch.rand_like(out_net)

    loss_net = F.mse_loss(out_net, test_target)
    loss_comparison = F.mse_loss(out_comparison, test_target)

    assert loss_net == loss_comparison, "Error: The loss of the network is not equal to the comparison network"

    optimizer_net.zero_grad()
    optimizer_comparison.zero_grad()

    loss_net.backward()
    loss_comparison.backward()


    optimizer_net.step()
    optimizer_comparison.step()

    assert torch.equal(net(data), net_comparison(data)), "Error: The network is not equal to the comparison network after backward"