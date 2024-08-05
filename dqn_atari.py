import random
import json
from types import SimpleNamespace
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from q_network import QNetwork
from utils import parse_args, make_env, Linear_schedule, Exponential_schedule, process_infos

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
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    print(f"Using device {device}")

    envs = gym.vector.SyncVectorEnv(
        [make_env(f"{args.wandb_project_name}/{args.exp_name}", args.seed + i, i, args.capture_video, run_name, args.video_path) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        optimize_memory_usage=True,
        handle_timeout_termination=False
    )
    start_time = time.time()

    obs, _ = envs.reset(seed=args.seed)
    #Could also use linear schedule
    epsilon = Exponential_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps)

    for global_step in range(args.total_timesteps):

        #Exploration or exploitation
        if random.random() < epsilon(global_step):
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()
        next_obs, rewards, terminated, truncated, infos = envs.step(actions)

        #Track episodic summaries if final episode
        process_infos(infos, writer, epsilon(global_step), global_step)
        
        #Add experience to replay buffer
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(truncated):
            if d:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminated, infos)

        obs = next_obs

        if global_step > args.learning_starts:
            if global_step == args.plasticity_injection and args.plasticity_injection != 0:
                print("Injecting plasticity")
                q_network.inject_plasticity()
                target_network.inject_plasticity()
            
            #Sanity check for plasticity injection
            if (global_step == args.plasticity_injection / 2 or global_step == args.plasticity_injection + args.plasticity_injection / 2):
                assert q_network.check_plasticity_status(), "Plasticity injection failed in QNetwork"
                assert target_network.check_plasticity_status(), "Plasticity injection failed in TargetNetwork"

            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    # Using double Q-learning to compute target values
                    next_actions = q_network(data.next_observations).argmax(dim=1, keepdim=True)
                    target_max = target_network(data.next_observations).gather(1, next_actions).squeeze()
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if global_step % args.target_network_update_freq == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )
                    


    #--After training--
    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.pth"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")

        from dqn_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            f"{args.wandb_project_name}/{args.exp_name}",
            eval_episode=10,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            video_path=args.video_path,
            device=device,
            epsilon=0,
        )

        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

    envs.close()
    writer.close()
