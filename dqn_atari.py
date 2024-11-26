import random
import json
from types import SimpleNamespace
import time
import inspect
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import torch.optim as optim
import wandb

from stable_baselines3.common.buffers import ReplayBuffer

import network
from network import Plastic
from utils import parse_args, Linear_schedule, Exponential_schedule, process_infos
from environments import Concept_Drift_Env, make_env, drifts, MnistDataset, Cifar100Dataset

if __name__ == "__main__":
    EVAL = False
    cfg = parse_args().config

    torch.set_float32_matmul_precision("medium")


    # Dynamically register all classes in environments.drifts that inherit from Concept_Drift_Env
    for name, obj in inspect.getmembers(drifts, inspect.isclass):
        if issubclass(obj, Concept_Drift_Env) and obj is not Concept_Drift_Env:
            print(f"Registering {name}...")
            gym.register(id=f"{name.lower()}_custom-v0", entry_point=f"environments.drifts:{name}")

    try:
        with open(cfg) as f:
            args_dict = json.load(f)
    except FileNotFoundError:
        print("Config file not found")
        exit(1)

    print("Loaded config: ", cfg)
    print(args_dict)

    args = SimpleNamespace(**args_dict)

    run_name = f"{args.wandb_project_name}/{args.exp_name}__{args.architecture}__{args.dataset}__{int(time.time())}"
    if args.track:

        wandb.init(
            project=args.wandb_project_name,
            group=args.exp_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    print(f"Using device {device}")

    if args.dataset == "mnist":
        dataset = MnistDataset()
    elif args.dataset == "cifar100":
        dataset = Cifar100Dataset()
    else:
        raise ValueError(f"Dataset '{args.dataset}' not supported")

    envs = gym.vector.SyncVectorEnv(
        [make_env(f"{args.exp_name}", args.seed + i, i, dataset, args.capture_video, run_name, args.video_path) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    try:
        network_class = getattr(network, args.architecture)
    except AttributeError:
        raise ValueError(f"Network '{args.architecture}' not found")
    
    cache_folder = f"runs/.tmp_{run_name.replace('/', '_')}"
    import os
    os.makedirs(cache_folder, exist_ok=True)

    print(f"Using network '{args.architecture}'")
    q_network = network_class(envs, args.total_timesteps, args.num_retrains, track=args.track, cache_folder=cache_folder).to(device)

    assert isinstance(q_network, Plastic), "Network must inherit from Injectable"

    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    scaler = GradScaler()
    target_network = network_class(envs, args.total_timesteps, args.num_retrains, track=False, cache_folder=cache_folder).to(device)

    assert isinstance(target_network, type(q_network)), "Target network and Q Network must be of same type"
    target_network.load_state_dict(q_network.state_dict())

    q_network.every_init()
    target_network.every_init()

    import platform
    os_name = platform.system()

    cuda_version = 0
    if torch.cuda.is_available():
        try:
            cuda_version = float(f"{torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]}")
        except IndexError:
            cuda_version = float(f"{torch.cuda.get_device_capability(0)[0]}.0")

    if os_name == "Linux" and cuda_version >= 7.0 and args.use_compile:
        q_network = torch.compile(q_network)
        target_network = torch.compile(target_network)
        print("Using Compiled Model")
    else:
        print(f"OS '{os_name}' or GPU CUDA Capability '{cuda_version}' not supported for model compilation") if args.use_compile else print("Not using Compiled Model")

    for concept_drift in range(args.num_retrains):

        q_network.every_drift(concept_drift)

        print(f"Concept drift {concept_drift + 1}/{args.num_retrains}")

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

        if isinstance(envs.envs[0].unwrapped, Concept_Drift_Env):
            envs.envs[0].unwrapped.inject_drift()
        else:
            gym.logger.warn("Concept drift not applied")
        

        obs, _ = envs.reset(seed=args.seed)

        #Could also use linear schedule
        epsilon = Exponential_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps)

        for global_step in range(args.total_timesteps):

            q_network.every_step(global_step)

            #Exploration or exploitation
            if random.random() < epsilon(global_step):
                actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            else:
                with autocast(dtype=torch.bfloat16):
                    q_values = q_network(torch.Tensor(obs).to(device), global_step)
                    actions = torch.argmax(q_values, dim=1).cpu().numpy()
            next_obs, rewards, terminated, truncated, infos = envs.step(actions)

            #Track episodic summaries if final episode
            process_infos(infos, epsilon(global_step), global_step, args.track)
            
            #Add experience to replay buffer
            real_next_obs = next_obs.copy()
            for idx, d in enumerate(truncated):
                if d:
                    real_next_obs[idx] = infos["final_observation"][idx]
            rb.add(obs, real_next_obs, actions, rewards, terminated, infos)

            obs = next_obs

            if global_step > args.learning_starts:
                if global_step % args.train_frequency == 0:
                    data = rb.sample(args.batch_size)
                    with torch.no_grad():
                        # Using double Q-learning to compute target values
                        next_actions = q_network(data.next_observations).argmax(dim=1, keepdim=True)
                        target_max = target_network(data.next_observations).gather(1, next_actions).squeeze()
                        td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                    
                    old_val = q_network(data.observations).gather(1, data.actions).squeeze()

                    try:
                        with autocast(dtype=torch.bfloat16):
                            loss = F.mse_loss(td_target, old_val)
                    except:
                        with autocast(dtype=torch.float16):
                            loss = F.mse_loss(td_target, old_val)
                    

                    if global_step % 100 == 0 and args.track:
                        wandb.log({
                            "losses/td_loss": loss,
                            "losses/q_values": old_val.mean().item(),
                            "charts/SPS": int(global_step / (time.time() - start_time))
                        },
                        step=global_step)

                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                if global_step % args.target_network_update_freq == 0:
                    for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                        target_network_param.data.copy_(
                            args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                        )

        #--After training--
        rb.reset()
        if args.save_model:
            model_path = f"runs/{run_name}/{args.exp_name}.pth"
            try:
                torch.save(q_network.state_dict(), model_path)
                print(f"model saved to {model_path}")
            except FileNotFoundError:
                print("Model path not found")

        if EVAL:#TODO Does not work with current label shuffling
            from dqn_eval import evaluate

            episodic_returns = evaluate(
                make_env,
                f"{args.wandb_project_name}/{args.exp_name}",
                eval_episode=100,
                run_name=f"{run_name}-eval",
                model=q_network,
                video_path=args.video_path,
                device=device,
                epsilon=0,
            )

            for idx, episodic_return in enumerate(episodic_returns):
                print(f"eval_episode={idx}, episodic_return={episodic_return}")
                if args.track:
                    wandb.log({"eval/episodic_return": episodic_return}, step=idx)

    #--After all concept drifts--
    import shutil
    shutil.rmtree(cache_folder)
    envs.close()
    if args.track:
        wandb.finish()
