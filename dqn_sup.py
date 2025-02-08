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


import network
from network import Plastic
from utils import parse_args, Linear_schedule, Exponential_schedule, process_infos, log, add_log, StepwiseConstantLR
from environments import Concept_Drift_Env, make_env, drifts, MnistDataset, Cifar100Dataset, CompositeDataset

def main(): 
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
    elif args.dataset == "composite":
        dataset = CompositeDataset()
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

    scaler = GradScaler()

    q_network.every_init()

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
        print("Using Compiled Model")
    else:
        print(f"OS '{os_name}' or GPU CUDA Capability '{cuda_version}' not supported for model compilation") if args.use_compile else print("Not using Compiled Model")

    from torchvision import datasets
    x_train = datasets.CIFAR100(root=".", train=True, download=True).data
    y_train = datasets.CIFAR100(root=".", train=True, download=True).targets

    x_train = np.mean(x_train, axis=3)

    x_test = datasets.CIFAR100(root=".", train=False, download=True).data
    y_test = datasets.CIFAR100(root=".", train=False, download=True).targets

    x_test = np.mean(x_test, axis=3)

    #Build dataloader
    from torch.utils.data import DataLoader, TensorDataset
    import torch.nn as nn

    x_train = torch.tensor(x_train).float()
    x_train = x_train.unsqueeze(0).repeat(4, 1, 1, 1).permute(1, 0, 2, 3)
    #reshape the image to be (4, x, 32, 32) -> (4, x, 84, 84)
    x_train = F.interpolate(x_train, size=(84, 84), mode="bilinear", align_corners=False)
    y_train = torch.tensor(y_train).long()

    x_test = torch.tensor(x_test).float()
    x_test = x_test.unsqueeze(0).repeat(4, 1, 1, 1).permute(1, 0, 2, 3)
    #reshape the image to be (4, x, 32, 32) -> (4, x, 84, 84)
    x_test = F.interpolate(x_test, size=(84, 84), mode="bilinear", align_corners=False)
    y_test = torch.tensor(y_test).long()

    dataset_total = TensorDataset(x_test, y_test)
    dataloader_total = DataLoader(dataset_total, batch_size=args.batch_size, shuffle=False)
    EPSILON = 5

    for concept_drift in range(args.num_retrains):

        
        x_samples = x_train[y_train < (10 * (concept_drift + 1))]
        y_samples = y_train[y_train < (10 * (concept_drift + 1))]
        y_samples_sorted = torch.argsort(torch.unique(y_samples, return_counts=True)[1], descending=True)
        print(y_samples_sorted)

        current_max_class = torch.max(y_samples)
        print(f"Current max class: {current_max_class}")

        x_samples_test = x_test[y_test < (10 * (concept_drift + 1))]
        y_samples_test = y_test[y_test < (10 * (concept_drift + 1))]

        dataset = TensorDataset(x_samples, y_samples)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        dataset_test = TensorDataset(x_samples_test, y_samples_test)
        dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)

        #constant learning rate when args.learning_rate is a float, otherwise use a scedueld learning rate
        if isinstance(args.learning_rate, float):
            rate= args.learning_rate
        else:
            rate = args.learning_rate[0]['lr']

        optimizer = optim.Adam(q_network.parameters(), lr=rate)

        criterion = nn.CrossEntropyLoss()

        scheduler = StepwiseConstantLR(optimizer, args.learning_rate, args.train_frequency)

        q_network.every_drift(concept_drift)

        print(f"Concept drift {concept_drift + 1}/{args.num_retrains}")

        start_time = time.time()

        #Could also use linear schedule
        epsilon = Exponential_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps)
        global_step = 0
        best_acc = 0
        if args.early_stopping:
            q_network.load_state_dict(torch.load(f"{cache_folder}/{args.exp_name}_best.pth") if os.path.exists(f"{cache_folder}/{args.exp_name}_best.pth") else q_network.state_dict())
        while global_step < args.total_timesteps / args.train_frequency:
            q_network.every_step(global_step)
            if args.track:
                add_log("charts/total_step", global_step + concept_drift * args.total_timesteps)

            for batch_idx, (inputs, targets) in enumerate(dataloader):
                global_step += 1
                if global_step > args.total_timesteps / args.train_frequency:
                    break
                # Move inputs and targets to the same device as the model (if using GPU)
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = q_network(inputs)
                #create output mask to only select the first "current_max_class" output neurons
                mask = torch.zeros_like(outputs, dtype=torch.bool).to(device)
                mask[:, :current_max_class] = True

                masked_outputs = torch.where(mask, outputs, torch.tensor(-float('inf')).to(device))
                loss = criterion(masked_outputs, targets)

                #Track episodic summaries if final episode
                process_infos({}, epsilon(global_step), global_step, args.track)

                if global_step % 100 == 0 and args.track:
                    add_log("losses/td_loss", loss)
                    add_log("charts/SPS", int(global_step / (time.time() - start_time)))
                    add_log("charts/learning_rate", scheduler.get_lr()[0])
                    add_log("charts/concept_drift", concept_drift)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scheduler.step()
                scaler.update()
                if args.track:
                    log()


            accuracy_current = 0
            accuracy_total = 0
            accuracy_train = 0
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(dataloader_test):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = q_network(inputs)
                    accuracy_current += (outputs.argmax(1) == targets).float().mean().item()
                
                for batch_idx, (inputs, targets) in enumerate(dataloader_total):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = q_network(inputs)
                    accuracy_total += (outputs.argmax(1) == targets).float().mean().item()

                for batch_idx, (inputs, targets) in enumerate(dataloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = q_network(inputs)
                    accuracy_train += (outputs.argmax(1) == targets).float().mean().item()

            accuracy_current *= (100 / len(dataloader_test))
            accuracy_total *= (100 / len(dataloader_total))
            accuracy_train *= (100 / len(dataloader))
            add_log("losses/accuracy_current", accuracy_current)
            add_log("losses/accuracy_total", accuracy_total)
            add_log("losses/accuracy_train", accuracy_train)
            add_log("losses/max_class", current_max_class)
            
            print(f"Step {global_step}, Loss: {loss}, Current Acc: {accuracy_current}%, Total Acc: {accuracy_total}%")
            if accuracy_current > best_acc and args.early_stopping:
                best_acc = accuracy_current
                print("New best accuracy.. Saving model")
                torch.save(q_network.state_dict(), f"{cache_folder}/{args.exp_name}_best.pth")

        print(f"END OF CONCEPT DRIFT.. Best accuracy: {best_acc}%")
        del dataset
        del dataloader
        del dataset_test
        del dataloader_test
            
        #--After training--
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
                    add_log("charts/episodic_return", episodic_return)
                    log()

    #--After all concept drifts--
    import shutil
    shutil.rmtree(cache_folder)
    envs.close()
    if args.track:
        wandb.finish()


from torch.profiler import profile, ProfilerActivity, record_function
if __name__ == "__main__":
    main()