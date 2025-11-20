import argparse
import importlib
import os
from typing import Iterable, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np

import random
import json
from types import SimpleNamespace
import time
import inspect

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from stable_baselines3.common.buffers import ReplayBuffer

from network.base_network import build_networks
from utils import parse_args, Linear_schedule, Exponential_schedule
from environments import Concept_Drift_Env, make_env, drifts, MnistDataset


PATH_EXP_1 = R"C:\Users\Jako9\Desktop\shuffle_labels_custom__Large_DNN_Rigid__mnist__1763238547\shuffle_labels_custom_0"
PATH_EXP_2 = R"C:\Users\Jako9\Desktop\shuffle_labels_custom__Large_SNN_Rigid__mnist__1763271959\shuffle_labels_custom_0"



def parse_model_spec(spec: Optional[str]):
    if spec is None:
        return None
    if ":" in spec:
        module_name, attr = spec.split(":", 1)
    else:
        # allow module.ClassName style
        parts = spec.rsplit(".", 1)
        if len(parts) != 2:
            raise ValueError(
                f"--model-class must be module:attr or module.ClassName, got {spec}"
            )
        module_name, attr = parts
    mod = importlib.import_module(module_name)
    if not hasattr(mod, attr):
        raise AttributeError(f"{attr} not found in module {module_name}")
    return getattr(mod, attr)


def build_or_load_model(model_path: str, model_ctor_or_factory, num_classes: int, device: torch.device):
    obj = torch.load(model_path, map_location=device)
    if isinstance(obj, nn.Module):
        model = obj.to(device)
        return model

    if model_ctor_or_factory is None:
        raise ValueError(
            "Model file appears to be a state_dict. Provide --model-class to construct the model."
        )

    # Instantiate the model
    if isinstance(model_ctor_or_factory, type):
        # class
        try:
            model = model_ctor_or_factory(num_classes=num_classes)
        except TypeError:
            model = model_ctor_or_factory()
    else:
        # factory function
        try:
            model = model_ctor_or_factory(num_classes=num_classes)
        except TypeError:
            model = model_ctor_or_factory()

    if not isinstance(model, nn.Module):
        raise TypeError("--model-class must resolve to an nn.Module subclass or factory returning nn.Module")

    state_dict = obj
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] Missing keys when loading state_dict: {missing}")
    if unexpected:
        print(f"[warn] Unexpected keys when loading state_dict: {unexpected}")

    model = model.to(device)
    return model


def get_mnist_loader(batch_size: int, train: bool, num_workers: int = 2) -> DataLoader:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    ds = datasets.MNIST(root="./MNIST", train=train, download=True, transform=transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=num_workers, pin_memory=False)


def collect_params(model: nn.Module, only_head: bool = False) -> List[torch.nn.Parameter]:
    if not only_head:
        return [p for p in model.parameters() if p.requires_grad]

    last_linear = None
    for module in model.modules():
        if isinstance(module, nn.Linear):
            last_linear = module
    if last_linear is None:
        print("[warn] --only-head requested but no nn.Linear found; using all parameters instead.")
        return [p for p in model.parameters() if p.requires_grad]
    return [p for p in last_linear.parameters() if p.requires_grad]


def flatten_params(params: Iterable[torch.Tensor]) -> torch.Tensor:
    return torch.cat([p.reshape(-1) for p in params])


def hvp(loss_fn, params: List[torch.nn.Parameter], v: torch.Tensor) -> torch.Tensor:
    loss = loss_fn()
    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
    flat_grad = torch.cat([g.reshape(-1) for g in grads])

    grad_dot_v = (flat_grad * v).sum()

    hvps = torch.autograd.grad(grad_dot_v, params, retain_graph=True)
    flat_hvp = torch.cat([h.reshape(-1) for h in hvps])
    return flat_hvp


ess_eps = 1e-8

def lanczos_top_eigs(loss_fn, params: List[torch.nn.Parameter], k: int, m: Optional[int] = None, device: Optional[torch.device] = None) -> np.ndarray:
    if device is None:
        device = params[0].device

    sizes = [p.numel() for p in params]
    n = sum(sizes)

    if m is None:
        m = max(k + 10, k)
        m = min(m, k * 2)

    q = torch.randn(n, device=device)
    q /= (q.norm() + ess_eps)

    alphas = []
    betas = []
    Q = []

    beta_prev = torch.tensor(0.0, device=device)
    q_prev = torch.zeros_like(q)

    def H_mv(v_flat: torch.Tensor) -> torch.Tensor:
        return hvp(loss_fn, params, v_flat)

    for j in range(m):
        # w = H q - beta_prev * q_prev
        Hv = H_mv(q)
        if j > 0:
            w = Hv - beta_prev * q_prev
        else:
            w = Hv
        alpha = torch.dot(q, w)
        w = w - alpha * q

        if Q:
            for qi in Q:
                w = w - torch.dot(w, qi) * qi

        beta = w.norm()
        Q.append(q)
        alphas.append(alpha.item())
        betas.append(beta.item())

        if beta.item() < 1e-10:
            break
        q_prev = q
        q = w / (beta + ess_eps)
        beta_prev = beta

    m_eff = len(alphas)
    T = np.zeros((m_eff, m_eff), dtype=np.float64)
    for i in range(m_eff):
        T[i, i] = alphas[i]
        if i + 1 < m_eff:
            T[i, i + 1] = betas[i + 1] if (i + 1) < len(betas) else 0.0
            T[i + 1, i] = T[i, i + 1]

    evals = np.linalg.eigvalsh(T)
    evals_sorted = np.sort(evals)[::-1]
    topk = evals_sorted[:k]
    return topk


def _lanczos_tridiag(loss_fn, params: List[torch.nn.Parameter], m: int, device: Optional[torch.device] = None):
    if device is None:
        device = params[0].device

    n = sum(p.numel() for p in params)
    q = torch.randn(n, device=device)
    q /= (q.norm() + ess_eps)
    q_prev = torch.zeros_like(q)
    beta_prev = torch.tensor(0.0, device=device)
    alphas, betas, Q = [], [], []

    def H_mv(v_flat: torch.Tensor) -> torch.Tensor:
        return hvp(loss_fn, params, v_flat)

    for j in range(m):
        Hv = H_mv(q)
        if j > 0:
            w = Hv - beta_prev * q_prev
        else:
            w = Hv
        alpha = torch.dot(q, w)
        w = w - alpha * q
        if Q:
            for qi in Q:
                w = w - torch.dot(w, qi) * qi
        beta = w.norm()
        Q.append(q)
        alphas.append(float(alpha))
        betas.append(float(beta))
        if beta.item() < 1e-10:
            break
        q_prev = q
        q = w / (beta + ess_eps)
        beta_prev = beta

    m_eff = len(alphas)
    return alphas, betas, m_eff


def hessian_spectral_density_slq(loss_fn, params: List[torch.nn.Parameter], m: int, num_probes: int, grid_size: int = 200, grid_min: Optional[float] = None, grid_max: Optional[float] = None, sigma: Optional[float] = None, device: Optional[torch.device] = None):
    if device is None:
        device = params[0].device
    n_dim = sum(p.numel() for p in params)

    all_thetas = []
    all_weights = []
    for _ in range(num_probes):
        alphas, betas, m_eff = _lanczos_tridiag(loss_fn, params, m=m, device=device)
        T = np.zeros((m_eff, m_eff), dtype=np.float64)
        for i in range(m_eff):
            T[i, i] = alphas[i]
            if i + 1 < m_eff:
                T[i, i + 1] = betas[i + 1] if (i + 1) < len(betas) else 0.0
                T[i + 1, i] = T[i, i + 1]
        thetas, U = np.linalg.eigh(T)

        e1 = U[0, :]
        weights = (e1 ** 2)
        all_thetas.append(thetas)
        all_weights.append(weights)

    thetas_concat = np.concatenate(all_thetas) if len(all_thetas) else np.array([0.0])
    weights_concat = np.concatenate(all_weights) if len(all_weights) else np.array([1.0])

    if grid_min is None:
        grid_min = float(thetas_concat.min()) - 1e-6
    if grid_max is None:
        grid_max = float(thetas_concat.max()) + 1e-6
    grid = np.linspace(grid_min, grid_max, grid_size)
    if sigma is None:
        sigma = max(1e-6, 0.02 * (grid_max - grid_min))

    density = np.zeros_like(grid)
    inv_norm = 1.0 / (np.sqrt(2 * np.pi) * sigma)
    for thetas, weights in zip(all_thetas, all_weights):
        diff = grid[:, None] - thetas[None, :]
        kernels = np.exp(-0.5 * (diff / sigma) ** 2) * inv_norm
        density += (kernels @ weights) / n_dim
    density /= max(1, num_probes)
    return grid, density


def one_step_sgd(model: nn.Module, batch, device: torch.device, lr: float = 1e-2, weight_decay: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    model.train()
    x, y = batch
    x = x.to(device)
    y = y.to(device)
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    opt.zero_grad()
    loss.backward()
    opt.step()
    with torch.no_grad():
        acc = (logits.argmax(dim=1) == y).float().mean()
    return loss.detach(), acc.detach()


def main():
    torch.set_float32_matmul_precision("medium")
    cfg = parse_args().config

    # Register concept drift envs
    for name, obj in inspect.getmembers(drifts, inspect.isclass):
        if issubclass(obj, Concept_Drift_Env) and obj is not Concept_Drift_Env:
            gym.register(id=f"{name.lower()}_custom-v0", entry_point=f"environments.drifts:{name}")

    try:
        with open(cfg) as f:
            args_dict = json.load(f)
    except FileNotFoundError:
        print("Config file not found")
        return

    print("Loaded config:", cfg)
    args = SimpleNamespace(**args_dict)

    # Seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cpu")
    networks = build_networks()


    run_name = f"{args.wandb_project_name}/{args.exp_name}__{args.architecture}__{args.dataset}__{int(time.time())}"
    cache_folder = f"runs/.tmp_{run_name.replace('/', '_')}"
    os.makedirs(cache_folder, exist_ok=True)

    def analyze_one(tag: str, arch: str, weights_path: str, dataset_path: str):
        ds = MnistDataset()
        envs = gym.vector.SyncVectorEnv(
            [make_env(f"{args.exp_name}", args.seed + i, i, ds, False, run_name, args.video_path) for i in range(args.num_envs)]
        )
        assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
        envs.envs[0].unwrapped.load_drift(dataset_path)
        print("Loaded drift state into environment.")

        if arch not in networks:
            raise ValueError(f"Network '{arch}' not found")
        net_cls = networks[arch]
        q_net = net_cls(envs, args.total_timesteps, args.num_retrains, track=False, cache_folder=cache_folder).to(device)
        tgt_net = net_cls(envs, args.total_timesteps, args.num_retrains, track=False, cache_folder=cache_folder).to(device)
        tgt_net.load_state_dict(q_net.state_dict())

        if weights_path and os.path.isfile(weights_path):
            try:
                sd = torch.load(weights_path, map_location=device)
                q_net.load_state_dict(sd)
                tgt_net.load_state_dict(sd)
                print(f"[{tag}] Loaded state_dict from {weights_path}")
            except Exception as e:
                print(f"[{tag}] [warn] Failed to load state_dict: {e}")
        else:
            print(f"[{tag}] No weights loaded (path missing or invalid): {weights_path}")

        rb = ReplayBuffer(
            args.buffer_size,
            envs.single_observation_space,
            envs.single_action_space,
            device,
            optimize_memory_usage=True,
            handle_timeout_termination=False,
        )

        obs, _ = envs.reset(seed=args.seed)
        epsilon_sched = Exponential_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps)

        warmup_steps = max(args.learning_starts, args.batch_size * 2)
        for global_step in range(warmup_steps):
            eps_val = epsilon_sched(global_step)
            if random.random() < eps_val:
                actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            else:
                with torch.no_grad():
                    q_vals = q_net(torch.as_tensor(obs, dtype=torch.float32, device=device), global_step)
                    actions = q_vals.argmax(dim=1).cpu().numpy()
            next_obs, rewards, terminated, truncated, infos = envs.step(actions)
            real_next_obs = next_obs.copy()
            for idx, d in enumerate(truncated):
                if d and "final_observation" in infos:
                    real_next_obs[idx] = infos["final_observation"][idx]
            rb.add(obs, real_next_obs, actions, rewards, terminated, infos)
            obs = next_obs

        if rb.size() < args.batch_size:
            envs.close()
            raise RuntimeError(f"[{tag}] Not enough samples collected for a batch")

        data = rb.sample(args.batch_size)

        if isinstance(args.learning_rate, float):
            lr = args.learning_rate
        else:
            lr = args.learning_rate[0]['lr']
        optimizer = optim.Adam(q_net.parameters(), lr=lr)

        def td_loss_closure():
            with torch.no_grad():
                next_actions = q_net(data.next_observations).argmax(dim=1, keepdim=True)
                target_max = tgt_net(data.next_observations).gather(1, next_actions).squeeze()
                td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
            current_q = q_net(data.observations).gather(1, data.actions).squeeze()
            return F.mse_loss(td_target, current_q)

        params_head = collect_params(q_net, only_head=True)
        print(f"[{tag}] Estimating Hessian eigenvalues (pre-update, head params)...")
        pre_eigs = lanczos_top_eigs(td_loss_closure, params_head, k=100, m=200, device=device)

        density_probes = getattr(args, 'density_probes', 10)
        density_m = getattr(args, 'density_m', 64)
        density_grid = getattr(args, 'density_grid', 200)
        density_sigma = getattr(args, 'density_sigma', None)
        print(f"[{tag}] Estimating Hessian spectral density (pre-update)...")
        grid_pre, dens_pre = hessian_spectral_density_slq(
            td_loss_closure, params_head, m=density_m, num_probes=density_probes,
            grid_size=density_grid, sigma=density_sigma, device=device
        )

        loss = td_loss_closure()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"[{tag}] Performed one TD update: loss={loss.item():.4f}")

        print(f"[{tag}] Estimating Hessian eigenvalues (post-update, head params)...")
        post_eigs = lanczos_top_eigs(td_loss_closure, params_head, k=100, m=200, device=device)
        print(f"[{tag}] Estimating Hessian spectral density (post-update)...")
        grid_post, dens_post = hessian_spectral_density_slq(
            td_loss_closure, params_head, m=density_m, num_probes=density_probes,
            grid_size=density_grid, sigma=density_sigma, device=device
        )

        envs.close()
        return {
            'pre_eigs': pre_eigs,
            'post_eigs': post_eigs,
            'grid_pre': grid_pre,
            'dens_pre': dens_pre,
            'grid_post': grid_post,
            'dens_post': dens_post,
        }
    
    dnn = {
        'tag': 'Rigid-DNN',
        'arch': 'Large_DNN_Rigid',
        'weights_path': PATH_EXP_1 + ".pth",
        'dataset_path': PATH_EXP_1,
    }
    snn = {
        'tag': 'Rigid-SNN',
        'arch': 'Large_SNN_Rigid',
        'weights_path': PATH_EXP_2 + ".pth",
        'dataset_path': PATH_EXP_2,
    }

    res_dnn = analyze_one(**dnn)
    res_snn = analyze_one(**snn)

    # Plot top eigenvalues comparison (pre and post) for A vs B
    plt.figure(figsize=(9, 4))
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(1, len(res_dnn['pre_eigs']) + 1), res_dnn['pre_eigs'], label='A-Pre')
    plt.plot(np.arange(1, len(res_snn['pre_eigs']) + 1), res_snn['pre_eigs'], label='B-Pre')
    plt.xlabel("Rank")
    plt.ylabel("Eigenvalue")
    plt.title("Top eigenvalues (pre-update)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(np.arange(1, len(res_dnn['post_eigs']) + 1), res_dnn['post_eigs'], label='A-Post')
    plt.plot(np.arange(1, len(res_snn['post_eigs']) + 1), res_snn['post_eigs'], label='B-Post')
    plt.xlabel("Rank")
    plt.ylabel("Eigenvalue")
    plt.title("Top eigenvalues (post-update)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(9, 4))
    gridA_pre, densA_pre = res_dnn['grid_pre'], res_dnn['dens_pre']
    gridB_pre, densB_pre = res_snn['grid_pre'], res_snn['dens_pre']
    densB_pre_interp = densB_pre if np.allclose(gridA_pre, gridB_pre) else np.interp(gridA_pre, gridB_pre, densB_pre)
    plt.subplot(1, 2, 1)
    plt.plot(gridA_pre, densA_pre, label='A-Pre')
    plt.plot(gridA_pre, densB_pre_interp, label='B-Pre')
    plt.xlabel("Eigenvalue λ")
    plt.ylabel("Density")
    plt.title("Spectral density (pre)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    gridA_post, densA_post = res_dnn['grid_post'], res_dnn['dens_post']
    gridB_post, densB_post = res_snn['grid_post'], res_snn['dens_post']
    densB_post_interp = densB_post if np.allclose(gridA_post, gridB_post) else np.interp(gridA_post, gridB_post, densB_post)
    plt.subplot(1, 2, 2)
    plt.plot(gridA_post, densA_post, label='A-Post')
    plt.plot(gridA_post, densB_post_interp, label='B-Post')
    plt.xlabel("Eigenvalue λ")
    plt.ylabel("Density")
    plt.title("Spectral density (post)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()
