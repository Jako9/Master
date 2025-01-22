import argparse
import wandb
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

loggin_dict = {}

class StepwiseConstantLR(_LRScheduler):
    def __init__(self, optimizer: Optimizer, lr_schedule: list, train_frequency: int, last_epoch: int = -1):
        """
        Custom learning rate scheduler for constant learning rate with switches.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            lr_schedule (list): List of (timestep, lr) pairs, sorted by timestep.
                                E.g., [(0, 0.1), (10, 0.01), (20, 0.001)]
            last_epoch (int): The index of the last epoch. Defaults to -1.
        """
        if isinstance(lr_schedule, float):
            lr_schedule = [(0, lr_schedule)]
        else:
            lr_schedule = [(d['timestep'] / train_frequency, d['lr']) for d in lr_schedule]
        self.lr_schedule = sorted(lr_schedule, key=lambda x: x[0])
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        current_epoch = self.last_epoch
        for timestep, lr in reversed(self.lr_schedule):
            if current_epoch >= timestep:
                return [lr for _ in self.base_lrs]
        # Default to the first learning rate if no timestep is reached
        return [self.lr_schedule[0][1] for _ in self.base_lrs]


def parse_args():
    parser = argparse.ArgumentParser(description="DQN evaluation")
    parser.add_argument("--config", type=str, default="config.json", help="config file")
    args = parser.parse_args()
    return args

class Linear_schedule():
    def __init__(self, start_e: float, end_e: float, duration: int):
        self.start_e = start_e
        self.end_e = end_e
        self.slope = (self.end_e - self.start_e) / duration

    def __call__(self, t: int):
        return max(self.slope * t + self.start_e, self.end_e)
    
class Exponential_schedule():
    def __init__(self, start_e: float, end_e: float, duration: int):
        self.start_e = start_e
        self.end_e = end_e
        self.slope = (self.end_e / self.start_e) ** (1 / duration)

    def __call__(self, t: int):
        return max(self.start_e * (self.slope ** t), self.end_e)
    
def process_infos(infos, epsilon, global_step, track):
    if "final_info" in infos:
        for info in infos["final_info"]:
            if "episode" not in info:
                continue
            print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
            if track:
                add_log("charts/episodic_return", info["episode"]["r"])
                add_log("charts/episode_length", info["episode"]["l"])
                add_log("charts/epsilon", epsilon)
                

def log():
    global loggin_dict
    wandb.log(loggin_dict)
    _reset_log()

def add_log(key, value):
    global loggin_dict
    loggin_dict[key] = value

def _reset_log():
    global loggin_dict
    loggin_dict = {}
    