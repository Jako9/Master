import argparse
import wandb


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
                    wandb.log({
                        "charts/episodic_return": info["episode"]["r"],
                        "charts/episode_length": info["episode"]["l"],
                        "charts/epsilon": epsilon},
                        step=global_step)
    