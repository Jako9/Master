import argparse

import gymnasium as gym

import wandb

from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv
)


def parse_args():
    parser = argparse.ArgumentParser(description="DQN evaluation")
    parser.add_argument("--config", type=str, default="config.json", help="config file")
    args = parser.parse_args()
    return args

def make_env(env_id, seed, idx, capture_video, run_name, video_path):
    def thunk():
        if capture_video and idx  ==0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"{video_path}/{run_name}")
        else:
            env = gym.make(env_id)

        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)

        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        env.action_space.seed(seed)

        return env
    
    return thunk

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
    
def process_infos(infos, epsilon, global_step):
    if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                wandb.log({
                    "charts/episodic_return": info["episode"]["r"],
                    "charts/episode_length": info["episode"]["l"],
                    "charts/epsilon": epsilon},
                    step=global_step)
    