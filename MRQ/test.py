import functools

import gymnasium
import highway_env
import numpy as np
import torch
import tyro
from models import Encoder, Policy

gymnasium.register_envs(highway_env)


def make_env(render_mode: str | None = None):
    env = gymnasium.make("racetrack-v0", render_mode=render_mode)
    env = gymnasium.wrappers.FlattenObservation(env)
    if render_mode is not None:
        env = gymnasium.wrappers.RecordVideo(
            env,
            "tmp_videos",
            name_prefix="MRQ-racetrack-v0",
            episode_trigger=lambda e: True,
        )
    return env


def load_policy(encoder_ckpt: str, policy_ckpt: str):
    env = make_env()
    encoder = Encoder(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        pixel_obs=False,
    )
    encoder.load_state_dict(torch.load(encoder_ckpt))
    policy = Policy(
        action_dim=env.action_space.shape[0],
        zs_dim=encoder.zs_dim,
        discrete=False,
    )
    policy.load_state_dict(torch.load(policy_ckpt))
    return encoder, policy


@torch.no_grad()
def predict(encoder: Encoder, policy: Policy, state: np.ndarray):
    _state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    zs = encoder.zs(_state)
    action = policy.act(zs)
    return action.numpy()


def evaluate_policy(
    encoder_ckpt: str = "MRQ/checkpoint/MRQ+HighWay-racetrack-v0+0/encoder.pt",
    policy_ckpt: str = "MRQ/checkpoint/MRQ+HighWay-racetrack-v0+0/policy.pt",
    render_mode: str | None = None,
    n_episodes: int = 1,
):
    encoder, policy = load_policy(encoder_ckpt, policy_ckpt)
    predictor = functools.partial(predict, encoder=encoder, policy=policy)
    total_rewards = []
    for _ in range(n_episodes):
        env = make_env(render_mode)
        total_r, state, t1, t2 = 0, env.reset()[0], False, False
        while not (t1 or t2):
            action_scale = env.action_space.high
            action = predictor(state=state) * action_scale
            state, reward, t1, t2, _ = env.step(action)
            total_r += reward
        env.close()
        total_rewards.append(total_r)
    return np.mean(total_rewards)


if __name__ == "__main__":
    total_r = tyro.cli(evaluate_policy)
    print(f"mean episode reward: {total_r}")
