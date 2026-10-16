import argparse
from pathlib import Path

import numpy as np
import ray
import torch

from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env

from environment.marl_env import DevTeamEnv


ENV_NAME = "dev_team_env"


def env_creator(config):
    return DevTeamEnv(
        num_resources=config.get(
            "num_resources",
            2,
        ),
        max_steps=config.get(
            "max_steps",
            50,
        ),
    )


def get_action(module, obs):
    """
    Compute a deterministic action using RLlib's
    new RLModule inference API.
    """

    obs_tensor = torch.tensor(
        obs,
        dtype=torch.float32,
    ).unsqueeze(0)

    with torch.no_grad():
        output = module.forward_inference(
            {
                "obs": obs_tensor,
            }
        )

    # PPO's RLModule returns action-distribution
    # inputs (logits).
    logits = output[
        "action_dist_inputs"
    ]

    # Our MultiDiscrete action is [2, 2, 2, 2].
    # Therefore RLlib produces 8 logits:
    #
    # Developer:     logits 0:2
    # Tester:        logits 2:4
    # Documentation: logits 4:6
    # Reviewer:      logits 6:8
    #
    # For deterministic evaluation, select the
    # highest-logit action for each agent.

    chunks = torch.split(
        logits,
        [2, 2, 2, 2],
        dim=-1,
    )

    action = [
        int(torch.argmax(chunk, dim=-1).item())
        for chunk in chunks
    ]

    return np.array(
        action,
        dtype=np.int64,
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="results/checkpoints/ppo",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--resources",
        type=int,
        default=2,
    )

    args = parser.parse_args()

    register_env(
        ENV_NAME,
        env_creator,
    )

    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
    )

    checkpoint_path = Path(
        args.checkpoint
    ).resolve()

    print(
        "Loading checkpoint:",
        checkpoint_path,
    )

    algo = Algorithm.from_checkpoint(
        str(checkpoint_path)
    )

    # New RLlib API.
    module = algo.get_module(
        "default_policy"
    )

    results = []

    for seed in range(
        args.episodes
    ):

        env = DevTeamEnv(
            num_resources=args.resources
        )

        obs, info = env.reset(
            seed=seed
        )

        total_reward = 0.0

        while True:

            action = get_action(
                module,
                obs,
            )

            (
                obs,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            total_reward += reward

            if terminated or truncated:
                break

        results.append(
            {
                "success":
                    info["success"],

                "steps":
                    env.step_count,

                "reward":
                    total_reward,

                "waste":
                    env.wasted_allocations,

                "review_rejections":
                    env.review_rejections,
            }
        )

    success = np.mean(
        [
            r["success"]
            for r in results
        ]
    )

    steps = np.mean(
        [
            r["steps"]
            for r in results
        ]
    )

    reward = np.mean(
        [
            r["reward"]
            for r in results
        ]
    )

    waste = np.mean(
        [
            r["waste"]
            for r in results
        ]
    )

    rejection = np.mean(
        [
            r["review_rejections"]
            for r in results
        ]
    )

    print()
    print("=" * 60)
    print("PPO EVALUATION")
    print("=" * 60)

    print(
        f"Episodes: {args.episodes}"
    )

    print(
        f"Resources: {args.resources}"
    )

    print(
        f"Success rate: {success:.3f}"
    )

    print(
        f"Average steps: {steps:.2f}"
    )

    print(
        f"Average reward: {reward:.2f}"
    )

    print(
        "Average wasted allocations: "
        f"{waste:.2f}"
    )

    print(
        "Average review rejections: "
        f"{rejection:.2f}"
    )

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
