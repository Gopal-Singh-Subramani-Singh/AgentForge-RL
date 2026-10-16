import argparse
from pathlib import Path

import numpy as np
import ray
import torch

from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env

from environment.multi_agent_env import MultiAgentDevTeamEnv


ENV_NAME = "multi_agent_dev_team"

AGENTS = [
    "developer",
    "tester",
    "documentation",
    "reviewer",
]


def env_creator(config):
    return MultiAgentDevTeamEnv(config)


def get_action(module, obs):
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

    logits = output[
        "action_dist_inputs"
    ]

    action = torch.argmax(
        logits,
        dim=-1,
    ).item()

    return int(action)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="results/checkpoints/marl_ppo",
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

    modules = {
        agent: algo.get_module(
            f"{agent}_policy"
        )
        for agent in AGENTS
    }

    results = []

    for seed in range(
        args.episodes
    ):

        env = MultiAgentDevTeamEnv(
            {
                "num_resources":
                    args.resources,
                "max_steps": 50,
                "bug_probability": 0.20,
            }
        )

        observations, infos = env.reset(
            seed=seed
        )

        total_team_reward = 0.0

        while True:

            actions = {
                agent: get_action(
                    modules[agent],
                    observations[agent],
                )
                for agent in AGENTS
            }

            (
                observations,
                rewards,
                terminateds,
                truncateds,
                infos,
            ) = env.step(actions)

            # All agents receive the same team reward.
            # Count it only once.
            total_team_reward += rewards[
                "developer"
            ]

            if (
                terminateds["__all__"]
                or
                truncateds["__all__"]
            ):
                break

        base = env.base_env

        results.append(
            {
                "success":
                    infos["developer"][
                        "success"
                    ],

                "steps":
                    base.step_count,

                "reward":
                    total_team_reward,

                "waste":
                    base.wasted_allocations,

                "review_rejections":
                    base.review_rejections,
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
    print("MARL PPO EVALUATION")
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
