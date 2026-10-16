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

RESOURCES = [1, 2, 3, 4]

ALPHA = 0.25

EPISODES = 500


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

    return int(
        torch.argmax(
            logits,
            dim=-1,
        ).item()
    )


def evaluate(
    modules,
    resources,
):

    results = []

    for seed in range(
        EPISODES
    ):

        env = MultiAgentDevTeamEnv(
            {
                "num_resources":
                    resources,

                "max_steps":
                    50,

                "bug_probability":
                    0.20,

                "alpha":
                    ALPHA,
            }
        )

        observations, infos = env.reset(
            seed=seed
        )

        team_reward = 0.0

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
            ) = env.step(
                actions
            )

            # Common evaluation metric:
            # original project/team reward.
            team_reward += (
                infos["developer"][
                    "team_reward"
                ]
            )

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
                    team_reward,

                "waste":
                    base.wasted_allocations,

                "rejections":
                    base.review_rejections,
            }
        )

    return {
        "resources":
            resources,

        "success":
            np.mean(
                [
                    x["success"]
                    for x in results
                ]
            ),

        "steps":
            np.mean(
                [
                    x["steps"]
                    for x in results
                ]
            ),

        "reward":
            np.mean(
                [
                    x["reward"]
                    for x in results
                ]
            ),

        "waste":
            np.mean(
                [
                    x["waste"]
                    for x in results
                ]
            ),

        "rejections":
            np.mean(
                [
                    x["rejections"]
                    for x in results
                ]
            ),
    }


def main():

    register_env(
        ENV_NAME,
        env_creator,
    )

    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
    )

    checkpoint = Path(
        "results/checkpoints/"
        "marl_ppo_alpha_0_25"
    ).resolve()

    print(
        "Loading checkpoint:",
        checkpoint,
    )

    algo = Algorithm.from_checkpoint(
        str(checkpoint)
    )

    modules = {
        agent: algo.get_module(
            f"{agent}_policy"
        )
        for agent in AGENTS
    }

    results = []

    for resources in RESOURCES:

        print(
            f"Evaluating resources={resources}"
        )

        results.append(
            evaluate(
                modules,
                resources,
            )
        )

    print()
    print("=" * 88)
    print("RESOURCE SCARCITY COMPARISON")
    print(
        f"Training reward alpha = {ALPHA}"
    )
    print("=" * 88)

    print(
        f"{'Resources':>10} "
        f"{'Success':>10} "
        f"{'Steps':>10} "
        f"{'Team Reward':>14} "
        f"{'Waste':>10} "
        f"{'Rejects':>10}"
    )

    print("-" * 88)

    for r in results:

        print(
            f"{r['resources']:>10} "
            f"{r['success']:>10.3f} "
            f"{r['steps']:>10.2f} "
            f"{r['reward']:>14.2f} "
            f"{r['waste']:>10.2f} "
            f"{r['rejections']:>10.2f}"
        )

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
