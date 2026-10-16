import argparse
from pathlib import Path

import ray

from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec

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


def policy_mapping_fn(agent_id, *args, **kwargs):
    return f"{agent_id}_policy"


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--resources",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
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

    policies = {
        "developer_policy": PolicySpec(),
        "tester_policy": PolicySpec(),
        "documentation_policy": PolicySpec(),
        "reviewer_policy": PolicySpec(),
    }

    config = (
        PPOConfig()
        .environment(
            env=ENV_NAME,
            env_config={
                "num_resources": args.resources,
                "max_steps": 50,
                "bug_probability": 0.20,
                "alpha": args.alpha,
            },
        )
        .framework("torch")
        .env_runners(
            num_env_runners=0,
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=list(
                policies.keys()
            ),
        )
        .training(
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            train_batch_size=4000,
            minibatch_size=128,
            num_epochs=10,
        )
    )

    algo = config.build_algo()

    print()
    print("=" * 70)
    print("TRAINING MARL PPO")
    print(f"Reward alpha: {args.alpha}")
    print("=" * 70)

    for iteration in range(
        1,
        args.iterations + 1,
    ):

        result = algo.train()

        env_results = result.get(
            "env_runners",
            {},
        )

        team_reward = env_results.get(
            "episode_return_mean",
            float("nan"),
        )

        episode_len = env_results.get(
            "episode_len_mean",
            float("nan"),
        )

        print(
            f"Iteration {iteration:4d} | "
            f"Team Reward {team_reward:8.2f} | "
            f"Length {episode_len:6.2f}"
        )

    alpha_name = str(
        args.alpha
    ).replace(".", "_")

    checkpoint_dir = Path(
        f"results/checkpoints/"
        f"marl_ppo_alpha_{alpha_name}"
    ).resolve()

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = algo.save(
        str(checkpoint_dir)
    )

    print()
    print("=" * 70)
    print("MARL TRAINING COMPLETE")
    print("=" * 70)

    print(
        "Checkpoint:",
        checkpoint,
    )

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
