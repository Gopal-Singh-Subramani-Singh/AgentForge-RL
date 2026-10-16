import argparse
from pathlib import Path

import ray

from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig

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


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--resources",
        type=int,
        default=2,
    )

    args = parser.parse_args()

    # Register our custom Gymnasium environment
    # with RLlib.
    register_env(
        ENV_NAME,
        env_creator,
    )

    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
    )

    config = (
        PPOConfig()
        .environment(
            env=ENV_NAME,
            env_config={
                "num_resources":
                    args.resources,
                "max_steps": 50,
            },
        )
        .framework("torch")
        .env_runners(
            num_env_runners=0,
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
    print("TRAINING PPO")
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

        reward = env_results.get(
            "episode_return_mean",
            float("nan"),
        )

        episode_len = env_results.get(
            "episode_len_mean",
            float("nan"),
        )

        print(
            f"Iteration {iteration:4d} | "
            f"Reward {reward:8.2f} | "
            f"Length {episode_len:6.2f}"
        )

    checkpoint_dir = Path(
    "results/checkpoints/ppo"
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
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        "Checkpoint:",
        checkpoint,
    )

    algo.stop()

    ray.shutdown()


if __name__ == "__main__":
    main()
