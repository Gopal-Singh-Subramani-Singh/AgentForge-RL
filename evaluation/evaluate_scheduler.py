import argparse
import numpy as np

from environment.marl_env import DevTeamEnv
from schedulers.rule_based import RuleBasedScheduler
from schedulers.random_scheduler import RandomScheduler


def run_episode(
    scheduler,
    seed,
    num_resources,
):

    env = DevTeamEnv(
        num_resources=num_resources
    )

    obs, info = env.reset(
        seed=seed
    )

    total_reward = 0.0

    while True:

        action = scheduler.act(env)

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

    return {
        "success": info["success"],
        "steps": env.step_count,
        "reward": total_reward,
        "waste":
            env.wasted_allocations,
        "review_rejections":
            env.review_rejections,
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scheduler",
        choices=[
            "random",
            "rule",
        ],
        required=True,
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--resources",
        type=int,
        default=2,
    )

    args = parser.parse_args()

    if args.scheduler == "random":
        scheduler = RandomScheduler()
    else:
        scheduler = RuleBasedScheduler()

    results = []

    for seed in range(
        args.episodes
    ):
        results.append(
            run_episode(
                scheduler,
                seed,
                args.resources,
            )
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

    print(
        f"Scheduler: {args.scheduler}"
    )

    print(
        f"Resources: {args.resources}"
    )

    print(
        f"Episodes: {args.episodes}"
    )

    print(
        f"Success rate: "
        f"{success:.3f}"
    )

    print(
        f"Average steps: "
        f"{steps:.2f}"
    )

    print(
        f"Average reward: "
        f"{reward:.2f}"
    )

    print(
        f"Average wasted allocations: "
        f"{waste:.2f}"
    )

    print(
        f"Average review rejections: "
        f"{rejection:.2f}"
    )


if __name__ == "__main__":
    main()
