import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from environment.multi_agent_env import MultiAgentDevTeamEnv


AGENTS = [
    "developer",
    "tester",
    "documentation",
    "reviewer",
]


class PolicyNetwork(nn.Module):

    def __init__(
        self,
        obs_dim=10,
        hidden_dim=128,
        action_dim=2,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                obs_dim,
                hidden_dim,
            ),
            nn.Tanh(),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.Tanh(),

            nn.Linear(
                hidden_dim,
                action_dim,
            ),
        )

    def forward(self, obs):
        return self.net(obs)


def get_action(
    policy,
    obs,
):

    obs_tensor = torch.tensor(
        obs,
        dtype=torch.float32,
    ).unsqueeze(0)

    with torch.no_grad():

        logits = policy(
            obs_tensor
        )

    return int(
        torch.argmax(
            logits,
            dim=-1,
        ).item()
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=(
            "results/checkpoints/"
            "marl_grpo/checkpoint.pt"
        ),
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

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.25,
    )

    args = parser.parse_args()

    checkpoint_path = Path(
        args.checkpoint
    ).resolve()

    print(
        "Loading checkpoint:",
        checkpoint_path,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    hidden_dim = checkpoint.get(
        "hidden_dim",
        128,
    )

    policies = {}

    for agent in AGENTS:

        policy = PolicyNetwork(
            hidden_dim=hidden_dim
        )

        policy.load_state_dict(
            checkpoint[
                "policies"
            ][agent]
        )

        policy.eval()

        policies[agent] = policy

    results = []

    for seed in range(
        args.episodes
    ):

        env = MultiAgentDevTeamEnv(
            {
                "num_resources":
                    args.resources,

                "max_steps":
                    50,

                "bug_probability":
                    0.20,

                "alpha":
                    args.alpha,
            }
        )

        observations, infos = env.reset(
            seed=seed
        )

        team_reward = 0.0

        while True:

            actions = {
                agent: get_action(
                    policies[agent],
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

    success = np.mean(
        [
            x["success"]
            for x in results
        ]
    )

    steps = np.mean(
        [
            x["steps"]
            for x in results
        ]
    )

    reward = np.mean(
        [
            x["reward"]
            for x in results
        ]
    )

    waste = np.mean(
        [
            x["waste"]
            for x in results
        ]
    )

    rejections = np.mean(
        [
            x["rejections"]
            for x in results
        ]
    )

    print()
    print("=" * 72)
    print("GRPO-STYLE MARL EVALUATION")
    print("=" * 72)

    print(
        f"Episodes: {args.episodes}"
    )

    print(
        f"Resources: {args.resources}"
    )

    print(
        f"Alpha: {args.alpha}"
    )

    print(
        f"Success rate: {success:.3f}"
    )

    print(
        f"Average steps: {steps:.2f}"
    )

    print(
        f"Average team reward: {reward:.2f}"
    )

    print(
        f"Average wasted allocations: {waste:.2f}"
    )

    print(
        f"Average review rejections: {rejections:.2f}"
    )


if __name__ == "__main__":
    main()
