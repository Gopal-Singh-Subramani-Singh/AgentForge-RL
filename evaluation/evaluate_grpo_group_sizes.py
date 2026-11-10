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

GROUP_SIZES = [2, 4, 8]

EPISODES = 500
RESOURCES = 2
ALPHA = 0.25


class PolicyNetwork(nn.Module):

    def __init__(
        self,
        obs_dim=10,
        hidden_dim=128,
        action_dim=2,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs):
        return self.net(obs)


def get_action(policy, obs):

    obs_tensor = torch.tensor(
        obs,
        dtype=torch.float32,
    ).unsqueeze(0)

    with torch.no_grad():
        logits = policy(obs_tensor)

    return int(
        torch.argmax(
            logits,
            dim=-1,
        ).item()
    )


def load_policies(group_size):

    checkpoint_path = Path(
        f"results/checkpoints/"
        f"marl_grpo_g{group_size}/"
        f"checkpoint.pt"
    ).resolve()

    print(
        f"Loading G={group_size}: "
        f"{checkpoint_path}"
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
            checkpoint["policies"][agent]
        )

        policy.eval()

        policies[agent] = policy

    return policies


def evaluate(group_size):

    policies = load_policies(
        group_size
    )

    results = []

    for seed in range(EPISODES):

        env = MultiAgentDevTeamEnv(
            {
                "num_resources":
                    RESOURCES,
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
            ) = env.step(actions)

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
        "group_size":
            group_size,

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

    results = []

    for group_size in GROUP_SIZES:

        print(
            f"Evaluating group size "
            f"G={group_size}"
        )

        results.append(
            evaluate(group_size)
        )

    print()
    print("=" * 88)
    print("GRPO GROUP-SIZE ABLATION")
    print(
        f"Episodes={EPISODES}, "
        f"Resources={RESOURCES}, "
        f"Alpha={ALPHA}"
    )
    print("=" * 88)

    print(
        f"{'Group':>8} "
        f"{'Success':>10} "
        f"{'Steps':>10} "
        f"{'Team Reward':>14} "
        f"{'Waste':>10} "
        f"{'Rejects':>10}"
    )

    print("-" * 88)

    for r in results:

        print(
            f"{r['group_size']:>8} "
            f"{r['success']:>10.3f} "
            f"{r['steps']:>10.2f} "
            f"{r['reward']:>14.2f} "
            f"{r['waste']:>10.2f} "
            f"{r['rejections']:>10.2f}"
        )


if __name__ == "__main__":
    main()
