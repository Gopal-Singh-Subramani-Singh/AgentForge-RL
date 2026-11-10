from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from environment.multi_agent_env import (
    MultiAgentDevTeamEnv,
)


AGENTS = [
    "developer",
    "tester",
    "documentation",
    "reviewer",
]

EPISODES = 500
RESOURCES = 2
ALPHA = 0.25

CHECKPOINT = Path(
    "results/checkpoints/"
    "marl_grpo_role_aware_g8/"
    "checkpoint.pt"
).resolve()


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


def load_policies():

    checkpoint = torch.load(
        CHECKPOINT,
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

    return policies


def get_action(
    policy,
    obs,
    stochastic,
):

    obs_tensor = torch.tensor(
        obs,
        dtype=torch.float32,
    ).unsqueeze(0)

    with torch.no_grad():
        logits = policy(
            obs_tensor
        )

    if stochastic:

        dist = Categorical(
            logits=logits
        )

        return int(
            dist.sample().item()
        )

    return int(
        torch.argmax(
            logits,
            dim=-1,
        ).item()
    )


def evaluate(
    policies,
    stochastic,
):

    results = []

    for seed in range(
        EPISODES
    ):

        torch.manual_seed(
            10000 + seed
        )

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
                agent:
                    get_action(
                        policies[agent],
                        observations[agent],
                        stochastic,
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
                infos[
                    "developer"
                ][
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
                    infos[
                        "developer"
                    ][
                        "success"
                    ],

                "steps":
                    base.step_count,

                "reward":
                    team_reward,

                "waste":
                    base.wasted_allocations,

                "reject":
                    base.review_rejections,
            }
        )

    return {
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

        "reject":
            np.mean(
                [
                    x["reject"]
                    for x in results
                ]
            ),
    }


def main():

    print(
        "Loading:",
        CHECKPOINT,
    )

    policies = load_policies()

    print()
    print("=" * 96)
    print(
        "ROLE-AWARE GRPO "
        "GREEDY VS STOCHASTIC"
    )
    print("=" * 96)

    print(
        f"{'Mode':>12} "
        f"{'Success':>10} "
        f"{'Steps':>10} "
        f"{'Reward':>12} "
        f"{'Waste':>10} "
        f"{'Reject':>10}"
    )

    print("-" * 96)

    for stochastic in [
        False,
        True,
    ]:

        result = evaluate(
            policies,
            stochastic,
        )

        mode = (
            "sample"
            if stochastic
            else "greedy"
        )

        print(
            f"{mode:>12} "
            f"{result['success']:>10.3f} "
            f"{result['steps']:>10.2f} "
            f"{result['reward']:>12.2f} "
            f"{result['waste']:>10.2f} "
            f"{result['reject']:>10.2f}"
        )


if __name__ == "__main__":
    main()
