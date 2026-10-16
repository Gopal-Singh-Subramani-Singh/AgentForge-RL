import argparse
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


def select_action(
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

        dist = Categorical(
            logits=logits
        )

        action = dist.sample()

        log_prob = dist.log_prob(
            action
        )

    return (
        int(action.item()),
        float(log_prob.item()),
    )


def rollout(
    policies,
    seed,
    resources,
    alpha,
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
                alpha,
        }
    )

    observations, infos = env.reset(
        seed=seed
    )

    trajectory = {
        agent: {
            "obs": [],
            "actions": [],
            "old_log_probs": [],
        }
        for agent in AGENTS
    }

    team_return = 0.0

    while True:

        actions = {}

        for agent in AGENTS:

            action, log_prob = (
                select_action(
                    policies[agent],
                    observations[agent],
                )
            )

            actions[agent] = action

            trajectory[agent][
                "obs"
            ].append(
                observations[
                    agent
                ].copy()
            )

            trajectory[agent][
                "actions"
            ].append(
                action
            )

            trajectory[agent][
                "old_log_probs"
            ].append(
                log_prob
            )

        (
            observations,
            rewards,
            terminateds,
            truncateds,
            infos,
        ) = env.step(
            actions
        )

        # GRPO group score:
        # always use the original project/team reward.
        #
        # This makes the coordination objective directly
        # comparable with our PPO evaluation metric.
        team_return += (
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

    metrics = {
        "success":
            infos["developer"][
                "success"
            ],

        "steps":
            base.step_count,

        "team_return":
            team_return,

        "waste":
            base.wasted_allocations,

        "rejections":
            base.review_rejections,
    }

    return trajectory, metrics


def collect_group(
    policies,
    seed,
    group_size,
    resources,
    alpha,
):

    trajectories = []
    metrics = []

    # Every rollout begins from the same seeded
    # project state. Policy sampling creates the
    # alternative joint coordination strategies.
    for _ in range(
        group_size
    ):

        trajectory, result = rollout(
            policies=policies,
            seed=seed,
            resources=resources,
            alpha=alpha,
        )

        trajectories.append(
            trajectory
        )

        metrics.append(
            result
        )

    returns = np.array(
        [
            result["team_return"]
            for result in metrics
        ],
        dtype=np.float32,
    )

    mean = float(
        returns.mean()
    )

    std = float(
        returns.std()
    )

    if std < 1e-8:

        advantages = np.zeros_like(
            returns
        )

    else:

        advantages = (
            returns - mean
        ) / (
            std + 1e-8
        )

    return (
        trajectories,
        metrics,
        advantages,
    )


def update_policies(
    policies,
    optimizers,
    collected_groups,
    clip_epsilon,
    update_epochs,
    entropy_coef,
):

    losses = {
        agent: 0.0
        for agent in AGENTS
    }

    sample_counts = {
        agent: 0
        for agent in AGENTS
    }

    for _ in range(
        update_epochs
    ):

        for agent in AGENTS:

            all_obs = []
            all_actions = []
            all_old_log_probs = []
            all_advantages = []

            for (
                trajectories,
                advantages,
            ) in collected_groups:

                for traj_index, trajectory in enumerate(
                    trajectories
                ):

                    agent_data = trajectory[
                        agent
                    ]

                    length = len(
                        agent_data["actions"]
                    )

                    all_obs.extend(
                        agent_data["obs"]
                    )

                    all_actions.extend(
                        agent_data["actions"]
                    )

                    all_old_log_probs.extend(
                        agent_data[
                            "old_log_probs"
                        ]
                    )

                    all_advantages.extend(
                        [
                            advantages[
                                traj_index
                            ]
                        ] * length
                    )

            if not all_obs:
                continue

            obs_tensor = torch.tensor(
                np.asarray(
                    all_obs,
                    dtype=np.float32,
                ),
                dtype=torch.float32,
            )

            action_tensor = torch.tensor(
                all_actions,
                dtype=torch.long,
            )

            old_log_prob_tensor = (
                torch.tensor(
                    all_old_log_probs,
                    dtype=torch.float32,
                )
            )

            advantage_tensor = (
                torch.tensor(
                    all_advantages,
                    dtype=torch.float32,
                )
            )

            logits = policies[
                agent
            ](
                obs_tensor
            )

            dist = Categorical(
                logits=logits
            )

            new_log_probs = (
                dist.log_prob(
                    action_tensor
                )
            )

            entropy = (
                dist.entropy().mean()
            )

            ratio = torch.exp(
                new_log_probs
                - old_log_prob_tensor
            )

            unclipped = (
                ratio
                * advantage_tensor
            )

            clipped = (
                torch.clamp(
                    ratio,
                    1.0
                    - clip_epsilon,
                    1.0
                    + clip_epsilon,
                )
                * advantage_tensor
            )

            policy_objective = (
                torch.min(
                    unclipped,
                    clipped,
                ).mean()
            )

            loss = (
                -policy_objective
                - entropy_coef
                * entropy
            )

            optimizers[
                agent
            ].zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                policies[
                    agent
                ].parameters(),
                max_norm=1.0,
            )

            optimizers[
                agent
            ].step()

            losses[
                agent
            ] += float(
                loss.item()
            )

            sample_counts[
                agent
            ] += 1

    for agent in AGENTS:

        if sample_counts[
            agent
        ] > 0:

            losses[
                agent
            ] /= sample_counts[
                agent
            ]

    return losses


def evaluate_training_batch(
    all_metrics,
):

    success = np.mean(
        [
            x["success"]
            for x in all_metrics
        ]
    )

    steps = np.mean(
        [
            x["steps"]
            for x in all_metrics
        ]
    )

    reward = np.mean(
        [
            x["team_return"]
            for x in all_metrics
        ]
    )

    waste = np.mean(
        [
            x["waste"]
            for x in all_metrics
        ]
    )

    rejections = np.mean(
        [
            x["rejections"]
            for x in all_metrics
        ]
    )

    return {
        "success":
            success,
        "steps":
            steps,
        "reward":
            reward,
        "waste":
            waste,
        "rejections":
            rejections,
    }


def save_checkpoint(
    policies,
    args,
    iteration,
):

    checkpoint_dir = Path(
        "results/checkpoints/marl_grpo"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        checkpoint_dir
        / "checkpoint.pt"
    )

    torch.save(
        {
            "iteration":
                iteration,

            "resources":
                args.resources,

            "alpha":
                args.alpha,

            "group_size":
                args.group_size,

            "hidden_dim":
                args.hidden_dim,

            "policies":
                {
                    agent:
                        policies[
                            agent
                        ].state_dict()
                    for agent in AGENTS
                },
        },
        checkpoint_path,
    )

    return checkpoint_path


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--groups-per-iteration",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--group-size",
        type=int,
        default=4,
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

    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--clip-epsilon",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--update-epochs",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    np.random.seed(
        args.seed
    )

    torch.manual_seed(
        args.seed
    )

    policies = {
        agent: PolicyNetwork(
            hidden_dim=args.hidden_dim
        )
        for agent in AGENTS
    }

    optimizers = {
        agent: torch.optim.Adam(
            policies[
                agent
            ].parameters(),
            lr=args.lr,
        )
        for agent in AGENTS
    }

    print()
    print("=" * 78)
    print("TRAINING GRPO-STYLE MARL")
    print("=" * 78)
    print(
        f"Resources: {args.resources}"
    )
    print(
        f"Reward alpha: {args.alpha}"
    )
    print(
        f"Group size: {args.group_size}"
    )
    print(
        "Group score: common team/project return"
    )
    print("=" * 78)

    global_group = 0

    for iteration in range(
        1,
        args.iterations + 1,
    ):

        collected_groups = []
        all_metrics = []

        group_advantage_stds = []

        for group_index in range(
            args.groups_per_iteration
        ):

            # Unique project seed per group.
            # All candidate trajectories inside
            # that group use the same seed.
            seed = (
                args.seed
                + global_group
            )

            (
                trajectories,
                metrics,
                advantages,
            ) = collect_group(
                policies=policies,
                seed=seed,
                group_size=args.group_size,
                resources=args.resources,
                alpha=args.alpha,
            )

            collected_groups.append(
                (
                    trajectories,
                    advantages,
                )
            )

            all_metrics.extend(
                metrics
            )

            group_advantage_stds.append(
                float(
                    np.std(
                        advantages
                    )
                )
            )

            global_group += 1

        losses = update_policies(
            policies=policies,
            optimizers=optimizers,
            collected_groups=collected_groups,
            clip_epsilon=args.clip_epsilon,
            update_epochs=args.update_epochs,
            entropy_coef=args.entropy_coef,
        )

        stats = evaluate_training_batch(
            all_metrics
        )

        mean_loss = np.mean(
            list(
                losses.values()
            )
        )

        print(
            f"Iteration {iteration:3d} | "
            f"Success {stats['success']:.3f} | "
            f"Steps {stats['steps']:6.2f} | "
            f"Reward {stats['reward']:7.2f} | "
            f"Waste {stats['waste']:5.2f} | "
            f"Reject {stats['rejections']:5.2f} | "
            f"Loss {mean_loss:8.4f}"
        )

    checkpoint = save_checkpoint(
        policies=policies,
        args=args,
        iteration=args.iterations,
    )

    print()
    print("=" * 78)
    print("GRPO TRAINING COMPLETE")
    print("=" * 78)
    print(
        "Checkpoint:",
        checkpoint.resolve(),
    )


if __name__ == "__main__":
    main()
