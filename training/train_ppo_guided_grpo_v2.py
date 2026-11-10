from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from environment.multi_agent_env import MultiAgentDevTeamEnv


AGENTS = [
    "developer",
    "tester",
    "documentation",
    "reviewer",
]

# ------------------------------------------------------------
# Starting checkpoint
# ------------------------------------------------------------

SOURCE_CHECKPOINT = Path(
    "results/checkpoints/"
    "ppo_guided_grpo_g8/"
    "checkpoint.pt"
).resolve()

OUTPUT_DIR = Path(
    "results/checkpoints/"
    "ppo_guided_grpo_v2_best"
)

OUTPUT_CHECKPOINT = (
    OUTPUT_DIR / "checkpoint.pt"
)

# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------

SEED = 42
RESOURCES = 2
ALPHA = 0.25

# ------------------------------------------------------------
# Fine-tuning
# ------------------------------------------------------------

ITERATIONS = 15
GROUP_SIZE = 8
GROUPS_PER_ITERATION = 16

# Still 128 trajectories / iteration.
LR = 3e-5
CLIP_EPSILON = 0.10
UPDATE_EPOCHS = 4

# Very low exploration because the starting policy
# is already strong.
ENTROPY_COEF = 1e-4

# Extra coordination penalties used ONLY for
# GRPO credit assignment.
WASTE_PENALTY = 2.0
REJECT_PENALTY = 3.0

# ------------------------------------------------------------
# Validation / final evaluation
# ------------------------------------------------------------

VALIDATION_EPISODES = 200
FINAL_EPISODES = 500

# Held-out seeds, separate from training groups.
VALIDATION_SEED_START = 5000


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
        SOURCE_CHECKPOINT,
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

        policy.train()

        policies[agent] = policy

    return policies, hidden_dim


def sample_action(
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
):

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

    observations, _ = env.reset(
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

    role_returns = {
        agent: 0.0
        for agent in AGENTS
    }

    while True:

        actions = {}

        for agent in AGENTS:

            (
                action,
                log_prob,
            ) = sample_action(
                policies[agent],
                observations[agent],
            )

            actions[agent] = action

            trajectory[
                agent
            ]["obs"].append(
                observations[
                    agent
                ].copy()
            )

            trajectory[
                agent
            ]["actions"].append(
                action
            )

            trajectory[
                agent
            ][
                "old_log_probs"
            ].append(
                log_prob
            )

        (
            observations,
            _,
            terminateds,
            truncateds,
            infos,
        ) = env.step(
            actions
        )

        team_return += (
            infos[
                "developer"
            ][
                "team_reward"
            ]
        )

        for agent in AGENTS:

            role_returns[
                agent
            ] += (
                infos[
                    agent
                ][
                    "hybrid_reward"
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
            infos[
                "developer"
            ][
                "success"
            ],

        "steps":
            base.step_count,

        "team_return":
            team_return,

        "role_returns":
            role_returns,

        "waste":
            base.wasted_allocations,

        "rejections":
            base.review_rejections,
    }

    return trajectory, metrics


def collect_group(
    policies,
    seed,
):

    trajectories = []
    metrics = []

    for _ in range(
        GROUP_SIZE
    ):

        trajectory, result = rollout(
            policies,
            seed,
        )

        trajectories.append(
            trajectory
        )

        metrics.append(
            result
        )

    advantages = {}

    for agent in AGENTS:

        # Role-specific return plus explicit coordination
        # penalties.
        training_returns = np.asarray(
            [
                (
                    result[
                        "role_returns"
                    ][agent]
                    - WASTE_PENALTY
                    * result[
                        "waste"
                    ]
                    - REJECT_PENALTY
                    * result[
                        "rejections"
                    ]
                )
                for result
                in metrics
            ],
            dtype=np.float32,
        )

        std = float(
            training_returns.std()
        )

        if std < 1e-8:

            advantages[
                agent
            ] = np.zeros_like(
                training_returns
            )

        else:

            advantages[
                agent
            ] = (
                training_returns
                - training_returns.mean()
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
    groups,
):

    losses = {
        agent: []
        for agent in AGENTS
    }

    for _ in range(
        UPDATE_EPOCHS
    ):

        for agent in AGENTS:

            obs_list = []
            action_list = []
            old_log_prob_list = []
            advantage_list = []

            for (
                trajectories,
                advantages,
            ) in groups:

                for (
                    index,
                    trajectory,
                ) in enumerate(
                    trajectories
                ):

                    data = trajectory[
                        agent
                    ]

                    length = len(
                        data["actions"]
                    )

                    obs_list.extend(
                        data["obs"]
                    )

                    action_list.extend(
                        data["actions"]
                    )

                    old_log_prob_list.extend(
                        data[
                            "old_log_probs"
                        ]
                    )

                    advantage_list.extend(
                        [
                            advantages[
                                agent
                            ][index]
                        ]
                        * length
                    )

            obs_tensor = torch.tensor(
                np.asarray(
                    obs_list,
                    dtype=np.float32,
                ),
                dtype=torch.float32,
            )

            actions_tensor = (
                torch.tensor(
                    action_list,
                    dtype=torch.long,
                )
            )

            old_log_probs = (
                torch.tensor(
                    old_log_prob_list,
                    dtype=torch.float32,
                )
            )

            advantages_tensor = (
                torch.tensor(
                    advantage_list,
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
                    actions_tensor
                )
            )

            ratio = torch.exp(
                new_log_probs
                - old_log_probs
            )

            raw_objective = (
                ratio
                * advantages_tensor
            )

            clipped_objective = (
                torch.clamp(
                    ratio,
                    1.0
                    - CLIP_EPSILON,
                    1.0
                    + CLIP_EPSILON,
                )
                * advantages_tensor
            )

            objective = torch.min(
                raw_objective,
                clipped_objective,
            ).mean()

            entropy = (
                dist.entropy().mean()
            )

            loss = (
                -objective
                - ENTROPY_COEF
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
            ].append(
                float(
                    loss.item()
                )
            )

    return float(
        np.mean(
            [
                np.mean(
                    losses[agent]
                )
                for agent in AGENTS
            ]
        )
    )


def greedy_action(
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


def evaluate(
    policies,
    episodes,
    seed_start,
):

    results = []

    for i in range(
        episodes
    ):

        seed = seed_start + i

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

        observations, _ = env.reset(
            seed=seed
        )

        team_return = 0.0

        while True:

            actions = {
                agent:
                    greedy_action(
                        policies[
                            agent
                        ],
                        observations[
                            agent
                        ],
                    )
                for agent in AGENTS
            }

            (
                observations,
                _,
                terminateds,
                truncateds,
                infos,
            ) = env.step(
                actions
            )

            team_return += (
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
                    team_return,

                "waste":
                    base.wasted_allocations,

                "reject":
                    base.review_rejections,
            }
        )

    return {
        "success":
            float(
                np.mean(
                    [
                        x["success"]
                        for x in results
                    ]
                )
            ),

        "steps":
            float(
                np.mean(
                    [
                        x["steps"]
                        for x in results
                    ]
                )
            ),

        "reward":
            float(
                np.mean(
                    [
                        x["reward"]
                        for x in results
                    ]
                )
            ),

        "waste":
            float(
                np.mean(
                    [
                        x["waste"]
                        for x in results
                    ]
                )
            ),

        "reject":
            float(
                np.mean(
                    [
                        x["reject"]
                        for x in results
                    ]
                )
            ),
    }


def validation_score(
    result,
):

    # Success is mandatory.
    if result["success"] < 0.999:
        return -1e6

    # Selection score only.
    # Final reporting still uses normal environment metrics.
    return (
        result["reward"]
        - 3.0
        * result["waste"]
        - 5.0
        * result["reject"]
        - 0.25
        * result["steps"]
    )


def save_checkpoint(
    policies,
    hidden_dim,
    iteration,
    validation,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "method":
                "ppo_guided_grpo_v2",

            "source_checkpoint":
                str(
                    SOURCE_CHECKPOINT
                ),

            "iteration":
                iteration,

            "hidden_dim":
                hidden_dim,

            "resources":
                RESOURCES,

            "alpha":
                ALPHA,

            "group_size":
                GROUP_SIZE,

            "groups_per_iteration":
                GROUPS_PER_ITERATION,

            "lr":
                LR,

            "clip_epsilon":
                CLIP_EPSILON,

            "entropy_coef":
                ENTROPY_COEF,

            "waste_penalty":
                WASTE_PENALTY,

            "reject_penalty":
                REJECT_PENALTY,

            "validation":
                validation,

            "policies":
                {
                    agent:
                        policies[
                            agent
                        ].state_dict()
                    for agent in AGENTS
                },
        },
        OUTPUT_CHECKPOINT,
    )


def restore_state(
    policies,
    state,
):

    for agent in AGENTS:

        policies[
            agent
        ].load_state_dict(
            state[
                agent
            ]
        )


def main():

    np.random.seed(
        SEED
    )

    torch.manual_seed(
        SEED
    )

    policies, hidden_dim = (
        load_policies()
    )

    optimizers = {
        agent:
            torch.optim.Adam(
                policies[
                    agent
                ].parameters(),
                lr=LR,
            )
        for agent in AGENTS
    }

    print()
    print("=" * 86)
    print(
        "CONSTRAINED PPO-GUIDED "
        "GRPO V2"
    )
    print("=" * 86)

    print(
        "Starting checkpoint:",
        SOURCE_CHECKPOINT,
    )

    print(
        f"Iterations: {ITERATIONS}"
    )

    print(
        f"Trajectories / iteration: "
        f"{GROUP_SIZE * GROUPS_PER_ITERATION}"
    )

    print(
        f"LR: {LR}"
    )

    print(
        f"Clip: {CLIP_EPSILON}"
    )

    print(
        f"Entropy: {ENTROPY_COEF}"
    )

    print(
        "Training penalties: "
        f"waste={WASTE_PENALTY}, "
        f"reject={REJECT_PENALTY}"
    )

    # --------------------------------------------------------
    # Baseline validation before any new training
    # --------------------------------------------------------

    initial_validation = evaluate(
        policies,
        VALIDATION_EPISODES,
        VALIDATION_SEED_START,
    )

    best_score = validation_score(
        initial_validation
    )

    best_state = {
        agent:
            deepcopy(
                policies[
                    agent
                ].state_dict()
            )
        for agent in AGENTS
    }

    best_iteration = 0
    best_validation = (
        initial_validation.copy()
    )

    print()
    print(
        "Initial held-out validation:"
    )

    print(
        f"Success "
        f"{initial_validation['success']:.3f} | "
        f"Steps "
        f"{initial_validation['steps']:.2f} | "
        f"Reward "
        f"{initial_validation['reward']:.2f} | "
        f"Waste "
        f"{initial_validation['waste']:.2f} | "
        f"Reject "
        f"{initial_validation['reject']:.2f} | "
        f"Score "
        f"{best_score:.2f}"
    )

    global_group = 0

    for iteration in range(
        1,
        ITERATIONS + 1,
    ):

        groups = []
        metrics = []

        for _ in range(
            GROUPS_PER_ITERATION
        ):

            seed = (
                SEED
                + global_group
            )

            (
                trajectories,
                group_metrics,
                advantages,
            ) = collect_group(
                policies,
                seed,
            )

            groups.append(
                (
                    trajectories,
                    advantages,
                )
            )

            metrics.extend(
                group_metrics
            )

            global_group += 1

        loss = update_policies(
            policies,
            optimizers,
            groups,
        )

        train_steps = float(
            np.mean(
                [
                    x["steps"]
                    for x in metrics
                ]
            )
        )

        train_reward = float(
            np.mean(
                [
                    x["team_return"]
                    for x in metrics
                ]
            )
        )

        train_waste = float(
            np.mean(
                [
                    x["waste"]
                    for x in metrics
                ]
            )
        )

        train_reject = float(
            np.mean(
                [
                    x["rejections"]
                    for x in metrics
                ]
            )
        )

        validation = evaluate(
            policies,
            VALIDATION_EPISODES,
            VALIDATION_SEED_START,
        )

        score = validation_score(
            validation
        )

        improved = (
            score > best_score
        )

        if improved:

            best_score = score
            best_iteration = iteration

            best_validation = (
                validation.copy()
            )

            best_state = {
                agent:
                    deepcopy(
                        policies[
                            agent
                        ].state_dict()
                    )
                for agent in AGENTS
            }

            save_checkpoint(
                policies,
                hidden_dim,
                iteration,
                validation,
            )

        marker = (
            "  <-- BEST"
            if improved
            else ""
        )

        print(
            f"Iter {iteration:2d} | "
            f"Train steps "
            f"{train_steps:5.2f} "
            f"reward "
            f"{train_reward:6.2f} "
            f"waste "
            f"{train_waste:4.2f} "
            f"reject "
            f"{train_reject:4.2f} | "
            f"VAL steps "
            f"{validation['steps']:5.2f} "
            f"reward "
            f"{validation['reward']:6.2f} "
            f"waste "
            f"{validation['waste']:4.2f} "
            f"reject "
            f"{validation['reject']:4.2f} "
            f"score "
            f"{score:6.2f} | "
            f"loss "
            f"{loss:.4f}"
            f"{marker}"
        )

    # If no fine-tuned iteration beat iteration 0,
    # preserve the starting policy as the best model.
    if best_iteration == 0:

        restore_state(
            policies,
            best_state,
        )

        save_checkpoint(
            policies,
            hidden_dim,
            0,
            best_validation,
        )

    else:

        restore_state(
            policies,
            best_state,
        )

    print()
    print("=" * 86)
    print("BEST HELD-OUT CHECKPOINT")
    print("=" * 86)

    print(
        f"Iteration: {best_iteration}"
    )

    print(
        f"Success: "
        f"{best_validation['success']:.3f}"
    )

    print(
        f"Steps: "
        f"{best_validation['steps']:.2f}"
    )

    print(
        f"Reward: "
        f"{best_validation['reward']:.2f}"
    )

    print(
        f"Waste: "
        f"{best_validation['waste']:.2f}"
    )

    print(
        f"Reject: "
        f"{best_validation['reject']:.2f}"
    )

    # --------------------------------------------------------
    # Final comparison using the same 0..499 seeds used
    # by the previous GRPO evaluations.
    # --------------------------------------------------------

    final = evaluate(
        policies,
        FINAL_EPISODES,
        0,
    )

    print()
    print("=" * 86)
    print(
        "FINAL 500-EPISODE "
        "GREEDY EVALUATION"
    )
    print("=" * 86)

    print(
        f"Success: {final['success']:.3f}"
    )

    print(
        f"Steps:   {final['steps']:.2f}"
    )

    print(
        f"Reward:  {final['reward']:.2f}"
    )

    print(
        f"Waste:   {final['waste']:.2f}"
    )

    print(
        f"Reject:  {final['reject']:.2f}"
    )

    print()
    print(
        "MARL PPO target:"
    )

    print(
        "Success 1.000 | "
        "Steps 10.33 | "
        "Reward 45.77 | "
        "Waste 0.00"
    )

    print()
    print(
        "Checkpoint:",
        OUTPUT_CHECKPOINT.resolve(),
    )


if __name__ == "__main__":
    main()
