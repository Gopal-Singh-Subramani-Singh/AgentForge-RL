from pathlib import Path
import csv

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.utils.data import DataLoader, TensorDataset

import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env

from environment.multi_agent_env import MultiAgentDevTeamEnv


AGENTS = [
    "developer",
    "tester",
    "documentation",
    "reviewer",
]

# ============================================================
# Experiment configuration
# ============================================================

SEED = 42
RESOURCES = 2
ALPHA = 0.25

# PPO teacher
PPO_CHECKPOINT = Path(
    "results/checkpoints/"
    "marl_ppo_alpha_0_25"
).resolve()

# Behavioral cloning
BC_EPISODES = 500
BC_RANDOM_STATE_PROB = 0.20
BC_EPOCHS = 30
BC_BATCH_SIZE = 256
BC_LR = 1e-3

# GRPO
GRPO_ITERATIONS = 30
GROUP_SIZE = 8
GROUPS_PER_ITERATION = 16

# 16 * 8 = same 128 trajectories / iteration
GRPO_LR = 1e-4
CLIP_EPSILON = 0.20
UPDATE_EPOCHS = 4

# Lower exploration than from-scratch GRPO because
# we begin from a strong PPO-derived policy.
ENTROPY_START = 0.003
ENTROPY_END = 0.0005

HIDDEN_DIM = 128

# Evaluation
EVAL_EPISODES = 500

CHECKPOINT_DIR = Path(
    "results/checkpoints/"
    "ppo_guided_grpo_g8"
)

CHECKPOINT_PATH = (
    CHECKPOINT_DIR / "checkpoint.pt"
)

RESULTS_CSV = Path(
    "results/"
    "ppo_guided_grpo_results.csv"
)


# ============================================================
# Model
# ============================================================

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


# ============================================================
# PPO teacher
# ============================================================

def env_creator(config):
    return MultiAgentDevTeamEnv(config)


def teacher_action(
    module,
    obs,
):
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


def collect_teacher_dataset(
    teacher_modules,
):
    """
    Collect PPO-labeled observations.

    Most environment transitions follow PPO itself.
    A small fraction use random actions so that the
    student also sees states slightly outside the
    teacher's normal trajectory.

    The LABEL is always the PPO teacher action.
    """

    print()
    print("=" * 78)
    print("STAGE 1 — COLLECT PPO TEACHER DATA")
    print("=" * 78)

    data = {
        agent: {
            "obs": [],
            "actions": [],
        }
        for agent in AGENTS
    }

    rng = np.random.default_rng(
        SEED
    )

    for seed in range(
        BC_EPISODES
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

        while True:

            teacher_actions = {}

            for agent in AGENTS:

                action = teacher_action(
                    teacher_modules[
                        agent
                    ],
                    observations[
                        agent
                    ],
                )

                teacher_actions[
                    agent
                ] = action

                data[
                    agent
                ]["obs"].append(
                    observations[
                        agent
                    ].copy()
                )

                data[
                    agent
                ]["actions"].append(
                    action
                )

            # Occasionally enter a different state
            # while preserving PPO labels for BC.
            if (
                rng.random()
                < BC_RANDOM_STATE_PROB
            ):

                env_actions = {
                    agent:
                        int(
                            rng.integers(
                                0,
                                2,
                            )
                        )
                    for agent in AGENTS
                }

            else:

                env_actions = (
                    teacher_actions
                )

            (
                observations,
                _,
                terminateds,
                truncateds,
                _,
            ) = env.step(
                env_actions
            )

            if (
                terminateds["__all__"]
                or
                truncateds["__all__"]
            ):
                break

    for agent in AGENTS:

        print(
            f"{agent:15s}: "
            f"{len(data[agent]['actions'])} "
            f"training states"
        )

    return data


# ============================================================
# Behavioral cloning
# ============================================================

def behavioral_clone(
    policies,
    teacher_data,
):
    print()
    print("=" * 78)
    print("STAGE 2 — DISTILL MARL PPO INTO GRPO NETWORKS")
    print("=" * 78)

    loss_fn = nn.CrossEntropyLoss()

    for agent in AGENTS:

        obs = torch.tensor(
            np.asarray(
                teacher_data[
                    agent
                ]["obs"],
                dtype=np.float32,
            ),
            dtype=torch.float32,
        )

        actions = torch.tensor(
            teacher_data[
                agent
            ]["actions"],
            dtype=torch.long,
        )

        dataset = TensorDataset(
            obs,
            actions,
        )

        loader = DataLoader(
            dataset,
            batch_size=BC_BATCH_SIZE,
            shuffle=True,
        )

        optimizer = torch.optim.Adam(
            policies[
                agent
            ].parameters(),
            lr=BC_LR,
        )

        for epoch in range(
            1,
            BC_EPOCHS + 1,
        ):

            total_loss = 0.0
            total_correct = 0
            total_count = 0

            for (
                batch_obs,
                batch_actions,
            ) in loader:

                logits = policies[
                    agent
                ](
                    batch_obs
                )

                loss = loss_fn(
                    logits,
                    batch_actions,
                )

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                total_loss += (
                    float(
                        loss.item()
                    )
                    * len(
                        batch_actions
                    )
                )

                prediction = torch.argmax(
                    logits,
                    dim=-1,
                )

                total_correct += int(
                    (
                        prediction
                        == batch_actions
                    ).sum().item()
                )

                total_count += len(
                    batch_actions
                )

            if (
                epoch == 1
                or epoch % 5 == 0
                or epoch == BC_EPOCHS
            ):

                accuracy = (
                    total_correct
                    / total_count
                )

                avg_loss = (
                    total_loss
                    / total_count
                )

                print(
                    f"{agent:15s} "
                    f"epoch {epoch:2d} | "
                    f"loss {avg_loss:.4f} | "
                    f"accuracy "
                    f"{accuracy:.4f}"
                )


# ============================================================
# Policy action sampling
# ============================================================

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
        int(
            action.item()
        ),
        float(
            log_prob.item()
        ),
    )


# ============================================================
# GRPO rollout
# ============================================================

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
                policies[
                    agent
                ],
                observations[
                    agent
                ],
            )

            actions[
                agent
            ] = action

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

        team_reward = (
            infos[
                "developer"
            ][
                "team_reward"
            ]
        )

        team_return += (
            team_reward
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

    return (
        trajectory,
        metrics,
    )


# ============================================================
# GRPO group collection
# ============================================================

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

    # Role-aware group-relative advantage.
    advantages = {}

    for agent in AGENTS:

        returns = np.asarray(
            [
                result[
                    "role_returns"
                ][
                    agent
                ]
                for result
                in metrics
            ],
            dtype=np.float32,
        )

        std = float(
            returns.std()
        )

        if std < 1e-8:

            advantages[
                agent
            ] = np.zeros_like(
                returns
            )

        else:

            advantages[
                agent
            ] = (
                returns
                - returns.mean()
            ) / (
                std
                + 1e-8
            )

    return (
        trajectories,
        metrics,
        advantages,
    )


# ============================================================
# GRPO policy update
# ============================================================

def update_policies(
    policies,
    optimizers,
    collected_groups,
    entropy_coef,
):

    losses = {
        agent: []
        for agent in AGENTS
    }

    for _ in range(
        UPDATE_EPOCHS
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

                for (
                    traj_index,
                    trajectory,
                ) in enumerate(
                    trajectories
                ):

                    agent_data = (
                        trajectory[
                            agent
                        ]
                    )

                    length = len(
                        agent_data[
                            "actions"
                        ]
                    )

                    all_obs.extend(
                        agent_data[
                            "obs"
                        ]
                    )

                    all_actions.extend(
                        agent_data[
                            "actions"
                        ]
                    )

                    all_old_log_probs.extend(
                        agent_data[
                            "old_log_probs"
                        ]
                    )

                    all_advantages.extend(
                        [
                            advantages[
                                agent
                            ][
                                traj_index
                            ]
                        ]
                        * length
                    )

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
                    - CLIP_EPSILON,
                    1.0
                    + CLIP_EPSILON,
                )
                * advantage_tensor
            )

            policy_objective = (
                torch.min(
                    unclipped,
                    clipped,
                ).mean()
            )

            entropy = (
                dist.entropy().mean()
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
            ].append(
                float(
                    loss.item()
                )
            )

    return {
        agent:
            float(
                np.mean(
                    losses[
                        agent
                    ]
                )
            )
        for agent in AGENTS
    }


# ============================================================
# GRPO fine tuning
# ============================================================

def entropy_for_iteration(
    iteration,
):

    fraction = (
        (iteration - 1)
        / max(
            GRPO_ITERATIONS - 1,
            1,
        )
    )

    return (
        ENTROPY_START
        + fraction
        * (
            ENTROPY_END
            - ENTROPY_START
        )
    )


def grpo_fine_tune(
    policies,
):

    print()
    print("=" * 78)
    print("STAGE 3 — PPO-GUIDED GRPO FINE-TUNING")
    print("=" * 78)

    print(
        f"Iterations: "
        f"{GRPO_ITERATIONS}"
    )

    print(
        f"Group size: "
        f"{GROUP_SIZE}"
    )

    print(
        f"Groups / iteration: "
        f"{GROUPS_PER_ITERATION}"
    )

    print(
        "Trajectories / iteration: "
        f"{GROUP_SIZE * GROUPS_PER_ITERATION}"
    )

    print(
        "Credit assignment: "
        "0.25 individual + "
        "0.75 team"
    )

    print(
        "Entropy: "
        f"{ENTROPY_START} "
        f"-> {ENTROPY_END}"
    )

    optimizers = {
        agent:
            torch.optim.Adam(
                policies[
                    agent
                ].parameters(),
                lr=GRPO_LR,
            )
        for agent in AGENTS
    }

    global_group = 0

    for iteration in range(
        1,
        GRPO_ITERATIONS + 1,
    ):

        entropy_coef = (
            entropy_for_iteration(
                iteration
            )
        )

        collected_groups = []
        all_metrics = []

        for _ in range(
            GROUPS_PER_ITERATION
        ):

            group_seed = (
                SEED
                + global_group
            )

            (
                trajectories,
                metrics,
                advantages,
            ) = collect_group(
                policies,
                group_seed,
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

            global_group += 1

        losses = update_policies(
            policies,
            optimizers,
            collected_groups,
            entropy_coef,
        )

        success = float(
            np.mean(
                [
                    x["success"]
                    for x in all_metrics
                ]
            )
        )

        steps = float(
            np.mean(
                [
                    x["steps"]
                    for x in all_metrics
                ]
            )
        )

        reward = float(
            np.mean(
                [
                    x[
                        "team_return"
                    ]
                    for x in all_metrics
                ]
            )
        )

        waste = float(
            np.mean(
                [
                    x["waste"]
                    for x in all_metrics
                ]
            )
        )

        reject = float(
            np.mean(
                [
                    x[
                        "rejections"
                    ]
                    for x in all_metrics
                ]
            )
        )

        mean_loss = float(
            np.mean(
                list(
                    losses.values()
                )
            )
        )

        print(
            f"Iteration "
            f"{iteration:3d} | "
            f"Success "
            f"{success:.3f} | "
            f"Steps "
            f"{steps:6.2f} | "
            f"Reward "
            f"{reward:7.2f} | "
            f"Waste "
            f"{waste:5.2f} | "
            f"Reject "
            f"{reject:5.2f} | "
            f"Entropy "
            f"{entropy_coef:.5f} | "
            f"Loss "
            f"{mean_loss:.4f}"
        )


# ============================================================
# Evaluation
# ============================================================

def eval_action(
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
        EVAL_EPISODES
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

        observations, _ = env.reset(
            seed=seed
        )

        team_reward = 0.0

        while True:

            actions = {
                agent:
                    eval_action(
                        policies[
                            agent
                        ],
                        observations[
                            agent
                        ],
                        stochastic,
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


# ============================================================
# Checkpoint
# ============================================================

def save_checkpoint(
    policies,
):

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "method":
                "ppo_guided_grpo",

            "seed":
                SEED,

            "resources":
                RESOURCES,

            "alpha":
                ALPHA,

            "group_size":
                GROUP_SIZE,

            "groups_per_iteration":
                GROUPS_PER_ITERATION,

            "iterations":
                GRPO_ITERATIONS,

            "hidden_dim":
                HIDDEN_DIM,

            "policies":
                {
                    agent:
                        policies[
                            agent
                        ].state_dict()
                    for agent in AGENTS
                },
        },
        CHECKPOINT_PATH,
    )


# ============================================================
# Main
# ============================================================

def main():

    np.random.seed(
        SEED
    )

    torch.manual_seed(
        SEED
    )

    # --------------------------------------------------------
    # Load PPO teacher
    # --------------------------------------------------------

    register_env(
        "multi_agent_dev_team",
        env_creator,
    )

    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
    )

    print()
    print("=" * 78)
    print("PPO-GUIDED GRPO-STYLE MARL")
    print("=" * 78)

    print(
        "Teacher checkpoint:",
        PPO_CHECKPOINT,
    )

    teacher = (
        Algorithm.from_checkpoint(
            str(
                PPO_CHECKPOINT
            )
        )
    )

    teacher_modules = {
        agent:
            teacher.get_module(
                f"{agent}_policy"
            )
        for agent in AGENTS
    }

    # --------------------------------------------------------
    # Teacher data
    # --------------------------------------------------------

    teacher_data = (
        collect_teacher_dataset(
            teacher_modules
        )
    )

    # --------------------------------------------------------
    # Student policies
    # --------------------------------------------------------

    policies = {
        agent:
            PolicyNetwork(
                hidden_dim=
                    HIDDEN_DIM
            )
        for agent in AGENTS
    }

    # --------------------------------------------------------
    # PPO behavioral cloning
    # --------------------------------------------------------

    behavioral_clone(
        policies,
        teacher_data,
    )

    # Teacher no longer needed.
    teacher.stop()
    ray.shutdown()

    # --------------------------------------------------------
    # GRPO fine-tune
    # --------------------------------------------------------

    grpo_fine_tune(
        policies
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_checkpoint(
        policies
    )

    # --------------------------------------------------------
    # Final 500-episode evaluation
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print(
        "STAGE 4 — FINAL "
        "500-EPISODE EVALUATION"
    )
    print("=" * 78)

    greedy = evaluate(
        policies,
        stochastic=False,
    )

    sampled = evaluate(
        policies,
        stochastic=True,
    )

    print()
    print(
        f"{'Method':22s}"
        f"{'Success':>10s}"
        f"{'Steps':>10s}"
        f"{'Reward':>12s}"
        f"{'Waste':>10s}"
        f"{'Reject':>10s}"
    )

    print("-" * 74)

    for (
        name,
        result,
    ) in [
        (
            "PPO-GRPO greedy",
            greedy,
        ),
        (
            "PPO-GRPO sample",
            sampled,
        ),
    ]:

        print(
            f"{name:22s}"
            f"{result['success']:>10.3f}"
            f"{result['steps']:>10.2f}"
            f"{result['reward']:>12.2f}"
            f"{result['waste']:>10.2f}"
            f"{result['reject']:>10.2f}"
        )

    RESULTS_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RESULTS_CSV.open(
        "w",
        newline="",
    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow(
            [
                "method",
                "success",
                "steps",
                "reward",
                "waste",
                "reject",
            ]
        )

        for (
            name,
            result,
        ) in [
            (
                "ppo_grpo_greedy",
                greedy,
            ),
            (
                "ppo_grpo_sample",
                sampled,
            ),
        ]:

            writer.writerow(
                [
                    name,
                    result[
                        "success"
                    ],
                    result[
                        "steps"
                    ],
                    result[
                        "reward"
                    ],
                    result[
                        "waste"
                    ],
                    result[
                        "reject"
                    ],
                ]
            )

    print()
    print(
        "Checkpoint:",
        CHECKPOINT_PATH.resolve(),
    )

    print(
        "Results:",
        RESULTS_CSV.resolve(),
    )

    print()
    print("=" * 78)
    print("REFERENCE TARGETS")
    print("=" * 78)

    print(
        "MARL PPO broad evaluation:"
    )
    print(
        "  Steps  = 10.33"
    )
    print(
        "  Reward = 45.77"
    )
    print(
        "  Waste  = 0.00"
    )

    print()
    print(
        "Original GRPO G8 stochastic:"
    )
    print(
        "  Steps  = 15.18"
    )
    print(
        "  Reward = 39.84"
    )
    print(
        "  Waste  = 6.50"
    )
    print(
        "  Reject = 0.45"
    )


if __name__ == "__main__":
    main()
