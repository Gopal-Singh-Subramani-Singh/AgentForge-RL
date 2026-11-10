import csv
import random
from pathlib import Path

import numpy as np
import ray
import torch
import torch.nn as nn

from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env

from environment.marl_env import DevTeamEnv
from environment.multi_agent_env import MultiAgentDevTeamEnv

from schedulers.random_scheduler import RandomScheduler
from schedulers.rule_based import RuleBasedScheduler


AGENTS = [
    "developer",
    "tester",
    "documentation",
    "reviewer",
]

SEED = 0
RESOURCES = 2

PPO_CHECKPOINT = Path(
    "results/checkpoints/ppo"
).resolve()

MARL_PPO_CHECKPOINT = Path(
    "results/checkpoints/marl_ppo_alpha_0_25"
).resolve()

# Use the strongest fair-budget GRPO group-size model.
GRPO_CHECKPOINT = Path(
    "results/checkpoints/marl_grpo_g8/checkpoint.pt"
).resolve()


def central_env_creator(config):
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


def marl_env_creator(config):
    return MultiAgentDevTeamEnv(
        config
    )


def state_dict(env):
    return {
        "dev": env.dev_tasks,
        "untested": env.untested_work,
        "bugs": env.bugs,
        "docs": round(
            env.docs_progress,
            2,
        ),
        "docs_done":
            env.docs_complete,
    }


def central_ppo_action(
    module,
    obs,
):
    obs_tensor = torch.tensor(
        obs,
        dtype=torch.float32,
    ).unsqueeze(0)

    with torch.no_grad():
        output = (
            module.forward_inference(
                {"obs": obs_tensor}
            )
        )

    logits = output[
        "action_dist_inputs"
    ]

    chunks = torch.split(
        logits,
        [2, 2, 2, 2],
        dim=-1,
    )

    return np.array(
        [
            int(
                torch.argmax(
                    chunk,
                    dim=-1,
                ).item()
            )
            for chunk in chunks
        ],
        dtype=np.int64,
    )


def single_agent_action(
    module,
    obs,
):
    obs_tensor = torch.tensor(
        obs,
        dtype=torch.float32,
    ).unsqueeze(0)

    with torch.no_grad():
        output = (
            module.forward_inference(
                {"obs": obs_tensor}
            )
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


def grpo_sample_action(
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
        ).squeeze(0)

    distribution = (
        torch.distributions.Categorical(
            logits=logits
        )
    )

    return int(
        distribution.sample().item()
    )


def run_classic_scheduler(
    name,
    scheduler,
):
    random.seed(SEED)
    np.random.seed(SEED)

    env = DevTeamEnv(
        num_resources=RESOURCES,
        max_steps=50,
        bug_probability=0.20,
    )

    obs, _ = env.reset(
        seed=SEED
    )

    trace = []
    total_reward = 0.0

    while True:
        before = state_dict(env)

        action = scheduler.act(env)

        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        total_reward += reward

        trace.append(
            {
                "algorithm": name,
                "step": env.step_count - 1,
                "state": str(before),
                "requested": ", ".join(
                    info[
                        "requested_agents"
                    ]
                ),
                "allocated": ", ".join(
                    info[
                        "allocated_agents"
                    ]
                ),
                "waste_step":
                    info[
                        "wasted_allocations_step"
                    ],
            }
        )

        if terminated or truncated:
            break

    return trace, {
        "algorithm": name,
        "success": info["success"],
        "steps": env.step_count,
        "reward": total_reward,
        "waste":
            env.wasted_allocations,
        "rejections":
            env.review_rejections,
        **{
            f"{agent}_alloc":
                env.total_allocations[
                    agent
                ]
            for agent in AGENTS
        },
    }


def run_central_ppo(
    algo,
):
    env = DevTeamEnv(
        num_resources=RESOURCES,
        max_steps=50,
        bug_probability=0.20,
    )

    obs, _ = env.reset(
        seed=SEED
    )

    module = algo.get_module(
        "default_policy"
    )

    trace = []
    total_reward = 0.0

    while True:
        before = state_dict(env)

        action = central_ppo_action(
            module,
            obs,
        )

        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        total_reward += reward

        trace.append(
            {
                "algorithm":
                    "Centralized PPO",
                "step":
                    env.step_count - 1,
                "state":
                    str(before),
                "requested":
                    ", ".join(
                        info[
                            "requested_agents"
                        ]
                    ),
                "allocated":
                    ", ".join(
                        info[
                            "allocated_agents"
                        ]
                    ),
                "waste_step":
                    info[
                        "wasted_allocations_step"
                    ],
            }
        )

        if terminated or truncated:
            break

    return trace, {
        "algorithm":
            "Centralized PPO",
        "success":
            info["success"],
        "steps":
            env.step_count,
        "reward":
            total_reward,
        "waste":
            env.wasted_allocations,
        "rejections":
            env.review_rejections,
        **{
            f"{agent}_alloc":
                env.total_allocations[
                    agent
                ]
            for agent in AGENTS
        },
    }


def run_marl_ppo(
    algo,
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
                0.25,
        }
    )

    observations, _ = env.reset(
        seed=SEED
    )

    modules = {
        agent:
            algo.get_module(
                f"{agent}_policy"
            )
        for agent in AGENTS
    }

    trace = []
    team_reward = 0.0

    while True:
        base = env.base_env
        before = state_dict(base)

        actions = {
            agent:
                single_agent_action(
                    modules[agent],
                    observations[
                        agent
                    ],
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

        info = infos["developer"]

        team_reward += (
            info["team_reward"]
        )

        trace.append(
            {
                "algorithm":
                    "MARL PPO",
                "step":
                    base.step_count - 1,
                "state":
                    str(before),
                "requested":
                    ", ".join(
                        info[
                            "requested_agents"
                        ]
                    ),
                "allocated":
                    ", ".join(
                        info[
                            "allocated_agents"
                        ]
                    ),
                "waste_step":
                    info[
                        "wasted_allocations_step"
                    ],
            }
        )

        if (
            terminateds["__all__"]
            or
            truncateds["__all__"]
        ):
            break

    return trace, {
        "algorithm":
            "MARL PPO",
        "success":
            info["success"],
        "steps":
            base.step_count,
        "reward":
            team_reward,
        "waste":
            base.wasted_allocations,
        "rejections":
            base.review_rejections,
        **{
            f"{agent}_alloc":
                base.total_allocations[
                    agent
                ]
            for agent in AGENTS
        },
    }


def load_grpo():
    checkpoint = torch.load(
        GRPO_CHECKPOINT,
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


def run_grpo(
    policies,
):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env = MultiAgentDevTeamEnv(
        {
            "num_resources":
                RESOURCES,
            "max_steps":
                50,
            "bug_probability":
                0.20,
            "alpha":
                0.25,
        }
    )

    observations, _ = env.reset(
        seed=SEED
    )

    trace = []
    team_reward = 0.0

    while True:
        base = env.base_env
        before = state_dict(base)

        actions = {
            agent:
                grpo_sample_action(
                    policies[agent],
                    observations[
                        agent
                    ],
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

        info = infos["developer"]

        team_reward += (
            info["team_reward"]
        )

        trace.append(
            {
                "algorithm":
                    "GRPO-style MARL",
                "step":
                    base.step_count - 1,
                "state":
                    str(before),
                "requested":
                    ", ".join(
                        info[
                            "requested_agents"
                        ]
                    ),
                "allocated":
                    ", ".join(
                        info[
                            "allocated_agents"
                        ]
                    ),
                "waste_step":
                    info[
                        "wasted_allocations_step"
                    ],
            }
        )

        if (
            terminateds["__all__"]
            or
            truncateds["__all__"]
        ):
            break

    return trace, {
        "algorithm":
            "GRPO-style MARL",
        "success":
            info["success"],
        "steps":
            base.step_count,
        "reward":
            team_reward,
        "waste":
            base.wasted_allocations,
        "rejections":
            base.review_rejections,
        **{
            f"{agent}_alloc":
                base.total_allocations[
                    agent
                ]
            for agent in AGENTS
        },
    }


def print_trace(
    trace,
):
    print()
    print(
        "=" * 90
    )
    print(
        trace[0]["algorithm"]
    )
    print(
        "=" * 90
    )

    for row in trace:
        print(
            f"Step {row['step']:2d} | "
            f"requested: "
            f"{row['requested'] or '-':35s} | "
            f"allocated: "
            f"{row['allocated'] or '-'}"
        )


def main():
    register_env(
        "dev_team_env",
        central_env_creator,
    )

    register_env(
        "multi_agent_dev_team",
        marl_env_creator,
    )

    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
    )

    all_traces = []
    summaries = []

    # ---------------------------------
    # Random
    # ---------------------------------
    trace, summary = (
        run_classic_scheduler(
            "Random",
            RandomScheduler(),
        )
    )

    all_traces += trace
    summaries.append(summary)
    print_trace(trace)

    # ---------------------------------
    # Rule-based
    # ---------------------------------
    trace, summary = (
        run_classic_scheduler(
            "Rule-based",
            RuleBasedScheduler(),
        )
    )

    all_traces += trace
    summaries.append(summary)
    print_trace(trace)

    # ---------------------------------
    # Centralized PPO
    # ---------------------------------
    print(
        "\nLoading centralized PPO..."
    )

    ppo = Algorithm.from_checkpoint(
        str(PPO_CHECKPOINT)
    )

    trace, summary = (
        run_central_ppo(ppo)
    )

    all_traces += trace
    summaries.append(summary)
    print_trace(trace)

    ppo.stop()

    # ---------------------------------
    # MARL PPO
    # ---------------------------------
    print(
        "\nLoading MARL PPO..."
    )

    marl_ppo = (
        Algorithm.from_checkpoint(
            str(
                MARL_PPO_CHECKPOINT
            )
        )
    )

    trace, summary = (
        run_marl_ppo(
            marl_ppo
        )
    )

    all_traces += trace
    summaries.append(summary)
    print_trace(trace)

    marl_ppo.stop()

    # ---------------------------------
    # GRPO
    # ---------------------------------
    print(
        "\nLoading GRPO-style MARL..."
    )

    policies = load_grpo()

    trace, summary = run_grpo(
        policies
    )

    all_traces += trace
    summaries.append(summary)
    print_trace(trace)

    ray.shutdown()

    # ---------------------------------
    # Summary
    # ---------------------------------
    print()
    print(
        "=" * 110
    )
    print(
        "ALGORITHM SUMMARY — SAME SEED"
    )
    print(
        "=" * 110
    )

    header = (
        f"{'Algorithm':20s}"
        f"{'Success':>9s}"
        f"{'Steps':>8s}"
        f"{'Reward':>10s}"
        f"{'Waste':>8s}"
        f"{'Reject':>8s}"
        f"{'Dev':>7s}"
        f"{'Test':>7s}"
        f"{'Docs':>7s}"
        f"{'Review':>8s}"
    )

    print(header)
    print("-" * 110)

    for s in summaries:
        print(
            f"{s['algorithm']:20s}"
            f"{str(s['success']):>9s}"
            f"{s['steps']:>8d}"
            f"{s['reward']:>10.2f}"
            f"{s['waste']:>8d}"
            f"{s['rejections']:>8d}"
            f"{s['developer_alloc']:>7d}"
            f"{s['tester_alloc']:>7d}"
            f"{s['documentation_alloc']:>7d}"
            f"{s['reviewer_alloc']:>8d}"
        )

    # ---------------------------------
    # Save CSVs
    # ---------------------------------
    trace_path = Path(
        "results/"
        "allocation_traces.csv"
    )

    with trace_path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "algorithm",
                "step",
                "state",
                "requested",
                "allocated",
                "waste_step",
            ],
        )

        writer.writeheader()
        writer.writerows(
            all_traces
        )

    summary_path = Path(
        "results/"
        "allocation_summary.csv"
    )

    with summary_path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "algorithm",
                "success",
                "steps",
                "reward",
                "waste",
                "rejections",
                "developer_alloc",
                "tester_alloc",
                "documentation_alloc",
                "reviewer_alloc",
            ],
        )

        writer.writeheader()
        writer.writerows(
            summaries
        )

    print()
    print(
        "Saved:",
        trace_path,
    )

    print(
        "Saved:",
        summary_path,
    )


if __name__ == "__main__":
    main()
