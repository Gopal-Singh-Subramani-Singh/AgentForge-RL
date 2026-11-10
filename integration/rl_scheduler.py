from pathlib import Path

import numpy as np
import torch

from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env

from environment.multi_agent_env import MultiAgentDevTeamEnv


AGENTS = [
    "developer",
    "tester",
    "documentation",
    "reviewer",
]


class MARLPPOScheduler:
    """
    Adapter around the trained multi-agent PPO scheduler.

    Input:
        10-dimensional project-state observation.

    Output:
        Per-role binary actions:
            0 = WAIT
            1 = REQUEST COMPUTE
    """

    def __init__(
        self,
        checkpoint=(
            "results/checkpoints/"
            "marl_ppo_alpha_0_25"
        ),
        max_resources=2,
    ):
        self.checkpoint = Path(
            checkpoint
        ).resolve()

        self.max_resources = (
            max_resources
        )

        register_env(
            "multi_agent_dev_team",
            lambda config: MultiAgentDevTeamEnv(config),
        )

        print(
            "Loading MARL PPO scheduler:"
        )
        print(
            self.checkpoint
        )

        self.algo = (
            Algorithm.from_checkpoint(
                str(self.checkpoint)
            )
        )

        self.modules = {
            agent:
                self.algo.get_module(
                    f"{agent}_policy"
                )
            for agent in AGENTS
        }

    def _action(
        self,
        agent,
        observation,
    ):
        obs_tensor = torch.tensor(
            np.asarray(
                observation,
                dtype=np.float32,
            ),
            dtype=torch.float32,
        ).unsqueeze(0)

        module = self.modules[
            agent
        ]

        with torch.no_grad():
            output = (
                module.forward_inference(
                    {
                        "obs":
                            obs_tensor
                    }
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

    def decide(
        self,
        observation,
    ):
        observation = np.asarray(
            observation,
            dtype=np.float32,
        )

        if observation.shape != (10,):
            raise ValueError(
                "Scheduler observation "
                "must have shape (10,), "
                f"got {observation.shape}"
            )

        requested = []

        actions = {}

        for agent in AGENTS:
            action = self._action(
                agent,
                observation,
            )

            actions[
                agent
            ] = action

            if action == 1:
                requested.append(
                    agent
                )

        #
        # Real deployment safety:
        # respect the actual compute limit.
        #
        # If the learned agents request more
        # resources than available, keep the
        # first N according to a deterministic
        # dependency-aware priority.
        #
        priority = [
            "developer",
            "tester",
            "documentation",
            "reviewer",
        ]

        allocated = [
            agent
            for agent in priority
            if agent in requested
        ][
            :self.max_resources
        ]

        return {
            "actions":
                actions,

            "requested":
                requested,

            "allocated":
                allocated,
        }

    def close(self):
        if self.algo is not None:
            self.algo.stop()
            self.algo = None


def main():

    scheduler = MARLPPOScheduler(
        max_resources=2
    )

    #
    # Example state:
    #
    # dev_tasks       = 5/5
    # untested_work   = 0
    # bugs            = 0
    # docs_progress   = 0
    # docs_complete   = 0
    # review_rejected = 0
    # resources       = 2/4
    # dev_progress    = 0
    # test_progress   = 0
    # time            = 0
    #
    observation = np.array(
        [
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.5,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )

    decision = scheduler.decide(
        observation
    )

    print()
    print(
        "Scheduler decision:"
    )

    print(
        "Actions:",
        decision["actions"],
    )

    print(
        "Requested:",
        decision["requested"],
    )

    print(
        "Allocated:",
        decision["allocated"],
    )

    scheduler.close()


if __name__ == "__main__":
    main()
