import numpy as np
from gymnasium import spaces

from ray.rllib.env.multi_agent_env import MultiAgentEnv

from environment.marl_env import DevTeamEnv


AGENTS = [
    "developer",
    "tester",
    "documentation",
    "reviewer",
]


class MultiAgentDevTeamEnv(MultiAgentEnv):

    def __init__(self, config=None):

        super().__init__()

        config = config or {}

        self.alpha = float(
            config.get(
                "alpha",
                0.0,
            )
        )

        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(
                "alpha must be between 0.0 and 1.0"
            )

        self.base_env = DevTeamEnv(
            num_resources=config.get(
                "num_resources",
                2,
            ),
            max_steps=config.get(
                "max_steps",
                50,
            ),
            bug_probability=config.get(
                "bug_probability",
                0.20,
            ),
        )

        self.possible_agents = AGENTS.copy()
        self.agents = AGENTS.copy()

        self.observation_spaces = {
            agent: spaces.Box(
                low=0.0,
                high=1.0,
                shape=(10,),
                dtype=np.float32,
            )
            for agent in AGENTS
        }

        self.action_spaces = {
            agent: spaces.Discrete(2)
            for agent in AGENTS
        }

    def get_observation_space(
        self,
        agent_id,
    ):
        return self.observation_spaces[
            agent_id
        ]

    def get_action_space(
        self,
        agent_id,
    ):
        return self.action_spaces[
            agent_id
        ]

    def _snapshot(self):
        """
        Capture state before/after one environment step.
        """

        env = self.base_env

        return {
            "dev_tasks":
                env.dev_tasks,

            "bugs":
                env.bugs,

            "untested_work":
                env.untested_work,

            "docs_progress":
                env.docs_progress,

            "docs_complete":
                env.docs_complete,

            "review_complete":
                env.review_complete,
        }

    def _individual_rewards(
        self,
        before,
        after,
        info,
    ):
        """
        Compute role-specific contribution rewards.

        These intentionally mirror the role rewards
        already used by DevTeamEnv.
        """

        allocated = set(
            info.get(
                "allocated_agents",
                [],
            )
        )

        rewards = {
            agent: 0.0
            for agent in AGENTS
        }

        # -------------------------------------------------
        # Developer
        # -------------------------------------------------

        if "developer" in allocated:

            if after["bugs"] < before["bugs"]:
                # Fixed a known bug.
                rewards["developer"] += 3.0

            elif (
                after["dev_tasks"]
                < before["dev_tasks"]
            ):
                # Completed new development work.
                rewards["developer"] += 2.0

            else:
                # Received compute but made no progress.
                rewards["developer"] -= 0.5

        # -------------------------------------------------
        # Tester
        # -------------------------------------------------

        if "tester" in allocated:

            tested_something = (
                after["untested_work"]
                < before["untested_work"]
            )

            if tested_something:

                bug_discovered = (
                    after["bugs"]
                    > before["bugs"]
                )

                if bug_discovered:
                    rewards["tester"] += 1.0
                else:
                    rewards["tester"] += 1.5

            else:
                rewards["tester"] -= 0.5

        # -------------------------------------------------
        # Documentation
        # -------------------------------------------------

        if "documentation" in allocated:

            just_completed = (
                not before["docs_complete"]
                and after["docs_complete"]
            )

            made_progress = (
                after["docs_progress"]
                > before["docs_progress"]
            )

            if just_completed:
                rewards["documentation"] += 3.0

            elif made_progress:
                rewards["documentation"] += 0.5

            else:
                rewards["documentation"] -= 0.5

        # -------------------------------------------------
        # Reviewer
        # -------------------------------------------------

        if "reviewer" in allocated:

            just_approved = (
                not before["review_complete"]
                and after["review_complete"]
            )

            if just_approved:
                rewards["reviewer"] += 20.0

            elif info.get(
                "review_rejected",
                False,
            ):
                rewards["reviewer"] -= 5.0

            else:
                rewards["reviewer"] -= 0.5

        return rewards

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):

        obs, info = self.base_env.reset(
            seed=seed,
            options=options,
        )

        self.agents = AGENTS.copy()

        observations = {
            agent: obs.copy()
            for agent in AGENTS
        }

        infos = {
            agent: {
                "alpha": self.alpha,
            }
            for agent in AGENTS
        }

        return observations, infos

    def step(
        self,
        action_dict,
    ):

        joint_action = np.array(
            [
                action_dict.get(
                    agent,
                    0,
                )
                for agent in AGENTS
            ],
            dtype=np.int64,
        )

        before = self._snapshot()

        (
            obs,
            team_reward,
            terminated,
            truncated,
            info,
        ) = self.base_env.step(
            joint_action
        )

        after = self._snapshot()

        individual_rewards = (
            self._individual_rewards(
                before,
                after,
                info,
            )
        )

        # -------------------------------------------------
        # Hybrid reward:
        #
        # alpha = 0 -> pure team reward
        # alpha = 1 -> pure individual reward
        # -------------------------------------------------

        rewards = {
            agent:
                self.alpha
                * individual_rewards[agent]
                +
                (1.0 - self.alpha)
                * team_reward
            for agent in AGENTS
        }

        observations = {
            agent: obs.copy()
            for agent in AGENTS
        }

        terminateds = {
            agent: terminated
            for agent in AGENTS
        }

        terminateds["__all__"] = (
            terminated
        )

        truncateds = {
            agent: truncated
            for agent in AGENTS
        }

        truncateds["__all__"] = (
            truncated
        )

        infos = {
            agent: {
                **info,
                "team_reward":
                    team_reward,
                "individual_reward":
                    individual_rewards[
                        agent
                    ],
                "hybrid_reward":
                    rewards[agent],
                "alpha":
                    self.alpha,
            }
            for agent in AGENTS
        }

        if terminated or truncated:
            self.agents = []

        return (
            observations,
            rewards,
            terminateds,
            truncateds,
            infos,
        )
