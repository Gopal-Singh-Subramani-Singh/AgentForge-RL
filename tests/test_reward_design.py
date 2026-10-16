import pytest

from environment.multi_agent_env import (
    MultiAgentDevTeamEnv,
)


def make_actions(
    developer=0,
    tester=0,
    documentation=0,
    reviewer=0,
):
    return {
        "developer": developer,
        "tester": tester,
        "documentation": documentation,
        "reviewer": reviewer,
    }


def test_alpha_validation():

    with pytest.raises(ValueError):
        MultiAgentDevTeamEnv(
            {
                "alpha": -0.1,
            }
        )

    with pytest.raises(ValueError):
        MultiAgentDevTeamEnv(
            {
                "alpha": 1.1,
            }
        )


def test_shared_reward_alpha_zero():

    env = MultiAgentDevTeamEnv(
        {
            "num_resources": 2,
            "alpha": 0.0,
            "bug_probability": 0.0,
        }
    )

    env.reset(seed=42)

    (
        obs,
        rewards,
        terminateds,
        truncateds,
        infos,
    ) = env.step(
        make_actions(
            developer=1,
        )
    )

    values = list(
        rewards.values()
    )

    assert all(
        reward == values[0]
        for reward in values
    )

    assert (
        rewards["developer"]
        == infos["developer"][
            "team_reward"
        ]
    )


def test_individual_reward_alpha_one():

    env = MultiAgentDevTeamEnv(
        {
            "num_resources": 2,
            "alpha": 1.0,
            "bug_probability": 0.0,
        }
    )

    env.reset(seed=42)

    (
        obs,
        rewards,
        terminateds,
        truncateds,
        infos,
    ) = env.step(
        make_actions(
            developer=1,
        )
    )

    assert rewards["developer"] == 2.0

    assert rewards["tester"] == 0.0

    assert rewards[
        "documentation"
    ] == 0.0

    assert rewards["reviewer"] == 0.0


def test_hybrid_reward_alpha_half():

    env = MultiAgentDevTeamEnv(
        {
            "num_resources": 2,
            "alpha": 0.5,
            "bug_probability": 0.0,
        }
    )

    env.reset(seed=42)

    (
        obs,
        rewards,
        terminateds,
        truncateds,
        infos,
    ) = env.step(
        make_actions(
            developer=1,
        )
    )

    team = infos["developer"][
        "team_reward"
    ]

    expected_developer = (
        0.5 * 2.0
        + 0.5 * team
    )

    expected_tester = (
        0.5 * 0.0
        + 0.5 * team
    )

    assert rewards[
        "developer"
    ] == pytest.approx(
        expected_developer
    )

    assert rewards[
        "tester"
    ] == pytest.approx(
        expected_tester
    )
