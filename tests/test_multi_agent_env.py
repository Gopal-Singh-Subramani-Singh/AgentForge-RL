from environment.multi_agent_env import (
    MultiAgentDevTeamEnv,
)


def test_reset():

    env = MultiAgentDevTeamEnv(
        {
            "num_resources": 2,
        }
    )

    obs, infos = env.reset(
        seed=42
    )

    assert len(obs) == 4

    assert "developer" in obs
    assert "tester" in obs
    assert "documentation" in obs
    assert "reviewer" in obs

    assert obs["developer"].shape == (
        10,
    )


def test_step():

    env = MultiAgentDevTeamEnv(
        {
            "num_resources": 2,
        }
    )

    env.reset(seed=42)

    actions = {
        "developer": 1,
        "tester": 0,
        "documentation": 0,
        "reviewer": 0,
    }

    (
        obs,
        rewards,
        terminateds,
        truncateds,
        infos,
    ) = env.step(actions)

    assert "developer" in rewards

    assert (
        rewards["developer"]
        == rewards["tester"]
    )

    assert "__all__" in terminateds
    assert "__all__" in truncateds
