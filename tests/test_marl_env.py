from environment.marl_env import DevTeamEnv


def test_environment():

    env = DevTeamEnv(
        num_resources=2
    )

    obs, info = env.reset(seed=42)

    assert obs.shape == (10,)

    action = [1, 0, 0, 0]

    (
        obs,
        reward,
        terminated,
        truncated,
        info,
    ) = env.step(action)

    assert obs.shape == (10,)
    assert isinstance(reward, float)

    # Developer completed one task.
    assert env.dev_tasks == 4
    assert env.untested_work == 1


def test_resource_limit():

    env = DevTeamEnv(
        num_resources=2
    )

    env.reset(seed=42)

    action = [1, 1, 1, 1]

    (
        obs,
        reward,
        terminated,
        truncated,
        info,
    ) = env.step(action)

    assert (
        len(info["allocated_agents"])
        == 2
    )


def test_tester_requires_work():

    env = DevTeamEnv()

    env.reset(seed=42)

    (
        obs,
        reward,
        terminated,
        truncated,
        info,
    ) = env.step(
        [0, 1, 0, 0]
    )

    assert reward < 0
    assert env.tested_tasks == 0


def test_developer_creates_testable_work():

    env = DevTeamEnv()

    env.reset(seed=42)

    env.step(
        [1, 0, 0, 0]
    )

    assert env.untested_work == 1


def test_documentation_requires_progress():

    env = DevTeamEnv()

    env.reset(seed=42)

    env.step(
        [0, 0, 1, 0]
    )

    assert env.docs_progress == 0.0


def test_review_too_early_rejected():

    env = DevTeamEnv()

    env.reset(seed=42)

    env.step(
        [0, 0, 0, 1]
    )

    assert env.review_rejected is True
    assert env.review_complete is False


def test_successful_project():

    env = DevTeamEnv(
        num_resources=2,
        bug_probability=0.0,
    )

    env.reset(seed=42)

    # Develop and test all five tasks.
    for _ in range(5):

        env.step(
            [1, 0, 0, 0]
        )

        env.step(
            [0, 1, 0, 0]
        )

    # Documentation needs four work steps.
    for _ in range(4):

        env.step(
            [0, 0, 1, 0]
        )

    (
        obs,
        reward,
        terminated,
        truncated,
        info,
    ) = env.step(
        [0, 0, 0, 1]
    )

    assert env.dev_tasks == 0
    assert env.untested_work == 0
    assert env.bugs == 0
    assert env.docs_complete is True
    assert terminated is True
    assert info["success"] is True
