class RuleBasedScheduler:

    def act(self, env):

        action = [0, 0, 0, 0]

        # Developer fixes bugs or develops
        # unfinished tasks.
        if (
            env.bugs > 0
            or env.dev_tasks > 0
        ):
            action[0] = 1

        # Tester processes completed code.
        if env.untested_work > 0:
            action[1] = 1

        # Documentation can proceed once
        # development has started.
        if (
            env.completed_dev_tasks > 0
            and not env.docs_complete
        ):
            action[2] = 1

        # Reviewer only acts when project
        # appears ready.
        ready = (
            env.dev_tasks == 0
            and env.untested_work == 0
            and env.bugs == 0
            and env.docs_complete
        )

        if ready:
            action[3] = 1

        return action
