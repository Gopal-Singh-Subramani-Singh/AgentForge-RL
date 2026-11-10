class RandomScheduler:

    def act(self, env):
        return env.action_space.sample()
