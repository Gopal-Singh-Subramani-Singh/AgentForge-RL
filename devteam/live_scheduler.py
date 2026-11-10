from integration.rl_scheduler import (
    MARLPPOScheduler,
)


class LiveMARLScheduler:
    def __init__(
        self,
        events,
        max_resources=2,
    ):
        self.events = events

        self.scheduler = (
            MARLPPOScheduler(
                max_resources=(
                    max_resources
                )
            )
        )

        self.max_resources = (
            max_resources
        )

    def decide(
        self,
        observation,
    ):
        decision = (
            self.scheduler.decide(
                observation
            )
        )

        self.events.emit(
            "allocation",
            actions=decision.get(
                "actions",
                {},
            ),
            requested=decision.get(
                "requested",
                [],
            ),
            allocated=decision.get(
                "allocated",
                [],
            ),
            resources=(
                self.max_resources
            ),
        )

        return decision

    def close(self):
        self.scheduler.close()
