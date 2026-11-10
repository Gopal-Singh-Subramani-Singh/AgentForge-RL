import json
import time
from pathlib import Path


class EventLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.path.write_text(
            "",
            encoding="utf-8",
        )

    def emit(self, event_type, **data):
        event = {
            "time": time.time(),
            "type": event_type,
            **data,
        }

        with self.path.open(
            "a",
            encoding="utf-8",
        ) as f:
            f.write(
                json.dumps(event)
                + "\n"
            )

        return event
