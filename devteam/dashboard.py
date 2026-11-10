import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import (
    HTMLResponse,
)


EVENT_FILE = Path(
    "results/live/events.jsonl"
)

app = FastAPI()


@app.get("/api/events")
def events():
    if not EVENT_FILE.exists():
        return []

    items = []

    for line in (
        EVENT_FILE
        .read_text(
            encoding="utf-8"
        )
        .splitlines()
    ):
        try:
            items.append(
                json.loads(
                    line
                )
            )
        except Exception:
            pass

    return items[-1000:]


@app.get(
    "/",
    response_class=HTMLResponse,
)
def dashboard():
    return HTMLResponse(
        """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>MARL Dev Team</title>

<style>
body {
    margin: 0;
    padding: 24px;
    background: #0b1020;
    color: #edf2ff;
    font-family: system-ui, sans-serif;
}

h1, h2 {
    margin-top: 0;
}

.grid {
    display: grid;
    grid-template-columns:
        repeat(auto-fit, minmax(210px, 1fr));
    gap: 16px;
}

.card {
    background: #151c32;
    border: 1px solid #2a3558;
    border-radius: 14px;
    padding: 18px;
}

.big {
    font-size: 28px;
    font-weight: 700;
}

.muted {
    color: #9aa7c7;
}

.role {
    margin: 16px 0;
}

.label {
    display: flex;
    justify-content: space-between;
    gap: 12px;
}

.bar {
    height: 12px;
    margin: 7px 0;
    background: #242d49;
    border-radius: 8px;
    overflow: hidden;
}

.fill {
    height: 100%;
    width: 12%;
    background: #65708c;
    transition: width .3s;
}

.active .fill {
    width: 100%;
    background: #55d98b;
}

.running {
    color: #55d98b;
}

.waiting {
    color: #9aa7c7;
}

#events {
    max-height: 340px;
    overflow: auto;
    white-space: pre-wrap;
    font-family: monospace;
    font-size: 12px;
}

#quality {
    white-space: pre-wrap;
    max-height: 200px;
    overflow: auto;
}
</style>
</head>

<body>

<h1>Multi-Agent RL Software Team</h1>

<div class="grid">

<div class="card">
<div class="muted">Scheduler Step</div>
<div class="big" id="step">0</div>
</div>

<div class="card">
<div class="muted">Completed Tasks</div>
<div class="big" id="completed">0 / 0</div>
</div>

<div class="card">
<div class="muted">Compute Slots</div>
<div class="big" id="resources">2</div>
</div>

<div class="card">
<div class="muted">Known Bugs</div>
<div class="big" id="bugs">0</div>
</div>

<div class="card">
<div class="muted">Project</div>
<div class="big" id="status">WAITING</div>
</div>

</div>

<br>

<div class="card">
<h2>Live Resource Allocation</h2>
<div id="roles"></div>
</div>

<br>

<div class="grid">

<div class="card">
<h2>Current Work</h2>
<div id="work">
No task currently running.
</div>
</div>

<div class="card">
<h2>Latest Quality Result</h2>
<div id="quality">
No result yet.
</div>
</div>

</div>

<br>

<div class="card">
<h2>Event Stream</h2>
<div id="events"></div>
</div>

<script>
const roleNames = [
    "developer",
    "tester",
    "documentation",
    "reviewer"
];

function emptyState() {
    return {
        step: 0,
        completed: 0,
        total: 0,
        resources: 2,
        bugs: 0,
        allocated: [],
        tasks: {},
        quality: "",
        complete: false,
        maxSteps: false
    };
}

function esc(value) {
    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
}

function applyEvent(state, event) {
    if (event.type === "step") {
        state.step =
            event.step ?? state.step;

        state.completed =
            event.completed ?? state.completed;

        state.total =
            event.total ?? state.total;

        state.resources =
            event.resources ?? state.resources;

        state.bugs =
            event.bugs ?? state.bugs;
    }

    if (event.type === "allocation") {
        state.allocated =
            event.allocated || [];
    }

    if (event.type === "scheduler_fallback") {
        state.allocated = [
            event.role
        ];
    }

    if (event.type === "task_start") {
        state.tasks[event.role] =
            event.task_id + ": " +
            event.description;
    }

    if (event.type === "task_end") {
        delete state.tasks[
            event.role
        ];
    }

    if (event.type === "quality") {
        state.quality =
            event.output || "";
    }

    if (event.type === "complete") {
        state.complete = true;
        state.allocated = [];
    }

    if (event.type === "max_steps") {
        state.maxSteps = true;
        state.allocated = [];
    }
}

function renderRoles(state) {
    const root =
        document.getElementById("roles");

    root.innerHTML =
        roleNames.map(role => {
            const active =
                state.allocated.includes(role);

            const task =
                state.tasks[role] || "";

            return `
            <div class="role ${active ? "active" : ""}">
                <div class="label">
                    <strong>${role.toUpperCase()}</strong>
                    <span class="${active ? "running" : "waiting"}">
                        ${active ? "RUNNING" : "WAITING"}
                    </span>
                </div>

                <div class="bar">
                    <div class="fill"></div>
                </div>

                <div class="muted">
                    ${esc(task)}
                </div>
            </div>
            `;
        }).join("");
}

function render(events) {
    const state = emptyState();

    for (const event of events) {
        applyEvent(
            state,
            event
        );
    }

    document.getElementById(
        "step"
    ).textContent = state.step;

    document.getElementById(
        "completed"
    ).textContent =
        `${state.completed} / ${state.total}`;

    document.getElementById(
        "resources"
    ).textContent =
        state.resources;

    document.getElementById(
        "bugs"
    ).textContent =
        state.bugs;

    let status = "RUNNING";

    if (events.length === 0) {
        status = "WAITING";
    }

    if (state.complete) {
        status = "COMPLETE";
    }

    if (state.maxSteps) {
        status = "MAX STEPS";
    }

    document.getElementById(
        "status"
    ).textContent = status;

    const work =
        Object.entries(
            state.tasks
        )
        .map(
            ([role, task]) =>
                `<strong>${role}</strong>: ${esc(task)}`
        )
        .join("<br>");

    document.getElementById(
        "work"
    ).innerHTML =
        work ||
        "No task currently running.";

    document.getElementById(
        "quality"
    ).textContent =
        state.quality ||
        "No quality result yet.";

    document.getElementById(
        "events"
    ).textContent =
        events
        .slice(-35)
        .map(
            event =>
                `[${event.type}] ` +
                JSON.stringify(event)
        )
        .join("\\n");

    renderRoles(state);
}

async function refresh() {
    try {
        const response =
            await fetch(
                "/api/events"
            );

        const events =
            await response.json();

        render(events);

    } catch (error) {
        console.error(error);
    }
}

refresh();

setInterval(
    refresh,
    700
);
</script>

</body>
</html>
        """
    )
