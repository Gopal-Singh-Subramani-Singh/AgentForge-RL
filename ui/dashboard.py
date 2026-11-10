from pathlib import Path
import ast
import json
import subprocess
import sys

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]

TRACE_FILE = ROOT / "results/allocation_traces.csv"
SUMMARY_FILE = ROOT / "results/allocation_summary.csv"

AGENTS = [
    "developer",
    "tester",
    "documentation",
    "reviewer",
]

ROLE_NAMES = {
    "developer": "Developer",
    "tester": "Tester",
    "documentation": "Documentation",
    "reviewer": "Reviewer",
}

ROLE_ICONS = {
    "developer": "💻",
    "tester": "🧪",
    "documentation": "📚",
    "reviewer": "🔍",
}

PROJECTS = {
    "Bank Account": "bank",
    "Password Validator": "password",
    "Todo List": "todo",
    "Inventory Manager": "inventory",
}


st.set_page_config(
    page_title="MARL AI Software Team",
    page_icon="🤖",
    layout="wide",
)


st.markdown(
    """
<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
}

.hero {
    border: 1px solid #30363d;
    border-radius: 16px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 1rem;
}

.hero-title {
    font-size: 1.8rem;
    font-weight: 800;
}

.hero-sub {
    opacity: 0.8;
    margin-top: .25rem;
}

.status-banner {
    border-radius: 14px;
    padding: .9rem 1.1rem;
    font-size: 1.1rem;
    font-weight: 800;
    margin-bottom: .8rem;
}

.status-running {
    border: 1px solid #d29922;
}

.status-pass {
    border: 1px solid #2ea043;
}

.status-fail {
    border: 1px solid #f85149;
}

.agent-card {
    border: 1px solid #30363d;
    border-radius: 14px;
    padding: .95rem;
    min-height: 145px;
}

.agent-allocated {
    border: 2px solid #2ea043;
}

.agent-requested {
    border: 2px solid #d29922;
}

.agent-wait {
    opacity: .7;
}

.agent-name {
    font-size: 1.02rem;
    font-weight: 800;
}

.agent-state {
    margin-top: .65rem;
    font-size: 1rem;
    font-weight: 800;
}

.resource-slot {
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: .9rem;
    min-height: 105px;
}

.timeline-box {
    border-left: 3px solid #30363d;
    padding-left: .8rem;
    margin-bottom: .6rem;
}

.small-muted {
    opacity: .65;
    font-size: .88rem;
}

.metric-note {
    opacity: .7;
    font-size: .82rem;
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def load_results():
    traces = pd.read_csv(TRACE_FILE)
    summary = pd.read_csv(SUMMARY_FILE)
    return traces, summary


def parse_list(value):
    if isinstance(value, list):
        return value

    if pd.isna(value):
        return []

    value = str(value).strip()

    if not value:
        return []

    return [
        x.strip()
        for x in value.split(",")
        if x.strip()
    ]


def parse_state(value):
    if isinstance(value, dict):
        return value

    try:
        return ast.literal_eval(str(value))
    except Exception:
        return {}


def agent_card(role, requested, allocated, active_agent=None):

    if role in allocated:
        css = "agent-allocated"
        state = "ALLOCATED"
        detail = "Compute resource assigned"

    elif role in requested:
        css = "agent-requested"
        state = "REQUESTED"
        detail = "Requested compute"

    else:
        css = "agent-wait"
        state = "WAIT"
        detail = "No resource allocated"

    if active_agent == role:
        detail = "Currently executing"

    return f"""
<div class="agent-card {css}">
    <div style="font-size:2rem">
        {ROLE_ICONS[role]}
    </div>
    <div class="agent-name">
        {ROLE_NAMES[role]}
    </div>
    <div class="agent-state">
        {state}
    </div>
    <div class="small-muted" style="margin-top:.35rem">
        {detail}
    </div>
</div>
"""


def resource_slot(index, allocated):

    if index < len(allocated):
        role = allocated[index]

        return f"""
<div class="resource-slot">
    <div class="small-muted">
        Compute Slot {index + 1}
    </div>
    <div style="font-size:1.8rem;margin-top:.4rem">
        {ROLE_ICONS.get(role, "⚙️")}
    </div>
    <div style="font-weight:800">
        {ROLE_NAMES.get(role, role)}
    </div>
    <div class="small-muted">
        ACTIVE
    </div>
</div>
"""

    return f"""
<div class="resource-slot">
    <div class="small-muted">
        Compute Slot {index + 1}
    </div>
    <div style="font-size:1.8rem;margin-top:.4rem">
        ◻️
    </div>
    <div style="font-weight:800">
        Available
    </div>
    <div class="small-muted">
        IDLE
    </div>
</div>
"""


def run_live_project(project_key):

    cmd = [
        sys.executable,
        "-m",
        "integration.live_multi_project_runner",
        "--project",
        project_key,
    ]

    return subprocess.Popen(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )


traces, summary = load_results()


st.markdown(
    """
<div class="hero">
    <div class="hero-title">
        Multi-Agent RL AI Software Team
    </div>
    <div class="hero-sub">
        MARL PPO coordinates Developer, Tester, Documentation,
        and Reviewer agents under a two-resource compute limit.
    </div>
</div>
""",
    unsafe_allow_html=True,
)


tabs = st.tabs(
    [
        "🔴 Live AI Team",
        "🎛 Allocation Replay",
        "📊 Algorithm Comparison",
        "🧠 Research Summary",
    ]
)


# ============================================================
# LIVE AI TEAM
# ============================================================

with tabs[0]:

    left, right = st.columns(
        [2, 1]
    )

    with left:
        selected_project = st.selectbox(
            "Project",
            list(PROJECTS.keys()),
        )

    with right:
        st.metric(
            "Available compute resources",
            "2",
        )

    st.caption(
        "This mode runs the real Qwen2.5-Coder-7B model, "
        "the trained MARL PPO scheduler, pytest, and reviewer."
    )

    run_button = st.button(
        "▶ Run Live AI Team",
        type="primary",
        use_container_width=True,
    )

    if run_button:

        project_key = PROJECTS[
            selected_project
        ]

        status_placeholder = st.empty()

        status_placeholder.markdown(
            """
<div class="status-banner status-running">
    RUNNING — Loading model and scheduler
</div>
""",
            unsafe_allow_html=True,
        )

        metrics = st.columns(5)

        step_metric = metrics[0].empty()
        test_metric = metrics[1].empty()
        bug_metric = metrics[2].empty()
        docs_metric = metrics[3].empty()
        project_metric = metrics[4].empty()

        step_metric.metric(
            "Scheduling step",
            0,
        )

        test_metric.metric(
            "Tests",
            "Waiting",
        )

        bug_metric.metric(
            "Known bugs",
            0,
        )

        docs_metric.metric(
            "Documentation",
            "Pending",
        )

        project_metric.metric(
            "Project",
            "Running",
        )

        st.markdown(
            "### Resource allocation"
        )

        slot_columns = st.columns(2)

        slot_placeholders = [
            slot_columns[i].empty()
            for i in range(2)
        ]

        for i in range(2):
            slot_placeholders[
                i
            ].markdown(
                resource_slot(
                    i,
                    [],
                ),
                unsafe_allow_html=True,
            )

        st.markdown(
            "### Agent team"
        )

        role_columns = st.columns(4)

        role_placeholders = {
            role:
                role_columns[i].empty()
            for i, role in enumerate(
                AGENTS
            )
        }

        for role in AGENTS:
            role_placeholders[
                role
            ].markdown(
                agent_card(
                    role,
                    [],
                    [],
                ),
                unsafe_allow_html=True,
            )

        state_left, state_right = st.columns(
            [1.2, 1]
        )

        with state_left:

            st.markdown(
                "### PPO observation"
            )

            observation_placeholder = st.empty()

            st.markdown(
                "### Current project state"
            )

            state_placeholder = st.empty()

        with state_right:

            st.markdown(
                "### Decision"
            )

            decision_placeholder = st.empty()

            st.markdown(
                "### Activity timeline"
            )

            timeline_placeholder = st.empty()

        artifact_tabs = st.tabs(
            [
                "Generated app.py",
                "README.md",
                "Pytest",
                "Tester diagnosis",
                "Reviewer",
            ]
        )

        with artifact_tabs[0]:
            code_placeholder = st.empty()

        with artifact_tabs[1]:
            docs_placeholder = st.empty()

        with artifact_tabs[2]:
            tests_placeholder = st.empty()

        with artifact_tabs[3]:
            diagnosis_placeholder = st.empty()

        with artifact_tabs[4]:
            reviewer_placeholder = st.empty()

        with st.expander(
            "Backend runtime log",
            expanded=False,
        ):
            runtime_log_placeholder = st.empty()
        raw_runtime_lines = []

        process = run_live_project(
            project_key
        )

        timeline = []
        latest_requested = []
        latest_allocated = []
        latest_state = {}
        success = False
        final_steps = None
        last_test_passed = None
        active_agent = None

        for raw_line in process.stdout:

            line = raw_line.strip()

            if not line:
                continue

            try:
                event = json.loads(
                    line
                )
            except Exception:
                raw_runtime_lines.append(line)

                runtime_log_placeholder.code(
                    "\n".join(
                        raw_runtime_lines[-40:]
                    ),
                    language="text",
                )

                continue

            event_type = event.get(
                "event"
            )

            if event_type == "status":

                timeline.append(
                    event.get(
                        "message",
                        ""
                    )
                )

            elif event_type == "allocation":

                step = event.get(
                    "step",
                    0,
                )

                latest_requested = event.get(
                    "requested",
                    [],
                )

                latest_allocated = event.get(
                    "allocated",
                    [],
                )

                latest_state = event.get(
                    "state",
                    {},
                )

                observation = event.get(
                    "observation",
                    [],
                )

                step_metric.metric(
                    "Scheduling step",
                    step + 1,
                )

                bug_metric.metric(
                    "Known bugs",
                    latest_state.get(
                        "bugs",
                        0,
                    ),
                )

                docs_metric.metric(
                    "Documentation",
                    (
                        "Complete"
                        if latest_state.get(
                            "docs_complete"
                        )
                        else "Pending"
                    ),
                )

                for i in range(2):
                    slot_placeholders[
                        i
                    ].markdown(
                        resource_slot(
                            i,
                            latest_allocated,
                        ),
                        unsafe_allow_html=True,
                    )

                for role in AGENTS:
                    role_placeholders[
                        role
                    ].markdown(
                        agent_card(
                            role,
                            latest_requested,
                            latest_allocated,
                            active_agent,
                        ),
                        unsafe_allow_html=True,
                    )

                observation_labels = [
                    "dev_tasks",
                    "untested_work",
                    "bugs",
                    "docs_progress",
                    "docs_complete",
                    "review_rejected",
                    "resource_ratio",
                    "development_progress",
                    "test_progress",
                    "normalized_time",
                ]

                obs_df = pd.DataFrame(
                    {
                        "Feature":
                            observation_labels,
                        "Value":
                            observation,
                    }
                )

                observation_placeholder.dataframe(
                    obs_df,
                    use_container_width=True,
                    hide_index=True,
                )

                state_placeholder.json(
                    latest_state
                )

                decision_placeholder.markdown(
                    f"""
**Requested**

`{", ".join(latest_requested) or "None"}`

**Allocated**

`{", ".join(latest_allocated) or "None"}`

**Compute limit**

`2 agents`
"""
                )

                timeline.append(
                    f"Step {step + 1}: "
                    f"{', '.join(latest_allocated) or 'No agent'}"
                )

            elif event_type == "agent_start":

                active_agent = event.get(
                    "agent"
                )

                for role in AGENTS:
                    role_placeholders[
                        role
                    ].markdown(
                        agent_card(
                            role,
                            latest_requested,
                            latest_allocated,
                            active_agent,
                        ),
                        unsafe_allow_html=True,
                    )

                timeline.append(
                    f"{ROLE_NAMES.get(active_agent, active_agent)} started"
                )

            elif event_type == "agent_end":

                active_agent = None

            elif event_type == "artifact":

                filename = event.get(
                    "file"
                )

                content = event.get(
                    "content",
                    "",
                )

                if filename == "app.py":

                    code_placeholder.code(
                        content,
                        language="python",
                    )

                    timeline.append(
                        "Developer generated app.py"
                    )

                elif filename == "README.md":

                    docs_placeholder.markdown(
                        content
                    )

                    docs_metric.metric(
                        "Documentation",
                        "Complete",
                    )

                    timeline.append(
                        "Documentation generated README.md"
                    )

            elif event_type == "tests":

                passed = event.get(
                    "passed",
                    False,
                )

                last_test_passed = passed

                output = event.get(
                    "output",
                    "",
                )

                test_metric.metric(
                    "Tests",
                    (
                        "PASS"
                        if passed
                        else "FAIL"
                    ),
                )

                tests_placeholder.code(
                    output,
                    language="text",
                )

                if passed:

                    status_placeholder.markdown(
                        """
<div class="status-banner status-pass">
    TESTS PASSED — Awaiting final review
</div>
""",
                        unsafe_allow_html=True,
                    )

                    timeline.append(
                        "Tester: all tests passed"
                    )

                else:

                    status_placeholder.markdown(
                        """
<div class="status-banner status-fail">
    TEST FAILURE — Repair cycle triggered
</div>
""",
                        unsafe_allow_html=True,
                    )

                    timeline.append(
                        "Tester: failure detected"
                    )

            elif event_type == "diagnosis":

                diagnosis = event.get(
                    "text",
                    "",
                )

                diagnosis_placeholder.markdown(
                    diagnosis
                )

                timeline.append(
                    "Tester diagnosed the failure"
                )

                status_placeholder.markdown(
                    """
<div class="status-banner status-running">
    REPAIRING — Developer receives test feedback
</div>
""",
                    unsafe_allow_html=True,
                )

            elif event_type == "review":

                approved = event.get(
                    "approved",
                    False,
                )

                text = event.get(
                    "text",
                    "",
                )

                if approved:

                    reviewer_placeholder.success(
                        text
                    )

                    timeline.append(
                        "Reviewer approved project"
                    )

                else:

                    reviewer_placeholder.error(
                        text
                    )

                    timeline.append(
                        "Reviewer rejected project"
                    )

            elif event_type == "completed":

                success = event.get(
                    "success",
                    False,
                )

                final_steps = event.get(
                    "steps"
                )

                project_metric.metric(
                    "Project",
                    (
                        "COMPLETE"
                        if success
                        else "FAILED"
                    ),
                )

                if success:

                    status_placeholder.markdown(
                        """
<div class="status-banner status-pass">
    PROJECT COMPLETE — Reviewer approved
</div>
""",
                        unsafe_allow_html=True,
                    )

                else:

                    status_placeholder.markdown(
                        """
<div class="status-banner status-fail">
    PROJECT FAILED — Maximum scheduling steps reached
</div>
""",
                        unsafe_allow_html=True,
                    )

            timeline_html = ""

            for item in timeline[-12:]:

                timeline_html += (
                    '<div class="timeline-box">'
                    + item
                    + "</div>"
                )

            timeline_placeholder.markdown(
                timeline_html,
                unsafe_allow_html=True,
            )

        return_code = process.wait()

        if return_code != 0:
            runtime_log_placeholder.code(
                "\n".join(
                    raw_runtime_lines[-80:]
                ),
                language="text",
            )

            st.error(
                f"Live backend exited with code {return_code}. "
                "See Backend runtime log above."
            )

        st.divider()

        if success:

            st.success(
                f"{selected_project} completed successfully."
            )

            final_cols = st.columns(4)

            final_cols[0].metric(
                "Result",
                "SUCCESS",
            )

            final_cols[1].metric(
                "Scheduling steps",
                final_steps,
            )

            final_cols[2].metric(
                "Tests",
                (
                    "PASS"
                    if last_test_passed
                    else "N/A"
                ),
            )

            final_cols[3].metric(
                "Reviewer",
                "APPROVED",
            )

        else:

            st.error(
                f"{selected_project} did not complete."
            )


# ============================================================
# ALLOCATION REPLAY
# ============================================================

with tabs[1]:

    st.subheader(
        "Resource Allocation Replay"
    )

    algorithms = traces[
        "algorithm"
    ].drop_duplicates().tolist()

    default_index = (
        algorithms.index(
            "MARL PPO"
        )
        if "MARL PPO" in algorithms
        else 0
    )

    selected_algorithm = st.selectbox(
        "Algorithm",
        algorithms,
        index=default_index,
        key="replay_algorithm",
    )

    replay = (
        traces[
            traces["algorithm"]
            == selected_algorithm
        ]
        .sort_values("step")
        .reset_index(drop=True)
    )

    replay_step = st.slider(
        "Step",
        min_value=0,
        max_value=max(
            len(replay) - 1,
            0,
        ),
        value=0,
    )

    row = replay.iloc[
        replay_step
    ]

    requested = parse_list(
        row["requested"]
    )

    allocated = parse_list(
        row["allocated"]
    )

    state = parse_state(
        row["state"]
    )

    slots = st.columns(2)

    for i in range(2):
        slots[i].markdown(
            resource_slot(
                i,
                allocated,
            ),
            unsafe_allow_html=True,
        )

    role_cols = st.columns(4)

    for i, role in enumerate(
        AGENTS
    ):
        role_cols[i].markdown(
            agent_card(
                role,
                requested,
                allocated,
            ),
            unsafe_allow_html=True,
        )

    m = st.columns(5)

    m[0].metric(
        "Step",
        replay_step + 1,
    )

    m[1].metric(
        "Requested",
        len(requested),
    )

    m[2].metric(
        "Allocated",
        len(allocated),
    )

    m[3].metric(
        "Waste",
        int(
            row["waste_step"]
        ),
    )

    m[4].metric(
        "Bugs",
        state.get(
            "bugs",
            "-",
        ),
    )

    st.json(
        state
    )


# ============================================================
# ALGORITHM COMPARISON
# ============================================================

with tabs[2]:

    st.subheader(
        "Algorithm Comparison"
    )

    st.markdown(
        "### Same-seed baseline comparison"
    )

    st.caption(
        "All baseline schedulers below were evaluated on the same project instance "
        "to compare coordination behavior directly."
    )

    comparison = summary.copy()

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        "### Final learned-method comparison"
    )

    learned_final = pd.DataFrame(
        [
            {
                "Method": "MARL PPO",
                "Success": 1.000,
                "Steps": 10.33,
                "Team Reward": 45.77,
                "Waste": 0.00,
                "Review Rejects": 0.00,
                "Interpretation": "Best overall balance",
            },
            {
                "Method": "PPO-Guided GRPO",
                "Success": 1.000,
                "Steps": 9.88,
                "Team Reward": 43.18,
                "Waste": 0.20,
                "Review Rejects": 0.52,
                "Interpretation": "Fastest learned policy",
            },
            {
                "Method": "GRPO-style MARL G8",
                "Success": 1.000,
                "Steps": 15.18,
                "Team Reward": 39.84,
                "Waste": 6.50,
                "Review Rejects": 0.45,
                "Interpretation": "From-scratch GRPO",
            },
        ]
    )

    st.dataframe(
        learned_final,
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "MARL PPO achieved the strongest overall balance of reward, efficiency, "
        "and coordination. PPO-Guided GRPO achieved the fastest average completion "
        "time, showing that group-relative optimization was more effective as a "
        "fine-tuning method than as a from-scratch optimizer."
    )

    left, right = st.columns(2)

    with left:

        st.markdown(
            "#### Same-seed completion steps"
        )

        chart = (
            summary[
                [
                    "algorithm",
                    "steps",
                ]
            ]
            .set_index(
                "algorithm"
            )
        )

        st.bar_chart(
            chart
        )

    with right:

        st.markdown(
            "#### Same-seed wasted allocations"
        )

        chart = (
            summary[
                [
                    "algorithm",
                    "waste",
                ]
            ]
            .set_index(
                "algorithm"
            )
        )

        st.bar_chart(
            chart
        )

    st.markdown(
        "#### Resource allocation by role"
    )

    role_chart = (
        summary[
            [
                "algorithm",
                "developer_alloc",
                "tester_alloc",
                "documentation_alloc",
                "reviewer_alloc",
            ]
        ]
        .set_index(
            "algorithm"
        )
    )

    st.bar_chart(
        role_chart
    )


# ============================================================
# RESEARCH SUMMARY
# ============================================================

with tabs[3]:

    st.subheader(
        "Final Research Summary"
    )

    headline = st.columns(4)

    headline[0].metric(
        "Best balanced policy",
        "MARL PPO",
    )

    headline[1].metric(
        "Fastest learned policy",
        "PPO-Guided GRPO",
    )

    headline[2].metric(
        "Real live projects",
        "4",
    )

    headline[3].metric(
        "Compute resources",
        "2",
    )

    st.markdown(
        """
### Learned coordination

MARL PPO learned to coordinate four specialized software-development
agents under a two-resource compute constraint.

The learned policy consistently moved through a sensible workflow:

**Developer / Documentation → Tester → repair if needed → Reviewer**

### GRPO findings

From-scratch GRPO-style MARL learned successful coordination but was
less efficient than MARL PPO.

PPO-guided GRPO substantially improved completion speed and eliminated
the earlier greedy-deployment collapse.

MARL PPO achieved the best overall balance of reward, efficiency, and
coordination. PPO-Guided GRPO achieved the fastest average completion time,
demonstrating that GRPO was more effective as a fine-tuning method than as a
from-scratch optimizer.

### Real LLM deployment

The trained MARL PPO scheduler was transferred to a real
Qwen2.5-Coder-7B workflow.

Validated live projects:

- Bank Account — 12 tests passed
- Todo List — 8 tests passed
- Inventory Manager — 11 tests passed
- Password Validator — repair loop followed by 8 tests passed

The Password Validator demonstrates the complete recovery cycle:
test failure → diagnosis → developer repair → retest → reviewer approval.
"""
    )
