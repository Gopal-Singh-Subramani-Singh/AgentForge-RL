import json


FILES = {
    "Original":
        "results/planner_benchmark_before.json",

    "Compression":
        "results/planner_benchmark_compression.json",

    "Parallelism":
        "results/planner_benchmark_parallel.json",
}


def score(metrics):

    total = 0.0

    # --------------------------------------------------
    # 1. Structural validity: 30 points
    # --------------------------------------------------

    validity_errors = (
        metrics["invalid_roles"]
        + metrics["invalid_dependencies"]
        + int(metrics["duplicate_ids"])
    )

    if validity_errors == 0:
        total += 30

    else:
        total += max(
            0,
            30 - 10 * validity_errors,
        )

    # --------------------------------------------------
    # 2. Ordering correctness: 20 points
    # --------------------------------------------------

    ordering_errors = (
        metrics["ordering_errors"]
    )

    total += max(
        0,
        20 - 10 * ordering_errors,
    )

    # --------------------------------------------------
    # 3. Dependency depth: 20 points
    #
    # <= 5 excellent
    # 6-8 good
    # 9-12 moderate
    # >12 poor
    # --------------------------------------------------

    depth = metrics["max_depth"]

    if depth <= 5:
        total += 20

    elif depth <= 8:
        total += 16

    elif depth <= 12:
        total += 10

    else:
        total += 5

    # --------------------------------------------------
    # 4. Reviewer efficiency: 15 points
    # --------------------------------------------------

    reviews = metrics[
        "review_tasks"
    ]

    if reviews == 1:
        total += 15

    elif reviews == 2:
        total += 12

    elif reviews <= 4:
        total += 8

    else:
        total += 3

    # --------------------------------------------------
    # 5. Task-count efficiency: 15 points
    # --------------------------------------------------

    tasks = metrics[
        "task_count"
    ]

    if 6 <= tasks <= 15:
        total += 15

    elif 16 <= tasks <= 20:
        total += 11

    elif 21 <= tasks <= 30:
        total += 7

    else:
        total += 4

    return total


def main():

    datasets = {}

    for name, filename in (
        FILES.items()
    ):
        with open(filename) as f:
            datasets[name] = (
                json.load(f)
            )

    projects = list(
        datasets[
            "Original"
        ].keys()
    )

    print()
    print("=" * 78)
    print("PLANNER QUALITY SCORE")
    print("=" * 78)

    print(
        f"{'Project':24s}"
        f"{'Original':>12s}"
        f"{'Compression':>14s}"
        f"{'Parallelism':>14s}"
    )

    print("-" * 78)

    averages = {
        name: []
        for name in FILES
    }

    for project in projects:

        values = {}

        for version in FILES:

            metrics = (
                datasets[
                    version
                ][project][
                    "metrics"
                ]
            )

            value = score(
                metrics
            )

            values[
                version
            ] = value

            averages[
                version
            ].append(
                value
            )

        print(
            f"{project:24s}"
            f"{values['Original']:>12.1f}"
            f"{values['Compression']:>14.1f}"
            f"{values['Parallelism']:>14.1f}"
        )

    print("-" * 78)

    print(
        f"{'AVERAGE':24s}"
        f"{sum(averages['Original']) / len(projects):>12.1f}"
        f"{sum(averages['Compression']) / len(projects):>14.1f}"
        f"{sum(averages['Parallelism']) / len(projects):>14.1f}"
    )


if __name__ == "__main__":
    main()
