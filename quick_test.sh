cd /home/rawrer/Work/icaps/pspr-planner

cargo build --release --example bench_modes

export HDDL_PARSER=$HOME/.cargo/bin/hddl_analyzer
export LD_LIBRARY_PATH=$(dirname "$(find target/release -path '*scip_install/lib/libscip.so' | head -1)")
export PSPR_MAX_GROUNDINGS=4096

python3 - <<'PY'
import csv, os, statistics, subprocess, time
from pathlib import Path

root = Path.cwd()
ipc = root.parent / "hddl" / "test_data" / "ipc"
bench = root / "target" / "release" / "examples" / "bench_modes"
csv_path = root / "target" / "decomposition-comparison" / "results.csv"
repeats = 3
timeout = 30

with csv_path.open() as handle:
    rows = list(csv.DictReader(handle))
cases = list(dict.fromkeys(
    (row["domain"], row["problem"])
    for row in rows
    if row["mode"] == "hierarchical"
))

per_instance = []
successful = []
for domain, problem in cases:
    domain_file = ipc / domain / "domain.hddl"
    problem_file = ipc / domain / problem
    times = []
    output = ""
    for _ in range(repeats):
        start = time.perf_counter()
        try:
            result = subprocess.run(
                [str(bench), str(domain_file), str(problem_file)],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            elapsed = (time.perf_counter() - start) * 1000
            output = (result.stdout or result.stderr).strip().splitlines()
            output = output[-1] if output else ""
        except subprocess.TimeoutExpired:
            elapsed = timeout * 1000
            output = "timeout"
        times.append(elapsed)
        if output == "timeout":
            break

    median_ms = statistics.median(times)
    per_instance.append(median_ms)
    if "solved=true" in output:
        successful.append(median_ms)
    print(f"{domain}/{problem}: {median_ms:.0f} ms ({len(times)} run(s)) {output}")

print()
print(f"C+G median over {len(per_instance)} instances: {statistics.median(per_instance):.0f} ms")
if successful:
    print(f"C+G median over {len(successful)} successful instances: {statistics.median(successful):.0f} ms")
PY

python3 - <<'PY'
import csv, statistics

rows = [
    r for r in csv.DictReader(open("target/decomposition-comparison/results.csv"))
    if r["mode"] == "hierarchical"
]

for label, selected in [
    ("all", rows),
    ("solved", [r for r in rows if r["pspr_status"] == "true"]),
]:
    times = [int(r["pspr_ms"]) for r in selected]
    print(
        f"{label}: n={len(times)} "
        f"median={statistics.median(times):.1f} ms "
        f"mean={statistics.mean(times):.1f} ms "
        f"stdev={statistics.stdev(times):.1f} ms"
    )
PY
