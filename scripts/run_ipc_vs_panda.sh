#!/usr/bin/env bash
#
# Timeout-safe IPC runner for PSPR (hierarchical path) vs pandaPI.
#
# Runs the same case list used by the decomposition comparison harness, but
# executes each case independently so long-running instances do not destabilize
# one large terminal session.
#
# Usage:
#   scripts/run_ipc_vs_panda.sh
#
# Environment overrides:
#   IPC_DIR                 (default: ../hddl/test_data/ipc)
#   HDDL_PARSER             (default: $HOME/.cargo/bin/hddl_analyzer)
#   PSPR_BIN                (default: target/release/examples/bench_modes)
#   PANDA_PARSER            (default: pandaPIparser in PATH, then /tmp/pandaPI/...)
#   PANDA_GROUNDER          (default: pandaPIgrounder in PATH, then /tmp/pandaPI/...)
#   PANDA_ENGINE            (default: pandaPIengine in PATH, then /tmp/pandaPI/...)
#   PSPR_TIMEOUT            (default: 120 seconds)
#   PANDA_PARSE_TIMEOUT     (default: 120 seconds)
#   PANDA_GROUND_TIMEOUT    (default: 300 seconds)
#   PANDA_SEARCH_TIMEOUT    (default: 120 seconds)
#   PSPR_REPEATS            (default: 1)
#   PSPR_MAX_GROUNDINGS     (default: 4096)
#   OUT_DIR                 (default: target/panda-comparison)
#   CASE_FILTER             (optional regex over "Domain/problem", e.g. "Blocksworld|Depots")

set -uo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
IPC_DIR=${IPC_DIR:-"$repo_root/../hddl/test_data/ipc"}
HDDL_PARSER=${HDDL_PARSER:-"$HOME/.cargo/bin/hddl_analyzer"}
PSPR_BIN=${PSPR_BIN:-"$repo_root/target/release/examples/bench_modes"}

PANDA_PARSER=${PANDA_PARSER:-$(command -v pandaPIparser || true)}
PANDA_PARSER=${PANDA_PARSER:-/tmp/pandaPI/pandaPIparser/pandaPIparser}
PANDA_GROUNDER=${PANDA_GROUNDER:-$(command -v pandaPIgrounder || true)}
PANDA_GROUNDER=${PANDA_GROUNDER:-/tmp/pandaPI/pandaPIgrounder/pandaPIgrounder}
PANDA_ENGINE=${PANDA_ENGINE:-$(command -v pandaPIengine || true)}
PANDA_ENGINE=${PANDA_ENGINE:-/tmp/pandaPI/pandaPIengine/build/pandaPIengine}

PSPR_TIMEOUT=${PSPR_TIMEOUT:-120}
PANDA_PARSE_TIMEOUT=${PANDA_PARSE_TIMEOUT:-120}
PANDA_GROUND_TIMEOUT=${PANDA_GROUND_TIMEOUT:-300}
PANDA_SEARCH_TIMEOUT=${PANDA_SEARCH_TIMEOUT:-120}
PSPR_REPEATS=${PSPR_REPEATS:-1}
PSPR_MAX_GROUNDINGS=${PSPR_MAX_GROUNDINGS:-4096}
OUT_DIR=${OUT_DIR:-"$repo_root/target/panda-comparison"}
CASE_FILTER=${CASE_FILTER:-}

CASES=(
    'Blocksworld-GTOHP|p01.hddl'
    'Blocksworld-GTOHP|p02.hddl'
    'Blocksworld-GTOHP|p03.hddl'
    'Blocksworld-GTOHP|p04.hddl'
    'Blocksworld-GTOHP|p05.hddl'
    'Depots|p01.hddl'
    'Depots|p02.hddl'
    'Depots|p03.hddl'
    'Satellite-GTOHP|p01.hddl'
    'Satellite-GTOHP|p02.hddl'
    'Satellite-GTOHP|p03.hddl'
    'Transport|pfile01.hddl'
    'Transport|pfile02.hddl'
    'Transport|pfile03.hddl'
    'PO_Transport|pfile01.hddl'
    'PO_Satellite|1obs-1sat-1mod.hddl'
    'Freecell-Learned-ECAI-16|probfreecell-02-1.hddl'
    'Logistics-Learned-ECAI-16|probLOGISTICS-04-0.hddl'
    'Logistics-Learned-ECAI-16|probLOGISTICS-04-1.hddl'
    'Logistics-Learned-ECAI-16|probLOGISTICS-04-2.hddl'
    'AssemblyHierarchical|genericLinearProblem_depth01.hddl'
    'Barman-BDI|pfile01.hddl'
    'Barman-BDI|pfile02.hddl'
    'Robot|pfile_01_001.hddl'
    'Minecraft-Player|p-003-003-003-003.hddl'
    'Monroe_PO_1|pfile01-p-0014-fix-power-line-4.hddl'
    'PCP_1|p-pcp01.hddl'
)

for path in "$IPC_DIR" "$HDDL_PARSER" "$PANDA_PARSER" "$PANDA_GROUNDER" "$PANDA_ENGINE"; do
    if [[ ! -e $path ]]; then
        echo "missing: $path" >&2
        echo "set IPC_DIR / HDDL_PARSER / PANDA_PARSER / PANDA_GROUNDER / PANDA_ENGINE" >&2
        exit 2
    fi
done

if [[ ! -x $PSPR_BIN ]]; then
    echo "building $PSPR_BIN" >&2
    (cd "$repo_root" && HDDL_PARSER="$HDDL_PARSER" cargo build --release --example bench_modes) || exit 2
fi

scip_lib=$(find "$repo_root/target/release" -path '*scip_install/lib/libscip.so' -print -quit 2>/dev/null || true)
if [[ -n $scip_lib ]]; then
    scip_dir=$(dirname "$scip_lib")
    export LD_LIBRARY_PATH="$scip_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

mkdir -p "$OUT_DIR"
raw="$OUT_DIR/raw"
rm -rf "$raw"
mkdir -p "$raw"

csv="$OUT_DIR/results.csv"
printf 'domain,problem,pspr_status,pspr_goal,pspr_actions,pspr_methods,pspr_abstract_steps,pspr_tasks,pspr_ms,panda_status,panda_actions,panda_parse_ms,panda_ground_ms,panda_search_ms,panda_total_ms\n' >"$csv"

export HDDL_PARSER
export PSPR_MAX_GROUNDINGS

now_ms() {
    echo $(( $(date +%s%N) / 1000000 ))
}

median() {
    sort -n | awk '{ values[NR] = $1 } END {
        if (NR == 0) { print 0; exit }
        if (NR % 2) { print values[(NR + 1) / 2]; exit }
        print int((values[NR / 2] + values[NR / 2 + 1]) / 2)
    }'
}

field() {
    local key=$1
    local line=$2
    sed -n "s/.*\\b${key}=\([^ ]*\).*/\1/p" <<<"$line"
}

printf '| Domain | Problem | PSPR | PANDA |\n'
printf '| --- | --- | --- | --- |\n'

for case in "${CASES[@]}"; do
    domain=${case%%|*}
    problem=${case#*|}
    case_id="$domain/$problem"
    if [[ -n $CASE_FILTER ]] && ! [[ $case_id =~ $CASE_FILTER ]]; then
        continue
    fi
    tag=$(printf '%s_%s' "$domain" "$problem" | tr '/.' '__')
    domain_file="$IPC_DIR/$domain/domain.hddl"
    problem_file="$IPC_DIR/$domain/$problem"

    if [[ ! -f $domain_file || ! -f $problem_file ]]; then
        echo "warning: missing case files for $domain/$problem" >&2
        printf '%s,%s,missing,invalid,0,0,0,0,0,missing,-,0,0,0,0\n' "$domain" "$problem" >>"$csv"
        continue
    fi

    # -------- PSPR (bench_modes) --------
    pspr_status=timeout
    pspr_goal=invalid
    pspr_actions=0
    pspr_methods=0
    pspr_abstract_steps=0
    pspr_tasks=0
    pspr_times=()

    for ((i = 0; i < PSPR_REPEATS; i++)); do
        log="$raw/$tag.pspr.$i.log"
        start=$(now_ms)
        timeout "$PSPR_TIMEOUT" "$PSPR_BIN" "$domain_file" "$problem_file" >"$log" 2>&1
        code=$?
        stop=$(now_ms)
        pspr_times+=($((stop - start)))

        if [[ $code -eq 124 ]]; then
            pspr_status=timeout
            pspr_goal=timeout
            break
        fi

        line=$(grep -a 'mode=' "$log" | tail -1 || true)
        if [[ -z $line ]]; then
            pspr_status=error
            pspr_goal=invalid
            break
        fi

        solved=$(field solved "$line")
        pspr_goal=$(field goal "$line")
        pspr_actions=$(field actions "$line")
        pspr_methods=$(field methods "$line")
        pspr_abstract_steps=$(field abstract_steps "$line")
        pspr_tasks=$(field tasks "$line")

        if [[ $solved == true ]]; then
            pspr_status=solved
        else
            pspr_status=failed
        fi
    done

    pspr_ms=$(printf '%s\n' "${pspr_times[@]:-0}" | median)

    # -------- PANDA --------
    panda_status=error
    panda_actions=-
    panda_parse_ms=0
    panda_ground_ms=0
    panda_search_ms=0
    panda_total_ms=0

    htn="$raw/$tag.panda.htn"
    sas="$raw/$tag.panda.sas"
    parse_log="$raw/$tag.panda-parse.log"
    ground_log="$raw/$tag.panda-ground.log"
    engine_log="$raw/$tag.panda-engine.log"

    t0=$(now_ms)
    if timeout "$PANDA_PARSE_TIMEOUT" "$PANDA_PARSER" "$domain_file" "$problem_file" "$htn" >"$parse_log" 2>&1 && [[ -s $htn ]]; then
        t1=$(now_ms)
        panda_parse_ms=$((t1 - t0))
        if timeout "$PANDA_GROUND_TIMEOUT" "$PANDA_GROUNDER" -q "$htn" "$sas" >"$ground_log" 2>&1 && [[ -s $sas ]]; then
            t2=$(now_ms)
            panda_ground_ms=$((t2 - t1))
            timeout "$PANDA_SEARCH_TIMEOUT" "$PANDA_ENGINE" -t "$PANDA_SEARCH_TIMEOUT" "$sas" >"$engine_log" 2>&1
            panda_code=$?
            t3=$(now_ms)
            panda_total_ms=$((t3 - t0))

            if [[ $panda_code -eq 124 ]] || grep -aq 'Status: Timeout' "$engine_log"; then
                panda_status=timeout
            elif grep -aq 'Found solution of length' "$engine_log"; then
                panda_status=solved
                panda_actions=$(grep -a 'Total costs of actions' "$engine_log" | tail -1 | sed -n 's/.*actions: \([0-9][0-9]*\).*/\1/p')
                if [[ -z $panda_actions ]]; then
                    panda_actions=$(grep -a 'Found solution of length' "$engine_log" | tail -1 | sed -n 's/.*length \([0-9][0-9]*\).*/\1/p')
                fi
                [[ -n $panda_actions ]] || panda_actions=-
            elif grep -aq 'Status: Proven unsolvable' "$engine_log"; then
                panda_status=unsolvable
            else
                panda_status=failed
            fi

            search_s=$(grep -a 'Search time' "$engine_log" | tail -1 | sed -n 's/.*Search time \([0-9][0-9.]*\) seconds.*/\1/p')
            if [[ -n $search_s ]]; then
                panda_search_ms=$(awk -v s="$search_s" 'BEGIN { printf "%d", s * 1000 }')
            fi
        else
            t2=$(now_ms)
            panda_ground_ms=$((t2 - t1))
            panda_total_ms=$((t2 - t0))
            panda_status=ground-error
        fi
    else
        t1=$(now_ms)
        panda_parse_ms=$((t1 - t0))
        panda_total_ms=$((t1 - t0))
        panda_status=parse-error
    fi

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$domain" "$problem" \
        "$pspr_status" "$pspr_goal" "$pspr_actions" "$pspr_methods" "$pspr_abstract_steps" "$pspr_tasks" "$pspr_ms" \
        "$panda_status" "$panda_actions" "$panda_parse_ms" "$panda_ground_ms" "$panda_search_ms" "$panda_total_ms" \
        >>"$csv"

    pspr_cell="${pspr_status}:${pspr_goal}:${pspr_actions}@${pspr_ms}ms"
    panda_cell="${panda_status}:${panda_actions}@${panda_total_ms}ms"
    printf '| %s | %s | %s | %s |\n' "$domain" "$problem" "$pspr_cell" "$panda_cell"
done

python3 - "$csv" <<'PY'
import csv
import statistics
import sys

rows = list(csv.DictReader(open(sys.argv[1])))

def count(key, value):
    return sum(row[key] == value for row in rows)

print("\nPSPR summary:")
for status in sorted({row["pspr_status"] for row in rows}):
    print(f"  {status}: {count('pspr_status', status)}")

solved = [int(row["pspr_ms"]) for row in rows if row["pspr_status"] == "solved"]
if solved:
    print(f"  solved median ms: {statistics.median(solved):.0f}")

print("\nPANDA summary:")
for status in sorted({row["panda_status"] for row in rows}):
    print(f"  {status}: {count('panda_status', status)}")

panda_solved = [
    int(row["panda_total_ms"])
    for row in rows
    if row["panda_status"] == "solved"
]
if panda_solved:
    print(f"  solved median total ms: {statistics.median(panda_solved):.0f}")

print(f"\nCSV written to: {sys.argv[1]}")
PY
