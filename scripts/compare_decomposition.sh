#!/usr/bin/env bash
#
# Compare PSPR's hierarchical refinement (C+G) with pandaPI.
#
# pandaPI solves the full HTN problem and is the reference for complete plans.
#
# Usage:
#
#   cargo build --release --example bench_modes
#   scripts/compare_decomposition.sh
#
# The script writes target/decomposition-comparison/results.csv and prints a
# per-instance status table plus a compact summary.

set -uo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
IPC_DIR=${IPC_DIR:-"$repo_root/../hddl/test_data/ipc"}
PSPR_BIN=${PSPR_BIN:-"$repo_root/target/release/examples/bench_modes"}
HDDL_PARSER=${HDDL_PARSER:-"$HOME/.cargo/bin/hddl_analyzer"}
PANDA_PARSER=${PANDA_PARSER:-$(command -v pandaPIparser || true)}
PANDA_PARSER=${PANDA_PARSER:-/tmp/pandaPI/pandaPIparser/pandaPIparser}
PANDA_GROUNDER=${PANDA_GROUNDER:-$(command -v pandaPIgrounder || true)}
PANDA_GROUNDER=${PANDA_GROUNDER:-/tmp/pandaPI/pandaPIgrounder/pandaPIgrounder}
PANDA_ENGINE=${PANDA_ENGINE:-$(command -v pandaPIengine || true)}
PANDA_ENGINE=${PANDA_ENGINE:-/tmp/pandaPI/pandaPIengine/build/pandaPIengine}
TIME_LIMIT=${TIME_LIMIT:-60}
PSPR_TIMEOUT=${PSPR_TIMEOUT:-30}
PSPR_REPEATS=${PSPR_REPEATS:-3}
PSPR_MAX_GROUNDINGS=${PSPR_MAX_GROUNDINGS:-4096}
OUT_DIR=${OUT_DIR:-$repo_root/target/decomposition-comparison}

lock=$OUT_DIR/.lock
mkdir -p "$OUT_DIR"
exec 9>"$lock"
if ! flock -n 9; then
    echo "another comparison run holds $lock" >&2
    exit 2
fi

for path in "$IPC_DIR" "$PANDA_PARSER" "$PANDA_GROUNDER" "$PANDA_ENGINE" "$HDDL_PARSER"; do
    if [[ ! -e $path ]]; then
        echo "missing: $path" >&2
        echo "set IPC_DIR / PANDA_PARSER / PANDA_GROUNDER / PANDA_ENGINE / HDDL_PARSER" >&2
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
export HDDL_PARSER
export PSPR_MAX_GROUNDINGS

mkdir -p "$OUT_DIR"
raw="$OUT_DIR/raw"
rm -rf "$raw"
mkdir -p "$raw"
csv="$OUT_DIR/results.csv"
printf 'domain,problem,mode,pspr_status,pspr_goal,pspr_actions,pspr_methods,pspr_abstract_steps,pspr_tasks,pspr_ms,panda_status,panda_actions,panda_parse_ms,panda_ground_ms,panda_search_ms,panda_total_ms\n' >"$csv"

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

now_ms() {
    echo $(($(date +%s%N) / 1000000))
}

median() {
    sort -n | awk '{ values[NR] = $1 } END {
        if (NR == 0) { print 0; exit }
        if (NR % 2) { print values[(NR + 1) / 2]; exit }
        print int((values[NR / 2] + values[NR / 2 + 1]) / 2)
    }'
}

field() {
    local key=$1 line=$2
    sed -n "s/.*\b${key}=\([^ ]*\).*/\1/p" <<<"$line"
}

printf '| Domain | Problem | PSPR (C+G) | PANDA |\n'
printf '| --- | --- | --- | --- |\n'

for case in "${CASES[@]}"; do
    domain=${case%%|*}
    problem=${case#*|}
    tag=$(printf '%s_%s' "$domain" "$problem" | tr '/.' '__')
    domain_file="$IPC_DIR/$domain/domain.hddl"
    problem_file="$IPC_DIR/$domain/$problem"

    # ---------- PANDA, once per instance ----------
    htn="$raw/$tag.htn"
    sas="$raw/$tag.sas"
    parse_log="$raw/$tag.panda-parse.log"
    ground_log="$raw/$tag.panda-ground.log"
    engine_log="$raw/$tag.panda-engine.log"
    panda_status=error
    panda_actions=-
    parse_ms=-
    ground_ms=-
    search_ms=-
    total_ms=-

    start=$(now_ms)
    if "$PANDA_PARSER" "$domain_file" "$problem_file" "$htn" >"$parse_log" 2>&1 && [[ -s $htn ]]; then
        mid=$(now_ms)
        if timeout 300 "$PANDA_GROUNDER" -q "$htn" "$sas" >"$ground_log" 2>&1 && [[ -s $sas ]]; then
            grounded=$(now_ms)
            timeout $((TIME_LIMIT + 30)) "$PANDA_ENGINE" -t "$TIME_LIMIT" "$sas" >"$engine_log" 2>&1
            stop=$(now_ms)
            if grep -aq 'Found solution of length' "$engine_log"; then
                panda_status=solved
                panda_actions=$(grep -a 'Total costs of actions' "$engine_log" | tail -1 | sed -n 's/.*actions: \([0-9][0-9]*\).*/\1/p')
                if [[ -z $panda_actions ]]; then
                    panda_actions=$(grep -a 'Found solution of length' "$engine_log" | tail -1 | sed -n 's/.*length \([0-9][0-9]*\).*/\1/p')
                fi
                [[ -n $panda_actions ]] || panda_actions=-
            elif grep -aq 'Status: Proven unsolvable' "$engine_log"; then
                panda_status=unsolvable
            elif grep -aq 'Status: Timeout' "$engine_log"; then
                panda_status=timeout
            fi
            search_s=$(grep -a 'Search time' "$engine_log" | tail -1 | sed -n 's/.*Search time \([0-9][0-9.]*\) seconds.*/\1/p')
            [[ -n $search_s ]] && search_ms=$(awk -v s="$search_s" 'BEGIN { printf "%d", s * 1000 }')
            parse_ms=$((mid - start))
            ground_ms=$((grounded - mid))
            total_ms=$((stop - start))
        else
            panda_status=ground-error
            parse_ms=$((mid - start))
        fi
    else
        panda_status=parse-error
    fi

    # ---------- PSPR hierarchical ----------
    mode=hierarchical
    status=timeout
    goal=-
    actions=0
    methods=0
    abstract_steps=0
    tasks=0
    times=()
    for ((i = 0; i < PSPR_REPEATS; i++)); do
        log="$raw/$tag.$mode.$i.log"
        start=$(now_ms)
        timeout "$PSPR_TIMEOUT" "$PSPR_BIN" "$domain_file" "$problem_file" >"$log" 2>&1
        stop=$(now_ms)
        times+=($((stop - start)))
        line=$(grep -a 'mode=' "$log" | tail -1)
        if [[ -n $line ]]; then
            status=$(field solved "$line")
            goal=$(field goal "$line")
            actions=$(field actions "$line")
            methods=$(field methods "$line")
            abstract_steps=$(field abstract_steps "$line")
            tasks=$(field tasks "$line")
        fi
        if [[ $status == timeout ]]; then
            break
        fi
    done
    ms=$(printf '%s\n' "${times[@]}" | median)
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$domain" "$problem" "$mode" "$status" "$goal" "$actions" "$methods" \
        "$abstract_steps" "$tasks" "$ms" "$panda_status" "$panda_actions" \
        "$parse_ms" "$ground_ms" "$search_ms" "$total_ms" >>"$csv"

    # ---------- compact Markdown row ----------
    row="| $domain | $problem"
    line=$(awk -F, -v d="$domain" -v p="$problem" \
        'NR > 1 && $1 == d && $2 == p && $3 == "hierarchical" { print; exit }' "$csv")
    if [[ -z $line ]]; then
        row+=" | -"
    else
        IFS=, read -r _ _ _ status goal actions _ _ _ _ _ <<<"$line"
        if [[ $status == true ]]; then
            if [[ $goal == violated ]]; then
                row+=" | V($actions)"
            else
                row+=" | S($actions)"
            fi
        else
            row+=" | F"
        fi
    fi
    if [[ $panda_status == solved ]]; then
        row+=" | S($panda_actions)"
    elif [[ $panda_status == timeout ]]; then
        row+=" | T"
    else
        row+=" | F"
    fi
    row+=" |"
    echo "$row"
done

printf '\nS = solved and goal satisfied or absent; V = solved but goal violated; F = failed; T = timeout. Actions are primitive actions for PSPR/PANDA.\n'

# ---------- summary ----------
python3 - "$csv" <<'PY'
import csv
import statistics
import sys

rows = list(csv.DictReader(open(sys.argv[1])))
print()
print("| Technique | Solved | Goal satisfied/absent | Median ms | Mean ms |")
print("| --- | ---: | ---: | ---: | ---: | ---: |")
selected = [row for row in rows if row["mode"] == "hierarchical"]
solved = sum(row["pspr_status"] == "true" for row in selected)
goal = sum(row["pspr_goal"] in ("satisfied", "absent") for row in selected)
times = [int(row["pspr_ms"]) for row in selected]
print(
    f"| C+G | {solved}/{len(selected)} | {goal}/{len(selected)} | "
    f"{statistics.median(times):.0f} | {statistics.mean(times):.0f} |"
)
panda = rows
print()
print("PANDA reference (one run per instance):")
print("| Status | Count |")
print("| --- | ---: |")
for status in sorted({row["panda_status"] for row in panda}):
    print(f"| {status} | {sum(row['panda_status'] == status for row in panda)} |")
PY
