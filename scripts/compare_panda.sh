#!/usr/bin/env bash
#
# Empirical comparison of PSPR and the pandaPI planning system.
#
# PSPR plans the initial task network with compound tasks treated as opaque
# operators. pandaPI grounds the full HTN problem and searches for a
# decomposition into primitive actions. The two planners therefore solve
# different problems; the script reports both results side by side and never
# pretends the plan lengths are comparable.
#
# Usage:
#
#   cargo build --release --example bench
#   scripts/compare_panda.sh
#
# Environment:
#
#   IPC_DIR        IPC benchmark checkout (default: ../hddl/test_data/ipc)
#   PSPR_BIN       PSPR benchmark binary (default: target/release/examples/bench)
#   HDDL_PARSER    hddl_analyzer executable used by PSPR
#   PANDA_PARSER   pandaPIparser executable
#   PANDA_GROUNDER pandaPIgrounder executable
#   PANDA_ENGINE   pandaPIengine executable
#   TIME_LIMIT     pandaPI search time limit in seconds (default: 60)
#   PSPR_REPEATS   number of PSPR runs per instance (default: 5)
#   OUT_DIR        directory for logs and results.csv (default: target/panda-comparison)
#
# The script prints a Markdown table on stdout and writes the raw per-instance
# logs plus results.csv into OUT_DIR.

set -uo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
IPC_DIR=${IPC_DIR:-"$repo_root/../hddl/test_data/ipc"}
PSPR_BIN=${PSPR_BIN:-"$repo_root/target/release/examples/bench"}
HDDL_PARSER=${HDDL_PARSER:-"$HOME/.cargo/bin/hddl_analyzer"}
PANDA_PARSER=${PANDA_PARSER:-$(command -v pandaPIparser || true)}
PANDA_PARSER=${PANDA_PARSER:-/tmp/pandaPI/pandaPIparser/pandaPIparser}
PANDA_GROUNDER=${PANDA_GROUNDER:-$(command -v pandaPIgrounder || true)}
PANDA_GROUNDER=${PANDA_GROUNDER:-/tmp/pandaPI/pandaPIgrounder/pandaPIgrounder}
PANDA_ENGINE=${PANDA_ENGINE:-$(command -v pandaPIengine || true)}
PANDA_ENGINE=${PANDA_ENGINE:-/tmp/pandaPI/pandaPIengine/build/pandaPIengine}
TIME_LIMIT=${TIME_LIMIT:-60}
PSPR_REPEATS=${PSPR_REPEATS:-5}
OUT_DIR=${OUT_DIR:-"$repo_root/target/panda-comparison"}

for path in "$IPC_DIR" "$PANDA_PARSER" "$PANDA_GROUNDER" "$PANDA_ENGINE" "$HDDL_PARSER"; do
    if [[ ! -e $path ]]; then
        echo "missing: $path" >&2
        echo "set IPC_DIR / PANDA_PARSER / PANDA_GROUNDER / PANDA_ENGINE / HDDL_PARSER" >&2
        exit 2
    fi
done

if [[ ! -x $PSPR_BIN ]]; then
    echo "building $PSPR_BIN" >&2
    (cd "$repo_root" && HDDL_PARSER="$HDDL_PARSER" cargo build --release --example bench) || exit 2
fi

# scip-sys links the bundled SCIP dynamically; make the library visible for
# the benchmark process.
scip_lib=$(find "$repo_root/target/release" -path '*scip_install/lib/libscip.so' -print -quit 2>/dev/null || true)
if [[ -n $scip_lib ]]; then
    scip_dir=$(dirname "$scip_lib")
    export LD_LIBRARY_PATH="$scip_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export HDDL_PARSER

mkdir -p "$OUT_DIR"
raw="$OUT_DIR/raw"
rm -rf "$raw"
mkdir -p "$raw"
csv="$OUT_DIR/results.csv"
printf 'domain,problem,pspr_status,pspr_goal,pspr_steps,pspr_ms,panda_status,panda_actions,panda_parse_ms,panda_ground_ms,panda_search_ms,panda_total_ms\n' >"$csv"

# Keep this list in sync with tests/ipc.rs. The list is deliberately small:
# every entry is an IPC instance that exercises a different encoding feature
# (totally ordered, partial order, state goals, numeric tasks, learned models).
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
    # field key line -> value of key=value, with values kept verbatim
    local key=$1 line=$2
    sed -n "s/.*\b${key}=\([^ ]*\).*/\1/p" <<<"$line"
}

printf '| Domain | Problem | PSPR solved | PSPR goal | PSPR steps | PSPR ms | PANDA status | PANDA actions | PANDA search ms | PANDA total ms |\n'
printf '| --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: |\n'

for case in "${CASES[@]}"; do
    domain=${case%%|*}
    problem=${case#*|}
    tag=$(printf '%s_%s' "$domain" "$problem" | tr '/.' '__')
    domain_file="$IPC_DIR/$domain/domain.hddl"
    problem_file="$IPC_DIR/$domain/$problem"

    # ---------- PSPR ----------
    pspr_status=timeout
    pspr_goal=-
    pspr_steps=-
    pspr_times=()
    for ((i = 0; i < PSPR_REPEATS; i++)); do
        log="$raw/$tag.pspr.$i.log"
        start=$(now_ms)
        timeout 300 "$PSPR_BIN" "$domain_file" "$problem_file" >"$log" 2>&1
        stop=$(now_ms)
        pspr_times+=($((stop - start)))
        line=$(grep -a 'solved=' "$log" | tail -1)
        if [[ -n $line ]]; then
            pspr_status=$(field solved "$line")
            pspr_goal=$(field goal "$line")
            pspr_steps=$(field steps "$line")
        fi
    done
    pspr_ms=$(printf '%s\n' "${pspr_times[@]}" | median)

    # ---------- PANDA ----------
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
                panda_actions=$(grep -a 'Found solution of length' "$engine_log" | tail -1 | sed -n 's/.*length \([0-9][0-9]*\).*/\1/p')
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

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$domain" "$problem" "$pspr_status" "$pspr_goal" "$pspr_steps" "$pspr_ms" \
        "$panda_status" "$panda_actions" "$parse_ms" "$ground_ms" "$search_ms" "$total_ms" >>"$csv"

    printf '| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n' \
        "$domain" "$problem" "$pspr_status" "$pspr_goal" "$pspr_steps" "$pspr_ms" \
        "$panda_status" "$panda_actions" "$search_ms" "$total_ms"
done

# ---------- summary ----------
read -r total pspr_solved pspr_valid panda_solved common panda_total pspr_total < <(
    awk -F, '
        NR == 1 { next }
        {
            n++
            if ($3 == "true") pspr_solved++
            if ($3 == "true" && ($4 == "satisfied" || $4 == "absent")) pspr_valid++
            if ($7 == "solved") panda_solved++
            if ($3 == "true" && $7 == "solved" && $6 + 0 > 0 && $12 + 0 > 0) {
                common++
                panda_total += $12
                pspr_total += $6
            }
        }
        END {
            printf "%d %d %d %d %d %d %d\n",
                n, pspr_solved, pspr_valid, panda_solved, common, panda_total, pspr_total
        }
    ' "$csv"
)

median_ratio=$(awk -F, 'NR > 1 && $3 == "true" && $7 == "solved" && $6 + 0 > 0 && $12 + 0 > 0 {
        printf "%.6f\n", $12 / $6
    }' "$csv" | sort -n | awk '
        { values[NR] = $1 }
        END {
            if (NR == 0) { print "0" }
            else if (NR % 2) { printf "%.2f", values[(NR + 1) / 2] }
            else { printf "%.2f", (values[NR / 2] + values[NR / 2 + 1]) / 2 }
        }')

printf '\n%d instances: PSPR solved %d, of which %d passed the state-goal check (or had no goal); pandaPI solved %d.\n' \
    "$total" "$pspr_solved" "$pspr_valid" "$panda_solved"
if ((common > 0)); then
    printf 'On the %d instances solved by both, PSPR took %d ms in total and pandaPI %d ms (aggregate %.1fx); the median per-instance ratio was %sx.\n' \
        "$common" "$pspr_total" "$panda_total" \
        "$(awk -v p="$panda_total" -v s="$pspr_total" 'BEGIN { printf "%.1f", p / s }')" \
        "$median_ratio"
fi
printf 'PSPR milliseconds are medians of the repeated runs; pandaPI runs once per instance and its total includes parsing, grounding and search.\n'
