#!/usr/bin/env bash
#
# Build the pandaPI toolchain (parser, grounder, engine) for the comparison
# harness in scripts/run_ipc_vs_panda.sh.
#
# The build is self-contained: gengetopt is compiled into
# "$PANDA_ROOT/local" and nothing is installed system-wide.
#
# Usage:
#   scripts/install_panda.sh
#
# Environment overrides:
#   PANDA_ROOT       (default: /tmp/pandaPI)
#   GENGETOPT_VER    (default: 2.23)
#   JOBS             (default: number of CPUs)
#   USE_OPENCL       (default: no; the grounder only needs the CPU path and
#                     many systems ship the OpenCL runtime without headers)
#
# After a successful run the binaries are at:
#   $PANDA_ROOT/pandaPIparser/pandaPIparser
#   $PANDA_ROOT/pandaPIgrounder/pandaPIgrounder
#   $PANDA_ROOT/pandaPIengine/build/pandaPIengine
#
# scripts/run_ipc_vs_panda.sh uses exactly these paths by default.

set -uo pipefail

PANDA_ROOT=${PANDA_ROOT:-/tmp/pandaPI}
GENGETOPT_VER=${GENGETOPT_VER:-2.23}
JOBS=${JOBS:-$(nproc)}
USE_OPENCL=${USE_OPENCL:-no}

PARSER_REPO=https://github.com/panda-planner-dev/pandaPIparser.git
GROUNDER_REPO=https://github.com/panda-planner-dev/pandaPIgrounder.git
ENGINE_REPO=https://github.com/panda-planner-dev/pandaPIengine.git

log() { printf '[install-panda] %s\n' "$*"; }
die() { printf '[install-panda] error: %s\n' "$*" >&2; exit 1; }

need() {
    for tool in "$@"; do
        command -v "$tool" >/dev/null 2>&1 || die "missing required tool: $tool"
    done
}

need git make g++ gcc cmake flex bison zip unzip patch tar

mkdir -p "$PANDA_ROOT/src" "$PANDA_ROOT/local"
PREFIX="$PANDA_ROOT/local"
export PATH="$PREFIX/bin:$PATH"

# ---------------------------------------------------------------- gengetopt --
if [[ -x $PREFIX/bin/gengetopt ]]; then
    log "gengetopt already installed in $PREFIX"
else
    tarball="$PANDA_ROOT/src/gengetopt-$GENGETOPT_VER.tar.xz"
    srcdir="$PANDA_ROOT/src/gengetopt-$GENGETOPT_VER"
    if [[ ! -f $tarball ]]; then
        log "downloading gengetopt $GENGETOPT_VER"
        curl -fsSL -o "$tarball" \
            "https://ftp.gnu.org/gnu/gengetopt/gengetopt-$GENGETOPT_VER.tar.xz" \
            || die "cannot download gengetopt"
    fi
    if [[ ! -d $srcdir ]]; then
        log "extracting gengetopt"
        tar -xJf "$tarball" -C "$PANDA_ROOT/src" || die "cannot extract gengetopt"
    fi
    log "building gengetopt"
    (cd "$srcdir" \
        && ./configure --prefix="$PREFIX" >"$PANDA_ROOT/gengetopt-configure.log" 2>&1 \
        && make -j"$JOBS" >"$PANDA_ROOT/gengetopt-make.log" 2>&1 \
        && make install >>"$PANDA_ROOT/gengetopt-make.log" 2>&1) \
        || die "gengetopt build failed (see $PANDA_ROOT/gengetopt-*.log)"
    [[ -x $PREFIX/bin/gengetopt ]] || die "gengetopt was not installed"
fi

# ------------------------------------------------------------------- clones --
clone() {
    local repo=$1 dir=$2
    if [[ -d $dir/.git ]]; then
        log "$(basename "$dir") already cloned"
    else
        log "cloning $(basename "$dir")"
        git clone --depth 1 "$repo" "$dir" >"$PANDA_ROOT/clone-$(basename "$dir").log" 2>&1 \
            || die "cannot clone $repo"
    fi
}

cd "$PANDA_ROOT" || die "cannot enter $PANDA_ROOT"
clone "$PARSER_REPO" "$PANDA_ROOT/pandaPIparser"
clone "$GROUNDER_REPO" "$PANDA_ROOT/pandaPIgrounder"
clone "$ENGINE_REPO" "$PANDA_ROOT/pandaPIengine"

# ------------------------------------------------------------------- parser --
if [[ -x $PANDA_ROOT/pandaPIparser/pandaPIparser ]]; then
    log "pandaPIparser already built"
else
    log "building pandaPIparser"
    (cd "$PANDA_ROOT/pandaPIparser" && make -j"$JOBS") \
        >"$PANDA_ROOT/parser-build.log" 2>&1 \
        || die "pandaPIparser build failed (see $PANDA_ROOT/parser-build.log)"
fi

# ----------------------------------------------------------------- grounder --
if [[ -x $PANDA_ROOT/pandaPIgrounder/pandaPIgrounder ]]; then
    log "pandaPIgrounder already built"
else
    log "fetching grounder submodules"
    (cd "$PANDA_ROOT/pandaPIgrounder" && git submodule update --init --recursive) \
        >"$PANDA_ROOT/submodule.log" 2>&1 \
        || die "cannot fetch pandaPIgrounder submodules"

    log "building cpddl (this takes a few minutes)"
    (
        cd "$PANDA_ROOT/pandaPIgrounder/cpddl" || exit 1
        # The Makefile patch removes -lrt (glibc >= 2.34) and applies two
        # portability patches to the bundled boruvka.
        git apply ../0002-makefile.patch 2>/dev/null
        make USE_OPENCL="$USE_OPENCL" -j"$JOBS" boruvka opts bliss lpsolve || exit 1
        make USE_OPENCL="$USE_OPENCL" -j"$JOBS" || exit 1
    ) >"$PANDA_ROOT/cpddl-build.log" 2>&1 \
        || die "cpddl build failed (see $PANDA_ROOT/cpddl-build.log)"

    # GCC 13+ no longer transitively includes <algorithm>; parser.cpp uses
    # std::sort and std::for_each.
    if ! grep -q '^#include <algorithm>' "$PANDA_ROOT/pandaPIgrounder/src/parser.cpp"; then
        log "applying <algorithm> include fix to grounder parser.cpp"
        python3 - "$PANDA_ROOT/pandaPIgrounder/src/parser.cpp" <<'PY'
import sys
from pathlib import Path
path = Path(sys.argv[1])
text = path.read_text()
if '#include <algorithm>' not in text:
    text = text.replace('#include <cassert>\n', '#include <algorithm>\n#include <cassert>\n', 1)
    path.write_text(text)
PY
    fi

    log "building pandaPIgrounder"
    (cd "$PANDA_ROOT/pandaPIgrounder/src" && make USE_OPENCL="$USE_OPENCL" -j"$JOBS") \
        >"$PANDA_ROOT/grounder-build.log" 2>&1 \
        || die "pandaPIgrounder build failed (see $PANDA_ROOT/grounder-build.log)"
fi

# ------------------------------------------------------------------- engine --
if [[ -x $PANDA_ROOT/pandaPIengine/build/pandaPIengine ]]; then
    log "pandaPIengine already built"
else
    log "building pandaPIengine"
    mkdir -p "$PANDA_ROOT/pandaPIengine/build"
    (
        cd "$PANDA_ROOT/pandaPIengine/build" || exit 1
        cmake ../src -DCMAKE_BUILD_TYPE=Release || exit 1
        make -j"$JOBS" || exit 1
    ) >"$PANDA_ROOT/engine-build.log" 2>&1 \
        || die "pandaPIengine build failed (see $PANDA_ROOT/engine-build.log)"
fi

# ------------------------------------------------------------------- verify --
missing=0
for binary in \
    "$PANDA_ROOT/pandaPIparser/pandaPIparser" \
    "$PANDA_ROOT/pandaPIgrounder/pandaPIgrounder" \
    "$PANDA_ROOT/pandaPIengine/build/pandaPIengine"; do
    if [[ -x $binary ]]; then
        log "ok: $binary"
    else
        log "MISSING: $binary"
        missing=$((missing + 1))
    fi
done
[[ $missing -eq 0 ]] || die "$missing pandaPI binaries are missing"

log "done; run scripts/run_ipc_vs_panda.sh (add PANDA_ROOT to your environment if you changed it)"
