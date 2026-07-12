#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
app_root="${repo_root}/Hera"

cd "$app_root"

BIN="hera-core"

if [[ -f "./target/release/${BIN}" ]] && ! file "./target/release/${BIN}" | grep -q "ELF"; then
    echo "[imaginos] ${BIN}: binary is not ELF (corrupt or wrong arch) — removing it"
    rm -f "./target/release/${BIN}"
fi

if [[ -f "./target/release/${BIN}" ]]; then
    set +e
    timeout 5 "./target/release/${BIN}" --help >/dev/null 2>&1
    probe_status=$?
    set -e
    if [[ "${probe_status}" != "0" && "${probe_status}" != "124" ]]; then
        echo "[imaginos] ${BIN}: binary cannot run on this host — removing it"
        rm -f "./target/release/${BIN}"
    fi
fi

if [[ ! -x "./target/release/${BIN}" ]]; then
    if [[ -x "${HOME}/bin/${BIN}" ]]; then
        echo "[imaginos] ${BIN}: using pre-deployed binary at ${HOME}/bin/${BIN}"
        exec "${HOME}/bin/${BIN}"
    elif [[ "${IMAGINOS_ALLOW_RELEASE_BUILD:-0}" == "1" ]]; then
        echo "[imaginos] ${BIN}: release binary missing; building because IMAGINOS_ALLOW_RELEASE_BUILD=1"
        cargo build --release --bin "${BIN}"
    else
        echo "[imaginos] ${BIN}: release binary missing; refusing compile-on-boot"
        echo "[imaginos] build ${BIN} ahead of time or set IMAGINOS_ALLOW_RELEASE_BUILD=1 for an emergency rebuild"
        exit 1
    fi
fi

exec "./target/release/${BIN}"
