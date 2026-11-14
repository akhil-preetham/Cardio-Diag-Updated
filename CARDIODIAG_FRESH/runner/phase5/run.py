"""Execute the isolated CARDIODIAG Phase 5 programme."""
from __future__ import annotations

from runner.phase5.data_audit.audit import run as run_data_audit
from runner.phase5.reliability.run import run as run_reliability
from runner.phase5.reporting.build_reports import run as run_reports
from runner.phase5.uci_track.run import run as run_uci
from runner.phase5.zas_track.run import run as run_zas


def main() -> None:
    run_data_audit()
    run_zas()
    run_uci()
    run_reliability()
    run_reports()
    print("Phase 5 complete")


if __name__ == "__main__":
    main()
