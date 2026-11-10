"""Run mocked checks by default; --live also requires LABRAG_RUN_DAY4_LIVE=1."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.evaluation.day4_runner import run_cases, export_report

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live",action="store_true")
    parser.add_argument("--audit-absence",action="store_true",help="Exhaustively read selected namespace for TC17; requires --live and explicit environment opt-in")
    parser.add_argument("--output",type=Path,default=Path("evaluation/day4_results.json"))
    args=parser.parse_args()
    report=run_cases(mode="live" if args.live else "mock",audit_corpus_absence=args.audit_absence)
    export_report(report,args.output)
    print(report["mode"],report["summary"])
    raise SystemExit(1 if report["summary"]["Failed"] else 0)
