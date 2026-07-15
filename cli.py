#!/usr/bin/env python3
"""CLI interface for resume generation and optimization.

Usage:
  # Scenario 2: Generate resume from personal profile
  python cli.py generate --scenario profile_only --profile "我是一名Python开发者..."

  # Scenario 4: Generate resume adapted to a JD
  python cli.py generate --scenario jd_only --profile "我是一名..." --jd "招聘高级算法工程师..."

  # Scenario 1: Optimize resume with JD
  python cli.py audit --file resume.pdf --jd "招聘高级算法工程师..."

  # Scenario 3: Optimize resume for target role
  python cli.py audit --file resume.pdf --target "我想转做运营"

  # From stdin
  cat profile.txt | python cli.py generate --scenario profile_only
"""

import argparse
import json
import sys
import os
from pathlib import Path

# Ensure the server package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from schemas import ResumeScenario, UserStage, JobFamily
from resume_service import generate_resume_service, resume_audit_and_optimize_service
from resume_io import extract_text_from_bytes
from schemas import GenerateRequest, AuditAndOptimizeRequest


def _read_optional_text(value: str | None, from_file: bool = False) -> str | None:
    """Read text from string or stdin."""
    if value:
        return value
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    return None


def _resolve_scenario(scenario: str) -> ResumeScenario:
    try:
        return ResumeScenario(scenario)
    except ValueError:
        valid = ", ".join(s.value for s in ResumeScenario)
        sys.exit(f"Invalid scenario '{scenario}'. Valid: {valid}")


def _resolve_job_family(family: str | None) -> JobFamily | None:
    if not family:
        return None
    try:
        return JobFamily(family)
    except ValueError:
        valid = ", ".join(f.value for f in JobFamily)
        sys.exit(f"Invalid job_family '{family}'. Valid: {valid}")


def _resolve_user_stage(stage: str | None) -> UserStage | None:
    if not stage:
        return None
    try:
        return UserStage(stage)
    except ValueError:
        sys.exit(f"Invalid user_stage '{stage}'. Valid: student, working")


def cmd_generate(args):
    """Handle scenarios 2 & 4: resume generation.

    Uses asyncio.run() (Python 3.10+) instead of deprecated
    get_event_loop().run_until_complete().
    """
    import asyncio

    async def _run():
        # Read inputs
        profile = _read_optional_text(args.profile) or ""
        jd_text = _read_optional_text(args.jd, from_file=args.jd_file)
        target = _read_optional_text(args.target)

        if not profile:
            print("Error: personal profile is required for generation scenarios.", file=sys.stderr)
            print("Provide --profile TEXT or pipe via stdin.", file=sys.stderr)
            sys.exit(1)

        scenario = _resolve_scenario(args.scenario)
        if scenario not in (ResumeScenario.profile_only, ResumeScenario.jd_only):
            print(f"Error: scenario '{scenario}' is not a generation scenario. Use 'audit' command.", file=sys.stderr)
            sys.exit(1)

        job_family = _resolve_job_family(args.job_family)
        user_stage = _resolve_user_stage(args.user_stage)

        # Build request
        req = GenerateRequest(
            scenario=scenario,
            personal_profile=profile,
            jd_text=jd_text,
            job_family=job_family,
            user_stage=user_stage,
            target_description=target,
            template=args.template or "new_standard",
        )

        # Call service
        try:
            response = await generate_resume_service(req)
        except Exception as e:
            print(f"Error during generation: {e}", file=sys.stderr)
            sys.exit(1)

        # Output results
        _print_generation_result(response, json_output=args.json)

    asyncio.run(_run())


def cmd_audit(args):
    """Handle scenario 1 and 3: resume audit + optimization."""
    import asyncio

    async def _run():
        # Read resume
        resume_text = None
        file_path = args.file
        resume_content = _read_optional_text(args.content)

        if args.file:
            path = Path(args.file)
            if not path.exists():
                print(f"Error: file not found: {args.file}", file=sys.stderr)
                sys.exit(1)
            with open(path, "rb") as f:
                raw = f.read()
            resume_text = extract_text_from_bytes(raw, path.name)
            if not resume_text or len(resume_text.strip()) < 10:
                print(f"Warning: file '{path}' appears empty or unreadable.", file=sys.stderr)
        elif resume_content:
            resume_text = resume_content
        else:
            # Try stdin
            resume_text = _read_optional_text(None)
            if not resume_text:
                print("Error: provide resume via --file, --content, or stdin.", file=sys.stderr)
                sys.exit(1)

        # Read JD and target
        jd_text = _read_optional_text(args.jd, from_file=args.jd_file)
        target = _read_optional_text(args.target)

        scenario = _resolve_scenario(args.scenario)
        if scenario not in (ResumeScenario.optimize_with_jd, ResumeScenario.profile_with_target):
            print(f"Error: scenario '{scenario}' is not an audit scenario. Use 'generate' command.", file=sys.stderr)
            sys.exit(1)

        # For scenario 3 (profile_with_target), treat resume_content as resume input
        req = AuditAndOptimizeRequest(
            resume_content=resume_text,
            file_path=file_path,
            jd_text=jd_text,
            style=args.style,
            template=args.template or "new_standard",
        )

        try:
            response = await resume_audit_and_optimize_service(req)
        except Exception as e:
            print(f"Error during audit: {e}", file=sys.stderr)
            sys.exit(1)

        _print_audit_result(response, json_output=args.json)

    asyncio.run(_run())


def _print_generation_result(response, json_output=False):
    """Pretty print a generation result."""
    print("=" * 60)
    print("  Resume Generated")
    print("=" * 60)

    # Score
    score = response.score
    print(f"\nTotal Score: {score.total} / 100")
    print(f"  Fabrication:  {'PASS' if score.fabrication == 100 else 'FAIL (fabrication detected)'}")
    print(f"  Readability:  {score.readability}")
    print(f"  Completeness: {score.completeness}")
    print(f"  Expression:   {score.expression}")
    print(f"  Response:     {score.response}")

    # Missing fields
    if response.missing_fields:
        print(f"\nMissing Fields:")
        for mf in response.missing_fields:
            print(f"  - {mf.label} ({mf.field}): {mf.reason}")

    # Conflicts
    if response.conflicts:
        print(f"\nTime Conflicts:")
        for c in response.conflicts:
            print(f"  - {c.description}")

    # Fabrication
    if response.fabrication_report.fabrication_found:
        print(f"\nWARNING - Possible Fabrication Detected:")
        for detail in response.fabrication_report.details:
            print(f"  - [{detail.type}] {detail.content}: {detail.reason}")

    # Generation direction
    if response.generation_direction:
        print(f"\nDirection: {response.generation_direction}")

    # Message
    if response.message:
        print(f"\n{response.message}")

    # Files
    if response.files:
        print(f"\nOutput Files:")
        for fmt, path in response.files.items():
            if path:
                print(f"  [{fmt}] {path}")

    # JSON output if requested
    if json_output:
        output = {
            "score": response.score.model_dump(),
            "missing_fields": [mf.model_dump() for mf in response.missing_fields],
            "conflicts": [c.model_dump() for c in response.conflicts],
            "fabrication_report": response.fabrication_report.model_dump(),
            "generation_direction": response.generation_direction,
            "message": response.message,
            "files": response.files,
        }
        print("\n--- JSON ---")
        print(json.dumps(output, ensure_ascii=False, indent=2))


def _print_audit_result(response, json_output=False):
    """Pretty print an audit/optimization result."""
    print("=" * 60)
    print("  Resume Audit Report")
    print("=" * 60)

    # Score
    score = response.final_score
    print(f"\nTotal Score: {score}")

    # Audit report
    audit = response.audit_report
    if isinstance(audit, dict):
        for key in ("overall_score", "dimension_scores", "summary"):
            if key in audit:
                print(f"\n{key}:")
                print(json.dumps(audit[key], ensure_ascii=False, indent=2))

    # Changes
    if response.changes:
        print(f"\nOptimization Changes ({len(response.changes)}):")
        for i, change in enumerate(response.changes, 1):
            print(f"\n  {i}. [{change.get('reason', 'N/A')}]")
            print(f"     Before: {change.get('before', '')[:100]}")
            print(f"     After:  {change.get('after', '')[:100]}")

    # User report
    user_report = response.user_report
    if user_report:
        print(f"\n--- User Report ---")
        print(json.dumps(user_report, ensure_ascii=False, indent=2))

    # Files
    if response.files:
        print(f"\nOutput Files:")
        for fmt, path in response.files.items():
            if path:
                print(f"  [{fmt}] {path}")

    # JSON output if requested
    if json_output:
        output = {
            "final_score": response.final_score,
            "audit_report": response.audit_report,
            "changes": response.changes,
            "user_report": response.user_report,
            "files": response.files,
        }
        print("\n--- JSON ---")
        print(json.dumps(output, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Resume Copilot CLI — generate and optimize resumes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # --- generate (scenario 2 & 4) ---
    gen_parser = subparsers.add_parser("generate", help="Generate resume from profile")
    gen_parser.add_argument(
        "--scenario", choices=["profile_only", "jd_only"], required=True,
        help="Profile only (scenario 2) or JD + profile (scenario 4)",
    )
    gen_parser.add_argument("--profile", "-p", help="Personal profile text")
    gen_parser.add_argument("--jd", "-j", help="Job description text")
    gen_parser.add_argument("--jd-file", help="Job description file path")
    gen_parser.add_argument("--target", "-t", help="Target role description")
    gen_parser.add_argument("--job-family", help=f"Target industry: {', '.join(f.value for f in JobFamily)}")
    gen_parser.add_argument("--user-stage", choices=["student", "working"], help="User stage")
    gen_parser.add_argument("--template", default="new_standard", help="Output template")
    gen_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # --- audit (scenario 1 & 3) ---
    audit_parser = subparsers.add_parser("audit", help="Audit and optimize existing resume")
    audit_parser.add_argument(
        "--scenario", choices=["optimize_with_jd", "profile_with_target"], required=True,
        help="With JD (scenario 1) or target role (scenario 3)",
    )
    audit_parser.add_argument("--file", "-f", help="Resume file path (PDF/TXT)")
    audit_parser.add_argument("--content", "-c", help="Resume text content")
    audit_parser.add_argument("--jd", "-j", help="Job description text")
    audit_parser.add_argument("--jd-file", help="Job description file path")
    audit_parser.add_argument("--target", "-t", help="Target role description")
    audit_parser.add_argument("--style", choices=["aggressive", "conservative"], default="aggressive")
    audit_parser.add_argument("--template", default="new_standard", help="Output template")
    audit_parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "audit":
        cmd_audit(args)


if __name__ == "__main__":
    main()
