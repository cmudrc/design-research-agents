from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo


CALLOUT_START = "<!-- release-callout:start -->"
CALLOUT_END = "<!-- release-callout:end -->"
NEW_YORK = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class Milestone:
    title: str
    html_url: str
    due_on: datetime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh the monthly release callout in README.md."
    )
    parser.add_argument(
        "--readme",
        default="README.md",
        help="Path to the README file that should be updated.",
    )
    parser.add_argument(
        "--repo",
        help="GitHub repository in owner/name form. Defaults to GITHUB_REPOSITORY or origin remote.",
    )
    parser.add_argument(
        "--today",
        help="Override the current date for testing (YYYY-MM-DD).",
    )
    return parser.parse_args()


def run_gh_api(route: str) -> list[dict[str, object]]:
    output = subprocess.check_output(
        ["gh", "api", route],
        text=True,
    )
    data = json.loads(output)
    if not isinstance(data, list):
        raise RuntimeError(f"Expected a list response from gh api for {route}.")
    return data


def resolve_repo(explicit_repo: str | None) -> str:
    if explicit_repo:
        return explicit_repo
    env_repo = os.environ.get("GITHUB_REPOSITORY")
    if env_repo:
        return env_repo
    remote_url = subprocess.check_output(
        ["git", "config", "--get", "remote.origin.url"],
        text=True,
    ).strip()
    for prefix in ("https://github.com/", "git@github.com:"):
        if remote_url.startswith(prefix):
            repo = remote_url[len(prefix) :]
            if repo.endswith(".git"):
                repo = repo[:-4]
            return repo
    raise RuntimeError("Could not infer owner/repo from remote.origin.url.")


def resolve_today(raw_today: str | None) -> date:
    env_today = raw_today or os.environ.get("RELEASE_README_TODAY")
    if env_today:
        return date.fromisoformat(env_today)
    return datetime.now(NEW_YORK).date()


def load_open_milestones(repo: str) -> list[Milestone]:
    route = f"repos/{repo}/milestones?state=open&per_page=100"
    raw_items = run_gh_api(route)
    milestones: list[Milestone] = []
    for item in raw_items:
        due_on = item.get("due_on")
        html_url = item.get("html_url")
        title = item.get("title")
        if not due_on or not html_url or not title:
            continue
        milestones.append(
            Milestone(
                title=str(title),
                html_url=str(html_url),
                due_on=datetime.fromisoformat(str(due_on).replace("Z", "+00:00")),
            )
        )
    if not milestones:
        raise RuntimeError(f"No open milestones with due dates found for {repo}.")
    milestones.sort(key=lambda milestone: milestone.due_on)
    return milestones


def select_current_milestone(milestones: list[Milestone], today: date) -> Milestone:
    for milestone in milestones:
        if milestone.due_on.date() >= today:
            return milestone
    return milestones[-1]


def tracked_work_month(due_date: date) -> str:
    previous_month_last_day = due_date.replace(day=1) - timedelta(days=1)
    return previous_month_last_day.strftime("%B %Y")


def format_callout(milestone: Milestone) -> str:
    due_date = milestone.due_on.date()
    return "\n".join(
        [
            CALLOUT_START,
            "> [!IMPORTANT]",
            f"> Current monthly release: [{milestone.title}]({milestone.html_url})  ",
            f"> Due: {due_date.strftime('%B %-d, %Y')}  ",
            f"> Tracks: {tracked_work_month(due_date)} work",
            CALLOUT_END,
        ]
    )


def update_readme(readme_path: Path, callout: str) -> bool:
    original_text = readme_path.read_text()
    block_pattern = re.compile(
        rf"{re.escape(CALLOUT_START)}.*?{re.escape(CALLOUT_END)}",
        re.DOTALL,
    )
    if block_pattern.search(original_text):
        updated_text = block_pattern.sub(callout, original_text)
    else:
        lines = original_text.splitlines(keepends=True)
        insert_at = 1
        while insert_at < len(lines) and lines[insert_at].lstrip().startswith("[!["):
            insert_at += 1
        if insert_at < len(lines) and lines[insert_at].strip() == "":
            insert_at += 1
        updated_text = "".join(lines[:insert_at]) + callout + "\n\n" + "".join(lines[insert_at:])
    if updated_text == original_text:
        return False
    readme_path.write_text(updated_text)
    return True


def main() -> int:
    args = parse_args()
    repo = resolve_repo(args.repo)
    today = resolve_today(args.today)
    milestones = load_open_milestones(repo)
    current = select_current_milestone(milestones, today)
    callout = format_callout(current)
    changed = update_readme(Path(args.readme), callout)
    if changed:
        print(f"Updated {args.readme} for {repo} -> {current.title}")
    else:
        print(f"No README changes needed for {repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
