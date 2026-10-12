"""Read Student OS originals without changing task or completion state."""

import datetime as dt
import hashlib
import importlib.util
from pathlib import Path
import re
from zoneinfo import ZoneInfo

_spec = importlib.util.spec_from_file_location(
    "student_os_reconciliation", Path(__file__).resolve().parents[1] / "telegram/reconciliation.py"
)
rules = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rules)
DEADLINES = f"{rules.PRIVATE}/截止事项记录.md"
RULES = f"{rules.PRIVATE}/Student OS 规则.md"


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def snapshot(vault):
    vault = Path(vault).resolve()
    sources, issues, documents = {}, [], {}

    def read(relative):
        try:
            text = rules._path(vault, relative).read_text(encoding="utf-8")
            sources[relative] = digest(text)
            return text
        except (OSError, ValueError) as error:
            issues.append({"path": relative, "code": "unreadable", "error": str(error)})
            return None

    overview = read(rules.OVERVIEW)
    for relative in (RULES, rules.LOG):
        content = read(relative)
        if content is not None:
            documents[relative] = content
    paths = set()
    if overview is not None:
        for line in overview.splitlines():
            if re.match(r"^\s*-\s*(?:真实计划|计划|课程入口)：", line):
                for target in rules.LINK.findall(line):
                    if not rules._resolve(vault, target.strip()):
                        issues.append({"path": rules.OVERVIEW, "code": "unresolved_registered_source", "target": target})
        try:
            paths.update(rules.active_sources(vault))
        except (OSError, ValueError) as error:
            issues.append({"code": "active_sources_failed", "error": str(error)})
    paths.add(DEADLINES)
    tasks, obligations = [], []
    today = dt.datetime.now(ZoneInfo("Europe/London")).date()
    for relative in sorted(paths):
        content = read(relative)
        if content is None:
            continue
        fenced = False
        anchors = {}
        for number, raw in enumerate(content.splitlines(), 1):
            if raw.lstrip().startswith(("```", "~~~")):
                fenced = not fenced
            if fenced:
                continue
            match = rules.TASK.match(raw)
            is_deadline = bool(rules.DEADLINE.search(raw))
            if not match or not (is_deadline or rules.TAG.search(raw)):
                continue
            if re.search(r"(?:^|\s)#student-os/archived(?=\s|$)", raw):
                continue
            anchor = rules.BLOCK.search(raw)
            ref = relative + "#^" + anchor[1] if anchor else None
            status = {" ": "completion_unknown", "x": "recorded_complete", "X": "recorded_complete", "-": "cancelled"}.get(match[2], "custom_status")
            item = {"ref": ref, "path": relative, "line": number, "raw": raw,
                    "version": digest(raw), "status": status, "checkbox": match[2]}
            if ref:
                anchors.setdefault(ref, []).append(item)
            if is_deadline:
                due = re.search(r"📅\s*(\d{4}-\d{2}-\d{2})", raw)
                item.update(date=due[1] if due else None, timezone="Europe/London", days_until=None)
                try:
                    if not due:
                        raise ValueError("Missing dated obligation")
                    item["days_until"] = (dt.date.fromisoformat(due[1]) - today).days
                except ValueError:
                    issues.append({"path": relative, "line": number, "code": "invalid_deadline_date"})
                # Date-only records remain date-only; do not invent midnight deadlines.
                exact = re.match(r"\s*(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2})[｜|]", match[3])
                item["local_time"] = exact[2] if exact else None
                if exact and due and exact[1] != due[1]:
                    issues.append({"path": relative, "line": number, "code": "conflicting_deadline_dates"})
                obligations.append(item)
            else:
                tasks.append(item)
        for ref, items in anchors.items():
            if len(items) > 1:
                issues.append({"path": relative, "ref": ref, "code": "duplicate_task_identity"})
                for item in items:
                    item["ref"] = None
    for relative, expected in list(sources.items()):
        try:
            if digest(rules._path(vault, relative).read_text(encoding="utf-8")) != expected:
                issues.append({"path": relative, "code": "changed_during_read"})
        except (OSError, ValueError):
            issues.append({"path": relative, "code": "unreadable_after_read"})
    return {"captured_at": dt.datetime.now(ZoneInfo("Europe/London")).isoformat(),
            "vault": str(vault), "status": "partial" if issues else "read",
            "sources": sources, "tasks": tasks, "obligations": obligations,
            "context_documents": documents, "issues": issues,
            "not_checked": ["live_calendar", "Moodle_or_submission_receipts", "Computer_History", "learning_handoffs", "time_limited_commitments"]}
