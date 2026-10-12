"""Deterministic weekly task projection; canonical checkboxes remain the owners."""

import datetime as dt
import re


def counts(tasks):
    active = [t for t in tasks if t["state"] != "cancelled"]
    return {"total": len(active), **{key: sum(t["state"] == key for t in active)
            for key in ("done", "recorded", "unknown", "planned")}}


def task_position(task, start, end):
    raw = task.get("raw", task.get("title", ""))
    title = task.get("title", raw)
    explicit = re.search(r"\bWeek\s+(\d+)\b", title, re.I)
    date = re.search(r"\d{4}-\d{2}-\d{2}", title)
    if explicit and task.get("kind") != "obligation":
        number = int(explicit[1])
        return str(number), "原任务 Week " + str(number), None
    if date:
        value, basis = date[0], "截止／必到日期" if task.get("kind") == "obligation" else "课堂／任务日期"
    else:
        mark = re.search(r"⏳\s*(\d{4}-\d{2}-\d{2})", raw) or re.search(r"📅\s*(\d{4}-\d{2}-\d{2})", raw) or re.search(r"✅\s*(\d{4}-\d{2}-\d{2})", raw)
        value, basis = (mark[1], {"⏳": "原计划日期", "📅": "截止日期", "✅": "完成记录日期"}[mark[0][0]]) if mark else (None, "原任务未标周次或日期")
    if not value:
        return "unassigned", basis, None
    day = dt.date.fromisoformat(value)
    return (str((day - start).days // 7 + 1) if start and start <= day <= end else "outside"), basis, value


def project(courses, tasks, obligations, progress, start, end, current, observations=None):
    start, end = dt.date.fromisoformat(start) if start else None, dt.date.fromisoformat(end) if end else None
    weeks = []
    if start and end:
        for n in range((end - start).days // 7 + 1):
            first = start + dt.timedelta(weeks=n)
            weeks.append({"key": str(n + 1), "number": n + 1, "start": first.isoformat(), "end": min(end, first + dt.timedelta(days=6)).isoformat()})
    result, seen = [], set()
    for course in courses:
        initial = (observations or {}).get(course["id"], {}).get("personal", course["personal"])
        own = [t for t in tasks if t["id"] in course["task_ids"]]
        # Dated obligations have their own canonical originals. Include each once.
        own += [{**t, "id": t.get("ref") or t["path"] + ":" + str(t["line"]),
                 "title": re.sub(r"^\s*[-*+]\s+\[[^]]\]\s*", "", t["raw"]), "kind": "obligation"}
                for t in obligations if re.search(r"(?:[｜|]|\s)" + re.escape(course["id"]) + r"[｜|]", t["raw"])]
        for task in own:
            if task["id"] in seen:
                continue
            seen.add(task["id"])
            week, basis, date = task_position(task, start, end)
            if week.isdigit() and not any(w["key"] == week for w in weeks):
                week = "outside"
            report = next((r for r in progress if r["task_id"] == task["id"]), None)
            observed = bool(initial.get("task_ref") and initial["task_ref"] == task.get("ref"))
            future = (date and date > current.date().isoformat()) or any(w["key"] == week and w["start"] > current.date().isoformat() for w in weeks)
            state = "done" if task["status"] == "recorded_complete" else "cancelled" if task["status"] == "cancelled" else "recorded" if report or observed else "planned" if future else "unknown"
            result.append({**task, "course_id": course["id"], "week": week, "date": date, "date_basis": basis,
                           "state": state, "record": report["payload"]["summary"] if report else initial["summary"] if observed else ""})
        course["task_progress"] = counts([t for t in result if t["course_id"] == course["id"]])
    return {"weeks": weeks, "tasks": result, "counts": counts(result),
            "basis": "完成按原任务勾选；有记录表示已有学习痕迹。周次来自原任务的 Week、课堂日期或排程日期，未记录完成不等于没学。"}
