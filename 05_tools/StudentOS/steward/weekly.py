"""Deterministic weekly task projection; canonical checkboxes remain the owners."""

import datetime as dt
import hashlib
import re

GROUPS = {
    "lecture": ("Lecture", "正式笔记与独立复盘分别核验；笔记完成不代替独立理解。"),
    "seminar": ("Seminar 与课后练习", "按下面的题目范围、实际产物和协作步骤逐项核验。"),
    "other": ("准备、提交与其他要求", "按原要求验收；正式提交须有对应凭证。"),
    "recommended": ("推荐内容", "单独记录进度，不计入本周核心收工条件。"),
}


def task_group(task):
    text = task.get("title", task.get("raw", ""))
    if task.get("kind") != "obligation" and re.search(r"(?:^|[｜|])\s*(?:教师)?(?:推荐|建议练习|可选|选做|延伸|optional\b|recommended\b)", text, re.I):
        return "recommended"
    if task.get("kind") == "obligation" or re.search(r"课前|准备|bootcamp", text, re.I):
        return "other"
    if re.search(r"大课|\blecture\b", text, re.I):
        return "lecture"
    if re.search(r"seminar|lab\s*\d|小班|课后|练习|journal|notebook", text, re.I):
        return "seminar"
    return "other"


def completion_condition(task):
    text = task.get("title", task.get("raw", ""))
    text = re.sub(r"\s+(?:#student-os/\S+|\^[\w-]+|[⏳✅📅🛫➕]\s*\d{4}-\d{2}-\d{2})", "", text)
    text = re.sub(r"（原课时[^）]*）", "", text).strip()
    if "复盘本讲内容" in text:
        return "能独立解释本讲核心内容、适用条件与关键例题，保留尚未理解的问题。"
    return text.split("：", 1)[-1] if "：" in text else text


def counts(tasks):
    active = [t for t in tasks if t["state"] != "cancelled"]
    return {"total": len(active), **{key: sum(t["state"] == key for t in active)
            for key in ("done", "recorded", "unknown", "planned")}}


def task_position(task, start, end):
    raw = task.get("raw", task.get("title", ""))
    title = task.get("title", raw)
    explicit = re.search(r"(?:\b|_)Week[ _-]*0*(\d+)(?=\b|_)", title, re.I)
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


def bundle(course, week, tasks):
    academic = course.get("group", "degree") in ("degree", "future")
    scope = course.get("week_scopes", {}).get(week, {})
    groups = []
    expected = course.get("week_events", {}).get(week, [])
    for key, (label, condition) in GROUPS.items():
        own = [t for t in tasks if t["group"] == key and t["state"] != "cancelled"]
        if own or key in expected:
            groups.append({"key": key, "label": label, "condition": condition,
                           "task_ids": [t["id"] for t in own], "counts": counts(own),
                           "missing": not own and key in expected})
    core = [t for t in tasks if t["group"] != "recommended" and t["state"] != "cancelled"]
    optional = [t for t in tasks if t["group"] == "recommended" and t["state"] != "cancelled"]
    remaining = [t for t in core if t["state"] != "done"]
    missing = [g["label"] for g in groups if g["missing"]]
    scope_ids = set(scope.get("task_refs", []))
    active_ids = {t.get("ref") for t in core}
    verified = bool(scope.get("reviewed") and scope.get("sources") and scope_ids == active_ids and None not in active_ids
                    and not scope.get("needs_review") and not scope.get("historical")
                    and not any(t.get("detail_needs_review") for t in core))
    complete = bool(core) and not remaining
    ready = complete and verified and not missing and week.isdigit()
    # Finish the first-pass note, then practise; the review can use that evidence.
    def order(task):
        if task["state"] == "planned":
            return 4
        if task["group"] == "lecture":
            return 3 if "复盘" in task.get("title", "") else 0
        return 1 if task["group"] == "seminar" else 2
    following = min(remaining, key=order) if remaining else None
    return {"course_id": course["id"], "week": week, "groups": groups, "academic": academic,
            "core": counts(core), "recommended": counts(optional),
            "remaining_ids": [t["id"] for t in remaining], "next_task_id": following["id"] if following else None,
            "ready": ready, "scope": scope, "scope_verified": verified,
            "missing": missing,
            "status": "本周已登记任务完成" if complete and not academic else "本周核心已收工" if ready else "已列核心完成 · 范围待核对" if complete else
                      f"还需核验 {len(remaining)} 项核心事项" if remaining else "本周要求待补齐"}


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
            layout = course.get("task_layout", {}).get(task.get("ref"), {})
            if layout.get("week"):
                week, basis = str(layout["week"]), layout["basis"]
            if week.isdigit() and not any(w["key"] == week for w in weeks):
                week = "outside"
            report = next((r for r in progress if r["task_id"] == task["id"]), None)
            detail = course.get("task_details", {}).get(task.get("ref"), {})
            changed = bool(detail and (detail.get("needs_review") or detail.get("task_hash") != hashlib.sha256(task["title"].encode()).hexdigest()))
            if changed:
                detail = {}
            observed = bool(initial.get("task_ref") and initial["task_ref"] == task.get("ref"))
            future = (date and date > current.date().isoformat()) or any(w["key"] == week and w["start"] > current.date().isoformat() for w in weeks)
            state = "done" if task["status"] == "recorded_complete" else "cancelled" if task["status"] == "cancelled" else "recorded" if report or observed else "planned" if future else "unknown"
            result.append({**task, "course_id": course["id"], "week": week, "date": date, "date_basis": basis,
                           "group": task_group(task), "done_when": detail.get("done_when") or completion_condition(task),
                           "short_title": detail.get("short_title", ""), "checkpoints": detail.get("checkpoints", []),
                           "detail_needs_review": changed, "detail_sources": detail.get("sources", []),
                           "record_next": report["payload"].get("next_step", "") if report else "",
                           "state": state, "record": report["payload"]["summary"] if report else initial["summary"] if observed else ""})
        own_result = [t for t in result if t["course_id"] == course["id"]]
        course["task_progress"] = counts(own_result)
        course["core_progress"] = counts([t for t in own_result if t["group"] != "recommended"])
        course["recommended_progress"] = counts([t for t in own_result if t["group"] == "recommended"])
    bundles = []
    for course in courses:
        own = [t for t in result if t["course_id"] == course["id"]]
        keys = set(t["week"] for t in own) | set(course.get("week_events", {})) | set(course.get("week_scopes", {}))
        for key in sorted(keys, key=lambda k: int(k) if k.isdigit() else 1000 if k == "unassigned" else 1001):
            bundles.append(bundle(course, key, [t for t in own if t["week"] == key]))
    return {"weeks": weeks, "tasks": result, "counts": counts(result), "bundles": bundles,
            "core_counts": counts([t for t in result if t["group"] != "recommended"]),
            "recommended_counts": counts([t for t in result if t["group"] == "recommended"]),
            "basis": "完成按原任务勾选；有记录表示已有学习痕迹。周次来自原任务的 Week、课堂日期或排程日期，未记录完成不等于没学。"}
