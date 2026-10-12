"""Read-only course positions, with separate schedule, evidence and learner state."""

import datetime as dt
import hashlib
import json
import re
import weekly

PRIVATE = "99_学习情况记录/管家"
POSITION = PRIVATE + "/课程定位.json"
TIMETABLE = "99_学习情况记录/课表/2026-27/课程安排.json"
EVENTS = "99_学习情况记录/课表/2026-27/逐次课程.json"
TIMETABLE_NOTE = "99_学习情况记录/课表/2026-27/我的课表.md"


def read_json(vault, relative, default, issues):
    try:
        value = json.loads((vault / relative).read_text())
        if not isinstance(value, type(default)):
            raise ValueError("Unexpected record structure")
        return value
    except (OSError, ValueError) as error:
        issues.append({"path": relative, "message": str(error)})
        return default


def source_bytes(vault, reference):
    """Fingerprint the cited section, so unrelated edits do not invalidate a fact."""
    relative, _, section = reference.partition("#")
    path = (vault / relative).resolve()
    if not path.is_relative_to(vault.resolve()) or not path.is_file():
        raise ValueError("missing")
    if not section:
        return path.read_bytes()
    lines = path.read_text().splitlines()
    headings = [(i, len(m[1]), m[2]) for i, line in enumerate(lines)
                if (m := re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line))]
    matches = [(i, depth) for i, depth, title in headings if title == section]
    if len(matches) != 1:
        raise ValueError("section_missing_or_ambiguous")
    start, depth = matches[0]
    end = next((i for i, level, _ in headings if i > start and level <= depth), len(lines))
    return "\n".join(lines[start:end]).strip().encode()


def references(vault, refs):
    result = []
    for ref in refs:
        item = dict(ref)
        try:
            content = source_bytes(vault, item["path"])
            item["status"] = "unchanged" if hashlib.sha256(content).hexdigest() == item.get("sha256") else "changed"
        except (OSError, ValueError) as error:
            item["status"] = str(error)
        result.append(item)
    return result


def fact(vault, value, today):
    result = dict(value or {})
    result["sources"] = references(vault, result.get("sources", []))
    result["needs_review"] = any(r["status"] != "unchanged" for r in result["sources"])
    result["historical"] = bool(result.get("valid_until") and today > result["valid_until"])
    return result


def belongs(task, course):
    if course.get("task_prefix"):
        return task.get("title", "").startswith(course["task_prefix"] + "｜")
    return bool(course.get("plan") and task.get("path") == course["plan"])


def overview(vault, state, progress, current):
    issues = []
    table = read_json(vault, TIMETABLE, {}, issues)
    events = read_json(vault, EVENTS, [], issues)
    dossier = read_json(vault, POSITION, {}, issues)
    day = current.date().isoformat()
    term = next((t for t, start in table.get("term_starts", {}).items()
                 if start <= day <= table.get("term_ends", {}).get(t, "")), None)
    week = ((current.date() - dt.date.fromisoformat(table["term_starts"][term])).days // 7 + 1) if term else None
    term_weeks = ((dt.date.fromisoformat(table["term_ends"][term]) - dt.date.fromisoformat(table["term_starts"][term])).days // 7 + 1) if term else None
    stage = f"{'秋季' if term == 'AT' else '冬季'}学期 · 第 {week} 周" if term else "学期外 · 按已公布课表与实际停点接续"
    specifications = {c["id"]: c for c in dossier.get("courses", [])}
    codes = list(dict.fromkeys(table.get("registered_courses", []) + list(specifications)))
    courses = []
    for code in codes:
        spec = specifications.get(code, {})
        series = [s for s in table.get("series", []) if s["course"] == code]
        is_degree = code in table.get("registered_courses", [])
        future = is_degree and bool(series) and all(table.get("term_starts", {}).get(s["term"], "") > day for s in series)
        item = {**spec, "id": code, "title": spec.get("title") or (series[0]["title"] if series else code),
                "group": "future" if future else "degree" if is_degree else spec.get("group", "other"),
                "task_prefix": spec.get("task_prefix", code if is_degree else None)}
        own_tasks = [t for t in state["tasks"] if belongs(t, item)]
        own_events = sorted((e for e in events if e["course"] == code), key=lambda e: e["start_iso"])
        past = [e for e in own_events if dt.datetime.fromisoformat(e["end_iso"]) <= current]
        upcoming = [e for e in own_events if dt.datetime.fromisoformat(e["end_iso"]) > current]
        latest = past[-1] if past else None
        following = upcoming[0] if upcoming else None
        item["schedule"] = {"last": latest, "next": following, "source": TIMETABLE_NOTE,
                            "verified_at": table.get("verified_at"), "meaning": "已保存的课表安排，不代表出席或实际授课范围"}
        school = fact(vault, spec.get("school"), day)
        if not school.get("summary"):
            school["summary"] = f"冬季开课 · {following['date']} 首次安排" if future and following else "实际授课主题尚无核验记录"
        item["school"] = school
        personal = fact(vault, spec.get("personal"), day)
        personal.setdefault("summary", "尚无已记录学习进度")
        next_step = spec.get("next_step", "开课前按已公布要求准备" if future else "从实际课堂记录确定接续点")
        next_task = next((t for t in own_tasks if spec.get("next_ref") and t.get("ref") == spec["next_ref"]), None)
        open_tasks = [t for t in own_tasks if t["status"] not in ("recorded_complete", "cancelled")]
        if not spec.get("next_ref") and len(open_tasks) == 1:
            next_task = open_tasks[0]
        if spec.get("mode") == "finite":
            units = [t for t in own_tasks if t["status"] != "cancelled"]
            item["unit_progress"] = {"completed": sum(t["status"] == "recorded_complete" for t in units), "total": len(units)}
            consecutive = []
            for task in own_tasks:
                if task["status"] != "recorded_complete":
                    break
                consecutive.append(task)
            next_task = next((t for t in own_tasks if t["status"] not in ("recorded_complete", "cancelled")), None)
            personal = {"summary": "原计划连续完成至 " + consecutive[-1]["title"] if consecutive else "原计划尚无连续完成记录；实际学过范围待核对",
                        "sources": [{"path": spec["plan"], "label": "任务原件", "status": "live"}], "needs_review": False}
            next_step = next_task["title"] if next_task else "原计划没有开放任务，按记录核对接续" if own_tasks else "原计划未读到可核验任务"
        reports = [r for r in progress if belongs({"title": r["payload"]["task_title"], "path": r["payload"]["task_path"]}, item)]
        reports.sort(key=lambda r: r["created_at"], reverse=True)
        report_day = dt.datetime.fromisoformat(reports[0]["created_at"].replace("Z", "+00:00")).astimezone(current.tzinfo).date().isoformat() if reports else ""
        if reports and report_day >= personal.get("as_of", ""):
            record = reports[0]
            personal = {"summary": record["payload"]["summary"], "as_of": report_day,
                        "label": "最新本人报告", "detail": "本次任务：" + record["payload"]["task_title"],
                        "sources": [{"path": record["export_path"], "label": "学习停点原记录", "status": "live"}], "needs_review": False}
            next_step = record["payload"]["next_step"]
            next_task = next((t for t in own_tasks if t["id"] == record["task_id"]), None)
        if next_task and next_task["status"] in ("recorded_complete", "cancelled"):
            next_step = "原下一步已记录完成或取消；按最新停点选接续任务"
            next_task = None
        item.update(personal=personal, next_step=next_step, next_task=next_task,
                    task_ids=[t["id"] for t in own_tasks], school=school)
        courses.append(item)
    plan = weekly.project(courses, state["tasks"], state.get("obligations", []), progress,
                          table.get("term_starts", {}).get(term), table.get("term_ends", {}).get(term), current, specifications)
    return {"stage": stage, "term": term, "week": week, "term_weeks": term_weeks, "as_of": day, "courses": courses, "weekly": plan,
            "reconciled_at": dossier.get("reconciled_at"), "issues": issues, "source_paths": [POSITION, TIMETABLE, EVENTS],
            "note_path": PRIVATE + "/课程总览.md", "dossier_path": POSITION,
            "boundary": "学校列区分课表与已核验材料；我的进度只按学习记录。没有记录不等于没学。"}


def markdown(data):
    lines = ["---", "type: student-os-course-overview", "generated: true", "---", "", "# 我的课程与学习位置", "",
             f"{data['stage']} · {data['as_of']}（Europe/London）", "", data["boundary"], "",
             "此页由管家从原件生成。任务勾选保留在原计划，课堂停点通过管家记录。", ""]
    for key, label in (("degree", "本学期正式课程"), ("foundation", "当前编程基础"), ("future", "后续学期"), ("other", "有余力时接续"), ("queued", "确定待学")):
        group = [c for c in data["courses"] if c["group"] == key]
        if not group:
            continue
        lines.extend(["## " + label, "", "| 课程 | 学校进度／课程范围 | 我学到哪里 | 接下来 |", "|---|---|---|---|"])
        for c in group:
            def cell(value):
                return str(value).replace("|", "／").replace("\n", " ")
            school = ("上次核验，待更新：" if c["school"].get("historical") or c["school"].get("needs_review") else "") + c["school"]["summary"]
            personal = ("来源有更新，待对账：" if c["personal"].get("needs_review") else "") + c["personal"]["summary"]
            lines.append("| " + " | ".join(cell(x) for x in (c["id"] + " · " + c["title"], school, personal, c["next_step"])) + " |")
        lines.append("")
    lines.extend(["## 依据与核验时间", ""])
    for c in data["courses"]:
        lines.extend(["### " + c["id"], ""])
        for key, label in (("school", "学校"), ("personal", "我的进度")):
            value = c[key]
            lines.append(f"- {label}：截至 {value.get('as_of') or '任务原件当前读取'}。" + ("来源有变化，需重新对账。" if value.get("needs_review") else "") + ("已超出当时核验周次。" if value.get("historical") else ""))
            if value.get("detail"):
                lines.append("  " + value["detail"])
            lines.extend(f"  - [[{r['path']}|{r.get('label') or r['path'].split('/')[-1]}]]" for r in value.get("sources", []))
        if c["schedule"]["next"]:
            event = c["schedule"]["next"]
            lines.append(f"- 下一次课表安排：{event['date']} {event['start']}–{event['end']}，{event['type']}，{event['room']}。[[{TIMETABLE_NOTE}|课表原件]]")
        if c.get("next_task"):
            task = c["next_task"]
            lines.append(f"- [[{task.get('ref') or task['path']}|下一步任务原件]]")
        lines.append("")
    return "\n".join(lines)
