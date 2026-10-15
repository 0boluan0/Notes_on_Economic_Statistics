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


def replay_successor(tasks, after_id):
    """Use the original course order, never historical completion as replay progress."""
    after = next((t for t in tasks if t["id"] == after_id), None)
    if not after or "本轮重看" not in after["title"] or after["status"] != "recorded_complete":
        return None
    def number(task):
        match = re.search(r"\bLecture\s+(\d+)\b", task["title"], re.I)
        return int(match[1]) if match else None
    units = [t for t in tasks if "本轮重看" not in t["title"]]
    numbers = [number(t) for t in units]
    if None in numbers or len(set(numbers)) != len(numbers) or number(after) not in numbers:
        return None
    following = next((t for t in units[numbers.index(number(after)) + 1:] if t["status"] != "cancelled"), None)
    if not following:
        return None
    existing = [t for t in tasks if "本轮重看" in t["title"] and number(t) == number(following)]
    if len(existing) > 1:
        return None
    anchor = "replay-" + hashlib.sha256((after_id + "\n" + following["id"]).encode()).hexdigest()[:24]
    return {"after_task_id": after_id, "source": following, "existing": existing[0] if existing else None, "anchor": anchor}


def overview(vault, state, progress, current, resumptions=None):
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
        item["active"] = bool(own_tasks)
        if own_tasks and item["group"] == "queued":
            item["group"] = "other"
        plan_text = ""
        if spec.get("plan"):
            try:
                plan_text = source_bytes(vault, spec["plan"]).decode("utf-8")
                front = plan_text.split("---", 2)[1] if plan_text.startswith("---\n") else ""
                if re.search(r"^kind:\s*finite-course\s*$", front, re.M):
                    item["mode"] = "finite"
            except (OSError, ValueError):
                pass  # The canonical snapshot reports missing active plans.
        own_events = sorted((e for e in events if e["course"] == code), key=lambda e: e["start_iso"])
        past = [e for e in own_events if dt.datetime.fromisoformat(e["end_iso"]) <= current]
        upcoming = [e for e in own_events if dt.datetime.fromisoformat(e["end_iso"]) > current]
        latest = past[-1] if past else None
        following = upcoming[0] if upcoming else None
        item["schedule"] = {"last": latest, "next": following, "source": TIMETABLE_NOTE,
                            "verified_at": table.get("verified_at"), "meaning": "已保存的课表安排，不代表出席或实际授课范围"}
        item["week_events"] = {}
        for event in own_events:
            if event.get("term") != term or event.get("type") not in ("Lecture", "Seminar"):
                continue
            key = str(event["week"])
            item["week_events"].setdefault(key, [])
            group = event["type"].lower()
            if group not in item["week_events"][key]:
                item["week_events"][key].append(group)
        item["week_scopes"] = {}
        for key, value in spec.get("week_scopes", {}).items():
            if not key.isdigit() or not isinstance(value, dict) or not isinstance(value.get("reviewed", False), bool):
                raise ValueError("Invalid weekly scope record")
            item["week_scopes"][key] = fact(vault, value, day)
        for layout in item.get("task_layout", {}).values():
            if not isinstance(layout.get("week"), int) or not 1 <= layout["week"] <= 52 or not layout.get("basis"):
                raise ValueError("Invalid task week mapping")
        item["task_details"] = {}
        for ref, value in spec.get("task_details", {}).items():
            if not isinstance(value, dict) or not isinstance(value.get("checkpoints", []), list):
                raise ValueError("Invalid task checkpoint record")
            if any(not isinstance(point, dict) or not point.get("label") or not point.get("condition") for point in value.get("checkpoints", [])):
                raise ValueError("Invalid task checkpoint")
            item["task_details"][ref] = fact(vault, value, day)
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
        if item.get("mode") == "finite":
            replay = [t for t in own_tasks if "本轮重看" in t["title"] and t["status"] != "cancelled"]
            units = [t for t in own_tasks if t["status"] != "cancelled" and t not in replay]
            item["unit_progress"] = {"completed": sum(t["status"] == "recorded_complete" for t in units), "total": len(units)}
            consecutive = []
            for task in units:
                if task["status"] != "recorded_complete":
                    break
                consecutive.append(task)
            next_task = next((t for t in own_tasks if t["status"] not in ("recorded_complete", "cancelled")), None)
            personal = {"summary": "原计划连续完成至 " + consecutive[-1]["title"] if consecutive else "原计划尚无连续完成记录；实际学过范围待核对",
                        "sources": [{"path": spec["plan"], "label": "任务原件", "status": "live"}], "needs_review": False, "label": "原计划完成记录"}
            next_step = next_task["title"] if next_task else "原计划没有开放任务，按记录核对接续" if own_tasks else "原计划未读到可核验任务"
            if replay:
                done = [t for t in replay if t["status"] == "recorded_complete"]
                item["replay_progress"] = {"completed": len(done), "registered": len(replay)}
                latest = done[-1] if done else replay[0]
                unit = re.search(r"Lecture\s*\d+(?:\s+[^，；]+)?", latest["title"])
                personal.update(summary=("本轮重看已完成：" if done else "本轮重看进行中：") + (unit[0] if unit else latest["title"]), label="本轮重看")
                next_task = next((t for t in replay if t["status"] == "completion_unknown"), None)
                next_step = next_task["title"] if next_task else "本轮已登记的重看任务已完成；下一讲按本轮重看计划接续。"
                following = replay_successor(own_tasks, replay[-1]["id"]) if not next_task else None
                criterion = re.search(r"^完成标准[：:]\s*(.+)$", plan_text, re.M) if spec.get("plan") else None
                if (following and not following["existing"] and criterion and "^" + following["anchor"] not in plan_text
                        and following["source"]["status"] in ("completion_unknown", "recorded_complete")):
                    item["continuation"] = {"after_task_id": following["after_task_id"], "source_task_id": following["source"]["id"],
                        "title": following["source"]["title"], "done_when": criterion[1]}
                    next_step = "接着学：" + following["source"]["title"]
                resumption = (resumptions or {}).get(code, {})
                if resumption.get("after_task_id") == replay[-1]["id"]:
                    # Rejoining an unfinished original ends the replay prefix, without duplicating it.
                    start = next((i for i, t in enumerate(units) if t["title"] == resumption.get("source_title")), None)
                    item.pop("continuation", None)
                    next_task = next((t for t in units[start:] if t["status"] == "completion_unknown"), None) if start is not None else None
                    next_step = next_task["title"] if next_task else "课程清单已没有开放学习项。" if start is not None else "课程顺序有变化，接续点待核对。"
                    item["replay_resumed"] = True
                    recent = [(m[1], t) for t in units[start:] if t["status"] == "recorded_complete"
                              and (m := re.search(r"✅\s*(\d{4}-\d{2}-\d{2})", t.get("raw", "")))
                              and resumption.get("joined_on") and m[1] >= resumption["joined_on"]] if start is not None else []
                    if recent:
                        date, finished = max(recent, key=lambda pair: pair[0])
                        personal.update(summary="最近完成：" + finished["title"], as_of=date, label="原任务完成记录")
            school = {"summary": f"{len(units)} 讲有限课程 · 按个人节奏学习", "sources": personal["sources"], "needs_review": False}
        else:
            completed = [(m[1], t) for t in own_tasks if t["status"] == "recorded_complete"
                         and (m := re.search(r"✅\s*(\d{4}-\d{2}-\d{2})", t.get("raw", "")))]
            if completed:
                date, task = max(completed, key=lambda pair: pair[0])
                if date >= personal.get("as_of", ""):
                    personal = {"summary": "最近完成：" + task["title"], "as_of": date, "label": "原任务完成记录",
                                "sources": [{"path": task.get("ref") or task["path"], "label": "任务原件", "status": "live"}], "needs_review": False}
        reports = [r for r in progress if r["task_id"] in {t["id"] for t in open_tasks}
                   and belongs({"title": r["payload"]["task_title"], "path": r["payload"]["task_path"]}, item)]
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
    scope_paths = list(dict.fromkeys(ref["path"].split("#", 1)[0] for course in courses
                                    for scope in [*course["week_scopes"].values(), *course["task_details"].values()] for ref in scope.get("sources", [])))
    return {"stage": stage, "term": term, "week": week, "term_weeks": term_weeks, "as_of": day, "courses": courses, "weekly": plan,
            "reconciled_at": dossier.get("reconciled_at"), "issues": issues, "source_paths": [POSITION, TIMETABLE, EVENTS, *scope_paths],
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
    lines.extend(["## 每周收尾", "", "核心事项与推荐内容分别计数。完成只读取原任务；没有记录不等于没有做过。", ""])
    tasks = {t["id"]: t for t in data.get("weekly", {}).get("tasks", [])}
    for entry in data.get("weekly", {}).get("bundles", []):
        if not entry["week"].isdigit() or int(entry["week"]) > (data.get("week") or 0):
            continue
        lines.extend([f"### {entry['course_id']} · Week {entry['week']}", "",
                      f"{entry['status']} · 核心 {entry['core']['done']}/{entry['core']['total']} · 推荐 {entry['recommended']['done']}/{entry['recommended']['total']}", ""])
        if not entry["scope_verified"]:
            lines.extend(["本周完整要求仍待核对，不能仅凭已列任务宣称整周完成。", ""])
        for group in entry["groups"]:
            lines.append(f"**{group['label']}** — {group['condition']}")
            if group["missing"]:
                lines.append("- 已有课表安排，实际练习／要求尚未整理。")
            for task_id in group["task_ids"]:
                task = tasks[task_id]
                status = "已完成" if task["state"] == "done" else "有学习记录" if task["state"] == "recorded" else "待核验"
                label = task["title"].replace("|", "／").replace("]", "")
                lines.append(f"- {status}：[[{task.get('ref') or task['path']}|{label}]]")
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
