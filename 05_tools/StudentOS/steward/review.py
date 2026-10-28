"""Finite, source-bound review attempts. No model calls or canonical task writes."""

import copy
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re
import uuid
from zoneinfo import ZoneInfo
from state import rules

ZONE = ZoneInfo("Europe/London")
PACKS = Path(__file__).parent / "review_packs"
PRIVATE = "99_学习情况记录/管家/复习"
PHASES = ("diagnostic", "feedback", "repair", "retest", "delayed", "system_wait")


def packed(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,120}", value):
        raise ValueError("Invalid review identifier")
    return value


def validate_pack(pack):
    identifier(pack["lesson_id"])
    if not pack.get("title") or not pack.get("version") or not pack.get("source_refs"):
        raise ValueError("题包缺少标题、版本或来源")
    sources = {s["id"] for s in pack["source_refs"]}
    objectives = {o["id"] for o in pack["objectives"]}
    if len(sources) != len(pack["source_refs"]) or len(objectives) != len(pack["objectives"]):
        raise ValueError("重复的来源或目标 ID")
    ids = set()
    for task in pack["tasks"]:
        identifier(task["id"])
        identifier(task["family_id"])
        if task["id"] in ids or task["phase"] not in ("diagnostic", "retest", "delayed"):
            raise ValueError("重复题目 ID 或无效阶段")
        ids.add(task["id"])
        if not task.get("items") or not set(task["objective_ids"]) <= objectives:
            raise ValueError("题目缺少项目或引用未知目标")
        if task["phase"] == "delayed" and (type(task.get("delay_index")) is not int or task["delay_index"] < 1):
            raise ValueError("延迟题必须指定正整数 delay_index")
        item_ids = set()
        for item in task["items"]:
            identifier(item["id"])
            if item["id"] in item_ids:
                raise ValueError("重复项目 ID")
            item_ids.add(item["id"])
            kind = item["kind"]
            if kind not in ("choice", "number", "text_exact", "explain_self"):
                raise ValueError("不支持的判题类型")
            if kind == "explain_self":
                if not item.get("rubric"):
                    raise ValueError("解释题需要自评标准")
            elif not isinstance(item.get("accepted"), list) or not item["accepted"]:
                raise ValueError("脚本题需要 accepted")
            if kind == "choice":
                options = [o["value"] for o in item["options"]]
                if len(set(options)) != len(options) or not set(item["accepted"]) <= set(options):
                    raise ValueError("选择题答案不在选项中")
            if kind == "number":
                numbers = [*item["accepted"], item.get("tolerance", 0)]
                if any(type(n) not in (float, int) or not math.isfinite(n) for n in numbers) or numbers[-1] < 0:
                    raise ValueError("数值答案或容差无效")
        for route in task.get("error_routes", []):
            if route["item_id"] not in item_ids or route["source_ref"] not in sources:
                raise ValueError("补救路线引用未知项目或来源")
    if not any(t["phase"] == "diagnostic" for t in pack["tasks"]):
        raise ValueError("题包没有诊断题")
    return pack


class Review:
    def __init__(self, store, clock=None, packs=None, write_private=None):
        self.store, self.db, self.vault = store, store.db, store.vault
        self.clock = clock or (lambda: dt.datetime.now(ZONE))
        self.packs = Path(packs) if packs is not None else PACKS
        self.write_private = write_private
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS review_sessions (
                id TEXT PRIMARY KEY, lesson_id TEXT NOT NULL, finished INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL, data TEXT NOT NULL);
            CREATE UNIQUE INDEX IF NOT EXISTS review_one_active
                ON review_sessions(lesson_id) WHERE finished=0;
            CREATE TABLE IF NOT EXISTS review_seen (
                lesson_id TEXT NOT NULL, task_id TEXT NOT NULL, session_id TEXT NOT NULL,
                PRIMARY KEY(lesson_id,task_id));
            CREATE TABLE IF NOT EXISTS review_schedule (lesson_id TEXT PRIMARY KEY, data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS review_checkpoints (
                session_id TEXT NOT NULL, event_id TEXT NOT NULL, data TEXT NOT NULL,
                PRIMARY KEY(session_id,event_id));
        """)

    def timestamp(self):
        return self.clock().astimezone(ZONE).isoformat()

    def _pack(self, lesson_id):
        identifier(lesson_id)
        return validate_pack(json.loads((self.packs / (lesson_id + ".json")).read_text()))

    def source_issues(self, pack):
        issues = []
        for source in pack["source_refs"]:
            relative = Path(source["path"])
            target = (self.vault / relative).resolve()
            try:
                if relative.is_absolute() or not target.is_relative_to(self.vault):
                    raise ValueError("来源必须位于 vault 内")
                actual = hashlib.sha256(target.read_bytes()).hexdigest()
                if actual != source["sha256"]:
                    issues.append({"id": source["id"], "path": source["path"], "status": "changed"})
            except (OSError, ValueError):
                issues.append({"id": source["id"], "path": source["path"], "status": "missing_or_invalid"})
        return issues

    def _check_sources(self, pack):
        if self.source_issues(pack):
            raise ValueError("复习来源已变化或不可读；已保存作答和报告保留，请先核对并更新题包。")

    def _schedule(self, lesson_id):
        row = self.db.execute("SELECT data FROM review_schedule WHERE lesson_id=?", (lesson_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def _seen(self, lesson_id, task_id):
        return self.db.execute("SELECT 1 FROM review_seen WHERE lesson_id=? AND task_id=?", (lesson_id, task_id)).fetchone() is not None

    def catalog(self):
        lessons = []
        for path in sorted(self.packs.glob("*.json")):
            try:
                pack = validate_pack(json.loads(path.read_text()))
                lesson_id = pack["lesson_id"]
                active = self.db.execute("SELECT id FROM review_sessions WHERE lesson_id=? AND finished=0", (lesson_id,)).fetchone()
                schedule = self._schedule(lesson_id) or {}
                due_at = schedule.get("due_at")
                lessons.append({k: pack.get(k) for k in ("lesson_id", "title", "objectives", "coverage_limits", "estimated_minutes")} | {
                    "source_issues": self.source_issues(pack), "active_session_id": active[0] if active else None,
                    "due_at": due_at, "due": bool(due_at and dt.datetime.fromisoformat(due_at) <= self.clock()),
                    "status": "active" if active else schedule.get("status", "not_started"),
                    "latest_report": schedule.get("latest_report"), "schedule": schedule or None})
            except (ValueError, KeyError, OSError, TypeError) as error:
                lessons.append({"lesson_id": path.stem, "title": path.stem, "status": "invalid_pack", "source_issues": [{"status": str(error)}]})
        return {"lessons": lessons}

    def _load(self, session_id):
        identifier(session_id)
        row = self.db.execute("SELECT data FROM review_sessions WHERE id=?", (session_id,)).fetchone()
        if not row:
            raise ValueError("找不到此复习会话")
        return json.loads(row[0])

    def _save(self, session):
        session["updated_at"] = self.timestamp()
        self.db.execute("UPDATE review_sessions SET data=?,finished=? WHERE id=?",
                        (packed(session), int(session["stage"] == "complete"), session["id"]))

    def _task(self, session, task_id):
        return next(t for t in session["pack"]["tasks"] if t["id"] == task_id)

    def _present(self, session, task_id):
        self._check_sources(session["pack"])
        if self._seen(session["lesson_id"], task_id):
            raise ValueError("该题已见过；需要补充新题，不能作为新题重复测量。")
        self.db.execute("INSERT INTO review_seen VALUES (?,?,?)", (session["lesson_id"], task_id, session["id"]))
        session.update(stage="question", current_task_id=task_id, draft={}, paused=False)

    def _start(self, lesson_id):
        active = self.db.execute("SELECT id FROM review_sessions WHERE lesson_id=? AND finished=0", (lesson_id,)).fetchone()
        if active:
            return self._load(active[0])
        pack = self._pack(lesson_id)
        self._check_sources(pack)
        schedule = self._schedule(lesson_id)
        phase, index = ("delayed", schedule["delay_index"]) if schedule else ("diagnostic", 0)
        if schedule and (not schedule.get("due_at") or dt.datetime.fromisoformat(schedule["due_at"]) > self.clock()):
            raise ValueError("尚未到单次复习时间，或新题库存已耗尽。")
        tasks = [t["id"] for t in pack["tasks"] if t["phase"] == phase and (phase != "delayed" or t["delay_index"] == index)
                 and not self._seen(lesson_id, t["id"])]
        if not tasks:
            raise ValueError("新题库存已耗尽，需要补充题包。")
        session = {"id": "review-" + uuid.uuid4().hex, "lesson_id": lesson_id, "pack": pack,
                   "started_at": self.timestamp(), "stage": "question", "phase": phase, "round": index,
                   "queue": tasks, "attempts": {}, "attempt_order": [], "retest_from": {}, "family_states": {}, "prompt_exposures": {}, "timing": {p: 0 for p in PHASES},
                   "draft": {}, "paused": False, "report_path": None, "schedule": None}
        self.db.execute("INSERT INTO review_sessions VALUES (?,?,0,?,?)",
                        (session["id"], lesson_id, session["started_at"], packed(session)))
        self._present(session, tasks[0])
        return session

    def _grade(self, task, answers):
        if not isinstance(answers, dict) or set(answers) != {i["id"] for i in task["items"]}:
            raise ValueError("请回答本题全部项目")
        results = {}
        for item in task["items"]:
            answer, kind = answers[item["id"]], item["kind"]
            if type(answer) not in (str, int, float) or isinstance(answer, str) and (not answer.strip() or len(answer) > 20000):
                raise ValueError("作答必须是非空文字或有限数值")
            if isinstance(answer, (int, float)) and not math.isfinite(answer):
                raise ValueError("数值不能是 NaN 或 Infinity")
            correct = None
            if kind == "explain_self":
                if not isinstance(answer, str):
                    raise ValueError("解释题请先保存原话")
            elif kind == "number":
                try:
                    number = float(answer)
                except (ValueError, TypeError):
                    number = math.nan
                correct = math.isfinite(number) and any(abs(number - n) <= item.get("tolerance", 0) for n in item["accepted"])
            elif kind == "choice":
                if answer not in [o["value"] for o in item["options"]]:
                    raise ValueError("请选择有效选项")
                correct = answer in item["accepted"]
            else:
                correct = str(answer).strip() in [str(a).strip() for a in item["accepted"]]
            results[item["id"]] = {"answer": answer, "correct": correct, "evidence": "pending_self" if kind == "explain_self" else "script"}
        return results

    def _refresh_family(self, session, task_id):
        task, attempt = self._task(session, task_id), session["attempts"][task_id]
        values = [r["correct"] for r in attempt["results"].values()]
        if None in values:
            state = "pending_self"
        elif not all(values):
            state = "needs_study" if task["phase"] == "retest" else "failed"
        else:
            state = "needs_retest" if task_id in session.get("prompt_exposures", {}) else "passed"
        attempt["status"] = state
        session["family_states"][task["family_id"]] = {"status": state, "task_id": task_id}

    def _pending_self(self, session):
        return any(r["correct"] is None for a in session["attempts"].values() for r in a["results"].values())

    def _actions(self, session):
        ready = session["stage"] == "feedback" and not self._pending_self(session) and not self.source_issues(session["pack"])
        failed = [v["task_id"] for v in session["family_states"].values() if v["status"] == "failed"]
        needs_retest = [v["task_id"] for v in session["family_states"].values() if v["status"] == "needs_retest"]
        return {"can_finish": ready and not failed and not needs_retest and all(t in session["attempts"] for t in session["queue"]),
                "can_repair": failed if ready else [], "can_defer": failed + needs_retest if ready else [],
                "can_retest": needs_retest if ready else [session["repair_task_id"]] if session["stage"] == "repair" and not self.source_issues(session["pack"]) else []}

    def _submit(self, session, task_id, answers):
        self._check_sources(session["pack"])
        if task_id in session["attempts"]:
            if session["attempts"][task_id]["answers"] != answers:
                raise ValueError("本题已有已保存作答；重复提交不能改写原记录。")
            return
        if session["stage"] != "question" or session["current_task_id"] != task_id:
            raise ValueError("当前不是这道作答题")
        task = self._task(session, task_id)
        session["attempts"][task_id] = {"answers": answers, "results": self._grade(task, answers), "submitted_at": self.timestamp()}
        session["attempt_order"].append(task_id)
        self._refresh_family(session, task_id)
        session.update(current_task_id=None, stage="feedback", draft={})
        remaining = [t for t in session["queue"] if t not in session["attempts"]]
        if remaining:
            self._present(session, remaining[0])

    def _self_grade(self, session, task_id, ratings):
        self._check_sources(session["pack"])
        if session["stage"] != "feedback" or task_id not in session["attempts"]:
            raise ValueError("请先完成全部题目，再对照标准自评")
        attempt = session["attempts"][task_id]
        expected = {i["id"] for i in self._task(session, task_id)["items"] if i["kind"] == "explain_self"}
        if not isinstance(ratings, dict) or set(ratings) != expected or any(type(v) is not bool for v in ratings.values()):
            raise ValueError("请为全部解释项目选择是否达到标准")
        changed = False
        for item_id, correct in ratings.items():
            result = attempt["results"][item_id]
            if result["correct"] is not None and result["correct"] != correct:
                raise ValueError("已保存自评不能改写；可在后续新题中重新观察")
            changed = changed or result["correct"] is None
            result.update(correct=correct, evidence="self_assessed", self_assessed=True)
        if changed and session["family_states"][self._task(session, task_id)["family_id"]]["task_id"] == task_id:
            self._refresh_family(session, task_id)

    def _repair(self, session, task_id):
        if session.get("repair_task_id") == task_id and session["stage"] == "repair" or task_id in session["retest_from"]:
            return
        if task_id not in self._actions(session)["can_repair"]:
            raise ValueError("当前没有这道题的待修补缺口")
        session.update(stage="repair", repair_task_id=task_id, current_task_id=None, draft={})

    def _retest(self, session, task_id):
        if task_id in session["retest_from"]:
            return
        if task_id not in self._actions(session)["can_retest"]:
            raise ValueError("请先完成自评，再进入材料修补或开始提示影响后的新题重测")
        family = self._task(session, task_id)["family_id"]
        new = next((t for t in session["pack"]["tasks"] if t["phase"] == "retest" and t["family_id"] == family and not self._seen(session["lesson_id"], t["id"])), None)
        if new:
            self._present(session, new["id"])
        else:
            session["family_states"][family].update(status="needs_study", reason="重测新题库存已耗尽，需要补题或补课")
            session.update(stage="feedback", current_task_id=None)
        session["retest_from"][task_id] = new["id"] if new else None

    def _defer(self, session, task_id, reason):
        actions = self._actions(session)
        if task_id not in actions["can_defer"] + actions["can_retest"]:
            existing = next((v for v in session["family_states"].values() if v["task_id"] == task_id and v.get("reason") == reason), None)
            if existing:
                return
            raise ValueError("只有当前待修补缺口可以延期")
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 1000:
            raise ValueError("请说明延期原因")
        family = self._task(session, task_id)["family_id"]
        state = session["family_states"][family]
        state.update(status="needs_verification" if state["status"] == "needs_retest" else "needs_study", reason=reason)
        session.update(stage="feedback", current_task_id=None)

    def _checkpoint(self, session, value):
        event_id = identifier(value["event_id"])
        phase, seconds = value["phase"], value["seconds"]
        if phase not in PHASES or type(seconds) not in (int, float) or not math.isfinite(seconds) or not 0 <= seconds <= 300:
            raise ValueError("无效计时片段（每次 0–300 秒）")
        checkpoint = {k: value.get(k) for k in ("phase", "seconds", "draft", "paused")}
        if checkpoint["draft"] is not None and (not isinstance(checkpoint["draft"], dict) or len(packed(checkpoint["draft"])) > 100000):
            raise ValueError("无效草稿")
        if checkpoint["paused"] is not None and type(checkpoint["paused"]) is not bool:
            raise ValueError("无效暂停状态")
        row = self.db.execute("SELECT data FROM review_checkpoints WHERE session_id=? AND event_id=?", (session["id"], event_id)).fetchone()
        if row:
            if row[0] != packed(checkpoint):
                raise ValueError("计时事件 ID 已用于另一片段")
            return
        if session["stage"] == "complete":
            raise ValueError("已结束的报告不追加计时")
        self.db.execute("INSERT INTO review_checkpoints VALUES (?,?,?)", (session["id"], event_id, packed(checkpoint)))
        session["timing"][phase] += seconds
        if checkpoint["draft"] is not None:
            session["draft"] = checkpoint["draft"]
        if checkpoint["paused"] is not None:
            session["paused"] = checkpoint["paused"]

    def _finish(self, session):
        if session["stage"] == "complete":
            return
        if not self._actions(session)["can_finish"]:
            raise ValueError("请先完成作答、自评，以及缺口修补或明确延期")
        needs_study = any(s["status"] == "needs_study" for s in session["family_states"].values())
        needs_verification = any(s["status"] == "needs_verification" for s in session["family_states"].values())
        next_index = session["round"] + 1
        available = any(t["phase"] == "delayed" and t["delay_index"] == next_index and not self._seen(session["lesson_id"], t["id"]) for t in session["pack"]["tasks"])
        days = 1 if session["round"] == 0 or needs_study or needs_verification else 6
        due_date = self.clock().astimezone(ZONE).date() + dt.timedelta(days=days)
        due_at = dt.datetime.combine(due_date, dt.time(7), tzinfo=ZONE)
        report_path = f"{PRIVATE}/{session['id']}.md"
        schedule = {"delay_index": next_index, "due_at": due_at.isoformat() if available else None,
                    "due_date": due_date.isoformat() if available else None,
                    "status": "needs_study" if needs_study else "needs_verification" if needs_verification else "scheduled", "requires_new_items": not available,
                    "pilot_intervals_days": [1, 6], "latest_report": report_path}
        if not available:
            schedule["status"] = "inventory_exhausted"
        session.update(stage="complete", current_task_id=None, finished_at=self.timestamp(), report_path=report_path, schedule=schedule, draft={})
        self.db.execute("INSERT OR REPLACE INTO review_schedule VALUES (?,?)", (session["lesson_id"], packed(schedule)))

    def _routes(self, session, task_id):
        task, attempt = self._task(session, task_id), session["attempts"][task_id]
        sources = {s["id"]: s for s in session["pack"]["source_refs"]}
        return [copy.deepcopy(route) | {"source": copy.deepcopy(sources[route["source_ref"]])}
                for route in task.get("error_routes", []) if attempt["results"][route["item_id"]]["correct"] is False]

    def _response(self, session):
        current = session.get("current_task_id")
        task = copy.deepcopy(self._task(session, current)) if current and session["stage"] == "question" else None
        if task:
            for key in ("error_routes", "repair_target"):
                task.pop(key, None)
            for item in task["items"]:
                for key in ("accepted", "tolerance", "rubric", "explanation"):
                    item.pop(key, None)
        reveal = all(t in session["attempts"] for t in session["queue"])
        feedback = []
        if reveal:
            for task_id in session["attempt_order"]:
                attempt = session["attempts"][task_id]
                original = self._task(session, task_id)
                state = session["family_states"][original["family_id"]]
                feedback.append({"task": original, "results": attempt["results"], "status": attempt["status"],
                                 "needs_self_grade": any(r["correct"] is None for r in attempt["results"].values()),
                                 "needs_repair": state["task_id"] == task_id and state["status"] == "failed",
                                 "needs_retest": state["task_id"] == task_id and state["status"] == "needs_retest",
                                 "prompt_exposure": copy.deepcopy(session.get("prompt_exposures", {}).get(task_id)),
                                 "family_status": state["status"], "reason": state.get("reason"), "routes": self._routes(session, task_id)})
        repair = None
        if session["stage"] == "repair":
            repair = {"task_id": session["repair_task_id"], "routes": self._routes(session, session["repair_task_id"])}
        phase = self._task(session, current)["phase"] if current else session["phase"]
        return {"session": {k: session.get(k) for k in ("id", "lesson_id", "stage", "round", "started_at", "updated_at", "finished_at", "current_task_id")} | {
                    "phase": phase, "status": "finished" if session["stage"] == "complete" else "active",
                    "task_count": len(session["queue"]), "answered_count": sum(t in session["attempts"] for t in session["queue"])},
                "title": session["pack"]["title"], "task": task, "feedback": feedback, "repair": repair,
                "timing": session["timing"], "draft": session["draft"], "paused": session["paused"],
                "schedule": session["schedule"], "report_path": session["report_path"], "actions": self._actions(session),
                "coverage_limits": session["pack"].get("coverage_limits", []), "objectives": session["pack"]["objectives"],
                "source_issues": self.source_issues(session["pack"])}

    def _write_report(self, session):
        path = rules._path(self.vault, session["report_path"])
        states = session["family_states"]
        labels = {o["id"]: o["label"] for o in session["pack"]["objectives"]}
        def task_label(task_id):
            return "、".join(labels[oid] for oid in self._task(session, task_id)["objective_ids"])
        gaps = [f"- {task_label(s['task_id'])}：{s.get('reason', '重测仍有缺口，需要重新学习')}" for s in states.values() if s["status"] == "needs_study"]
        verification = [f"- {task_label(s['task_id'])}：{s['reason']}" for s in states.values() if s["status"] == "needs_verification"]
        tested = {oid for task_id in session["attempts"] for oid in self._task(session, task_id)["objective_ids"]}
        all_objectives = {o["id"] for o in session["pack"]["objectives"]}
        active = sum(v for k, v in session["timing"].items() if k != "system_wait")
        results = [r for a in session["attempts"].values() for r in a["results"].values()]
        scripts = [r for r in results if r["evidence"] == "script"]
        selves = [r for r in results if r["evidence"] == "self_assessed"]
        exposures = session.get("prompt_exposures", {})
        exposure_lines = []
        if exposures:
            unprompted = [r for task_id, a in session["attempts"].items() if task_id not in exposures
                          for r in a["results"].values() if r["evidence"] == "script"]
            exposure_lines = ["## 提示影响", "",
                              *[f"- {task_label(task_id)}：{note['reason']}（记录于 {note['recorded_at']}）。该次作答保留原判分，不计为无提示独立正确。"
                                for task_id, note in exposures.items()], "",
                              f"未记录提示影响的脚本项目通过 {sum(r['correct'] for r in unprompted)}/{len(unprompted)}；此结果仍只支持所测样例。", ""]
        audit = {"source_refs": session["pack"]["source_refs"], "version": session["pack"]["version"],
                 "attempts": session["attempts"], "family_states": states}
        for field in ("prompt_exposures", "question_revisions"):
            if session.get(field):
                audit[field] = session[field]
        wall = (dt.datetime.fromisoformat(session["finished_at"]) - dt.datetime.fromisoformat(session["started_at"])).total_seconds()
        next_review = (f"从 {session['schedule']['due_date']} 起可复习；伦敦 07:00 起开放，不代表固定提醒或日历安排。"
                       if session["schedule"].get("due_date") else "新题库存已耗尽，需要补充题包后再安排下一次复习。")
        lines = [f"# {session['pack']['title']} · 复习记录", "", f"会话：{session['id']}；轮次：{session['round']}（0 为首次诊断）", "",
                 "## 仍需学习", "", *(gaps or ["本轮已测项目无未处理错误；解释项目仅为本人自评。"]), "",
                 *(["## 仍待新题确认", "", *verification, ""] if verification else []),
                 "## 时间", "", f"记录的可见用时（估计）{active:.1f} 秒；系统等待 {session['timing']['system_wait']:.1f} 秒；会话跨度 {wall:.1f} 秒。暂停和离线间隔不推算为练习，页面可见不等于持续专注。", "",
                 *[f"- {p}：{v:.1f} 秒" for p, v in session["timing"].items()], "", "## 覆盖", "",
                 f"已测目标：{'、'.join(labels[o] for o in sorted(tested))}。未测目标：{'、'.join(labels[o] for o in sorted(all_objectives-tested)) or '无（仅限题包所列范围）'}。", "",
                 f"全部已提交尝试：客观脚本项目通过 {sum(r['correct'] for r in scripts)}/{len(scripts)}；解释项目本人自评达到标准 {sum(r['correct'] for r in selves)}/{len(selves)}。初次与修补后作答分别保留。", "",
                 *exposure_lines,
                 *[f"- {limit}" for limit in session["pack"].get("coverage_limits", [])], "",
                 "脚本通过只支持这些测试样例；本人解释自评不等于独立核验。未修改课程任务、正式笔记或掌握状态。", "",
                 "## 后续单次复习", "", f"{next_review}状态：{session['schedule']['status']}。",
                 "下一伦敦日／成功后再隔 6 个日历日是试点参数，并非普适最优间隔或必须满 24 小时的界限。", "", "<details><summary>原始作答与证据</summary>", "", "```json", packed(audit), "```", "", "</details>", ""]
        content = "\n".join(lines)
        if self.write_private:
            self.write_private(session["report_path"], content, immutable=True)
        else:
            if not rules._ignored(self.vault, session["report_path"]):
                raise ValueError("复习记录必须位于 Git 忽略的私有目录")
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and path.read_text() != content:
                raise ValueError("报告有外部修改，保留原文")
            if not path.exists():
                rules._atomic_write(path, content)

    def handle(self, value):
        if not isinstance(value, dict):
            raise ValueError("复习请求必须为对象")
        op = value.get("op")
        if op == "catalog":
            return self.catalog()
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            session = self._start(value["lesson_id"]) if op == "start" else self._load(value["session_id"])
            if session["stage"] == "complete" and op not in ("get", "finish", "checkpoint"):
                raise ValueError("本次复习已结束；请查看报告或等待后续新题")
            if op == "submit":
                self._submit(session, value["task_id"], value["answers"])
            elif op == "self-grade":
                self._self_grade(session, value["task_id"], value["ratings"])
            elif op == "repair":
                self._repair(session, value["task_id"])
            elif op == "retest":
                self._retest(session, value["task_id"])
            elif op == "defer":
                self._defer(session, value["task_id"], value["reason"])
            elif op == "checkpoint":
                self._checkpoint(session, value)
            elif op == "finish":
                self._finish(session)
            elif op not in ("start", "get"):
                raise ValueError("Unknown review operation")
            if op != "get":
                self._save(session)
        if session["stage"] == "complete":
            self._write_report(session)
        return self._response(session)
