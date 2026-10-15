import datetime as dt
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from service import Service, PRIVATE, ZONE
from state import RULES, DEADLINES, rules
from steward import Store
from test_steward import FakeServer


class Planner(FakeServer):
    def __init__(self, plan):
        super().__init__()
        self.plan = plan

    def create(self, model, instructions=None):
        return super().create(model)

    def start(self, thread_id, prompt, output_schema=None):
        self.prompt = prompt
        return super().start(thread_id, prompt)

    def finish(self, thread_id, turn_id):
        self.turns = [{"id": turn_id, "status": "completed", "items": [{"type": "agentMessage", "text": json.dumps(self.plan)}]}]
        if self.crash == "after_turn":
            raise SystemExit("receipt lost after model finished")
        return self.read(thread_id)


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.vault = self.root / "vault"
        self.vault.mkdir()
        subprocess.run(["git", "init", "-q", str(self.vault)], check=True, capture_output=True)
        self.write(".gitignore", "99_学习情况记录/\n")
        self.write(rules.OVERVIEW, "# Overall\n## Active\n- 状态：active\n- 真实计划：[[active]]\n")
        self.write(RULES, "# Student OS rules\n")
        self.write(rules.LOG, "# User reports\n")
        self.write(DEADLINES, "- [ ] Submission pending #student-os/deadline 📅 2026-10-09 ^deadline\n")
        self.write("active.md", "- [ ] Demo unit: practise and save a result #student-os/task ^demo\n")
        self.write("99_学习情况记录/2026-10-08——Thu.md", "# Today\n- **09:00–11:00｜学习 [[active#^demo|Demo]]。**\n")
        self.current = dt.datetime(2026, 10, 8, 10, tzinfo=ZONE)
        self.open()
        self.task_id = self.service.source_state()["tasks"][0]["id"]

    def write(self, path, value):
        p = self.vault / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(value)

    def open(self, planner=None):
        self.store = Store(self.root / "runtime", self.vault, self.root / "codex")
        self.service = Service(self.store, clock=lambda: self.current, client_factory=planner,
                               register=lambda vault, native, title, model: {"conversation_id": native})

    def reopen(self, planner=None):
        self.store.close()
        self.open(planner)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def progress(self, **extra):
        return dict(kind="record_progress", task_id=self.task_id, job_key="", summary="I tried exercise 2, still uncertain.",
                    next_step="Reopen exercise 2 and compare the output.", source_quote="I tried exercise 2", evidence=[], **extra)

    def replay_course(self):
        from curriculum import POSITION, TIMETABLE, EVENTS
        self.replay_path = "99_学习情况记录/学习计划/Python.md"
        self.write(rules.OVERVIEW, f"# Overall\n## Python\n- 状态：active\n- 真实计划：[[{self.replay_path}]]\n")
        self.write(POSITION, json.dumps({"courses": [{"id": "MIT 6.100L", "plan": self.replay_path, "group": "other"}]}))
        self.write(TIMETABLE, "{}")
        self.write(EVENTS, "[]")
        self.write(self.replay_path, "---\nkind: finite-course\n---\n\n完成标准：看完这一讲，并且你自己判断已经理解。\n\n"
                   "## 本轮重启\n\n- [x] MIT 6.100L｜本轮重看 Lecture 01 Introduction，理解后关闭 #student-os/task ✅ 2026-10-08 ^replay-01\n\n"
                   "## 课程清单\n\n- [x] MIT 6.100L｜Lecture 01: [[01|Introduction]] #student-os/task ✅ 2026-08-07\n"
                   "- [x] MIT 6.100L｜Lecture 02: [[02|Strings, Input-Output and Branching]] #student-os/task ✅ 2026-08-07\n"
                   "- [x] MIT 6.100L｜Lecture 03: [[03|Iteration]] #student-os/task ✅ 2026-08-10\n"
                   "- [ ] MIT 6.100L｜Lecture 04: [[04|Loops]] #student-os/task\n")
        self.write("01.md", "# Introduction\nOLD INTRODUCTION MATERIAL\n")
        self.write("02.md", "# Strings\nCURRENT STRINGS MATERIAL\n")
        self.write("03.md", "# Iteration\nFUTURE ITERATION MATERIAL\n")
        return self.service.source_state()["tasks"]

    def continuation_action(self, tasks):
        return {"kind": "continue_course", "course_id": "MIT 6.100L", "task_id": tasks[0]["id"], "source_quote": "继续下一讲"}

    def test_direct_course_continuation_never_calls_a_model_and_deduplicates(self):
        tasks = self.replay_course()
        self.reopen(lambda *_: self.fail("Navigation must not call a model"))
        action = self.continuation_action(tasks)
        before = (self.vault / self.replay_path).read_text()
        projected = self.service.view()["curriculum"]["courses"][0]
        self.assertIn("Lecture 02", projected["continuation"]["title"])
        self.assertEqual((self.vault / self.replay_path).read_text(), before, "Projection must not register planned work")
        first = self.service.execute("direct-1", action, action["source_quote"])
        self.reopen(lambda *_: self.fail("Recovery must not call a model"))
        again = self.service.execute("direct-2", action, action["source_quote"])
        self.assertEqual(first["task_id"], again["task_id"])
        self.assertEqual((self.vault / self.replay_path).read_text().count("本轮重看 Lecture 02"), 1)
        view = self.service.view()
        self.assertEqual(view["curriculum"]["courses"][0]["next_task"]["id"], first["task_id"])
        self.assertEqual(view["focus"]["task"]["id"], first["task_id"])
        self.assertEqual(len(self.service.learning_records()), 0, "Choosing a lecture is not learning evidence")
        materials = self.service.material_context(first["task"])
        self.assertIn("02.md", [s["path"] for s in materials])
        self.assertNotIn("01.md", [s["path"] for s in materials])
        self.assertNotIn("03.md", [s["path"] for s in materials])

    def test_replay_registration_recovers_a_crash_after_writing_without_duplicates(self):
        tasks = self.replay_course()
        action = self.continuation_action(tasks)
        original = rules._atomic_write
        def crash_after_write(path, value):
            original(path, value)
            raise SystemExit("receipt lost")
        with patch.object(rules, "_atomic_write", crash_after_write), self.assertRaises(SystemExit):
            self.service.execute("crash-step", action, action["source_quote"])
        self.reopen(lambda *_: self.fail("Recovery must not call a model"))
        result = self.service.execute("crash-step", action, action["source_quote"])
        self.assertEqual(result["task"]["status"], "completion_unknown")
        self.assertEqual((self.vault / self.replay_path).read_text().count("本轮重看 Lecture 02"), 1)

    def test_closed_replay_is_never_recreated_by_a_stale_next_button(self):
        tasks = self.replay_course()
        action = self.continuation_action(tasks)
        result = self.service.execute("create-next", action, action["source_quote"])
        self.service.execute("close-next", {"kind": "close_task", "task_id": result["task_id"], "source_quote": "不再重看这一讲"}, "不再重看这一讲")
        with self.assertRaisesRegex(ValueError, "关闭|取消"):
            self.service.execute("stale-click", action, action["source_quote"])
        self.assertEqual((self.vault / self.replay_path).read_text().count("本轮重看 Lecture 02"), 1)
        self.assertNotIn("continuation", self.service.view()["curriculum"]["courses"][0])

    def test_replay_meets_unfinished_original_without_creating_a_second_checkbox(self):
        tasks = self.replay_course()
        plan = self.vault / self.replay_path
        plan.write_text(plan.read_text().replace("- [x] MIT 6.100L｜Lecture 02", "- [ ] MIT 6.100L｜Lecture 02").replace(" #student-os/task ✅ 2026-08-07\n- [x] MIT 6.100L｜Lecture 03", " #student-os/task\n- [x] MIT 6.100L｜Lecture 03"))
        before = plan.read_bytes()
        action = self.continuation_action(tasks)
        result = self.service.execute("join-original", action, action["source_quote"])
        self.assertNotIn("本轮重看", result["task"]["title"])
        self.assertEqual(plan.read_bytes(), before)
        self.reopen(lambda *_: self.fail("Projection must not call a model"))
        course = self.service.view()["curriculum"]["courses"][0]
        self.assertEqual(course["next_task"]["id"], result["task_id"])
        # Once that original is complete, the normal course sequence continues past historical entries.
        plan.write_text(plan.read_text().replace(result["task"]["raw"], result["task"]["raw"].replace("[ ]", "[x]", 1) + " ✅ 2026-10-08"))
        course = self.service.view()["curriculum"]["courses"][0]
        self.assertIn("Lecture 04", course["next_task"]["title"])
        self.assertIn("Lecture 02", course["personal"]["summary"], "After joining the original, completion must not fall back to the old replay position")

    def test_saved_next_replay_request_recovers_without_another_manager_turn(self):
        tasks = self.replay_course()
        history = (self.vault / self.replay_path).read_text().split("## 课程清单", 1)[1]
        body = ("MIT 6.100L 这一讲的本轮重看已完成。请按本轮学习计划确认下一讲的范围和收工条件，并准备接续课堂。\n\n"
                f"讨论的原任务：MIT 6.100L · 本轮重看下一讲\n原记录：[[{self.replay_path}]]")
        planner = Planner({"reply": "接续 Lecture 02。", "actions": [{"kind": "prepare_classroom", "task_id": tasks[2]["id"],
                           "source_quote": "请按本轮学习计划确认下一讲的范围和收工条件，并准备接续课堂。"}]})
        self.reopen(planner)
        # Replay the persisted decision from the screenshot, including its exact failure.
        self.store.db.execute("INSERT INTO requests(key,body,phase,plan,error,created_at,updated_at) VALUES (?,?,?,?,?,?,?)",
            ("next-replay", body, "needs_attention", json.dumps(planner.plan),
             "这个任务已记录完成或取消，请选择当前学习单元。", self.current.isoformat(), self.current.isoformat()))
        self.store.db.commit()
        result = self.service.chat("next-replay")
        self.assertEqual(result["phase"], "done", result)
        after = (self.vault / self.replay_path).read_text()
        self.assertEqual(after.split("## 课程清单", 1)[1], history)
        self.assertEqual(after.count("- [ ] MIT 6.100L｜本轮重看 Lecture 02"), 1)
        self.assertEqual(planner.starts, 1, "Only the independent teacher may use a model; no manager planning turn")
        current = self.service.view()["focus"]["task"]
        self.assertIn("本轮重看 Lecture 02", current["title"])
        self.assertEqual(current["status"], "completion_unknown")
        self.reopen(planner)
        self.assertEqual(self.service.chat("next-replay")["phase"], "done")
        self.assertEqual(planner.starts, 1)
        self.assertEqual(self.service.chat("same-click-new-request", body)["phase"], "done")
        self.assertEqual(planner.starts, 1, "A repeated built-in request reuses the teacher without asking a planner")

    def test_record_replay_restart_and_no_fabricated_completion(self):
        before = (self.vault / "active.md").read_bytes()
        action = self.progress()
        first = self.service.execute("progress-1", action, action["source_quote"])
        self.reopen()
        second = self.service.execute("progress-1", action, action["source_quote"])
        self.assertEqual(first, second)
        self.assertEqual(len(self.service.learning_records()), 1)
        self.assertEqual(before, (self.vault / "active.md").read_bytes())
        self.assertEqual(first["mastery"], "unassessed")
        self.assertIn("本人报告", (self.vault / first["path"]).read_text())
        state = self.service.view()
        self.assertEqual(state["focus"]["next_step"], action["next_step"])

    def test_saved_learning_position_reappears_in_course_home_after_restart(self):
        from curriculum import POSITION, TIMETABLE, EVENTS
        self.write(POSITION, json.dumps({"courses": [{"id": "Demo", "plan": "active.md", "group": "foundation"}]}))
        self.write(TIMETABLE, "{}")
        self.write(EVENTS, "[]")
        action = self.progress()
        self.service.execute("position", action, action["source_quote"])
        self.reopen()
        data = self.service.view()["curriculum"]
        self.assertEqual(data["courses"][0]["personal"]["summary"], action["summary"])
        self.assertIn(action["summary"], (self.vault / data["note_path"]).read_text())

    def test_course_read_recovers_after_corruption_and_restart_without_a_model(self):
        from curriculum import POSITION, TIMETABLE, EVENTS
        def no_model(*args):
            self.fail("Routine maintenance must not invoke a model")
        self.reopen(no_model)
        original = {"courses": [{"id": "Demo", "plan": "active.md", "personal": {"summary": "Known stop"}}]}
        self.write(POSITION, json.dumps(original))
        self.write(TIMETABLE, "{}")
        self.write(EVENTS, "[]")
        self.assertEqual(self.service.view()["curriculum"]["courses"][0]["personal"]["summary"], "Known stop")
        self.write(POSITION, "partial sync {")
        self.reopen(no_model)
        cached = self.service.view()["curriculum"]
        self.assertIn("cached_at", cached)
        self.assertEqual(cached["courses"][0]["personal"]["summary"], "Known stop")
        self.assertTrue(cached["courses"][0]["personal"]["needs_review"])
        original["courses"][0]["personal"]["summary"] = "New verified stop"
        self.write(POSITION, json.dumps(original))
        recovered = self.service.view()["curriculum"]
        self.assertNotIn("cached_at", recovered)
        self.assertEqual(recovered["courses"][0]["personal"]["summary"], "New verified stop")
        (self.vault / "active.md").unlink()
        partial = self.service.view()
        self.assertEqual(partial["source_status"], "partial")
        self.assertIn("cached_at", partial["curriculum"], "An unreadable task source cannot replace the good course snapshot with empty progress")

    def test_missing_evidence_and_material_path_escape_are_rejected(self):
        action = self.progress()
        for path in ("missing.py", "../outside.md"):
            action["evidence"] = [path]
            with self.assertRaises(ValueError):
                self.service.execute("bad-" + str(len(path)), action, action["source_quote"])
        self.assertEqual(len(self.service.learning_records()), 0)

    def test_original_task_and_deadline_edits_reach_weekly_view_after_restart(self):
        from curriculum import POSITION, TIMETABLE, EVENTS
        def no_model(*args):
            self.fail("Source reconciliation must not invoke a model")
        self.reopen(no_model)
        self.write(POSITION, json.dumps({"courses": [{"id": "ST447", "task_prefix": "ST447", "plan": "active.md"}]}))
        self.write(TIMETABLE, json.dumps({"term_starts": {"AT": "2026-09-28"}, "term_ends": {"AT": "2026-12-11"}}))
        self.write(EVENTS, "[]")
        self.write("active.md", "- [ ] ST447｜Week 2 Exercise 1 practice #student-os/task ^exercise\n")
        before = self.service.view()["curriculum"]["weekly"]
        self.assertEqual(before["counts"]["total"], 1)
        self.write("active.md", "- [x] ST447｜Week 2 Exercise 1 practice #student-os/task ✅ 2026-10-08 ^exercise\n")
        self.write(DEADLINES, "- [ ] 2026-10-09 17:00｜ST447｜Submit Exercise 1 PDF #student-os/deadline 📅 2026-10-09 ^submission\n")
        self.reopen(no_model)
        view = self.service.view()
        weekly = view["curriculum"]["weekly"]
        self.assertEqual(weekly["counts"]["total"], 2)
        self.assertEqual(weekly["counts"]["done"], 1)
        submission = next(t for t in weekly["tasks"] if t.get("kind") == "obligation")
        self.assertEqual(submission["week"], "2")
        self.assertNotEqual(submission["state"], "done", "Completed practice never closes its separate submission")
        self.assertEqual(view["obligations"][0]["local_time"], "17:00")
        self.write(DEADLINES, "- [x] 2026-10-16 17:00｜ST447｜Submit Exercise 1 PDF #student-os/deadline 📅 2026-10-16 ✅ 2026-10-08 ^submission\n")
        self.reopen(no_model)
        weekly = self.service.view()["curriculum"]["weekly"]
        self.assertEqual(weekly["counts"]["done"], 2)
        self.assertEqual(next(t for t in weekly["tasks"] if t.get("kind") == "obligation")["week"], "3")

    def test_record_does_not_overwrite_external_edits_after_export_failure(self):
        action = self.progress()
        path = f"{PRIVATE}/学习交接/conflict.md"
        self.write(path, "User-owned text, preserve me.\n")
        with self.assertRaises(ValueError):
            self.service.execute("conflict", action, action["source_quote"])
        self.assertEqual((self.vault / path).read_text(), "User-owned text, preserve me.\n")
        self.assertEqual(self.store.db.execute("SELECT state FROM operations WHERE key='conflict'").fetchone()[0], "needs_attention")

    def test_quoted_source_cannot_authorize_a_mutation(self):
        action = self.progress()
        with self.assertRaises(ValueError):
            self.service.execute("wrong-source", action, "What should I do today?")
        self.assertEqual(len(self.service.learning_records()), 0)

    def test_durable_chat_replays_saved_decision_without_new_model_turn(self):
        planner = Planner({"reply": "Here is your next step.", "actions": []})
        self.reopen(planner)
        first = self.service.chat("chat-1", "What is next?")
        self.reopen(planner)
        second = self.service.chat("chat-1", "What is next?")
        self.assertEqual(first["reply"], second["reply"])
        self.assertEqual((planner.creates, planner.starts), (1, 1))
        self.assertIn("持久记录", planner.prompt)

    def test_manager_can_reconcile_live_course_progress_with_a_receipt(self):
        from curriculum import POSITION, TIMETABLE, EVENTS
        self.write(POSITION, json.dumps({"courses": [{"id": "Demo", "plan": "active.md", "group": "foundation"}]}))
        self.write(TIMETABLE, "{}")
        self.write(EVENTS, "[]")
        self.service.view()
        self.write("active.md", "- [x] Demo unit: practise and save a result #student-os/task ✅ 2026-10-08 ^demo\n")
        before = (self.vault / "active.md").read_bytes()
        quote = "已完成但看板还没更新，请核对同步"
        planner = Planner({"reply": "正在核对课程进度。", "actions": [{"kind": "refresh_dashboard", "source_quote": quote}]})
        self.reopen(planner)
        result = self.service.chat("refresh-course", quote)
        self.assertEqual(result["phase"], "done", result)
        self.assertEqual(result["receipts"][0]["courses"][0]["done"], 1)
        self.assertIn("已同步", result["reply"])
        self.assertEqual((self.vault / "active.md").read_bytes(), before)
        self.assertEqual(planner.starts, 1)
        self.reopen(planner)
        saved = self.service.view()["chat"][-1]
        self.assertEqual(saved["receipts"][0]["kind"], "refresh_dashboard")
        self.assertEqual(saved["receipts"][0]["result"], result["receipts"][0])

    def test_chat_crash_after_model_result_resumes_same_turn(self):
        planner = Planner({"reply": "Saved decision", "actions": []})
        planner.crash = "after_turn"
        self.reopen(planner)
        with self.assertRaises(SystemExit):
            self.service.chat("crash", "What next?")
        self.reopen(planner)
        planner.crash = None
        result = self.service.chat("crash")
        self.assertEqual(result["phase"], "done")
        self.assertEqual((planner.creates, planner.starts, planner.resumes), (1, 1, 1))

    def test_partial_batch_preserves_receipt_and_does_not_duplicate_progress(self):
        action = self.progress()
        bad = {"kind": "cancel_commitment", "commitment_id": "does-not-exist", "source_quote": action["source_quote"]}
        planner = Planner({"reply": "Proposal", "actions": [action, bad]})
        self.reopen(planner)
        result = self.service.chat("batch", action["source_quote"])
        self.assertEqual(result["phase"], "needs_attention")
        self.assertNotIn("reply", result)
        self.reopen(planner)
        self.service.chat("batch")
        self.assertEqual(len(self.service.learning_records()), 1)
        self.assertEqual(planner.starts, 1)

    def test_cancelled_chat_cannot_resume(self):
        planner = Planner({"reply": "Proposal", "actions": []})
        planner.crash = "after_turn"
        self.reopen(planner)
        with self.assertRaises(SystemExit):
            self.service.chat("stop", "Help me plan")
        self.service.cancel_request("stop")
        planner.crash = None
        result = self.service.chat("stop")
        self.assertEqual(result["phase"], "cancelled")
        self.assertEqual(planner.starts, 1)

    def test_classroom_reuses_original_job_until_a_handoff(self):
        fake = FakeServer()
        self.reopen(fake)
        first = self.service.prepare_classroom(self.task_id)
        self.reopen(fake)
        second = self.service.prepare_classroom(self.task_id)
        self.assertEqual(first["job_key"], second["job_key"])
        self.assertEqual((fake.creates, fake.starts), (1, 1))
        stored = self.store.get(first["job_key"])
        self.assertEqual(stored["request"]["scope"], "learning")
        self.assertEqual(stored["request"]["task"]["version"], self.service.task(self.task_id)["version"])

    def test_preparation_resumes_from_durable_learning_handoff(self):
        from curriculum import POSITION, TIMETABLE, EVENTS
        self.write(POSITION, json.dumps({"courses": [{"id": "Demo", "plan": "active.md", "group": "foundation"}]}))
        self.write(TIMETABLE, "{}")
        self.write(EVENTS, "[]")
        action = self.progress()
        before = (self.vault / "active.md").read_bytes()
        self.service.execute("handoff", action, action["source_quote"])
        self.reopen(FakeServer())
        lesson = self.service.prepare_classroom(self.task_id)
        prompt = self.store.get(lesson["job_key"])["request"]["prompt"]
        self.assertIn(action["summary"], prompt)
        self.assertIn(action["next_step"], prompt)
        self.assertIn('"record_next"', prompt)
        self.assertEqual(before, (self.vault / "active.md").read_bytes())

    def test_due_notices_stop_at_expiry_and_cannot_revive_after_cancel(self):
        action = {"kind": "save_commitment", "summary": "Daily practice", "source_quote": "Practise for 21 days",
                  "starts_at": "2026-10-08T00:00:00+01:00", "ends_at": "2026-10-29T00:00:00+00:00", "daily_time": "08:00"}
        self.service.execute("training", action, action["source_quote"])
        self.assertTrue(self.service.commitment_state()[0]["due"])
        self.current = dt.datetime(2026, 10, 29, 0, tzinfo=ZONE)
        self.assertEqual(self.service.commitment_state()[0]["status"], "expired")
        self.assertFalse(self.service.commitment_state()[0]["due"])
        self.reopen()
        self.service.execute("training", action, action["source_quote"])
        self.assertFalse(self.service.commitment_state()[0]["due"])

    def test_cancelled_classroom_requires_explicit_restart(self):
        fake = FakeServer()
        self.reopen(fake)
        first = self.service.prepare_classroom(self.task_id)
        self.store.cancel(first["job_key"])
        with self.assertRaises(RuntimeError):
            self.service.prepare_classroom(self.task_id)
        self.assertEqual(fake.creates, 1)
        second = self.service.prepare_classroom(self.task_id, restart_key=first["job_key"])
        self.assertNotEqual(first["job_key"], second["job_key"])
        self.assertEqual(fake.creates, 2)
        self.assertEqual(self.store.get(first["job_key"])["state"], "cancelled")

    def test_changed_task_does_not_silently_reuse_stale_classroom(self):
        fake = FakeServer()
        self.reopen(fake)
        self.service.prepare_classroom(self.task_id)
        self.write("active.md", "- [ ] Revised unit scope #student-os/task ^demo\n")
        with self.assertRaisesRegex(ValueError, "原任务已更新"):
            self.service.prepare_classroom(self.task_id)
        self.assertEqual(fake.creates, 1)

    def test_material_selection_uses_matching_week(self):
        root = "03_Computer_Science/06_ST445_Test"
        self.write("active.md", "- [ ] ST445 Week 2 preparation #student-os/task ^demo\n")
        self.write(root + "/Week 01 Preparation.md", "Wrong week\n")
        self.write(root + "/Week 02 Preparation.md", "Correct week\n[wrong](Week%2001%20Lab.ipynb)\n[right](Week%2002%20Lab.ipynb)\n")
        self.write(root + "/Week 01 Lab.ipynb", json.dumps({"cells": [{"cell_type": "markdown", "source": ["Wrong notebook"]}]}))
        self.write(root + "/Week 02 Lab.ipynb", json.dumps({"cells": [{"cell_type": "markdown", "source": ["Correct notebook"]}]}))
        paths = [s["path"] for s in self.service.material_context(self.service.task(self.task_id))]
        self.assertIn(root + "/Week 02 Lab.ipynb", paths)
        self.assertFalse(any("Week 01" in p for p in paths))

    def test_cancelled_commitment_is_not_recreated_by_replay(self):
        action = {"kind": "save_commitment", "summary": "Practice", "source_quote": "Try for 7 days",
                  "starts_at": "2026-10-08T00:00:00+01:00", "ends_at": "2026-10-15T00:00:00+01:00", "daily_time": "08:00"}
        self.service.execute("training", action, action["source_quote"])
        cancel = {"kind": "cancel_commitment", "commitment_id": "training", "source_quote": "Cancel practice"}
        self.service.execute("cancel", cancel, cancel["source_quote"])
        self.service.execute("training", action, action["source_quote"])
        self.assertEqual(self.service.commitment_state()[0]["status"], "cancelled")
        self.assertFalse(self.service.commitment_state()[0]["due"])

    def test_weekly_report_preserves_manual_changes_and_activity_boundary(self):
        report = self.service.view()["weekly_report"]
        file = self.vault / report["path"]
        self.assertIn("不推断独立掌握", file.read_text())
        file.write_text("My manual note\n")
        result = self.service.view()["weekly_report"]
        self.assertEqual(result["status"], "needs_attention")
        self.assertEqual(file.read_text(), "My manual note\n")


if __name__ == "__main__":
    unittest.main()
