import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from review import Review, ZONE, validate_pack
from steward import Store


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.vault = root / "vault"
        self.vault.mkdir()
        subprocess.run(["git", "init", "-q", str(self.vault)], check=True)
        (self.vault / ".gitignore").write_text("99_学习情况记录/\n")
        (self.vault / "source.md").write_text("Synthetic source, not learner evidence.")
        (self.vault / "tasks.md").write_text("- [ ] Original task #student-os/task\n")
        self.packs = root / "packs"
        self.packs.mkdir()
        self.current = dt.datetime(2026, 10, 17, 10, tzinfo=ZONE)
        self.pack = {"lesson_id": "synthetic", "title": "Synthetic review", "version": "1",
                     "source_refs": [{"id": "source", "path": "source.md", "sha256": hashlib.sha256((self.vault / "source.md").read_bytes()).hexdigest()}],
                     "objectives": [{"id": f"o{n}", "label": f"Objective {n}", "scope": "synthetic"} for n in range(3)],
                     "coverage_limits": ["Does not establish application mastery."], "tasks": []}
        for family in ("a", "b"):
            for phase, delay in (("diagnostic", 0), ("retest", 0), ("delayed", 1), ("delayed", 2)):
                ident = f"{family}-{phase}-{delay}"
                items = [{"id": "choice", "kind": "choice", "prompt": "Pick", "options": [{"value": "yes", "label": "Yes"}, {"value": "no", "label": "No"}], "accepted": ["yes"], "explanation": "Answer key"},
                         {"id": "number", "kind": "number", "prompt": "Compute", "accepted": [2], "tolerance": .01, "explanation": "Two"},
                         {"id": "text", "kind": "text_exact", "prompt": "Name", "accepted": ["stable"], "explanation": "Exact token"},
                         {"id": "why", "kind": "explain_self", "prompt": "Why", "rubric": ["Includes a boundary"], "explanation": "Reference reasoning"}]
                self.pack["tasks"].append({"id": ident, "family_id": family, "phase": phase, "delay_index": delay,
                    "scenario": ident, "prompt": "Synthetic only", "objective_ids": ["o0" if family == "a" else "o1"], "items": items,
                    "error_routes": [{"item_id": i["id"], "source_ref": "source", "page": 1, "note_heading": "Boundary", "repair_prompt": "Read, then reconstruct"} for i in items]})
        self.save_pack()
        self.store = Store(root / "state", self.vault, root / "codex")
        self.engine = Review(self.store, clock=lambda: self.current, packs=self.packs)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def save_pack(self):
        (self.packs / "synthetic.json").write_text(json.dumps(self.pack))

    def call(self, op, **kwargs):
        return self.engine.handle({"op": op, **kwargs})

    def start(self):
        view = self.call("start", lesson_id="synthetic")
        self.sid = view["session"]["id"]
        return view

    def answer(self, view, correct=True):
        return self.call("submit", session_id=self.sid, task_id=view["task"]["id"],
                         answers={"choice": "yes" if correct else "no", "number": "2.001", "text": " stable ", "why": "My original explanation"})

    def finish_questions(self, correct=True):
        view = self.call("get", session_id=self.sid)
        while view["session"]["stage"] == "question":
            view = self.answer(view, correct)
        for item in view["feedback"]:
            if item["needs_self_grade"]:
                view = self.call("self-grade", session_id=self.sid, task_id=item["task"]["id"], ratings={"why": True})
        return view

    def test_initial_attempt_has_no_answers_and_feedback_waits_for_all(self):
        view = self.start()
        self.assertEqual(view["session"]["round"], 0)
        self.assertEqual(view["feedback"], [])
        for item in view["task"]["items"]:
            self.assertFalse(set(item) & {"accepted", "tolerance", "rubric", "explanation"})
        view = self.answer(view)
        self.assertEqual(view["feedback"], [])
        with self.assertRaises(ValueError):
            self.call("self-grade", session_id=self.sid, task_id="a-diagnostic-0", ratings={"why": True})
        view = self.answer(view)
        self.assertEqual(len(view["feedback"]), 2)
        self.assertFalse(view["actions"]["can_finish"])
        self.assertEqual(view["feedback"][0]["results"]["why"]["correct"], None)

    def test_self_assessment_is_not_script_evidence(self):
        self.start()
        view = self.finish_questions()
        self.assertEqual(view["feedback"][0]["results"]["why"]["evidence"], "self_assessed")
        self.assertEqual(view["feedback"][0]["results"]["number"]["evidence"], "script")
        self.assertTrue(view["actions"]["can_finish"])
        with self.assertRaises(ValueError):
            self.call("self-grade", session_id=self.sid, task_id="a-diagnostic-0", ratings={"why": False})

    def test_exact_text_preserves_case_unless_answer_key_lists_variant(self):
        task = {"items": [{"id": "token", "kind": "text_exact", "accepted": ["None"]}]}
        self.assertTrue(self.engine._grade(task, {"token": " None "})["token"]["correct"])
        self.assertFalse(self.engine._grade(task, {"token": "none"})["token"]["correct"])
        task["items"][0]["accepted"].append("none")
        self.assertTrue(self.engine._grade(task, {"token": "none"})["token"]["correct"])

    def expose_current_prompt(self):
        session = self.engine._load(self.sid)
        task_id = session["current_task_id"]
        session["prompt_exposures"][task_id] = {"reason": "后续问题提前提示了结果", "recorded_at": self.current.isoformat()}
        session["question_revisions"] = [{"recorded_at": self.current.isoformat(), "from_version": "1", "to_version": "2",
                                           "task_ids": [task_id], "prior_tasks": [copy.deepcopy(self.engine._task(session, task_id))]}]
        with self.store.db:
            self.engine._save(session)
        return task_id

    def test_prompted_correct_attempt_requires_direct_new_retest_and_keeps_audit(self):
        view = self.start()
        task_id = self.expose_current_prompt()
        self.answer(view)
        self.assertEqual(self.engine._load(self.sid)["family_states"]["a"]["status"], "pending_self")
        view = self.finish_questions()
        self.assertFalse(view["actions"]["can_finish"])
        self.assertEqual(view["actions"]["can_retest"], [task_id])
        self.assertEqual(view["actions"]["can_repair"], [])
        self.assertEqual(view["actions"]["can_defer"], [task_id])
        self.assertTrue(view["feedback"][0]["needs_retest"])
        self.assertTrue(view["feedback"][0]["results"]["choice"]["correct"])
        self.assertEqual(view["feedback"][0]["family_status"], "needs_retest")
        self.assertIn("提前提示", view["feedback"][0]["prompt_exposure"]["reason"])
        with self.assertRaises(ValueError):
            self.call("finish", session_id=self.sid)
        retest = self.call("retest", session_id=self.sid, task_id=task_id)
        self.assertEqual(self.call("retest", session_id=self.sid, task_id=task_id)["task"], retest["task"])
        self.answer(retest)
        view = self.call("self-grade", session_id=self.sid, task_id=retest["task"]["id"], ratings={"why": True})
        self.assertEqual(view["feedback"][-1]["family_status"], "passed")
        self.assertIsNone(view["feedback"][-1]["prompt_exposure"])
        self.assertEqual(view["feedback"][0]["status"], "needs_retest")
        self.assertFalse(view["feedback"][0]["needs_retest"])
        final = self.call("finish", session_id=self.sid)
        report = (self.vault / final["report_path"]).read_text()
        self.assertIn("不计为无提示独立正确", report)
        self.assertIn("客观脚本项目通过 9/9", report)
        self.assertIn("未记录提示影响的脚本项目通过 6/6", report)
        self.assertIn('"prompt_exposures"', report)
        self.assertIn('"question_revisions"', report)
        self.assertNotIn("question_revisions", final)

    def test_prompted_wrong_attempt_still_requires_repair(self):
        view = self.start()
        task_id = self.expose_current_prompt()
        self.answer(view, False)
        view = self.finish_questions()
        self.assertEqual(view["feedback"][0]["family_status"], "failed")
        self.assertFalse(view["feedback"][0]["results"]["choice"]["correct"])
        self.assertEqual(view["actions"]["can_repair"], [task_id])
        self.assertEqual(view["actions"]["can_retest"], [])
        with self.assertRaises(ValueError):
            self.call("retest", session_id=self.sid, task_id=task_id)
        self.call("repair", session_id=self.sid, task_id=task_id)
        retest = self.call("retest", session_id=self.sid, task_id=task_id)
        self.answer(retest, False)
        view = self.call("self-grade", session_id=self.sid, task_id=retest["task"]["id"], ratings={"why": True})
        self.assertEqual(view["feedback"][-1]["family_status"], "needs_study")

    def test_prompted_correct_attempt_can_be_deferred_without_retry_reset(self):
        self.start()
        self.finish_questions()
        self.call("finish", session_id=self.sid)
        self.current += dt.timedelta(days=1)
        self.start()
        task_id = self.expose_current_prompt()
        self.finish_questions()
        reason = "稍后做未提示的新题"
        self.call("defer", session_id=self.sid, task_id=task_id, reason=reason)
        self.call("defer", session_id=self.sid, task_id=task_id, reason=reason)
        view = self.call("self-grade", session_id=self.sid, task_id=task_id, ratings={"why": True})
        self.assertEqual(view["feedback"][0]["family_status"], "needs_verification")
        self.assertTrue(view["actions"]["can_finish"])
        final = self.call("finish", session_id=self.sid)
        self.assertEqual(final["schedule"]["status"], "needs_verification")
        self.assertEqual(final["schedule"]["due_at"], "2026-10-19T07:00:00+01:00")
        report = (self.vault / final["report_path"]).read_text()
        self.assertIn("## 仍待新题确认", report)
        self.assertIn(reason, report)
        self.assertIn("未记录提示影响的脚本项目通过 3/3", report)

    def test_duplicate_submit_and_start_do_not_add_attempts(self):
        view = self.start()
        first_id = view["task"]["id"]
        after = self.answer(view)
        again = self.answer(view)
        self.assertEqual(again["task"]["id"], after["task"]["id"])
        self.assertEqual(self.call("start", lesson_id="synthetic")["session"]["id"], self.sid)
        with self.assertRaises(ValueError):
            self.call("submit", session_id=self.sid, task_id=first_id, answers={"choice": "no"})
        self.assertEqual(len(self.engine._load(self.sid)["attempts"]), 1)

    def test_wrong_repair_new_retest_wrong_stops(self):
        self.start()
        view = self.finish_questions(False)
        task_id = view["actions"]["can_repair"][0]
        repair = self.call("repair", session_id=self.sid, task_id=task_id)
        self.assertEqual(repair["repair"]["routes"][0]["source"]["path"], "source.md")
        retest = self.call("retest", session_id=self.sid, task_id=task_id)
        self.assertEqual(retest["task"]["phase"], "retest")
        self.answer(retest, False)
        view = self.call("self-grade", session_id=self.sid, task_id=retest["task"]["id"], ratings={"why": True})
        self.assertEqual(view["feedback"][-1]["family_status"], "needs_study")
        self.assertNotIn(retest["task"]["id"], view["actions"]["can_repair"])
        # Retrying an older self-grade must not restore the old failure as current.
        view = self.call("self-grade", session_id=self.sid, task_id=task_id, ratings={"why": True})
        self.assertEqual(view["feedback"][-1]["family_status"], "needs_study")
        for task_id in view["actions"]["can_defer"]:
            view = self.call("defer", session_id=self.sid, task_id=task_id, reason="稍后作为学习任务修补")
        result = self.call("finish", session_id=self.sid)
        self.assertEqual(result["schedule"]["status"], "needs_study")
        report = (self.vault / result["report_path"]).read_text()
        self.assertIn("重测仍有缺口", report)
        self.assertIn("稍后作为学习任务修补", report)
        self.assertEqual((self.vault / "tasks.md").read_text(), "- [ ] Original task #student-os/task\n")

    def test_successful_retest_does_not_promote_initial_wrong_to_correct(self):
        self.start()
        view = self.finish_questions(False)
        for task_id in list(view["actions"]["can_repair"]):
            self.call("repair", session_id=self.sid, task_id=task_id)
            retest = self.call("retest", session_id=self.sid, task_id=task_id)
            self.answer(retest)
            view = self.call("self-grade", session_id=self.sid, task_id=retest["task"]["id"], ratings={"why": True})
        self.assertTrue(view["actions"]["can_finish"])
        self.assertFalse(view["feedback"][0]["results"]["choice"]["correct"])
        self.assertEqual(view["feedback"][0]["family_status"], "passed")

    def test_cross_day_schedule_and_finite_inventory(self):
        self.start()
        self.finish_questions()
        first = self.call("finish", session_id=self.sid)
        self.assertEqual(dt.datetime.fromisoformat(first["schedule"]["due_at"]), (self.current + dt.timedelta(days=1)).replace(hour=7))
        with self.assertRaises(ValueError):
            self.start()
        self.current += dt.timedelta(days=1)
        view = self.start()
        self.assertEqual(view["task"]["delay_index"], 1)
        self.finish_questions()
        second = self.call("finish", session_id=self.sid)
        self.assertEqual(dt.datetime.fromisoformat(second["schedule"]["due_at"]), (self.current + dt.timedelta(days=6)).replace(hour=7))
        self.current += dt.timedelta(days=6)
        view = self.start()
        self.assertEqual(view["task"]["delay_index"], 2)
        self.finish_questions()
        final = self.call("finish", session_id=self.sid)
        self.assertIsNone(final["schedule"]["due_at"])
        self.assertEqual(final["schedule"]["status"], "inventory_exhausted")
        self.assertTrue(final["schedule"]["requires_new_items"])
        with self.assertRaises(ValueError):
            self.start()

    def test_delayed_failure_uses_next_unseen_set_next_day(self):
        self.start()
        self.finish_questions()
        self.call("finish", session_id=self.sid)
        self.current += dt.timedelta(days=1)
        self.start()
        view = self.finish_questions(False)
        for task_id in view["actions"]["can_defer"]:
            self.call("defer", session_id=self.sid, task_id=task_id, reason="Need study")
        finished = self.call("finish", session_id=self.sid)
        self.assertEqual(finished["schedule"]["delay_index"], 2)
        self.assertEqual(dt.datetime.fromisoformat(finished["schedule"]["due_at"]), (self.current + dt.timedelta(days=1)).replace(hour=7))

    def test_checkpoint_is_idempotent_and_resume_keeps_draft(self):
        view = self.start()
        value = {"session_id": self.sid, "event_id": "tick-1", "phase": "diagnostic", "seconds": 10,
                 "draft": {"task_id": view["task"]["id"], "answers": {"why": "unfinished"}, "ratings": {}}, "paused": True}
        self.call("checkpoint", **value)
        self.call("checkpoint", **value)
        self.store.close()
        root = Path(self.tmp.name)
        self.store = Store(root / "state", self.vault, root / "codex")
        self.engine = Review(self.store, clock=lambda: self.current, packs=self.packs)
        self.current += dt.timedelta(days=2)
        resumed = self.call("start", lesson_id="synthetic")
        self.assertEqual(resumed["timing"]["diagnostic"], 10)
        self.assertEqual(resumed["draft"], value["draft"])
        self.assertTrue(resumed["paused"])
        with self.assertRaises(ValueError):
            self.call("checkpoint", **(value | {"seconds": 12}))
        self.call("checkpoint", **(value | {"event_id": "wait", "phase": "system_wait", "seconds": 2}))
        self.assertEqual(self.call("get", session_id=self.sid)["timing"]["system_wait"], 2)

    def test_source_changed_blocks_answers_but_preserves_records(self):
        view = self.start()
        self.answer(view)
        (self.vault / "source.md").write_text("Changed")
        view = self.call("get", session_id=self.sid)
        self.assertEqual(view["source_issues"][0]["status"], "changed")
        with self.assertRaises(ValueError):
            self.answer(view)
        self.assertEqual(len(self.engine._load(self.sid)["attempts"]), 1)
        self.assertEqual(self.call("start", lesson_id="synthetic")["session"]["id"], self.sid)

    def test_pack_answer_changes_cannot_change_existing_attempt_key(self):
        view = self.start()
        self.pack["tasks"][0]["items"][0]["accepted"] = ["no"]
        self.save_pack()
        self.answer(view)
        self.assertTrue(self.engine._load(self.sid)["attempts"][view["task"]["id"]]["results"]["choice"]["correct"])

    def test_finish_crash_report_write_is_recoverable_without_rescheduling(self):
        self.start()
        self.finish_questions()
        with patch.object(self.engine, "_write_report", side_effect=OSError("disk unavailable")):
            with self.assertRaises(OSError):
                self.call("finish", session_id=self.sid)
        saved = self.engine._load(self.sid)
        self.assertEqual(saved["stage"], "complete")
        self.current += dt.timedelta(hours=3)
        recovered = self.call("finish", session_id=self.sid)
        self.assertEqual(saved["schedule"], recovered["schedule"])
        self.assertTrue((self.vault / recovered["report_path"]).is_file())
        self.assertIn("未测目标：Objective 2", (self.vault / recovered["report_path"]).read_text())

    def test_missing_unseen_retest_is_explicit_and_does_not_repeat(self):
        self.start()
        view = self.finish_questions(False)
        for task_id in view["actions"]["can_repair"]:
            family = task_id.split("-")[0]
            with self.store.db:
                self.store.db.execute("INSERT INTO review_seen VALUES (?,?,?)", ("synthetic", f"{family}-retest-0", "older-session"))
            self.call("repair", session_id=self.sid, task_id=task_id)
            view = self.call("retest", session_id=self.sid, task_id=task_id)
        self.assertTrue(view["actions"]["can_finish"])
        self.assertIn("库存已耗尽", view["feedback"][0]["reason"])

    def test_invalid_inputs_do_not_mutate(self):
        view = self.start()
        with self.assertRaises(ValueError):
            self.call("submit", session_id=self.sid, task_id=view["task"]["id"], answers={})
        with self.assertRaises(ValueError):
            self.call("checkpoint", session_id=self.sid, event_id="bad", phase="diagnostic", seconds=float("nan"))
        self.assertEqual(self.engine._load(self.sid)["attempts"], {})
        invalid = copy.deepcopy(self.pack)
        invalid["tasks"][0]["items"][1]["tolerance"] = -1
        with self.assertRaises(ValueError):
            validate_pack(invalid)

    def test_repair_and_retest_retries_keep_the_current_new_question(self):
        self.start()
        view = self.finish_questions(False)
        task_id = view["actions"]["can_repair"][0]
        self.call("repair", session_id=self.sid, task_id=task_id)
        self.assertEqual(self.call("repair", session_id=self.sid, task_id=task_id)["session"]["stage"], "repair")
        retest = self.call("retest", session_id=self.sid, task_id=task_id)
        self.assertEqual(self.call("retest", session_id=self.sid, task_id=task_id)["task"], retest["task"])
        self.assertEqual(self.call("repair", session_id=self.sid, task_id=task_id)["task"], retest["task"])

    def test_due_window_uses_london_calendar_date_across_dst(self):
        self.current = dt.datetime(2026, 10, 24, 23, 30, tzinfo=ZONE)
        self.start()
        self.finish_questions()
        result = self.call("finish", session_id=self.sid)
        self.assertEqual(result["schedule"]["due_at"], "2026-10-25T07:00:00+00:00")
        self.current = dt.datetime(2026, 10, 25, 7, tzinfo=ZONE)
        self.assertTrue(self.call("catalog")["lessons"][0]["due"])
        self.assertEqual(self.start()["session"]["round"], 1)

    def test_finished_report_remains_readable_when_source_changes(self):
        self.start()
        self.finish_questions()
        first = self.call("finish", session_id=self.sid)
        (self.vault / "source.md").write_text("New material")
        view = self.call("get", session_id=self.sid)
        self.assertEqual(view["report_path"], first["report_path"])
        self.assertEqual(view["source_issues"][0]["status"], "changed")

    def test_failed_submission_transaction_rolls_back_presentation(self):
        view = self.start()
        with patch.object(self.engine, "_present", side_effect=RuntimeError("process interruption")):
            with self.assertRaises(RuntimeError):
                self.answer(view)
        state = self.engine._load(self.sid)
        self.assertEqual(state["attempts"], {})
        self.assertEqual(state["current_task_id"], view["task"]["id"])
        self.assertEqual(self.answer(view)["session"]["answered_count"], 1)


if __name__ == "__main__":
    unittest.main()
