"""Behavioral checks use disposable Git vaults and a mocked classifier only."""

import datetime as dt
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from zoneinfo import ZoneInfo

import reconciliation as subject


NOW = dt.datetime(2026, 9, 29, 20, 0, tzinfo=ZoneInfo("Europe/London"))


class ReconciliationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="student-os-reconcile-test-")
        self.addCleanup(self.temporary.cleanup)
        self.vault = Path(self.temporary.name) / "vault"
        self.vault.mkdir()
        self.state = Path(self.temporary.name) / "state"
        subprocess.run(["git", "init", "-q", str(self.vault)], check=True, capture_output=True)
        self.write(".gitignore", "/99_学习情况记录/\n")
        self.plan = "99_学习情况记录/学习计划/Python.md"
        self.raw = "- [ ] Python｜完成第 1–9 单元练习 #student-os/task ⏳ 2026-09-30 📅 2026-10-01 ^python-core"
        self.write(self.plan, "---\nstatus: active\n---\n# Python\n\n## 任务原件\n" + self.raw + "\n\n## 完成与证据\n保留可运行代码和输出。\n")
        self.write(subject.OVERVIEW, "# Overall\n\n## 学校责任\n\n#### Python\n- 状态：当前推进\n- 真实计划：[[学习计划/Python]]\n")
        self.write(subject.WORKBENCH, "# Workbench\n")
        self.clock = patch.object(subject, "_now", return_value=NOW)
        self.clock.start()
        self.addCleanup(self.clock.stop)

    def write(self, relative, text):
        path = self.vault / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def read(self, relative):
        return (self.vault / relative).read_text(encoding="utf-8")

    def classify(self, status="complete", date=None, phrase=None, mutate=None):
        def fake(vault, text, codex_path, candidates, contexts, now):
            if mutate:
                mutate()
            candidate = next(row for row in candidates if row["path"] == self.plan)
            return {"reply": "", "updates": [{"candidate_id": candidate["id"], "status": status,
                    "evidence": text, "actual_date": date, "date_evidence": phrase}]}
        return fake

    def process(self, text="第一组都做完了", update_id=1, **kwargs):
        with patch.object(subject, "classify_reply", side_effect=self.classify(**kwargs)):
            return subject.process_reply(self.vault, text, "/fake/codex", self.state, update_id)

    def test_complete_undated_preserves_dates_and_block_anchor(self):
        reply, report = self.process()
        self.assertIn(self.raw.replace("[ ]", "[x]"), self.read(self.plan))
        self.assertNotIn("✅", self.read(self.plan))
        self.assertIn("完成日期已留空", reply)
        self.assertEqual(len(report["applied"]), 1)
        self.assertIn("第一组都做完了", self.read(subject.LOG))

    def test_explicit_today_date_is_inserted_before_block_anchor(self):
        self.process(text="今天第一组都做完了", date="2026-09-29", phrase="今天")
        self.assertIn("📅 2026-10-01 ✅ 2026-09-29 ^python-core", self.read(self.plan))

    def test_partial_does_not_close_coarse_task(self):
        _, report = self.process(text="前两节做完了", status="partial")
        self.assertIn(self.raw, self.read(self.plan))
        self.assertEqual(report["items"][0]["result"], "partial")
        self.assertEqual(report["applied"], [])
        self.assertIn("部分完成", self.read(subject.LOG))

    def test_replay_classifies_and_logs_only_once(self):
        with patch.object(subject, "classify_reply", side_effect=self.classify()) as classifier:
            first = subject.process_reply(self.vault, "做好了", "/fake/codex", self.state, 31)
            second = subject.process_reply(self.vault, "做好了", "/fake/codex", self.state, 31)
        self.assertEqual(classifier.call_count, 1)
        self.assertEqual(first[0], second[0])
        self.assertTrue(second[1]["duplicate"])
        self.assertEqual(self.read(subject.LOG).count("<!-- telegram-reconciliation:"), 1)

    def test_update_id_collision_rejects_different_input(self):
        self.process()
        _, report = subject.process_reply(self.vault, "另一个消息", "/fake/codex", self.state, 1)
        self.assertEqual(report["disposition"], "reconciliation_unavailable")

    def test_cas_keeps_concurrent_edit(self):
        def mutate():
            self.write(self.plan, self.read(self.plan).replace("完成第 1–9", "完成第 1–10"))
        _, report = self.process(mutate=mutate)
        self.assertEqual(report["items"][0]["result"], "conflict")
        self.assertIn("[ ] Python｜完成第 1–10", self.read(self.plan))

    def test_duplicate_raw_line_is_not_changed(self):
        self.write(self.plan, self.read(self.plan) + self.raw + "\n")
        _, report = self.process()
        self.assertEqual(report["items"][0]["result"], "conflict")
        self.assertEqual(self.read(self.plan).count(self.raw), 2)

    def test_tracked_source_is_manual_even_if_model_says_complete(self):
        self.plan = "Course/EC400.md"
        self.write(self.plan, "# EC400\n" + self.raw + "\n")
        self.write(subject.OVERVIEW, "# Overall\n#### EC400\n- 状态：当前推进\n- 课程入口：[[Course/EC400]]\n")
        subprocess.run(["git", "add", "--", self.plan], cwd=self.vault, check=True, capture_output=True)
        _, report = self.process()
        self.assertEqual(report["items"][0]["result"], "manual")
        self.assertIn(self.raw, self.read(self.plan))

    def test_inactive_overview_or_plan_removes_candidates(self):
        self.write(self.plan, self.read(self.plan).replace("status: active", "status: paused"))
        self.assertEqual(subject.load_candidates(self.vault)[0], [])
        self.write(self.plan, self.read(self.plan).replace("status: paused", "status: active"))
        self.write(subject.OVERVIEW, self.read(subject.OVERVIEW).replace("当前推进", "已完成"))
        self.assertEqual(subject.load_candidates(self.vault)[0], [])

    def test_deadline_tag_is_excluded(self):
        self.write(subject.WORKBENCH, "# Workbench\n- [ ] 提交 #student-os/task #student-os/deadline\n- [ ] 普通任务 #student-os/task ^normal\n")
        rows, _ = subject.load_candidates(self.vault)
        self.assertEqual(len(rows), 2)
        self.assertFalse(any("提交" in row["raw"] for row in rows))

    def test_schema_unknown_id_fails_without_mutation(self):
        result = {"reply": "", "updates": [{"candidate_id": "invented", "status": "complete", "evidence": "好了",
                  "actual_date": None, "date_evidence": None}]}
        with patch.object(subject, "classify_reply", return_value=result):
            _, report = subject.process_reply(self.vault, "好了", "/fake/codex", self.state, 1)
        self.assertEqual(report["disposition"], "classification_failed")
        self.assertIn(self.raw, self.read(self.plan))

    def test_unstated_date_and_nonquoted_evidence_fail(self):
        _, report = self.process(date="2026-09-29", phrase="今天")
        self.assertEqual(report["disposition"], "classification_failed")
        self.assertIn(self.raw, self.read(self.plan))
        candidate = subject.load_candidates(self.vault)[0][0]
        bad = {"reply": "", "updates": [{"candidate_id": candidate["id"], "status": "complete", "evidence": "编造证据",
               "actual_date": None, "date_evidence": None}]}
        with self.assertRaises(ValueError):
            subject._validate(bad, [candidate], "好了", NOW)

    def test_context_contains_completion_criteria(self):
        rows, contexts = subject.load_candidates(self.vault)
        self.assertEqual(rows[0]["heading"], "Python / 任务原件")
        self.assertIn("保留可运行代码和输出", contexts[self.plan])

    def test_classifier_failure_logs_reply_and_keeps_task(self):
        with patch.object(subject, "classify_reply", side_effect=RuntimeError("unavailable")):
            reply, report = subject.process_reply(self.vault, "好了", "/fake/codex", self.state, 1)
        self.assertEqual(report["disposition"], "classification_failed")
        self.assertIn("自动识别未成功", reply)
        self.assertIn(self.raw, self.read(self.plan))
        self.assertIn("好了", self.read(subject.LOG))

    def test_prepared_receipt_recovers_after_task_write_before_log(self):
        with patch.object(subject, "classify_reply", side_effect=self.classify()), patch.object(subject, "_append_log", side_effect=OSError("crash")):
            _, report = subject.process_reply(self.vault, "好了", "/fake/codex", self.state, 71)
            self.assertEqual(report["disposition"], "reconciliation_unavailable")
        self.assertIn(self.raw.replace("[ ]", "[x]"), self.read(self.plan))
        with patch.object(subject, "classify_reply", side_effect=AssertionError("must not reclassify")):
            _, report = subject.process_reply(self.vault, "好了", "/fake/codex", self.state, 71)
        self.assertEqual(report["items"][0]["result"], "already_applied")
        self.assertTrue(report["logged"])

    def test_private_log_requirement_is_checked_before_classification(self):
        self.write(".gitignore", "")
        with patch.object(subject, "classify_reply") as classifier:
            _, report = subject.process_reply(self.vault, "好了", "/fake/codex", self.state, 1)
        self.assertEqual(report["disposition"], "reconciliation_unavailable")
        classifier.assert_not_called()

    def test_recent_outgoing_context_excludes_stale_notifications(self):
        self.state.mkdir()
        (self.state / "latest-evening.json").write_text(json.dumps({"date": "2026-09-29", "text": "Python 第一组完成了吗？"}), encoding="utf-8")
        (self.state / "latest-morning.json").write_text(json.dumps({"date": "2026-09-01", "text": "旧计划"}), encoding="utf-8")
        def fake(vault, text, codex_path, candidates, contexts, now):
            outgoing = contexts["__recent_conversation__"]["outgoing"]
            self.assertEqual(len(outgoing), 1)
            self.assertIn("第一组", outgoing[0]["text"])
            return {"reply": "", "updates": []}
        with patch.object(subject, "classify_reply", side_effect=fake):
            _, report = subject.process_reply(self.vault, "收到了", "/fake/codex", self.state, 1)
        self.assertEqual(report["applied"], [])
        self.assertIn(self.raw, self.read(self.plan))

    def test_cli_uses_read_only_stdin_no_model_override(self):
        rows, contexts = subject.load_candidates(self.vault)
        def fake_run(command, **kwargs):
            self.assertIn("--ephemeral", command)
            self.assertEqual(command[command.index("--sandbox") + 1], "read-only")
            self.assertNotIn("--model", command)
            self.assertNotIn("-m", command)
            self.assertEqual(command[-1], "-")
            self.assertIn("user_statement", kwargs["input"])
            Path(command[command.index("--output-last-message") + 1]).write_text(json.dumps({"reply": "", "updates": []}))
            return subprocess.CompletedProcess(command, 0)
        with patch.object(subject.subprocess, "run", side_effect=fake_run):
            result = subject.classify_reply(self.vault, "你好", "/fake/codex", rows, contexts, NOW)
        self.assertEqual(result["updates"], [])


if __name__ == "__main__":
    unittest.main()
