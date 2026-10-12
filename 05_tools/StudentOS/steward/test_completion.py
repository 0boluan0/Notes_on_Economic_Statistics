import datetime as dt
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from completion import checked_line
from service import Service, ZONE
from state import rules, RULES, DEADLINES
from steward import Store
from test_service import Planner


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.vault = self.root / "vault"
        self.vault.mkdir()
        subprocess.run(["git", "init", "-q", str(self.vault)], check=True)
        self.plan = "99_学习情况记录/学习计划/demo.md"
        self.raw = "- [ ] MIT 18.01｜Lecture 5: watch and understand #student-os/task ^demo"
        self.claim = "I watched Lecture 5 and understood implicit differentiation."
        self.write(".gitignore", "99_学习情况记录/\n")
        self.write(rules.OVERVIEW, "# Overall\n## Active\n### Demo\n- 状态：active\n- 真实计划：[[" + self.plan + "]]\n")
        self.write(RULES, "# Rules\n")
        self.write(rules.LOG, "# Reports\n")
        self.deadline = "- [ ] 2026-10-09 17:00｜ST447｜提交 Exercise 1 PDF #student-os/deadline 📅 2026-10-09 ^submit"
        self.write(DEADLINES, self.deadline + "\n")
        self.write(self.plan, "# Course\n完成标准：看完并理解。\n" + self.raw + "\n")
        self.server = Planner(self.verdict())
        self.open()

    def open(self):
        self.store = Store(self.root / "runtime", self.vault, self.root / "codex")
        self.service = Service(self.store, client_factory=self.server, clock=lambda: dt.datetime(2026, 10, 9, 12, tzinfo=ZONE))
        self.checks = self.service.completion

    def restart(self):
        self.store.close()
        self.open()

    def write(self, path, content):
        file = self.vault / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(content)

    def request(self, **changes):
        return {"path": self.plan, "raw": self.raw, "report": self.claim, "date": "", "evidence": [], **changes}

    def verdict(self, **changes):
        return {"verdict": "supported", "basis": "user_report", "reason": "明确报告本讲已看完并理解。", "missing": "",
                "citations": [{"source": "report", "quote": self.claim}], "receipt_source": "", "receipt_subject": "", "receipt_status": "", **changes}

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_self_report_receipt_restart_and_unknown_date(self):
        first = self.checks.run("complete", self.request())
        self.assertEqual(first["phase"], "complete")
        completed = (self.vault / self.plan).read_text()
        self.assertIn("[x]", completed)
        self.assertNotIn("✅", completed)
        receipt = (self.vault / first["path"]).read_text()
        self.assertIn("user_report", receipt)
        self.assertNotIn("#student-os/task", receipt)
        self.restart()
        self.assertEqual(first, self.checks.run("complete"))
        self.assertEqual((self.server.creates, self.server.starts), (1, 1))
        with self.assertRaises(ValueError):
            self.checks.run("complete", self.request(report="different"))

    def test_insufficient_evidence_survives_restart_without_a_model(self):
        self.server.plan = self.verdict(verdict="insufficient", missing="请说明实际完成范围。")
        before = (self.vault / self.plan).read_bytes()
        result = self.checks.run("pending", self.request())
        self.restart()
        self.service.client_factory = lambda _: self.fail("A pending record read must not run a model")
        context = self.checks.context(self.plan, self.raw)
        self.assertEqual(context["latest"]["phase"], "insufficient")
        self.assertEqual(self.checks.run("pending"), result)
        self.assertEqual(before, (self.vault / self.plan).read_bytes())

    def test_external_result_cannot_be_closed_from_claim(self):
        result = self.checks.run("submission", self.request(path=DEADLINES, raw=self.deadline))
        self.assertEqual(result["phase"], "insufficient")
        self.assertNotIn("[x]", (self.vault / DEADLINES).read_text())

    def test_receipt_closes_only_matching_submission(self):
        proof = "99_学习情况记录/receipt.txt"
        self.write(proof, "Moodle\nST447 Exercise 1\nSubmission status: Submitted for grading\n")
        self.server.plan = self.verdict(basis="external", citations=[{"source": "e0", "quote": "Submitted for grading"}],
                                      receipt_source="e0", receipt_subject="ST447 Exercise 1", receipt_status="Submitted for grading")
        before = (self.vault / self.plan).read_bytes()
        self.checks.run("receipt", self.request(path=DEADLINES, raw=self.deadline, evidence=[proof], date="2026-10-09"))
        self.assertIn("✅ 2026-10-09 ^submit", (self.vault / DEADLINES).read_text())
        self.assertEqual(before, (self.vault / self.plan).read_bytes())

    def test_notes_cannot_close_from_report_alone(self):
        raw = "- [ ] ST447｜Lecture 2 正式双语笔记 #student-os/task ^notes"
        self.write(self.plan, raw + "\n")
        result = self.checks.run("notes", self.request(raw=raw))
        self.assertEqual(result["phase"], "insufficient")

    def test_other_course_receipt_cannot_close_this_submission(self):
        proof = "99_学习情况记录/wrong-receipt.txt"
        self.write(proof, "ST445 Exercise 1\nSubmitted for grading\n")
        self.server.plan = self.verdict(basis="external", citations=[{"source": "e0", "quote": "Submitted for grading"}],
                                      receipt_source="e0", receipt_subject="ST445 Exercise 1", receipt_status="Submitted for grading")
        result = self.checks.run("wrong-course", self.request(path=DEADLINES, raw=self.deadline, evidence=[proof]))
        self.assertEqual(result["phase"], "insufficient")
        self.assertNotIn("[x]", (self.vault / DEADLINES).read_text())

    def test_fabricated_citation_cannot_write(self):
        self.server.plan = self.verdict(citations=[{"source": "report", "quote": "invented proof"}])
        with self.assertRaisesRegex(ValueError, "不在实际证据"):
            self.checks.run("invented", self.request())
        self.assertNotIn("[x]", (self.vault / self.plan).read_text())

    def test_changed_evidence_and_changed_task_block_write(self):
        proof = "99_学习情况记录/evidence.md"
        self.write(proof, "Answer and output")
        finish = self.server.finish
        def alter(*args):
            result = finish(*args)
            self.write(proof, "Changed evidence")
            return result
        self.server.finish = alter
        with self.assertRaisesRegex(ValueError, "证据已变化"):
            self.checks.run("changed", self.request(evidence=[proof]))
        self.assertNotIn("[x]", (self.vault / self.plan).read_text())
        self.server.finish = finish
        with patch.object(self.checks, "apply", side_effect=SystemExit("interrupt before write")):
            with self.assertRaises(SystemExit):
                self.checks.run("task-change", self.request())
        self.write(self.plan, self.raw.replace("Lecture 5", "Lecture 6") + "\n")
        with self.assertRaises(ValueError):
            self.checks.run("task-change")
        self.assertIn("Lecture 6", (self.vault / self.plan).read_text())

    def test_model_interruption_resumes_exact_turn(self):
        self.server.crash = "after_turn"
        with self.assertRaises(SystemExit):
            self.checks.run("interrupted", self.request(date="2026-10-08"))
        self.restart()
        self.server.crash = None
        result = self.checks.run("interrupted")
        self.assertEqual(result["phase"], "complete")
        self.assertEqual((self.server.creates, self.server.starts, self.server.resumes), (1, 1, 1))

    def test_interrupt_after_write_recovers_receipt_without_reticking(self):
        update = self.checks.update
        def interrupt(key, **values):
            if values.get("phase") == "complete":
                raise SystemExit("after task write")
            return update(key, **values)
        with patch.object(self.checks, "update", side_effect=interrupt):
            with self.assertRaises(SystemExit):
                self.checks.run("after-write", self.request(date="2026-10-08"))
        after = (self.vault / self.plan).read_bytes()
        self.restart()
        self.assertEqual(self.checks.run("after-write")["phase"], "complete")
        self.assertEqual(after, (self.vault / self.plan).read_bytes())
        self.assertEqual(self.server.starts, 1)

    def test_cancel_discards_late_verdict(self):
        finish = self.server.finish
        def cancel(*args):
            result = finish(*args)
            self.checks.cancel("cancelled")
            return result
        self.server.finish = cancel
        self.assertEqual(self.checks.run("cancelled", self.request())["phase"], "cancelled")
        self.assertNotIn("[x]", (self.vault / self.plan).read_text())

    def test_invalid_paths_dates_and_duplicate_tasks_fail_closed(self):
        for evidence in (["../outside.txt"], ["missing.txt"]):
            with self.assertRaises(ValueError):
                self.checks.run("bad", self.request(evidence=evidence))
        with self.assertRaises(ValueError):
            self.checks.run("future", self.request(date="2026-10-10"))
        self.write(self.plan, self.raw + "\n" + self.raw + "\n")
        with self.assertRaises(ValueError):
            self.checks.run("duplicate", self.request())
        self.assertEqual(self.server.starts, 0)

    def test_archive_during_check_prevents_completion(self):
        finish = self.server.finish
        def archive(*args):
            result = finish(*args)
            self.write(rules.OVERVIEW, "# Overall\n")
            return result
        self.server.finish = archive
        with self.assertRaises(ValueError):
            self.checks.run("archive", self.request())
        self.assertNotIn("[x]", (self.vault / self.plan).read_text())


if __name__ == "__main__":
    unittest.main()
