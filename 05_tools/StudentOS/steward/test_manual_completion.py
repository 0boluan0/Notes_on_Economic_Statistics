import unittest
from unittest.mock import patch

from completion import checked_line
import test_completion as fixture
from state import DEADLINES


class ManualCompletionTests(unittest.TestCase):
    setUp = fixture.CompletionTests.setUp
    tearDown = fixture.CompletionTests.tearDown
    open = fixture.CompletionTests.open
    restart = fixture.CompletionTests.restart
    write = fixture.CompletionTests.write
    request = fixture.CompletionTests.request
    verdict = fixture.CompletionTests.verdict

    def record(self, key="manual", **changes):
        self.service.client_factory = lambda _: self.fail("Manual controls must never call a model")
        return self.checks.record(key, self.request(intent="complete", report="", **changes))

    def test_complete_without_evidence_then_undo_and_restart(self):
        before = (self.vault / self.plan).read_text()
        result = self.record(date="2026-10-09")
        self.assertEqual(result["phase"], "complete")
        receipt = (self.vault / result["path"]).read_text()
        self.assertIn("user_report", receipt)
        self.assertNotIn("#student-os/task", receipt)
        self.restart()
        self.assertEqual(result, self.checks.record("manual"))
        complete = checked_line(self.raw, "2026-10-09")
        result = self.checks.record("undo", self.request(raw=complete, intent="reopen", report=""))
        self.assertEqual(result["phase"], "reopened")
        self.assertEqual(before, (self.vault / self.plan).read_text())
        self.assertEqual(self.checks.cancel("undo")["phase"], "reopened")
        self.restart()
        self.assertEqual(result, self.checks.record("undo"))

    def test_human_submission_confirmation_is_not_external_verification(self):
        result = self.record(path=DEADLINES, raw=self.deadline)
        self.assertEqual(result["verdict"]["basis"], "user_report")
        self.assertEqual(result["phase"], "complete")
        self.assertEqual((self.server.creates, self.server.starts), (0, 0))

    def test_stale_request_does_not_replace_user_edits(self):
        changed = (self.vault / self.plan).read_text().replace("watch and understand", "watch, practice and understand")
        self.write(self.plan, changed)
        with self.assertRaisesRegex(ValueError, "原任务已变化"):
            self.record()
        self.assertEqual(changed, (self.vault / self.plan).read_text())

    def test_recovery_after_canonical_write(self):
        with patch.object(self.checks, "update", side_effect=RuntimeError("interrupted")):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                self.record()
        self.assertEqual(self.checks.row("manual")["phase"], "applying")
        self.restart()
        self.assertEqual(self.checks.record("manual")["phase"], "complete")
        self.assertEqual((self.vault / self.plan).read_text().count("[x]"), 1)
        with self.assertRaisesRegex(ValueError, "内容已改变"):
            self.checks.record("manual", self.request(intent="complete", report="Changed"))

    def test_manual_decision_supersedes_pending_automatic_check(self):
        self.server.plan = self.verdict(verdict="insufficient", missing="Pending")
        self.checks.run("old", self.request())
        self.checks.update("old", phase="retryable")
        self.record()
        self.assertEqual(self.checks.row("old")["phase"], "cancelled")
        self.checks.record("undo", self.request(raw=checked_line(self.raw, ""), intent="reopen", report=""))
        self.assertEqual(self.checks.run("old")["phase"], "cancelled")
        self.assertIn(self.raw, (self.vault / self.plan).read_text())

    def test_old_rejection_remains_history_without_blocking_the_user(self):
        self.server.plan = self.verdict(verdict="insufficient", missing="Evidence")
        self.checks.run("old", self.request())
        self.assertEqual(self.record()["phase"], "complete")
        self.assertEqual(self.checks.row("old")["phase"], "insufficient")
        self.assertEqual(self.service.view()["tasks"][0]["status"], "recorded_complete")


if __name__ == "__main__":
    unittest.main()
