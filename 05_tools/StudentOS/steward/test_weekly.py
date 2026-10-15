import copy
import datetime as dt
import hashlib
import unittest
from zoneinfo import ZoneInfo

from weekly import project, task_position, task_group, bundle


class WeeklyTests(unittest.TestCase):
    def setUp(self):
        self.current = dt.datetime(2026, 10, 9, 12, tzinfo=ZoneInfo("Europe/London"))
        self.start = dt.date(2026, 9, 28)
        self.end = dt.date(2026, 12, 11)

    def test_week_markers_and_lesson_dates_keep_original_placement(self):
        cases = [
            ({"title": "ST445｜Week 1 Lab", "raw": "Week 1 Lab ⏳ 2026-10-09"}, "1", "原任务 Week 1"),
            ({"title": "ST447｜2026-10-02 大课：笔记", "raw": "Note ⏳ 2026-10-09"}, "1", "课堂／任务日期"),
            ({"title": "Read", "raw": "Read ⏳ 2026-10-09"}, "2", "原计划日期"),
            ({"title": "Read", "raw": "Read"}, "unassigned", "原任务未标周次或日期"),
            ({"title": "Final", "raw": "Final 📅 2027-08-06"}, "outside", "截止日期"),
            ({"title": "2026-10-09｜ST445｜Week 2 prep", "raw": "Week 2 prep", "kind": "obligation"}, "2", "截止／必到日期"),
        ]
        for task, key, basis in cases:
            result = task_position(task, self.start, self.end)
            self.assertEqual(result[:2], (key, basis))
        self.assertIsNone(task_position({"title": "Week 2 exercise", "raw": "Week 2 exercise"}, self.start, self.end)[2], "A week label is not an exact task date")
        self.assertEqual(task_position({"title": "ST445｜教师推荐：2_st445_week2_lab2.ipynb"}, self.start, self.end)[0], "2")

    def test_grouping_does_not_turn_optional_submission_into_optional_learning(self):
        for title, group in [("ST445｜教师推荐：Lab 2", "recommended"), ("ST447｜建议练习：Exercise 1", "recommended"),
                             ("ST445｜Week 1 Lab 1：可选 extension 另做", "seminar"),
                             ("ST445｜Week 1 大课：复盘", "lecture"), ("ST445｜Week 2 课前准备", "other")]:
            self.assertEqual(task_group({"title": title}), group)
        self.assertEqual(task_group({"title": "ST447｜可选提交", "kind": "obligation"}), "other")

    def test_closure_requires_complete_current_scope_but_not_recommendations(self):
        tasks = [{"id": "a", "ref": "p#a", "group": "lecture", "state": "done"},
                 {"id": "b", "ref": "p#b", "group": "seminar", "state": "done"},
                 {"id": "c", "ref": "p#c", "group": "recommended", "state": "unknown"}]
        scope = {"reviewed": True, "sources": [{"path": "teacher.md"}], "task_refs": ["p#a", "p#b"]}
        course = {"id": "ST445", "week_events": {"1": ["lecture", "seminar"]}, "week_scopes": {"1": scope}}
        result = bundle(course, "1", tasks)
        self.assertTrue(result["ready"])
        self.assertEqual(result["core"]["done"], 2)
        self.assertEqual(result["recommended"]["done"], 0)
        for field in ["needs_review", "historical"]:
            scope[field] = True
            self.assertFalse(bundle(course, "1", tasks)["ready"])
            scope[field] = False
        scope["reviewed"] = False
        self.assertFalse(bundle(course, "1", tasks)["ready"])
        scope["reviewed"] = True
        tasks[1]["state"] = "cancelled"
        result = bundle(course, "1", tasks)
        self.assertFalse(result["ready"])
        self.assertIn("Seminar 与课后练习", result["missing"])
        self.assertEqual(result["core"]["total"], 1)

    def test_checkpoint_report_never_completes_parent_and_stale_details_are_hidden(self):
        task = {"id": "a", "ref": "p#a", "path": "p", "title": "ST445｜Week 1 Lab 1", "status": "completion_unknown"}
        detail = {"task_hash": hashlib.sha256(task["title"].encode()).hexdigest(), "checkpoints": [{"label": "A", "condition": "Save"}]}
        course = {"id": "ST445", "task_ids": ["a"], "personal": {}, "task_details": {"p#a": detail}}
        report = [{"task_id": "a", "payload": {"summary": "Checkpoint A saved", "next_step": "Start B"}}]
        def view():return project([course], [task], [], report, "2026-09-28", "2026-12-11", self.current)
        data = view()
        self.assertEqual(data["tasks"][0]["record_next"], "Start B")
        self.assertEqual(data["counts"]["done"], 0)
        self.assertEqual(len(data["tasks"][0]["checkpoints"]), 1)
        task["title"] += " updated requirements"
        self.assertTrue(view()["tasks"][0]["detail_needs_review"])
        self.assertEqual(view()["tasks"][0]["checkpoints"], [])
        task["status"] = "recorded_complete"
        self.assertEqual(view()["counts"]["done"], 1)

    def test_completion_reports_plans_and_unknowns_are_not_conflated(self):
        def task(i, title, status):
            return {"id": i, "ref": "plan.md#^" + i, "title": title, "raw": title, "path": "plan.md", "status": status}
        tasks = [task("done", "ST445｜Week 1 Lab", "recorded_complete"),
                 task("partial", "ST445｜Week 2 prep", "completion_unknown"),
                 task("unknown", "ST445｜2026-10-06 review", "completion_unknown"),
                 task("future", "ST445｜2026-10-13 review", "completion_unknown"),
                 task("cancelled", "ST445｜Week 2 old plan", "cancelled")]
        before = copy.deepcopy(tasks)
        courses = [{"id": "ST445", "task_ids": [t["id"] for t in tasks], "personal": {"task_ref": "plan.md#^partial", "summary": "Some work exists"}}]
        due = {"raw": "- [ ] 2026-10-09 13:30 前｜ST445｜Prepare", "date": "2026-10-09", "path": "ddl.md", "line": 1,
               "ref": "ddl.md#^prep", "status": "completion_unknown"}
        plan = project(courses, tasks, [due, due], [], "2026-09-28", "2026-12-11", self.current)
        self.assertEqual(tasks, before)
        self.assertEqual(len(plan["weeks"]), 11)
        self.assertEqual(plan["counts"], {"total": 5, "done": 1, "recorded": 1, "unknown": 2, "planned": 1})
        self.assertEqual(len([t for t in plan["tasks"] if t.get("kind") == "obligation"]), 1)
        self.assertEqual(next(t for t in plan["tasks"] if t["id"] == "partial")["state"], "recorded")
        self.assertEqual(courses[0]["task_progress"], plan["counts"])


if __name__ == "__main__":
    unittest.main()
