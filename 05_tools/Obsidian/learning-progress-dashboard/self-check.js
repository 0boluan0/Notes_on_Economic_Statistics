const assert = require("assert");
const Module = require("module");

const originalLoad = Module._load;
Module._load = function load(request, parent, isMain) {
  if (request === "obsidian") {
    return {
      ItemView: class {},
      Modal: class {},
      Notice: class {},
      Plugin: class {},
      TFile: class {},
      normalizePath: (value) => value,
    };
  }
  return originalLoad(request, parent, isMain);
};

const {
  isOngoingKind,
  locateTaskLine,
  parseCompletedRecords,
  parseOverviewTracks,
  parsePlanText,
} = require("./main.js").__test;

const plan = parsePlanText(
  [
    "---",
    "student_os: learning-plan",
    "title: Demo Course",
    "track: self-directed",
    "status: active",
    "kind: finite-course",
    "---",
    "## 执行清单",
    "- [x] Lecture 1 #student-os/task",
    "- [ ] Lecture 2 #student-os/task",
    "## 最小自检",
    "- [ ] unrelated self-check",
  ].join("\n"),
  "plans/demo.md"
);
assert.equal(plan.total, 2);
assert.equal(plan.completed, 1);
assert.equal(plan.next.text, "Lecture 2");

const overview = parseOverviewTracks(
  [
    "## 自主成长",
    "### 当前推进",
    "### Demo Course",
    "- 状态：当前推进",
    "- 类型：有限课程",
    "- 真实计划：[[学习计划/Demo Course]]",
    "- 计划：[[Projects/Demo Project]]",
  ].join("\n")
);
assert.equal(overview.length, 1);
assert.equal(overview[0].planTarget, "学习计划/Demo Course");
assert.deepEqual(overview[0].planTargets, ["学习计划/Demo Course", "Projects/Demo Project"]);
assert.equal(overview[0].status, "active");

const completedRecords = parseCompletedRecords(
  [
    "## 计划外记录",
    "- [x] Useful unscheduled work #student-os/task ✅ 2026-08-02",
    "- [ ] Not completed #student-os/task",
    "## 输入箱",
    "- [x] Wrong section #student-os/task ✅ 2026-08-02",
  ].join("\n")
);
assert.equal(completedRecords.length, 1);
assert.equal(completedRecords[0].text, "Useful unscheduled work");
assert.equal(completedRecords[0].track, "unplanned");
assert.equal(isOngoingKind("continuous-capability"), true);
assert.equal(isOngoingKind("multi-stage-project"), false);

const raw = "- [ ] Lecture 2 #student-os/task";
assert.deepEqual(locateTaskLine(["heading", raw], 1, raw), { index: 1, reason: "exact" });
assert.deepEqual(locateTaskLine(["moved", "text", raw], 1, raw), { index: 2, reason: "moved" });
assert.equal(locateTaskLine([raw, raw], 8, raw).reason, "ambiguous");

console.log("learning-progress-dashboard self-check: ok");
