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
  WorkflowStore,
  isOngoingKind,
  locateTaskLine,
  materializeTodayTemplate,
  parseCompletedRecords,
  parseOverviewTracks,
  parsePlanText,
  scopeTaskQueries,
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
assert.equal(
  materializeTodayTemplate("---\ndate: pending\n---\n", new Date(2026, 7, 3)),
  "---\ndate: 2026-08-03\n---\n"
);

async function checkArchivedPlans() {
  const folder = "99_学习情况记录/学习计划";
  const overviewPath = "99_学习情况记录/Overview & Study Record.md";
  const workbenchPath = "99_学习情况记录/workbench.md";
  const coursePath = "Courses/Current course.md";
  const overviewText = [
    "## 学校责任",
    "### Current course",
    "- 状态：当前推进",
    `- 课程入口：[[${coursePath}]]`,
    "## 自主成长",
    ...["Current", "Archived", "Completed", "Paused", "Queued", "Overview closed", "Moved"].flatMap((name) => [
      `### ${name}`,
      `- 状态：${name === "Overview closed" ? "已归档" : name === "Paused" ? "暂停" : name === "Queued" ? "确定待学" : "当前推进"}`,
      `- 真实计划：[[${name === "Moved" ? "99_学习情况记录/archive" : folder}/${name}.md]]`,
    ]),
  ].join("\n");
  const documents = new Map([
    [overviewPath, overviewText],
    [workbenchPath, "## 一次性任务\n- [ ] Current admin #student-os/task\n- [ ] Personal note"],
    [coursePath, "# Course\n- [ ] Lecture review #student-os/task\n- [ ] Lecture notes #student-os/task\n- [ ] Later lecture #student-os/task"],
    [`${folder}/Current.md`, "---\nstudent_os: learning-plan\nstatus: active\n---\n- [ ] Current task #student-os/task"],
    [`${folder}/Archived.md`, "---\nstudent_os: learning-plan\nstatus: archived\n---\n- [ ] Old practice #student-os/task"],
    [`${folder}/Completed.md`, "---\nstudent_os: learning-plan\nstatus: completed\n---\n- [ ] Leftover task #student-os/task"],
    [`${folder}/Paused.md`, "---\nstudent_os: learning-plan\nstatus: active\n---\n- [ ] Paused task #student-os/task"],
    [`${folder}/Queued.md`, "---\nstudent_os: learning-plan\nstatus: queued\n---\n- [ ] Future task #student-os/task"],
    [`${folder}/Overview closed.md`, "---\nstudent_os: learning-plan\nstatus: active\n---\n- [ ] Stale active task #student-os/task"],
    [`${folder}/Orphan.md`, "---\nstudent_os: learning-plan\nstatus: active\n---\n- [ ] Unregistered task #student-os/task"],
    ["99_学习情况记录/archive/Moved.md", "---\nstudent_os: learning-plan\nstatus: active\n---\n- [ ] Moved task #student-os/task"],
  ]);
  const files = [...documents.keys()].map((path) => ({ path, extension: "md" }));
  const store = new WorkflowStore({
    vault: {
      getMarkdownFiles: () => files,
      getFileByPath: (path) => files.find((file) => file.path === path),
      cachedRead: async (file) => documents.get(file.path),
    },
    metadataCache: { getFirstLinkpathDest: (target) => files.find((file) => file.path === target) },
  });
  const choices = await store.taskChoices();
  assert.deepEqual(choices.map((task) => task.text).sort(),
    ["Current task", "Current admin", "Lecture review", "Lecture notes", "Later lecture"].sort(),
    "Quick record must use all current registered sources and exclude inactive or unregistered plans");
  const data = await store.readDashboardData();
  assert.deepEqual(data.active.map((track) => track.title).sort(), ["Current", "Current course"]);
  // The dialog may stay open while another workflow archives its source.
  documents.set(overviewPath, overviewText.replace("### Current\n- 状态：当前推进", "### Current\n- 状态：已归档"));
  assert.equal(await store.completeCanonicalTask(choices.find((task) => task.text === "Current task")), false);
  assert.ok(documents.get(`${folder}/Current.md`).includes("- [ ] Current task"));
  const todayView = 'User writing stays here\n```tasks\nnot done\ntag regex matches /^#student-os\\/task$/\n```\n```tasks\ndone\ntag regex matches /^#student-os\\/task$/\n```\n';
  const scoped = scopeTaskQueries(todayView, store.activeSourcePaths(await store.readDashboardData()));
  assert.equal(scopeTaskQueries(scoped, store.activeSourcePaths(await store.readDashboardData())), scoped);
  const pathPattern = scoped.match(/path regex matches \/(.+)\//)[1];
  const allowed = new RegExp(pathPattern);
  assert.equal(allowed.test(coursePath), true);
  assert.equal(allowed.test(workbenchPath), true);
  assert.equal(allowed.test(`${folder}/Current.md`), false, "A newly archived source must leave native task views too");
  assert.equal(allowed.test(`${folder}/Orphan.md`), false);
  assert.ok(scoped.startsWith('User writing stays here\n'));
  assert.ok(scoped.endsWith('```tasks\ndone\ntag regex matches /^#student-os\\/task$/\n```\n'));
  const archivedDone = parsePlanText(
    "---\nstudent_os: learning-plan\nstatus: archived\n---\n- [x] Historical result #student-os/task ✅ 2026-08-01",
    `${folder}/Archived done.md`
  );
  assert.equal(archivedDone.status, "archived");
  assert.equal(archivedDone.completed, 1, "Archiving must preserve completion evidence");
  const checkedTasks = parsePlanText(
    "---\nstudent_os: learning-plan\nstatus: active\n---\n- [x] Current stage #student-os/task",
    `${folder}/All checked.md`
  );
  assert.equal(checkedTasks.status, "active", "Task completion alone must not end a learning commitment");
}

checkArchivedPlans().then(() => {
  console.log("learning-progress-dashboard self-check: ok");
}).catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
