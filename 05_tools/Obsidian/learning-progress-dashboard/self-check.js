const assert = require("assert");
const Module = require("module");

const originalLoad = Module._load;
Module._load = function load(request, parent, isMain) {
  if (request === "obsidian") {
    return {
      ItemView: class {},
      Modal: class {},
      FuzzySuggestModal: class {},
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

const { homeModel, classroomMaterial, StewardBridge, StewardView, CompletionModal, renderedTask, positionFor, agendaFor, todayNotices, deadlineNotices, deadlineBoard, deadlineSupplements, selfStudyFor, studyUnitFor, learningContext, nearbyUnits, sameRef, sectionForPath, titleForDeadline, deadlineTime } = require("./steward-view.js");

const renderNode = (path, anchor, title) => ({ querySelector: selector => selector.includes("data-code-space-source-path")
  ? { dataset: { codeSpaceSourcePath: path } } : selector === ".task-block-link" ? (anchor ? { textContent: anchor } : null) : { textContent: title } });
const boundTask = { path: "private/plan.md", ref: "private/plan.md#^unit", title: "ST447｜Lecture 笔记", raw: "original" };
assert.equal(renderedTask(renderNode(boundTask.path, "^unit", "renamed visual label"), { tasks: [boundTask] }), boundTask);
assert.equal(renderedTask(renderNode("elsewhere.md", "^unit", boundTask.title), { tasks: [boundTask] }), null);
assert.equal(renderedTask(renderNode(boundTask.path, "^unit", boundTask.title), { tasks: [boundTask, boundTask] }), null, "An ambiguous source must never be ticked");
assert.equal(renderedTask(renderNode(boundTask.path, "", "ST447｜Lecture 笔记 #student-os/task"), { tasks: [boundTask] }), boundTask);

const observedPosition = { summary: "Observed scope", display: { week: 2, label: "Week 2", units: ["One"] } };
assert.equal(positionFor(observedPosition).week, 2);
for (const flag of ["historical", "needs_review"]) {
  const position = positionFor({ ...observedPosition, [flag]: true });
  assert.equal(position.week, null, "Old or changed evidence must not place a current progress marker");
  assert.equal(position.units, undefined, "Old evidence must not leave completed-unit badges visible");
  assert.equal(position.label, "Week 2", "Keep the last known scope readable, with its freshness made explicit");
  assert.ok(position.freshness);
}
assert.equal(positionFor({ summary: "New user report" }).week, null, "A new report without a precise week keeps its position unknown");
assert.equal(titleForDeadline("> - [ ] 2026-10-09 13:30 前 | Course | Prepare [[record#section|details]] #student-os/deadline ^id"), "Course | Prepare");
assert.equal(titleForDeadline("- [ ] 2026-10-09 13:30 前｜ST445｜Prepare ⏳ 2026-10-09 📅 2026-10-09"), "ST445｜Prepare");
assert.equal(deadlineTime({ raw: "- [ ] 2026-10-09 13:30 前 | Course | Prepare" }), "13:30前", "Compact event cards must retain the actual deadline time");
const stewardModel = homeModel({ focus: { task: { id: "task" }, next_step: "Continue exercise 2" },
  jobs: [{ key: "cancelled", state: "cancelled", request: { task: { id: "task" } } },
    { key: "closed", state: "registered", handed_off: true, request: { task: { id: "task" } } },
    { key: "ready", state: "registered", request: { task: { id: "task" } } }],
  obligations: [{ status: "completion_unknown", days_until: -3 }, { status: "recorded_complete", days_until: 0 }],
  commitments: [{ due: true }, { due: false }] });
assert.equal(stewardModel.job.key, "ready");
assert.equal(stewardModel.next, "Continue exercise 2");
assert.equal(stewardModel.obligations.length, 1);
assert.equal(stewardModel.notices.length, 1);

const dailyFixture = { today: { path: "99_学习情况记录/2026-10-09——Fri.md", text: `
<!-- student-os:day-spine:start -->
- **10:30–12:00｜ST443 小班，CBG.1.06。**
- **12:00–12:30｜午餐、休息。**
- **12:30–13:15｜LSE Library，[[record#^prep|ST445 课前补缺]]。**
- **13:30–15:00｜ST445 小班，CBG.2.05；15:00–17:00｜ST447 大课，CLM.5.02。**
- **17:30–19:00｜LSE Library，[[plan#^notes|ST447 笔记]]。**
<!-- student-os:day-spine:end -->
<!-- student-os:deadline-radar:start -->
- **D-0｜今天 13:30 前｜[[record#^prep|ST445 准备]]。** Original scope.
- **重要预约 D-4｜10 月 13 日 09:00–09:30｜[[meeting|科研交流]]。** Confirmed.
- **可选 D-0｜今天 17:00｜Exercise 1。** Optional, partial submission allowed.
<!-- student-os:deadline-radar:end -->` } };
const fridayMorning = new Date("2026-10-09T10:30:00Z");
const agenda = agendaFor(dailyFixture, fridayMorning);
assert.equal(agenda.current.label, "ST443 Seminar");
assert.equal(agenda.current.location, "CBG.1.06");
assert.deepEqual(agenda.upcoming.slice(0, 2).map(s => s.start), ["12:00", "12:30"]);
assert.equal(agenda.slots.find(s => s.start === "15:00").label, "ST447 Lecture", "Two classes on one Today line remain separate time slots");
assert.equal(agenda.slots.find(s => s.start === "17:30").ref, "plan#^notes");
assert.equal(agenda.upcoming[1].label, "ST445 课前补缺");
assert.equal(agendaFor(dailyFixture, new Date("2026-10-09T23:01:00Z")).slots.length, 0, "After London midnight, yesterday is not presented as today's plan");
const notices = todayNotices(dailyFixture, fridayMorning);
assert.equal(notices.length, 3);
assert.equal(notices[0].time, "今天 13:30 前");
assert.equal(notices[1].optional, true);
assert.equal(notices[2].title, "科研交流");
assert.equal(todayNotices({ ...dailyFixture, obligations: [{ ref: "record#^prep", status: "recorded_complete" }] }, fridayMorning).length, 2);

const due = { ref: "99_学习情况记录/截止事项记录.md#^prep", path: "99_学习情况记录/截止事项记录.md",
  raw: "- [ ] 2026-10-10 15:00｜Updated preparation 📅 2026-10-10", date: "2026-10-10", status: "completion_unknown" };
const linkedFixture = { ...dailyFixture, today: { ...dailyFixture.today, text: dailyFixture.today.text.replaceAll("record#^prep", "截止事项记录#^prep") }, obligations: [due] };
const linked = deadlineNotices(linkedFixture, fridayMorning);
assert.equal(linked.length, 3, "The canonical obligation replaces the old Today text, without hiding optional events");
assert.equal(linked[1].title, "Updated preparation");
assert.equal(linked[1].time, "10/10 15:00");
assert.equal(linked[1].days, 1);
assert.equal(deadlineNotices({ ...linkedFixture, obligations: [{ ...due, status: "recorded_complete" }] }, fridayMorning).length, 2);
assert.equal(deadlineNotices({ ...linkedFixture, obligations: [] }, fridayMorning).length, 2, "Deleted originals must not survive in a stale Today radar");
assert.equal(deadlineNotices({ ...linkedFixture, obligations: [due, { ...due, ref: due.ref.replace("prep", "new"), date: "2026-10-11" }] }, fridayMorning).length, 4, "New canonical deadlines appear even before Today is regenerated");
assert.equal(deadlineNotices({ ...linkedFixture, obligations: [{ ...due, date: "2026-11-10" }], today: dailyFixture.today }, fridayMorning).some(n => n.task), false);
assert.equal(deadlineNotices(linkedFixture, new Date("2026-10-10T23:01:00Z")).length, 0, "London rollover removes yesterday's supplements and leaves past originals in Deadlines");
assert.equal(sameRef("截止事项记录#^prep", due.ref), true);
assert.equal(sameRef("other/file.md#^prep", due.ref), false, "Matching block labels in other files are different records");
assert.equal(sectionForPath("99_学习情况记录/workbench.md#ST445", "today.md"), "workbench");
assert.equal(sectionForPath("today.md#Schedule", "today.md"), "today");
assert.equal(sectionForPath("yesterday.md", "today.md"), null);

const dated = (id, date, time = "", status = "completion_unknown") => ({ id, date, status,
  raw: `- [ ] ${date || ""} ${time}｜ST445｜提交 Problem Set 1（15%；开放日期待确认）` });
const deadlineFixture = { obligations: [dated("today", "2026-10-14", "17:00"), dated("past", "2026-10-14", "13:00"),
  dated("date-only", "2026-10-14"), dated("later", "2026-11-02", "12:00"), dated("no-date", null),
  dated("done", "2026-10-09", "17:00", "recorded_complete"), dated("cancelled", "2026-10-15", "12:00", "cancelled")] };
const originalDeadlines = JSON.stringify(deadlineFixture);
const deadlineGroups = deadlineBoard(deadlineFixture, new Date("2026-10-14T12:30:00Z"));
assert.deepEqual(deadlineGroups.upcoming.map(t => t.task.id), ["date-only", "today"]);
assert.deepEqual(deadlineGroups.past.map(t => t.task.id), ["past"], "London summer time, not UTC, decides whether the time has passed");
assert.equal(deadlineGroups.past[0].task.status, "completion_unknown", "Elapsed time never proves a task was missed");
assert.equal(deadlineGroups.upcoming[1].time, "17:00");
assert.equal(deadlineGroups.upcoming[1].title, "Problem Set 1");
assert.equal(deadlineGroups.undated[0].days, null);
assert.equal(deadlineGroups.later.length, 1); assert.equal(deadlineGroups.done.length, 1);
assert.equal(Object.values(deadlineGroups).flat().length, 6, "Cancelled obligations are excluded");
assert.equal(JSON.stringify(deadlineFixture), originalDeadlines, "Rendering does not change canonical facts");
assert.equal(deadlineBoard({ obligations: [dated("winter", "2026-10-26", "13:00")] }, new Date("2026-10-26T12:30:00Z")).upcoming.length, 1, "Winter time does not keep a summer offset");
assert.equal(deadlineBoard({ obligations: [dated("rollover", "2026-10-15")] }, new Date("2026-10-14T23:30:00Z")).upcoming[0].timing, "今天");
const finalProject = { ...dated("final", "2027-08-06", "17:00"), raw: "- [ ] 2027-08-06 17:00｜ST498｜提交小组报告（80%）＋个人贡献报告（20%）" };
assert.equal(deadlineBoard({ obligations: [finalProject] }, fridayMorning).later[0].title, "小组报告（80%）＋个人贡献报告（20%）", "Short titles retain both required deliverables");
const optionalSubmission = { ...dated("optional", "2026-10-24", "23:59"), optional: true,
  raw: "- [ ] 2026-10-24 23:59｜ST445｜提交 Formative PS1（可选、不计分） #student-os/submission-window 📅 2026-10-24 ^optional" };
const optionalBoard = deadlineBoard({ obligations: [optionalSubmission] }, new Date("2026-10-22T12:00:00Z"));
assert.equal(optionalBoard.upcoming[0].kind, "可选提交");
assert.equal(optionalBoard.upcoming[0].detail, "可选、不计分");
assert.equal(optionalBoard.upcoming[0].task.status, "completion_unknown");
assert.equal(deadlineNotices({ obligations: [optionalSubmission] }, new Date("2026-10-22T12:00:00Z"))[0].optional, true, "Optional submissions appear in the same near-deadline radar");
const supplemental = deadlineSupplements(`> [!info]+ 其他已公布提交节点与待定日期
> - **10 月 23 日 17:00｜ST447 Exercise 2｜可选提交**，以实际完成部分为准。[[截止事项记录#ST447|详情]]
> - **10 月 23 日｜ST445 formative PS1｜形成性反馈节点**，时刻待确认。[[截止事项记录#ST445|详情]]
> - **ST498 组队与偏好提交｜日期待学校确认**。星期与日期冲突。

> - **不属于此节的其他记录**`);
assert.equal(supplemental.length, 3);
assert.deepEqual(supplemental.map(t => t.kind), ["可选提交", "反馈节点", "日期待确认"]);
assert.equal(supplemental[0].time, "17:00"); assert.equal(supplemental[0].title, "ST447 Exercise 2");
assert.equal(supplemental[1].time, "时刻待确认"); assert.equal(supplemental[2].date, "待定");
assert.deepEqual(deadlineSupplements(), [], "Missing explanatory text does not fabricate optional deadlines");

const studyTasks = { tasks: [
  { course_id: "Demo", title: "Demo｜Lecture 01", path: "demo.md", raw: "unit1", state: "done", record: "" },
  { course_id: "Demo", title: "Demo｜Lecture 02", path: "demo.md", raw: "unit2", state: "recorded", record: "Completed the first example" },
  { course_id: "Demo", title: "Demo｜本轮重看 Lecture 01", path: "demo.md", raw: "replay", state: "done", record: "" },
  { course_id: "Demo", title: "Closed unit", state: "cancelled" },
  { course_id: "Other", title: "Other unit", state: "done" },
] };
const selfCourse = { id: "Demo", mode: "finite", unit_progress: { completed: 1, total: 2 }, personal: { summary: "Older position" },
  next_task: { path: "demo.md", raw: "unit2" }, next_step: "Continue the second lecture" };
const selfModel = selfStudyFor(selfCourse, studyTasks);
assert.equal(selfModel.units.length, 2, "Replay, closed and unrelated tasks are not added to the historical course total");
assert.equal(selfModel.recorded, 1); assert.equal(selfModel.finite, true);
assert.equal(selfModel.position, "Completed the first example", "A saved partial stopping point is visible even with zero whole-task completions");
assert.equal(selfModel.next, "Lecture 02");
assert.equal(selfStudyFor({ ...selfCourse, replay_progress: { completed: 1, registered: 1 }, personal: { summary: "Current replay completed" } }, studyTasks).position, "Current replay completed");
assert.equal(selfStudyFor({ ...selfCourse, mode: undefined, unit_progress: undefined }, studyTasks).finite, false, "Open-ended practice has no fabricated course percentage");
assert.equal(selfModel.defaultTask, studyTasks.tasks[1], "The route opens at the recorded next task");
assert.equal(selfStudyFor({ ...selfCourse, next_task: null, replay_progress: { completed: 1 } }, studyTasks).defaultTask, studyTasks.tasks[2], "A finished replay does not silently advance to the historical next lecture");
const ninth = studyUnitFor({ title: "MIT 18.01｜Lecture 9: Linear Approximation", state: "unknown" }, 7);
assert.equal(ninth.number, "09", "Lecture numbering is read from the source even when an exam slot is missing");
assert.equal(ninth.title, "Linear Approximation");
assert.equal(ninth.condition, "看完本讲视频，并确认已理解。", "MIT video lecture closure follows the existing Student OS rule");
assert.equal(studyUnitFor({ title: "MIT 18.01｜Lecture 9: Topic", done_when: "独立完成指定练习并保存推导", state: "unknown" }).condition, "独立完成指定练习并保存推导", "Explicit course requirements take precedence over the video default");
assert.equal(studyUnitFor({ title: "Demo｜Lecture 9: Topic", state: "unknown" }).hasCondition, false, "Other courses do not inherit a fabricated stopping condition");
assert.equal(ninth.state, "待确认", "An unchecked item is not described as unlearned");
const bundle = studyUnitFor({ title: "DSL Python｜完成第 1–9 单元练习：变量和循环；保存代码", state: "recorded" });
assert.equal(bundle.label, "第 1–9 单元"); assert.equal(bundle.lecture, false, "A bundled task is not one lecture");
assert.equal(studyUnitFor({ title: "DSL Python｜完成第 1–9 单元练习：变量和循环；保存代码", done_when: "变量和循环；保存代码", state: "recorded" }).condition, bundle.full, "A derived suffix must not drop the actual exercise requirement");
const replayUnit = studyUnitFor({ title: "MIT 6.100L｜本轮重看 Lecture 01 Introduction，记录实际停点；本人理解后完成", state: "done" });
assert.equal(replayUnit.title, "Introduction"); assert.equal(replayUnit.hasCondition, true);

const routeTask = { id: "lesson", path: "route.md", raw: "unit2", title: "Demo｜Lecture 02: Functions", status: "completion_unknown" };
const handoffTask = { ...routeTask, course_id: "Demo", record: "Solved the first example", record_next: "Try the second example", state: "recorded", done_when: "Explain and apply a function" };
const routeState = { tasks: [routeTask], jobs: [], focus: { task: { id: "unrelated" }, next_step: "Wrong course" },
  curriculum: { courses: [{ id: "Demo", personal: { summary: "Older position", needs_review: true }, next_task: routeTask, next_step: "Outdated suggestion" }], weekly: { tasks: [handoffTask] } } };
const context = learningContext(routeState, routeTask);
assert.equal(context.stop, handoffTask.record);
assert.equal(context.next, handoffTask.record_next, "Durable task handoff overrides old course and unrelated focus advice");
assert.equal(context.condition, handoffTask.done_when);
const noHandoff = { ...routeState, curriculum: { ...routeState.curriculum, weekly: { tasks: [{ ...handoffTask, record: "", record_next: "" }] } } };
assert.ok(learningContext(noHandoff, routeTask).stopLabel.includes("待更新"));
assert.equal(learningContext(noHandoff, routeTask).next, "进入课堂，先从已确认的停点接上。", "Old course advice cannot become the next learning step");
const route = [1, 2, 3, 4, 5].map(id => ({ id }));
assert.deepEqual(nearbyUnits(route, route[2]).map(t => t.id), [2, 3, 4]);
assert.deepEqual(nearbyUnits(route, route[0]).map(t => t.id), [1, 2]);

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
  const recordsPath = "99_学习情况记录/工作记录.md";
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
    [recordsPath, "# 工作记录\n## 一次性任务\n- [ ] Recorded admin #student-os/task ^recorded\n## 计划外记录\n"],
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
      process: async (file, transform) => documents.set(file.path, transform(documents.get(file.path))),
    },
    metadataCache: { initialized: true, getFirstLinkpathDest: (target) => files.find((file) => file.path === target) },
  });
  const choices = await store.taskChoices();
  assert.deepEqual(choices.map((task) => task.text).sort(),
    ["Current task", "Current admin", "Recorded admin ^recorded", "Lecture review", "Lecture notes", "Later lecture"].sort(),
    "Quick record must use all current registered sources and exclude inactive or unregistered plans");
  const data = await store.readDashboardData();
  assert.deepEqual(data.active.map((track) => track.title).sort(), ["Current", "Current course"]);
  // The dialog may stay open while another workflow archives its source.
  documents.set(overviewPath, overviewText.replace("### Current\n- 状态：当前推进", "### Current\n- 状态：已归档"));
  assert.ok(!(await store.taskChoices()).some(task => task.text === "Current task"));
  assert.ok(documents.get(`${folder}/Current.md`).includes("- [ ] Current task"));
  const todayView = 'User writing stays here\n```tasks\nnot done\ntag regex matches /^#student-os\\/task$/\n```\n```tasks\ndone\ntag regex matches /^#student-os\\/task$/\n```\n';
  const scoped = scopeTaskQueries(todayView, store.activeSourcePaths(await store.readDashboardData()));
  assert.equal(scopeTaskQueries(scoped, store.activeSourcePaths(await store.readDashboardData())), scoped);
  const pathPattern = scoped.match(/path regex matches \/(.+)\//)[1];
  const allowed = new RegExp(pathPattern);
  assert.equal(allowed.test(coursePath), true);
  assert.equal(allowed.test(workbenchPath), true);
  assert.equal(allowed.test(recordsPath), true);
  assert.equal(allowed.test(`${folder}/Current.md`), false, "A newly archived source must leave native task views too");
  assert.equal(allowed.test(`${folder}/Orphan.md`), false);
  assert.ok(scoped.startsWith('User writing stays here\n'));
  assert.ok(scoped.endsWith('```tasks\ndone\ntag regex matches /^#student-os\\/task$/\n```\n'));
  const retention = '(not done) OR ((status.type is DONE) AND (done on or after yesterday) AND (done on or before today))';
  const foldedView = `> [!todo]- Course\n>\n> \x60\x60\x60tasks\n> # student-os:active-sources\n> path regex matches /old/\n> ${retention}\n> tag regex matches /^#student-os\\/task$/\n> \x60\x60\x60\n`;
  const refreshed = scopeTaskQueries(foldedView, [recordsPath, coursePath]);
  assert.ok(refreshed.includes(`> ${retention}\n`), "Scope refresh must preserve retention in folded native views");
  assert.ok(!refreshed.includes('/old/'));
  assert.equal(scopeTaskQueries(refreshed, [recordsPath, coursePath]), refreshed);
  const workbenchBeforeColdCache = documents.get(workbenchPath);
  documents.set(workbenchPath, scoped);
  store.app.metadataCache.initialized = false;
  await store.syncCurrentTaskViews();
  assert.equal(documents.get(workbenchPath), scoped, "Do not rewrite source scopes before the initial vault index is available");
  store.app.metadataCache.initialized = true;
  const resolveWarmLink = store.resolveOverviewLink;
  store.resolveOverviewLink = () => null;
  await store.syncCurrentTaskViews();
  assert.equal(documents.get(workbenchPath), scoped, "An unresolved startup cache must preserve the existing course query scope");
  store.resolveOverviewLink = resolveWarmLink;
  documents.set(workbenchPath, workbenchBeforeColdCache);
  const workbenchBeforeCapture = documents.get(workbenchPath);
  await store.saveInput("A thought to clarify later");
  assert.ok(!documents.get(recordsPath).includes("- [x]"), "Intake cannot bypass completion checking");
  assert.ok(documents.get(recordsPath).includes("A thought to clarify later"));
  assert.equal(documents.get(workbenchPath), workbenchBeforeCapture, "Capture must not append history or prose to Workbench");
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

async function checkStewardLifecycle() {
  const bridge = new StewardBridge({});
  let refreshes = 0;
  bridge.refresh = async () => { refreshes++; return { focus: { task: { id: "new" } }, chat: [], jobs: [] }; };
  bridge.run = async () => { throw new Error("Opening the dashboard must not start an AI job"); };
  await bridge.start(); await bridge.start();
  assert.equal(refreshes, 2);
  await assert.rejects(() => bridge.call("chat", { body: "hello" }), /Codex/);
  await assert.rejects(() => bridge.call("prepare-focus"), /Codex/);
  await assert.rejects(() => bridge.call("act", { action: { kind: "check_completion" } }), /Codex/);
}

async function checkAutomaticRefresh() {
  const bridge = new StewardBridge({});
  const saved = { ...dailyFixture, tasks: [], obligations: [], curriculum: { source_paths: ["private/positions.json"],
    courses: [{ school: { sources: [{ path: "course.md#Scope" }] }, personal: {} }] } };
  bridge.lastState = saved;
  assert.equal(bridge.isSource("course.md"), true);
  assert.equal(bridge.isSource("private/positions.json"), true);
  for (const name of ["workbench", "deadlines", "截止事项记录", "Overview & Study Record"]) assert.equal(bridge.isSource(`99_学习情况记录/${name}.md`), true);
  assert.equal(bridge.isSource("99_学习情况记录/2026-10-10——Sat.md"), true);
  assert.equal(bridge.isSource("99_学习情况记录/管家/课程总览.md"), false, "Generated projections must not trigger their own refresh loop");
  bridge.call = async () => { throw new Error("temporarily unavailable"); };
  await assert.rejects(bridge.refresh());
  assert.equal(bridge.lastState, saved, "A failed refresh keeps the last readable state");
  assert.match(bridge.status, /保留上次记录/);
  let reads = 0, release;
  bridge.call = () => { reads++; return new Promise(resolve => { release = resolve; }); };
  const first = bridge.refresh(), second = bridge.refresh();
  assert.equal(reads, 1, "Concurrent file and timer refreshes share one read");
  const recovered = { ...saved, at: "recovered" };
  release(recovered);
  await Promise.all([first, second]);
  assert.equal(bridge.lastState, recovered);
  assert.equal(bridge.status, "记录已同步");
  const stale = bridge.refresh();
  void bridge.refresh(true);
  release({ ...saved, at: "in flight before file changed" });
  await stale;
  assert.equal(reads, 3, "A file change during a read queues one fresh read after that snapshot");
  const current = bridge.refresh();
  release({ ...saved, at: "latest source" });
  await current;
  assert.equal(bridge.lastState.at, "latest source");
  bridge.stop(); await bridge.refresh();
  assert.equal(reads, 3, "Unload cannot spawn another background read");
}

async function checkWorkspaceContext() {
  const view = { draft: "", input: { value: "What remains?" }, selectedCourse: "ST445", selectedWeek: "2", taskFilter: "recorded", section: "workbench", learningArea: "external",
    studyUnits: { "MIT 18.01": "unit-9" }, studyOpen: ["MIT 18.01"], studyRoutes: { "MIT 18.01": true },
    chatContext: { title: "ST445 preparation", ref: due.ref }, app: { workspace: { requestSaveLayout() {} } }, saveWorkspace() {}, render() {}, renderChatContext() {},
    bridge: { busy: false, lastState: { chat: [] }, run: async (command, request) => {
      assert.equal(command, "chat"); assert.ok(request.body.endsWith(`原记录：[[${due.ref}]]`)); return { phase: "done" };
    } } };
  const state = StewardView.prototype.getState.call(view);
  view.input.value = ""; view.section = "overview"; view.chatContext = null;
  await StewardView.prototype.setState.call(view, state);
  assert.equal(view.input.value, "What remains?"); assert.equal(view.section, "workbench"); assert.equal(view.chatContext.ref, due.ref);
  assert.equal(view.studyUnits["MIT 18.01"], "unit-9"); assert.deepEqual(view.studyOpen, ["MIT 18.01"], "Course expansion and selected unit survive workspace restoration");
  assert.equal(view.studyRoutes["MIT 18.01"], true, "Expanded historical routes survive workspace restoration");
  assert.equal(view.learningArea, "external", "The selected learning area survives workspace restoration");
  assert.equal(view.draft, "What remains?", "Historical chat drafts remain saved after removing the chat UI");
}

async function checkLearningNavigation() {
  const counts = { done: 0, total: 1, recorded: 0, unknown: 1, planned: 0 };
  const curriculum = { week: 4, courses: [
    { id: "ST443", title: "Machine Learning", group: "degree", task_progress: counts },
    { id: "ST447", title: "Data Science", group: "degree", task_progress: counts },
    { id: "MIT 18.06", title: "Linear Algebra", group: "foundation", active: true, task_progress: counts },
  ], weekly: { tasks: [], weeks: [{ key: "4", number: 4, start: "2026-10-19", end: "2026-10-25" }], bundles: [] }, issues: [] };
  const focus = { task: { id: "saved-focus" } }, visited = [];
  const view = { selectedCourse: "ST443", selectedWeek: "2", learningArea: "school", section: "overview",
    contentEl: { scrollTop: 900 },
    bridge: { lastState: { curriculum, focus } }, saveWorkspace() {}, render() {},
    board: { querySelector: selector => ({ scrollIntoView() { visited.push(selector); } }) } };
  StewardView.prototype.selectTasks.call(view, "ST447", "2");
  assert.equal(view.selectedWeek, null, "A different school course opens its current week instead of inheriting another course's old week");
  assert.equal(view.learningArea, "school");
  StewardView.prototype.selectTasks.call(view, "ST447", "3");
  assert.equal(view.selectedWeek, "3", "Selecting a historical week within the same course remains possible");
  StewardView.prototype.selectTasks.call(view, "MIT 18.06", "all");
  assert.equal(view.learningArea, "external", "Task entry selects the task's actual learning area");
  assert.equal(view.selectedWeek, "all");
  StewardView.prototype.selectLearningArea.call(view, "school");
  assert.equal(view.learningArea, "school");
  assert.equal(view.selectedCourse, "MIT 18.06", "Switching areas does not overwrite the remembered course or learning focus");
  assert.equal(view.bridge.lastState.focus, focus);
  assert.equal(view.contentEl.scrollTop, 0, "Switching areas starts at the top rather than retaining a long page's old scroll position");
  view.contentEl.scrollTop = 700;
  StewardView.prototype.returnToCourses.call(view);
  assert.equal(view.contentEl.scrollTop, 0, "Returning to courses reveals the first course cards below the fixed navigation");
  assert.equal(view.selectedCourse, null);
  const restored = { render() {} };
  await StewardView.prototype.setState.call(restored, { learningArea: "invalid" });
  assert.equal(restored.learningArea, "school", "Older or invalid workspace state starts in the school area");

  // Exercise the real area's visibility and control callbacks with only the DOM methods it uses.
  const nodes = [], actions = [];
  const node = (tag = "div", options = {}) => {
    const value = { tagName: tag.toUpperCase(), cls: options.cls || "", text: options.text || "", attrs: { ...options.attr }, dataset: {}, children: [],
      setAttribute(key, item) { this.attrs[key] = item; }, addEventListener() {},
      classList: {
        toggle(name, enabled) { const classes = new Set(value.cls.split(/\s+/).filter(Boolean)); if (enabled) classes.add(name); else classes.delete(name); value.cls = [...classes].join(" "); },
        contains(name) { return value.cls.split(/\s+/).includes(name); },
      },
      querySelectorAll(selector) { return this.children.flatMap(child => [...(selector.startsWith(".") && child.cls.split(/\s+/).includes(selector.slice(1)) ? [child] : []), ...child.querySelectorAll(selector)]); },
      appendChild(child) { this.children.push(child); child.parentElement = this; },
      createEl(childTag, childOptions) { const child = node(childTag, childOptions); this.appendChild(child); return child; },
      createDiv(childOptions) { return this.createEl("div", childOptions); }, createSpan(childOptions) { return this.createEl("span", childOptions); } };
    nodes.push(value); return value;
  };
  const rendered = { ...view, learningArea: "external", contentEl: node(), board: node(), classroom: node(),
    button(parent, label, action, cls) { const result = parent.createEl("button", { text: label, cls }); actions.push({ label, action, node: result }); return result; },
    renderSelfStudy() {}, renderWeekly() {}, selectTasks: (...args) => { rendered.selected = args; },
    selectLearningArea: area => { rendered.chosenArea = area; }, selectSection: section => { rendered.chosenSection = section; }, returnToCourses() {},
    updateNavigation() { StewardView.prototype.updateNavigation.call(this); },
  };
  StewardView.prototype.renderAppbar.call(rendered);
  const tabs = rendered.contentEl.querySelectorAll(".sos-appbar-primary-tab");
  assert.deepEqual(tabs.map(tab => tab.text), ["学校课程", "校外学习"]);
  assert.equal(tabs[0].attrs["aria-pressed"], "false"); assert.equal(tabs[1].attrs["aria-current"], "page");
  assert.equal(rendered.contentEl.classList.contains("is-learning-overview"), true);
  assert.equal(rendered.contentEl.classList.contains("is-external-area"), true);
  actions.find(a => a.label === "学校课程").action(); assert.equal(rendered.chosenArea, "school");
  actions.find(a => a.label === "校外学习").action(); assert.equal(rendered.chosenArea, "external");
  for (const [label, section] of [["工作台", "workbench"], ["期限", "deadlines"], ["今日", "today"]]) {
    actions.find(a => a.label === label).action(); assert.equal(rendered.chosenSection, section);
    rendered.section = section; rendered.updateNavigation();
    assert.equal(tabs.every(tab => tab.attrs["aria-current"] === "false"), true, "Management views must not leave a learning area visibly selected");
    assert.equal(rendered.contentEl.querySelectorAll(".sos-section-tab").filter(tab => tab.attrs["aria-current"] === "page")[0].dataset.section, section);
    assert.equal(rendered.contentEl.classList.contains("is-external-area"), false, "Management views must not inherit the bounded external workspace layout");
  }
  rendered.section = "overview"; rendered.updateNavigation();
  assert.equal(rendered.learningArea, "external", "Visiting management views preserves the remembered learning area");
  assert.equal(tabs[1].attrs["aria-current"], "page");
  StewardView.prototype.renderCurriculum.call(rendered, { curriculum }, {});
  assert.equal(nodes.find(n => n.cls === "sos-school").hidden, true);
  assert.equal(nodes.find(n => n.cls === "sos-external").hidden, false);
  assert.deepEqual(nodes.find(n => n.cls === "sos-learning-breadcrumb").children.map(n => n.text),
    ["学校学习", "›", "全部学校课程", "›", "第 4 周"], "The all-courses context still identifies the current week");
  assert.equal(nodes.some(n => n.cls === "sos-learning-nav"), false, "The curriculum must not add a duplicate area navigation row");
  rendered.selectedCourse = "ST443"; rendered.selectedWeek = "2";
  actions.find(a => a.node.dataset.course === "ST443").action();
  assert.deepEqual(rendered.selected, ["ST443", "2", "all"], "Clicking the selected course again does not toggle silently to all courses");
  actions.length = 0;
  StewardView.prototype.renderWeekly.call(rendered, node(), curriculum, "school");
  actions.find(a => a.label === "回到本周").action();
  assert.deepEqual(rendered.selected, ["ST443", null], "Back to this week preserves the currently selected school course");

  const todayDate = new Intl.DateTimeFormat("en-CA", { timeZone: "Europe/London", year: "numeric", month: "2-digit", day: "2-digit" }).format(new Date());
  const dueTitle = "ST445｜Formative PS1提交窗口关闭（可选、不计分）：完成并核对习题后提交";
  const compactDue = { obligations: [
    { ...dated("compact", todayDate, "23:59"), raw: `- [ ] ${todayDate} 23:59｜${dueTitle}` },
    { ...dated("second", todayDate, "13:30"), raw: `- [ ] ${todayDate} 13:30｜ST443｜准备课堂材料` },
  ] };
  const originalDue = JSON.stringify(compactDue);
  const noticeView = { ...rendered, board: node() };
  StewardView.prototype.renderDueToday.call(noticeView, compactDue);
  const dueButtons = noticeView.board.querySelectorAll(".sos-due-item");
  assert.equal(dueButtons.length, 2, "Compact deadline headlines retain every item due today");
  assert.deepEqual(noticeView.board.querySelectorAll(".sos-due-time").map(n => n.text), ["13:30", "23:59"]);
  assert.deepEqual(noticeView.board.querySelectorAll(".sos-due-label").map(n => n.text), ["ST443｜准备课堂材料", dueTitle.split("：")[0]]);
  assert.equal(dueButtons[1].attrs.title, dueTitle, "The tooltip retains the full original instruction");
  assert.equal(dueButtons[1].attrs["aria-label"], `23:59 ${dueTitle}`, "The accessible deadline retains its time and full instruction");
  actions.find(a => a.node === dueButtons[1]).action(); assert.equal(rendered.chosenSection, "deadlines");
  assert.equal(JSON.stringify(compactDue), originalDue, "Compact rendering must not change the original deadline");

  const externalEdit = "记录含有外部修改，已保留原文：99_学习情况记录/week-review/example.md";
  for (const error of [externalEdit, "周报读取失败：文件不可用"]) {
    const detailView = { ...rendered, board: node() };
    StewardView.prototype.renderDetails.call(detailView, { progress: [], commitments: [], weekly_report: { error } });
    const warning = detailView.board.querySelectorAll(".sos-record-warning")[0];
    assert.equal(warning.tagName, error === externalEdit ? "DETAILS" : "P");
    if (error === externalEdit) {
      assert.equal(warning.children[0].text, "同步提醒：周报有外部修改，原文已保留");
      assert.equal(warning.children[1].text, error, "The expandable reminder preserves the original diagnostic and path");
      assert.equal(warning.classList.contains("sos-error"), false);
    } else assert.equal(warning.text, error, "Other errors remain visible in full");
  }

  const mount = { empty() {}, appendChild(child) { this.child = child; } };
  const row = { tagName: "DIV", dataset: { key: "map-unit", taskId: "mapped-task" }, querySelector: () => mount, parentElement: { tagName: "DETAILS", open: false } };
  const unitBody = { tagName: "DETAILS", open: false, parentElement: row };
  mount.parentElement = unitBody;
  Object.defineProperty(row, "open", { set() { throw new Error("A map unit is not a details element"); } });
  const taskView = { section: "overview", board: { querySelectorAll: () => [row] }, taskDock: { appendChild(child) { this.child = child; } },
    taskDetail: { rowKey: "map-unit", controller: { task: { id: "mapped-task" } }, host: { draft: "Unfinished note" } } };
  StewardView.prototype.placeTaskDetail.call(taskView);
  assert.equal(mount.child, taskView.taskDetail.host); assert.equal(row.parentElement.open, true);
  assert.equal(unitBody.open, true, "Opening a map record expands the unit-body details inside its row");
  assert.equal(taskView.taskDock.hidden, true, "Mapped course records stay inline rather than appearing at the dashboard's top");
  const otherProjection = { ...row, dataset: { key: "weekly-same-task", taskId: "mapped-task" }, querySelector() { throw new Error("A map record must not jump into another projection"); } };
  taskView.board.querySelectorAll = () => [otherProjection];
  StewardView.prototype.placeTaskDetail.call(taskView);
  assert.equal(taskView.taskDock.hidden, true, "Browsing away from the map unit parks its record without exposing a top-level form");
  assert.equal(taskView.taskDock.child.draft, "Unfinished note", "The parked record retains its existing form and draft");
  assert.equal(taskView.taskDetail.rowKey, "map-unit", "Returning can still identify the original map unit");
  taskView.board.querySelectorAll = () => [row];
  unitBody.open = false;
  StewardView.prototype.placeTaskDetail.call(taskView);
  assert.equal(mount.child, taskView.taskDetail.host); assert.equal(unitBody.open, true);
  assert.equal(mount.child.draft, "Unfinished note", "Returning to the mapped unit reuses the unfinished form");
}

async function checkWorkspacePersistence() {
  const PluginClass = require("./main.js"), writes = [];
  let finish;
  const plugin = { savedData: { unrelated: "keep" }, saveData: data => {
    writes.push(data);
    return writes.length === 1 ? new Promise(resolve => { finish = resolve; }) : Promise.resolve();
  } };
  const first = PluginClass.prototype.saveStewardState.call(plugin, { study: { units: { Demo: "unit1" } } });
  await Promise.resolve();
  const second = PluginClass.prototype.saveStewardState.call(plugin, { study: { units: { Demo: "unit2" } } });
  await Promise.resolve();
  assert.equal(writes.length, 1, "State writes must be serialized so slow storage cannot restore an older unit");
  finish(); await Promise.all([first, second]);
  assert.equal(writes.at(-1).stewardWorkspace.study.units.Demo, "unit2");
  assert.equal(writes.at(-1).unrelated, "keep");
  await PluginClass.prototype.saveStewardState.call(plugin, plugin.savedData.stewardWorkspace);
  assert.equal(writes.length, 2, "Unchanged layout does not write another copy");
  let opened;
  plugin.stewardViewType = "student-os-steward-view";
  plugin.app = { workspace: { getLeavesOfType: () => [], getLeaf: () => ({ setViewState: async state => { opened = state; } }) } };
  await PluginClass.prototype.openSteward.call(plugin);
  assert.deepEqual(opened.state, plugin.savedData.stewardWorkspace, "A reopened view receives its durable workspace state");
  plugin.saveData = async () => { throw new Error("temporary storage failure"); };
  const retryState = { study: { units: { Demo: "unit3" } } };
  await assert.rejects(() => PluginClass.prototype.saveStewardState.call(plugin, retryState), /storage failure/);
  plugin.saveData = async data => { writes.push(data); };
  await PluginClass.prototype.saveStewardState.call(plugin, retryState);
  assert.equal(writes.at(-1).stewardWorkspace.study.units.Demo, "unit3", "An identical retry must recover a failed layout save");
}

async function checkLearningEntry() {
  const data = JSON.parse(JSON.stringify(routeState)), calls = [];
  const ready = { key: "ready", state: "registered", request: { task: routeTask } };
  const view = { bridge: { lastState: data, act: async action => {
    calls.push(action.kind); assert.equal(action.task_id, routeTask.id); assert.equal(action.next_step, handoffTask.record_next);
    data.focus = { task: routeTask }; return {};
  } }, openStudyMaterial: async task => { calls.push("open"); assert.equal(task.id, routeTask.id); } };
  await StewardView.prototype.beginTask.call(view, handoffTask);
  assert.deepEqual(calls, ["set_focus", "open"], "Both course types save the selected task and open its material without AI");
  calls.length = 0;
  await StewardView.prototype.beginTask.call(view, handoffTask);
  assert.deepEqual(calls, ["open"], "Repeated entry uses the existing focus");
  view.taskDetail = { controller: { task: { id: "another" } } };
  await assert.rejects(() => StewardView.prototype.beginTask.call(view, handoffTask), /正在填写/);
  view.taskDetail = null; data.tasks[0].status = "recorded_complete";
  await assert.rejects(() => StewardView.prototype.beginTask.call(view, handoffTask), /任务记录已变化/);
  assert.equal(view.startingLesson, false);
  data.tasks[0].status = "completion_unknown"; data.focus = null;
  view.bridge.act = async () => ({ error: "Save failed" });
  calls.length = 0;
  await assert.rejects(() => StewardView.prototype.beginTask.call(view, handoffTask), /Save failed/);
  assert.deepEqual(calls, [], "A failed focus write cannot prepare or enter a different task");
}

async function checkCourseContinuation() {
  const course = { id: "Python", continuation: { after_task_id: "replay-01", title: "Python｜Lecture 02" } };
  const target = { id: "replay-02", title: "Python｜本轮重看 Lecture 02", status: "completion_unknown" };
  const calls = [];
  let release;
  const view = { bridge: { busy: false, lastState: { tasks: [target], curriculum: { courses: [course] } },
    act: async action => { calls.push(action); await new Promise(resolve => { release = resolve; }); return { task_id: target.id }; } },
    selectStudyUnit: (c, t) => { assert.equal(c.id, course.id); assert.equal(t.id, target.id); calls.push("selected"); },
    beginTask: async task => { assert.equal(task.id, target.id); calls.push("entered"); },
    send: () => { throw new Error("A next-lecture click must never send a manager prompt"); } };
  const work = StewardView.prototype.continueCourse.call(view, course);
  await StewardView.prototype.continueCourse.call(view, course);
  assert.equal(calls.length, 1, "Double-clicking cannot register two replay tasks");
  release(); await work;
  assert.deepEqual(calls[0], { kind: "continue_course", course_id: course.id, task_id: "replay-01" });
  assert.deepEqual(calls.slice(1), ["selected", "entered"]);
  assert.equal(view.continuingCourse, false);
  view.bridge.act = async () => ({ error: "Sequence changed" });
  await assert.rejects(() => StewardView.prototype.continueCourse.call(view, course), /Sequence changed/);
  assert.equal(calls.length, 3, "A failed continuation cannot enter a historical completed task");
}

async function checkClassroomMaterials() {
  const makeFile = path => ({ path, extension: path.split(".").at(-1), basename: path.split("/").at(-1).replace(/\.[^.]+$/, "") });
  const plan = makeFile("99_学习情况记录/学习计划/MIT.md"), note = makeFile("MIT/02_Strings.md"), slides = makeFile("MIT/slides/lec02.pdf");
  const oldSlides = makeFile("MIT/slides/lec01.pdf"), solution = makeFile("MIT/ex02_sol.pdf");
  const files = [plan, note, slides, oldSlides, solution];
  const caches = new Map([[note.path, { links: [
    { link: solution.path, displayText: "Lecture 02 exercise solution" },
    { link: slides.path, displayText: "Lecture 02 slides" }
  ] }]]);
  const app = { vault: { getAbstractFileByPath: path => files.find(f => f.path === path), getFiles: () => files }, metadataCache: {
    getFileCache: file => caches.get(file.path),
    getFirstLinkpathDest: path => files.find(f => f.path === path || f.path === path + ".md")
  } };
  const task = { id: "unit", path: plan.path, title: "MIT 6.100L｜本轮重看 Lecture 02", raw: "- [ ] Lecture 02: [[MIT/02_Strings|Strings]]" };
  const job = { request: { task, materials: [oldSlides, plan, note, solution] } };
  assert.equal(classroomMaterial(app, job).file, slides, "The exact lesson's slides outrank unrelated prepared sources and answer sheets");
  const original = JSON.stringify(job);
  const savedRaw = task.raw;
  task.raw = "- [ ] 本轮重看 Lecture 02";
  assert.equal(classroomMaterial(app, job).file, slides, "An older saved classroom resolves its numbered lesson without re-preparation");
  task.raw = savedRaw;
  assert.equal(JSON.stringify(job), original, "Resolving materials is read-only");
  caches.set(note.path, { links: [{ link: "missing.pdf", displayText: "Lecture 02 slides" }] });
  const missing = classroomMaterial(app, job);
  assert.equal(missing.file, note);
  assert.match(missing.message, /讲义尚未定位/, "A missing PDF falls back to the exact lesson note with an explicit explanation");
  caches.set(note.path, { links: [{ link: slides.path, displayText: "Slides" }, { link: oldSlides.path, displayText: "Previous lecture slides" }] });
  assert.equal(classroomMaterial(app, job).file, slides, "Previous-lesson navigation is not the current handout");
  assert.equal(classroomMaterial(app, { request: { task: { ...task, title: "No lesson binding", raw: "- [ ] Study" }, materials: [plan, oldSlides] } }).file, undefined, "Never open the first prepared attachment without a lesson binding");

  const school = makeFile("ST445/2026_Course_Materials/02_Lecture_Notes/Week_03/ST445 - Week 03 - Lecture.pdf");
  files.push(school, makeFile("ST445/2026_Course_Materials/02_Lecture_Notes/Week_01/ST445 - Week 01 - Lecture.pdf"));
  const schoolTask = { id: "school", path: plan.path, title: "ST445｜Lecture", raw: "- [ ] ST445｜2026-10-13 Lecture" };
  const schoolJob = { request: { task: schoolTask, materials: [oldSlides] } };
  const weekly = [{ ...schoolTask, course_id: "ST445", group: "lecture", week: "3" }];
  assert.equal(classroomMaterial(app, schoolJob, weekly).file, school, "Match a school lecture to its explicit course and week, even if the old preparation omitted that PDF");
  weekly[0].raw += " changed";
  assert.equal(classroomMaterial(app, schoolJob, weekly).file, undefined, "Changed tasks must not reuse a stale week binding");
  files.push(makeFile("ST445/2026_Course_Materials/02_Lecture_Notes/ST445 - Chapter 02.pdf"));
  weekly[0] = { ...schoolTask, course_id: "ST445", group: "lecture", week: "2" };
  assert.equal(classroomMaterial(app, schoolJob, weekly).file, undefined, "A school week must never be treated as a chapter number");

  const main = {}, right = {}, leaves = [], events = [];
  const workspace = app.workspace = { rootSplit: main, getLeavesOfType: () => leaves, getLeaf: kind => {
    events.push(kind);
    const leaf = { getRoot: () => main, view: {}, openFile: async file => { leaf.view.file = file; events.push("open"); } };
    leaves.push(leaf); return leaf;
  }, revealLeaf: async leaf => { assert.equal(leaf.view.file, slides); events.push("reveal"); }, setActiveLeaf() {} };
  const view = { app, plugin: {}, bridge: { lastState: { jobs: [job], curriculum: { weekly: { tasks: [] } } } } };
  await StewardView.prototype.openStudyMaterial.call(view, task);
  leaves[0].view.page = 7;
  await StewardView.prototype.openStudyMaterial.call(view, task);
  assert.deepEqual(events, ["tab", "open", "reveal", "reveal"], "Repeated entry reuses the material tab without reopening or resetting its file");
  assert.equal(leaves[0].view.page, 7);
  assert.equal(leaves.length, 1);
}

async function checkTaskClosure() {
  const control = value => ({ value, setText(text) { this.text = text; } });
  let closed = false, saved;
  const modal = { task: boundTask, reportInput: control(""), dateInput: control("2026-10-09"),
    closeReason: control("This preparation is no longer needed"), files: ["unused-evidence.md"], resultEl: control(),
    close() { closed = true; }, plugin: { setTaskCompletion: async (task, completed, details) => { saved = details; } } };
  await CompletionModal.prototype.check.call(modal, "close");
  assert.equal(closed, true);
  assert.equal(saved.intent, "close"); assert.equal(saved.date, ""); assert.deepEqual(saved.evidence, []);
  closed = false;
  await CompletionModal.prototype.check.call(modal, "complete");
  assert.equal(closed, true, "An empty optional report must not block manual completion");
  assert.equal(saved.report, "");
}

Promise.all([checkArchivedPlans(), checkStewardLifecycle(), checkAutomaticRefresh(), checkWorkspaceContext(), checkLearningNavigation(), checkTaskClosure(), checkLearningEntry(), checkWorkspacePersistence(), checkCourseContinuation(), checkClassroomMaterials()]).then(() => {
  console.log("learning-progress-dashboard self-check: ok");
}).catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
