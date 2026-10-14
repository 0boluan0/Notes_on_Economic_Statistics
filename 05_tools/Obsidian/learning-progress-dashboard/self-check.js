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

const { homeModel, StewardBridge, StewardView, CompletionModal, renderedTask, positionFor, agendaFor, todayNotices, deadlineNotices, sameRef, sectionForPath, titleForDeadline, deadlineTime } = require("./steward-view.js");

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
  const task = { id: "learning-task" };
  let data = { focus: { task }, chat: [], jobs: [], commitments: [], obligations: [] };
  const calls = [];
  bridge.refresh = async () => data;
  bridge.run = async (command) => calls.push(command);
  await bridge.start();
  assert.deepEqual(calls, ["prepare-focus"]);
  calls.length = 0;
  data.jobs = [{ key: "original", state: "cancelled", request: { task } }];
  await bridge.start();
  assert.equal(calls.length, 0, "Periodic checks must not revive a cancelled classroom");
  data.jobs[0] = { key: "original", state: "registered", handed_off: true, request: { task } };
  await bridge.start();
  assert.equal(calls.length, 0, "A saved handoff needs an explicit next-class request");
  bridge.busy = true;
  bridge.activeTask = task.id;
  data.jobs[0] = { key: "original", state: "preparing", request: { task } };
  bridge.call = async (command, input) => { calls.push({ command, input }); };
  await bridge.cancelActive();
  assert.equal(calls[0].input.action.kind, "cancel_classroom");
  assert.equal(calls[0].input.action.job_key, "original", "Stop must cancel the running job, even while the bridge is busy");

  let metadataLoaded = false;
  let opened = null;
  const nativeId = "original-native-thread";
  const host = {
    ensureConversationMetadataLoaded: async ids => { assert.deepEqual(ids, [nativeId]); metadataLoaded = true; },
    getConversationById: async id => metadataLoaded ? { id, sessionId: id, messages: [{}, {}] } : null,
  };
  let stateDelivered = false;
  const savedState = { tabWorkspace: { existing: "preserve" } };
  const leaf = { view: {
    tabWorkspace: { isReady: false },
    getState: () => savedState,
    setState: async state => { assert.equal(state, savedState); stateDelivered = true; },
    getTabManager: () => ({ openConversation: async id => { opened = id; } }),
    getActiveTab: () => ({ state: { currentConversationId: opened, messages: [{}, {}] } }),
  } };
  const view = { app: {
    plugins: { plugins: { realclaudian: { chatHost: host } } },
    commands: { executeCommandById: async () => true },
    workspace: { getLeavesOfType: () => [leaf], revealLeaf: async () => {} },
  } };
  view.loadClassroom = StewardView.prototype.loadClassroom;
  await StewardView.prototype.openClassroom.call(view, { key: "saved-job", conversation_id: nativeId, thread_id: nativeId });
  assert.equal(opened, nativeId, "After application restart, load saved metadata before reading and opening the original classroom");
  assert.equal(stateDelivered, true, "A cold pane must finish restoring its exact saved workspace state before navigation");
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
  const view = { draft: "", input: { value: "What remains?" }, selectedCourse: "ST445", selectedWeek: "2", taskFilter: "recorded", section: "workbench",
    chatContext: { title: "ST445 preparation", ref: due.ref }, app: { workspace: { requestSaveLayout() {} } }, render() {}, renderChatContext() {},
    bridge: { busy: false, lastState: { chat: [] }, run: async (command, request) => {
      assert.equal(command, "chat"); assert.ok(request.body.endsWith(`原记录：[[${due.ref}]]`)); return { phase: "done" };
    } } };
  const state = StewardView.prototype.getState.call(view);
  view.input.value = ""; view.section = "overview"; view.chatContext = null;
  await StewardView.prototype.setState.call(view, state);
  assert.equal(view.input.value, "What remains?"); assert.equal(view.section, "workbench"); assert.equal(view.chatContext.ref, due.ref);
  await StewardView.prototype.send.call(view);
  assert.equal(view.input.value, ""); assert.equal(view.chatContext, null);
}

async function checkTaskClosure() {
  const control = value => ({ value, setText(text) { this.text = text; } });
  const modal = { task: boundTask, resumeKey: "old-incomplete-check", reportInput: control(""), dateInput: control("2026-10-09"),
    files: ["unused-evidence.md"], submit: control(), closeTask: control(), stop: control(), restartButton: control(), resultEl: control(),
    lockForm() {}, showResult() {}, plugin: { refreshViews: async () => {}, steward: { run: async (command, input) => {
      assert.equal(command, "completion-check");
      assert.notEqual(input.key, "old-incomplete-check", "Closing must not resume the previous completion argument");
      assert.equal(input.request.intent, "close");
      assert.equal(input.request.date, "");
      assert.deepEqual(input.request.evidence, []);
      assert.ok(input.request.report);
      return { phase: "closed", message: "Closed" };
    } } } };
  await CompletionModal.prototype.check.call(modal, "close");
  assert.equal(modal.terminalPhase, "closed");
  assert.equal(modal.submit.disabled, true);
  assert.equal(modal.closeTask.disabled, true);
  assert.equal(modal.stop.hidden, true);
}

Promise.all([checkArchivedPlans(), checkStewardLifecycle(), checkAutomaticRefresh(), checkWorkspaceContext(), checkTaskClosure()]).then(() => {
  console.log("learning-progress-dashboard self-check: ok");
}).catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
