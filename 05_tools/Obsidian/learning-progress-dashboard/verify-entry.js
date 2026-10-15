/* Live regression: click the actual dashboard button, including an already-open classroom.
   Set window.studentOsExpectedMaterial to an independently checked vault path to verify its exact handout. */
(async () => {
  const view = app.workspace.getLeavesOfType("student-os-steward-view")[0]?.view;
  if (!view?.bridge.lastState || view.bridge.busy) throw new Error("Student OS must be idle and open.");
  const checks = window.studentOsEntryVerification = { pending: true, failures: [] };
  const ensure = (ok, message) => { if (!ok) checks.failures.push(message); };
  const sleep = ms => new Promise(resolve => require("timers").setTimeout(resolve, ms));
  const data = view.bridge.lastState;
  const job = data.jobs.find(j => j.request?.task?.id === data.focus.task?.id && j.state === "registered" && !j.handed_off);
  if (!job) throw new Error("The current classroom must already be prepared.");
  const host = app.plugins.plugins.realclaudian.chatHost;
  const before = host.getConversationList().length;
  const initialTab = app.workspace.getLeavesOfType("claudian-view")[0]?.view.getActiveTab();
  const initialDraft = initialTab?.dom.inputEl?.textContent;
  const expected = window.studentOsExpectedMaterial;
  const taskFile = app.vault.getAbstractFileByPath(job.request.task.path);
  const taskBefore = await app.vault.read(taskFile);
  const detail = view.taskDetail;
  const detailBefore = detail && [...detail.host.querySelectorAll("input,textarea")].map(el => el.value);
  try {
    await app.workspace.revealLeaf(view.leaf);
    app.workspace.setActiveLeaf(view.leaf, { focus: true });
    view.classroom.scrollIntoView({ block: "center", behavior: "instant" });
    view.classroom.querySelector(".sos-enter").click();
    await Promise.resolve();
    const button = view.classroom.querySelector(".sos-enter");
    ensure(button.disabled && button.textContent.includes("打开"), "Clicking Continue gives no immediate visible opening feedback");
    for (let i = 0; i < 200 && view.openingClassroom; i++) await sleep(50);
    const leaf = app.workspace.getLeavesOfType("claudian-view")[0];
    const tab = leaf?.view.getActiveTab();
    const box = leaf?.view.containerEl.getBoundingClientRect();
    ensure(box?.width > 100 && box?.height > 100, "The classroom stayed hidden");
    if (leaf?.getRoot() === app.workspace.rightSplit) ensure(!app.workspace.rightSplit.collapsed, "The classroom sidebar is still collapsed");
    ensure(tab?.state.currentConversationId === job.conversation_id, "The button opened a different conversation");
    ensure(leaf?.view.containerEl.contains(leaf.view.containerEl.ownerDocument.activeElement), "Keyboard focus did not enter the classroom");
    if (initialTab?.state.currentConversationId === job.conversation_id) ensure(tab?.dom.inputEl?.textContent === initialDraft, "Opening replaced the unsent classroom draft");
    ensure(view.classroom.querySelector(".sos-entry-feedback")?.textContent.includes("课堂"), "The dashboard never confirms where the classroom opened");
    ensure(host.getConversationList().length === before, "Clicking Continue duplicated a classroom");
    if (expected) {
      const materialLeaves = ["pdf", "markdown"].flatMap(type => app.workspace.getLeavesOfType(type))
        .filter(l => l.getRoot() === app.workspace.rootSplit && (l.view.file?.path || l.getViewState().state?.file) === expected);
      ensure(materialLeaves.length === 1, "The exact handout was missing or opened in duplicate tabs");
      const material = materialLeaves[0], rect = material?.view.containerEl.getBoundingClientRect();
      ensure(rect?.width > 100 && rect?.height > 100, "The material did not stay visible beside the classroom");
      const materialId = material?.id, state = JSON.stringify(material?.view.getEphemeralState?.());
      const page = material?.view.containerEl.querySelector(".pdf-page-input")?.value;
      const classroomDraft = tab?.dom.inputEl?.textContent;
      await app.workspace.revealLeaf(view.leaf);
      view.classroom.querySelector(".sos-enter").click();
      for (let i = 0; i < 200 && view.openingClassroom; i++) await sleep(50);
      const after = ["pdf", "markdown"].flatMap(type => app.workspace.getLeavesOfType(type))
        .filter(l => (l.view.file?.path || l.getViewState().state?.file) === expected);
      ensure(after.length === 1 && after[0].id === materialId, "Repeated entry duplicated the material tab");
      ensure(JSON.stringify(material?.view.getEphemeralState?.()) === state, "Repeated entry reset the material reading position");
      ensure(material?.view.containerEl.querySelector(".pdf-page-input")?.value === page, "Repeated entry reset the PDF page");
      ensure(tab?.dom.inputEl?.textContent === classroomDraft, "Repeated entry changed the classroom draft");
      checks.material = expected;
      checks.materialLeafId = materialId;
      checks.page = page;
    }
    ensure(await app.vault.read(taskFile) === taskBefore, "Opening class changed a learning record");
    ensure(view.taskDetail === detail && JSON.stringify(detail && [...detail.host.querySelectorAll("input,textarea")].map(el => el.value)) === JSON.stringify(detailBefore), "Opening class changed the task detail or its draft");
    checks.messages = tab?.state.messages?.length;
    checks.sameConversation = tab?.state.currentConversationId === job.conversation_id;
  } catch (error) { checks.failures.push(error.message); }
  finally { checks.pending = false; }
  return JSON.stringify(checks);
})()
