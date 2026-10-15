/* Live read-only check: open an existing classroom and inspect persisted action receipts. */
(async () => {
  const leaf = app.workspace.getLeavesOfType("student-os-steward-view")[0], view = leaf?.view;
  if (!view?.bridge.lastState) throw new Error("Open Student OS first.");
  const checks = window.studentOsManagerVerification = { pending: true, failures: [] };
  const ensure = (ok, message) => { if (!ok) checks.failures.push(message); };
  try {
    await view.bridge.refresh();
    const data = view.bridge.lastState;
    const job = data.jobs.find(j => j.state === "registered" && j.request?.task?.id === data.focus.task?.id)
      || data.jobs.find(j => j.state === "registered" && !j.handed_off);
    if (!job) throw new Error("No existing prepared classroom available.");
    const before = data.tasks.map(t => [t.id, t.raw]);
    const host = app.plugins.plugins.realclaudian.chatHost;
    const originalCount = host.getConversationList().length;
    await view.openClassroom(job);
    const classroom = app.workspace.getLeavesOfType("claudian-view")[0].view;
    const tab = classroom.getActiveTab();
    ensure(tab.state.currentConversationId === job.conversation_id, "Opening changed the saved classroom identity");
    ensure(tab.state.messages.length >= 2, "Classroom history did not load");
    ensure(host.getConversationList().length === originalCount, "Opening duplicated a classroom");
    checks.classroom = { sameConversation: tab.state.currentConversationId === job.conversation_id,
      messages: tab.state.messages.length, conversationCountUnchanged: host.getConversationList().length === originalCount };
    await app.workspace.revealLeaf(leaf);
    view.section = "overview"; view.render();
    const latest = data.chat[data.chat.length - 1];
    if (latest?.receipts?.length) {
      ensure(!!view.messages.querySelector(":scope > .sos-outcome"), "Latest action receipt is hidden in history or prose");
      checks.visibleReceipts = latest.receipts.length;
    }
    checks.classroomTitle = view.classroom.querySelector("h3")?.textContent;
    ensure(!view.classroom.textContent.includes("registration not confirmed"), "Raw adapter error remains on the classroom card");
    await view.bridge.refresh();
    ensure(JSON.stringify(before) === JSON.stringify(view.bridge.lastState.tasks.map(t => [t.id, t.raw])), "Opening a classroom changed learning facts");
  } catch (error) { checks.failures.push(error.message); }
  finally { await app.workspace.revealLeaf(leaf); checks.pending = false; }
  return JSON.stringify(checks);
})()
