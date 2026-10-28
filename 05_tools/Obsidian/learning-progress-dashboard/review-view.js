const { ItemView, Notice } = require("obsidian");
const { randomUUID } = require("crypto");

const REVIEW_VIEW = "student-os-review-view";
const PHASES = { diagnostic: "独立检测", feedback: "核对与反馈", repair: "回看与修补", retest: "换题重测", delayed: "延迟复习", system_wait: "系统等待" };
const minutes = seconds => `${Math.floor(Math.max(0, seconds || 0) / 60)}:${String(Math.floor(Math.max(0, seconds || 0) % 60)).padStart(2, "0")}`;
const totalTime = timing => Object.values(timing || {}).reduce((sum, n) => sum + (Number(n) || 0), 0);

// A visible interval is an estimate of time on task, not proof of attention.
class ReviewClock {
  constructor() { this.pending = {}; this.last = null; }
  tick(at, phase, active) {
    const seconds = this.last === null ? 0 : Math.max(0, (at - this.last) / 1000);
    this.last = at;
    // Sleep and a suspended renderer must never become hours of study time.
    if (active && seconds <= 3) this.pending[phase] = (this.pending[phase] || 0) + seconds;
  }
  saved(phase, seconds) { this.pending[phase] = Math.max(0, (this.pending[phase] || 0) - seconds); }
}

class ReviewView extends ItemView {
  constructor(leaf, plugin) {
    super(leaf); this.plugin = plugin; this.clock = new ReviewClock();
    this.running = false; this.busy = false; this.draft = {}; this.sourceLeaves = new Set();
  }
  getViewType() { return REVIEW_VIEW; }
  getDisplayText() { return "Student OS · 课程复习"; }
  getIcon() { return "rotate-ccw"; }
  getState() { return {}; }
  async onOpen() {
    this.contentEl.addClass("sor-review");
    this.registerInterval(setInterval(() => { this.tick(); this.updateClock(); }, 1000));
    this.registerInterval(setInterval(() => { if (this.data?.session && !this.busy) void this.flush().catch(error => this.showError(error)); }, 10000));
    this.registerDomEvent(this.contentEl.ownerDocument, "visibilitychange", () => {
      this.tick();
      if (this.contentEl.ownerDocument.hidden) void this.flush().catch(error => this.showError(error));
    });
    await this.loadCatalog();
  }
  async onClose() {
    this.tick(); this.running = false;
    await this.flush().catch(error => new Notice(`复习最近片段尚未保存：${error.message}`));
  }
  call(input) { return this.transport ? this.transport(input) : this.plugin.steward.call("review", input); }
  async loadCatalog() {
    if (this.data?.session) { this.tick(); this.running = false; await this.flush(); }
    this.catalog = await this.call({ op: "catalog" });
    this.data = null; this.draft = {}; this.clock = new ReviewClock(); this.error = ""; this.render();
  }
  phase() {
    if (this.busy) return "system_wait";
    const session = this.data?.session;
    return session?.stage === "question" ? session.phase : session?.stage === "repair" ? "repair" : "feedback";
  }
  tick() {
    const doc = this.contentEl.ownerDocument;
    const active = this.app.workspace.activeLeaf;
    const visible = !doc.hidden && doc.hasFocus() && (active === this.leaf || this.sourceLeaves.has(active));
    this.clock.tick(performance.now(), this.phase(), !!this.data?.session && this.data.session.stage !== "complete" && this.running && visible);
  }
  async flush() {
    if (!this.data?.session || this.data.session.stage === "complete") return;
    if (this.flushing) return this.flushing;
    this.flushing = (async () => {
      this.tick();
      const budget = { ...this.clock.pending };
      // Keep exactly the same event on retry after an unknown write outcome.
      const snapshot = JSON.parse(JSON.stringify(this.draft));
      const events = this.pendingCheckpoints || Object.entries(this.clock.pending).filter(([, n]) => n > 0)
        .flatMap(([phase, seconds]) => {
          const chunks = [];
          while (seconds > 0) {
            const amount = Math.min(seconds, 300); seconds -= amount;
            chunks.push({ op: "checkpoint", session_id: this.data.session.id, event_id: randomUUID(), phase, seconds: amount,
              draft: snapshot, paused: !this.running });
          }
          return chunks;
        });
      if (!events.length) events.push({ op: "checkpoint", session_id: this.data.session.id, event_id: randomUUID(), phase: this.phase(), seconds: 0,
        draft: snapshot, paused: !this.running });
      this.pendingCheckpoints = events;
      while (events.length) {
        const event = events[0], result = await this.call(event);
        this.clock.saved(event.phase, event.seconds); events.shift();
        budget[event.phase] = Math.max(0, (budget[event.phase] || 0) - event.seconds);
        if (result.timing) this.data.timing = result.timing;
        if (!events.length) {
          for (const [phase, value] of Object.entries(budget)) {
            let remaining = value;
            while (remaining > 0) {
              const seconds = Math.min(remaining, 300); remaining -= seconds;
              events.push({ op: "checkpoint", session_id: this.data.session.id, event_id: randomUUID(), phase, seconds,
                draft: JSON.parse(JSON.stringify(this.draft)), paused: !this.running });
            }
          }
        }
        if (!events.length && (JSON.stringify(event.draft) !== JSON.stringify(this.draft) || event.paused !== !this.running)) {
          // A saved-but-unacknowledged event may predate edits made during recovery.
          events.push({ op: "checkpoint", session_id: this.data.session.id, event_id: randomUUID(), phase: this.phase(), seconds: 0,
            draft: JSON.parse(JSON.stringify(this.draft)), paused: !this.running });
        }
      }
      this.pendingCheckpoints = null;
    })().finally(() => { this.flushing = null; });
    return this.flushing;
  }
  async action(op, args = {}) {
    if (this.busy) return;
    this.tick();
    const wasRunning = this.running;
    this.busy = true; this.updateClock(); this.error = "";
    this.contentEl.querySelectorAll("button").forEach(button => { button.disabled = true; });
    try {
      await this.flush();
      const result = await this.call({ op, ...(this.data?.session ? { session_id: this.data.session.id } : {}), ...args });
      this.tick();
      this.data = result; this.draft = result.draft || {};
      this.running = op === "start" || wasRunning;
      if (result.session?.stage === "complete") {
        this.running = false; this.clock = new ReviewClock();
        if (!this.transport) void this.plugin.steward.refresh().catch(() => {});
      }
      this.render();
    } catch (error) {
      this.error = error.message;
      if (this.data?.session) {
        try {
          const saved = await this.call({ op: "get", session_id: this.data.session.id });
          if (saved.session?.stage !== this.data.session.stage || saved.session?.current_task_id !== this.data.session.current_task_id) {
            this.data = saved; this.draft = saved.draft || {};
          }
        } catch { /* The in-memory draft remains available for a retry. */ }
      }
      this.render();
    }
    finally { this.busy = false; this.updateClock(); }
  }
  showError(error) {
    this.error = error.message;
    if (this.errorEl) { this.errorEl.setText(`保存需要重试：${error.message}`); this.errorEl.hidden = false; }
  }
  button(parent, text, action, cls = "", op = "") {
    const button = parent.createEl("button", { text, cls });
    if (op) button.dataset.reviewAction = op;
    button.addEventListener("click", () => Promise.resolve().then(action).catch(error => this.showError(error)));
    return button;
  }
  prose(parent, text, cls = "sor-prose") { return parent.createEl("p", { text: text || "", cls }); }
  render() {
    this.contentEl.empty();
    const header = this.contentEl.createDiv({ cls: "sor-header" });
    const heading = header.createDiv(); heading.createEl("div", { cls: "sor-eyebrow", text: "STUDENT OS / REVIEW" });
    heading.createEl("h1", { text: "把学过的课用出来" });
    const nav = header.createDiv({ cls: "sor-actions" });
    this.button(nav, "返回我的学习", async () => {
      this.tick(); this.running = false; await this.flush(); await this.plugin.openSteward();
    });
    if (this.data?.session) this.button(nav, "复习首页", () => this.loadCatalog());
    this.errorEl = this.contentEl.createDiv({ cls: "sor-error", attr: { role: "alert" }, text: this.error || "" });
    this.errorEl.hidden = !this.error;
    if (!this.data?.session) { this.renderCatalog(); return; }
    const session = this.data.session;
    this.prose(this.contentEl, this.data.title || this.catalog?.lessons?.find(l => l.lesson_id === session.lesson_id)?.title || "ST445 第一讲 · 数据、类型与结构", "sor-subtitle");
    const steps = this.contentEl.createDiv({ cls: "sor-steps", attr: { "aria-label": "复习流程" } });
    for (const [label, match] of [["独立尝试", "question"], ["核对结果", "feedback"], ["回看 · 换题", "repair"], ["用时与后续", "complete"]]) {
      const active = session.stage === match || match === "repair" && session.phase === "retest";
      steps.createSpan({ text: label, cls: active ? "is-active" : "" });
    }
    const clock = this.contentEl.createDiv({ cls: "sor-clock" });
    this.clockEl = clock.createSpan();
    this.pauseButton = this.button(clock, "", async () => { this.tick(); this.running = !this.running; await this.flush(); this.updateClock(); }, "", "pause");
    this.pauseButton.hidden = session.stage === "complete";
    this.prose(clock, "记录当前复习及其讲义窗口的可见用时；可暂停，离开应用和休眠不补算。", "sor-muted");
    const issues = this.data.source_issues || [];
    if (issues.length) this.prose(this.contentEl, "材料版本已变化，作答暂停。已有答案保留，需更新题包后继续。", "sor-error");
    const main = this.contentEl.createDiv({ cls: "sor-body" });
    if (session.stage === "question") this.renderQuestion(main);
    else if (session.stage === "repair") this.renderRepair(main);
    else if (session.stage === "complete") this.renderReport(main);
    else this.renderFeedback(main);
    if (issues.length) main.querySelectorAll("button,input,select,textarea").forEach(el => { el.disabled = true; });
    this.updateClock();
  }
  updateClock() {
    if (!this.clockEl || !this.data?.session) return;
    const elapsed = totalTime(this.data.timing) + totalTime(this.clock.pending);
    this.clockEl.setText(`${minutes(elapsed)} · ${this.busy ? "正在保存" : this.running ? PHASES[this.phase()] || "复习" : "已暂停"}`);
    this.pauseButton?.setText(this.running ? "暂停" : "继续计时");
  }
  renderCatalog() {
    this.prose(this.contentEl, "先独立完成情境任务，再统一核对。需要时回到指定讲义，修补后换题重测；之后再检查能否独立使用。", "sor-subtitle");
    for (const lesson of this.catalog?.lessons || []) {
      const card = this.contentEl.createEl("section", { cls: "sor-card" });
      card.createEl("h2", { text: lesson.title });
      this.prose(card, lesson.description || "第一讲试点 · 检查数据表示、处理选择及解释。" );
      const list = card.createEl("ul");
      for (const goal of lesson.objectives || []) list.createEl("li", { text: goal.label });
      const due = lesson.due_at ? new Date(lesson.due_at).toLocaleString("zh-CN", { timeZone: "Europe/London", month: "long", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "";
      const active = lesson.active_session_id;
      const available = active || !lesson.due_at || new Date(lesson.due_at) <= new Date();
      const exhausted = ["needs_preparation", "exhausted", "inventory_exhausted"].includes(lesson.status);
      this.prose(card, active ? "上次答案和停点已保存，可以继续。" : exhausted ? "本试点的新题库存已用完；需要准备下一批，保留已有记录。" : due ? `下次延迟复习：${due}（伦敦时间）` : "首次作答后显示实际用时。检测完成后可按需要修补，也可以保存待学习项。", "sor-muted");
      const actions = card.createDiv({ cls: "sor-actions" });
      const start = this.button(actions, active ? "继续这次复习" : due ? "开始延迟复习" : "开始第一讲检测", () => this.action("start", { lesson_id: lesson.lesson_id }), "mod-cta", "start");
      start.disabled = !available || exhausted || !!lesson.source_issues?.length;
      if (lesson.latest_report) this.button(actions, "查看上次用时与结果", () => this.openPath(lesson.latest_report), "", "last-report");
      if (lesson.source_issues?.length) this.prose(card, "课程材料有更新，题包需要重新核验。", "sor-error");
      if (lesson.coverage_limits?.length) {
        const details = card.createEl("details"); details.createEl("summary", { text: "本次检查的范围" });
        lesson.coverage_limits.forEach(text => this.prose(details, text, "sor-muted"));
      }
    }
  }
  renderQuestion(parent) {
    const task = this.data.task, session = this.data.session;
    if (!task) { this.prose(parent, "题目尚未读取，请返回复习首页继续。"); return; }
    if (this.draft.task_id !== task.id) this.draft = { task_id: task.id, answers: {}, ratings: this.draft.ratings || {} };
    this.draft.answers ||= {};
    const card = parent.createEl("section", { cls: "sor-card" });
    this.prose(card, `${PHASES[session.phase] || "独立尝试"}${session.task_count ? ` · ${Math.min((session.answered_count || 0) + 1, session.task_count)} / ${session.task_count}` : ""}`, "sor-eyebrow");
    card.createEl("h2", { text: task.title || task.scenario });
    if (task.title && task.scenario) this.prose(card, task.scenario, "sor-prompt");
    this.prose(card, task.prompt, "sor-prompt");
    this.prose(card, "先留下自己的判断和理由。提交后进入下一题，完成这一轮再核对。", "sor-muted");
    const form = card.createEl("form");
    for (const item of task.items) {
      const row = form.createEl("label", { cls: "sor-field" }); row.createSpan({ text: item.prompt });
      let input;
      if (item.kind === "choice") {
        input = row.createEl("select", { attr: { required: "", "aria-label": item.prompt } });
        input.createEl("option", { text: "选择你的判断…", attr: { value: "" } });
        for (const option of item.options || []) input.createEl("option", { text: option.label, attr: { value: option.value } });
      } else if (item.kind === "explain_self") {
        input = row.createEl("textarea", { attr: { rows: "3", required: "", placeholder: "用自己的话说明；不知道也可以直接写不知道。", "aria-label": item.prompt } });
      } else {
        input = row.createEl("input", { attr: { type: "text", required: "", "aria-label": item.prompt, ...(item.kind === "number" ? { inputmode: "decimal" } : {}) } });
      }
      input.dataset.itemId = item.id; input.value = this.draft.answers[item.id] ?? "";
      input.addEventListener("input", () => { this.draft.answers[item.id] = input.value; });
      input.addEventListener("change", () => { this.draft.answers[item.id] = input.value; });
    }
    const submit = form.createEl("button", { text: "提交，继续", cls: "mod-cta", attr: { type: "submit" } });
    submit.dataset.reviewAction = "submit";
    form.addEventListener("submit", event => { event.preventDefault(); if (form.reportValidity()) void this.action("submit", { task_id: task.id, answers: { ...this.draft.answers } }); });
  }
  renderFeedback(parent) {
    parent.createEl("h2", { text: "这一轮，哪些已经能用？" });
    this.prose(parent, "先对照自己的解释。选择题由程序核对，解释依据清单由你自评；两种证据分别保存。", "sor-muted");
    const actions = this.data.actions || {};
    for (const entry of this.data.feedback || []) {
      const task = entry.task;
      const card = parent.createEl("section", { cls: `sor-card ${entry.needs_repair ? "needs-repair" : ""}` });
      card.dataset.feedbackTask = task.id;
      card.createEl("h3", { text: task.title || task.scenario });
      const status = entry.needs_self_grade ? "解释待核对" : entry.family_status === "needs_study" ? "留待正式学习" : entry.family_status === "needs_verification" ? "稍后用新题确认" : entry.needs_repair ? "需要修补" : entry.needs_retest ? "需要用新题确认" : entry.prompt_exposure ? "原题存在提示 · 已保留说明" : entry.status === "failed" ? "原尝试有缺口 · 已在后续处理" : "本次已通过核对";
      this.prose(card, status, "sor-badge");
      if (entry.prompt_exposure) this.prose(card, `题面提示影响：${entry.prompt_exposure.reason}。原作答保留，独立表现以后续新题为准。`, "sor-muted");
      this.draft.ratings ||= {}; this.draft.ratings[task.id] ||= {};
      for (const item of task.items) {
        const result = entry.results[item.id] || {};
        const row = card.createDiv({ cls: "sor-result" }); row.createEl("strong", { text: item.prompt });
        const original = item.options?.find(o => o.value === result.answer)?.label ?? result.answer;
        this.prose(row, `你的作答：${original ?? "未作答"}`);
        if (item.kind === "explain_self") {
          const list = row.createEl("ul");
          (item.rubric || []).forEach(text => list.createEl("li", { text }));
          if (result.correct === null || result.correct === undefined) {
            const label = row.createEl("label", { cls: "sor-field" }); label.createSpan({ text: "对照以上标准，你原来的解释是否覆盖关键理由？" });
            const select = label.createEl("select", { attr: { "aria-label": "解释自评" } }); select.dataset.ratingItem = item.id;
            select.createEl("option", { text: "选择…", attr: { value: "" } });
            select.createEl("option", { text: "覆盖了，理由能够对应这个情境", attr: { value: "yes" } });
            select.createEl("option", { text: "遗漏、说不清或不确定，需要修补", attr: { value: "no" } });
            select.value = this.draft.ratings[task.id][item.id] === true ? "yes" : this.draft.ratings[task.id][item.id] === false ? "no" : "";
            select.addEventListener("change", () => {
              if (select.value) this.draft.ratings[task.id][item.id] = select.value === "yes";
              else delete this.draft.ratings[task.id][item.id];
            });
          } else this.prose(row, result.correct ? "你自评：关键理由已覆盖" : "你自评：仍需修补", "sor-muted");
        } else {
          const expected = item.options ? item.options.filter(o => (item.accepted || []).includes(o.value)).map(o => o.label).join(" / ") : (item.accepted || []).join(" / ");
          this.prose(row, `${result.correct ? "✓ 程序核对通过" : "需要修正"}${expected ? ` · 参考判断：${expected}` : ""}`, result.correct ? "sor-pass" : "sor-fail");
        }
        if (item.explanation) this.prose(row, item.explanation, "sor-muted");
      }
      const buttons = card.createDiv({ cls: "sor-actions" });
      if (entry.needs_self_grade) this.button(buttons, "保存解释自评", () => {
        const ratings = this.draft.ratings[task.id];
        const pending = task.items.filter(i => i.kind === "explain_self" && entry.results[i.id]?.correct == null);
        if (pending.some(i => typeof ratings[i.id] !== "boolean")) throw new Error("请先核对这道任务的解释。");
        return this.action("self-grade", { task_id: task.id, ratings });
      }, "mod-cta", "self-grade");
      if (actions.can_repair?.includes(task.id)) {
        this.button(buttons, "回到讲义，修补这里", () => this.action("repair", { task_id: task.id }), "mod-cta", "repair");
        this.button(buttons, "留待正式学习", () => this.action("defer", { task_id: task.id, reason: "本人选择稍后作为学习任务修补" }), "", "defer");
      }
      if (entry.needs_retest && actions.can_retest?.includes(task.id)) {
        this.button(buttons, "换一道新题，确认能否独立完成", async () => {
          await this.hideSources(); await this.action("retest", { task_id: task.id });
        }, "mod-cta", "retest");
        this.button(buttons, "稍后再确认", () => this.action("defer", { task_id: task.id, reason: "原题存在提示，稍后用新题确认" }), "", "defer");
      }
    }
    if (actions.can_finish) this.button(parent, "结束本次，查看用时与后续", () => this.action("finish"), "mod-cta sor-finish", "finish");
    else this.prose(parent, "核对解释后，选择修补或保存待学习项，即可结束本次。", "sor-muted");
  }
  renderRepair(parent) {
    const repair = this.data.repair;
    const card = parent.createEl("section", { cls: "sor-card" });
    card.createEl("h2", { text: "带着这个问题回到材料" });
    for (const route of repair?.routes || []) {
      const row = card.createDiv({ cls: "sor-source" });
      this.prose(row, route.repair_prompt);
      const source = route.source || route.source_ref;
      const path = typeof source === "object" ? source.path : route.path;
      if (path) {
        const target = path + (route.page ? `#page=${route.page}` : route.note_heading ? `#${route.note_heading}` : "");
        this.button(row, route.page ? `打开讲义第 ${route.page} 页` : "打开对应材料", () => this.openPath(target, true), "", "source");
      }
    }
    this.prose(card, "回看时找出自己原判断第一次出错的位置。读完后关闭材料，再完成一个不同任务。", "sor-muted");
    this.draft.repair_notes ||= "";
    const label = card.createEl("label", { cls: "sor-field" }); label.createSpan({ text: "这次修补了什么？（可选）" });
    const input = label.createEl("textarea", { attr: { rows: "3", placeholder: "例如：规则要按字段含义决定，不能全表替换。" } });
    input.value = this.draft.repair_notes; input.addEventListener("input", () => { this.draft.repair_notes = input.value; });
    const buttons = card.createDiv({ cls: "sor-actions" });
    this.button(buttons, "材料已收起，开始换题重测", async () => {
      await this.hideSources(); await this.action("retest", { task_id: repair.task_id });
    }, "mod-cta", "retest");
    this.button(buttons, "暂时没弄明白，留待正式学习", () => this.action("defer", { task_id: repair.task_id, reason: "回看后仍需进一步学习" }), "", "defer");
  }
  async hideSources() {
    for (const leaf of this.sourceLeaves) {
      if (leaf.parent && leaf.view.file?.path === leaf.reviewSourcePath) leaf.detach();
    }
    this.sourceLeaves.clear(); await this.app.workspace.revealLeaf(this.leaf);
    this.app.workspace.setActiveLeaf(this.leaf, { focus: true });
  }
  async openPath(target, source = false) {
    const file = this.app.vault.getAbstractFileByPath(target.split("#")[0]);
    if (!file) throw new Error("对应材料没有找到，当前答案已保留。");
    await this.flush();
    // A dedicated split does not move or close a user's existing reading tab.
    let leaf = source ? [...this.sourceLeaves].find(l => l.parent) : null;
    if (!leaf) leaf = this.app.workspace.getLeaf("split", "vertical");
    if (source) { this.sourceLeaves.add(leaf); leaf.reviewSourcePath = file.path; }
    await leaf.openLinkText(target, "", { active: true });
    await this.app.workspace.revealLeaf(leaf);
    if (source && !leaf.view.containerEl.querySelector(".sor-return")) {
      const action = leaf.view.addAction?.("rotate-ccw", "返回复习，保留当前讲义页", async () => {
        await this.app.workspace.revealLeaf(this.leaf); this.app.workspace.setActiveLeaf(this.leaf, { focus: true });
      });
      action?.addClass("sor-return"); action?.createSpan({ text: "返回复习" });
    }
  }
  renderReport(parent) {
    const card = parent.createEl("section", { cls: "sor-card" });
    card.createEl("h2", { text: "这次复习花在哪里" });
    const table = card.createEl("table"), body = table.createEl("tbody");
    for (const [key, label] of Object.entries(PHASES)) {
      const row = body.createEl("tr"); row.createEl("th", { text: label }); row.createEl("td", { text: minutes(this.data.timing?.[key]) });
    }
    const total = body.createEl("tr"); total.createEl("th", { text: "合计" }); total.createEl("td", { text: minutes(totalTime(this.data.timing)) });
    this.prose(card, "这是本次记录的可见用时估计，包含核对和修补；暂停、离开应用和休眠不计入。它不代表持续专注时长。", "sor-muted");
    const unresolved = [...new Map((this.data.feedback || []).filter(f => f.family_status === "needs_study").map(f => [f.task.family_id, f])).values()];
    if (unresolved.length) {
      card.createEl("h3", { text: "仍需学习" });
      const list = card.createEl("ul"); unresolved.forEach(f => list.createEl("li", { text: f.task.title || f.task.scenario }));
      this.prose(card, "这些缺口已保留；本轮结束并不表示已经修复。", "sor-muted");
    }
    const pending = [...new Map((this.data.feedback || []).filter(f => f.family_status === "needs_verification").map(f => [f.task.family_id, f])).values()];
    if (pending.length) {
      card.createEl("h3", { text: "仍待新题确认" });
      const list = card.createEl("ul"); pending.forEach(f => list.createEl("li", { text: f.task.title || f.task.scenario }));
    }
    if ((this.data.feedback || []).some(f => f.prompt_exposure)) this.prose(card, "本轮包含题面提示影响的尝试；原作答与后续换题结果分别保留在完整记录中。", "sor-muted");
    const schedule = this.data.schedule;
    this.prose(card, schedule?.due_at ? `下次检查：${new Date(schedule.due_at).toLocaleString("zh-CN", { timeZone: "Europe/London" })}（伦敦时间）。到时在 Student OS 的课程复习入口出现。`
      : "下一批新题需要准备；已有作答、用时和待学习项均保留。" );
    this.prose(card, "后续间隔是试点参数，随实际表现调整；尚未检验的课程目标会继续保留为未知。", "sor-muted");
    const buttons = card.createDiv({ cls: "sor-actions" });
    if (this.data.report_path) this.button(buttons, "打开完整记录", () => this.openPath(this.data.report_path), "", "report");
    this.button(buttons, "返回复习首页", () => this.loadCatalog(), "mod-cta");
  }
}

module.exports = { REVIEW_VIEW, ReviewView, ReviewClock, minutes, totalTime };
