const { Notice, Menu } = require("obsidian");
const catalog = require("./course-catalog");
const { courseMatrix, comparisonSummary, relatedCourses } = require("./course-projections");
const { PRIVATE_RECORD, BINDINGS, isSchoolCourse, unique, unitOutcomes, parseRecords, updateRecord, completionFor, learnedOutcomes, coverageFor, courseCoverage, knowledgeIndex, knowledgeSelection, comparisonCandidates, inDomain, graphLayout } = require("./course-model");

const ROLES = { core: "共同主干", branch: "方向深化", alternative: "替代课程" };
const STATUS = { covered: "所列主题已覆盖", partial: "部分重合", extension: "进一步学习", new: "暂无学习记录", unknown: "范围待核对", overview: "概览与复习" };
const MATERIALS = { video: "视频", videos: "视频", notes: "讲义", slides: "课件", exercises: "习题", assignments: "作业", homework: "作业", solutions: "答案", textbook: "教材", code: "代码", labs: "实验", readings: "阅读材料", papers: "论文", projects: "项目", project: "项目", exams: "试卷" };
const DEPTH = { "full-outline": "逐节目录与知识映射", "topic-outline": "主题模块与知识映射", syllabus: "课程大纲" };
const COLORS = ["#6f8ba5", "#8e83ad", "#66988c", "#ae9269", "#798ca4", "#69918f", "#a78196", "#8c9a6b", "#8f8ec1", "#648d9d", "#af887a"];
const day = () => new Intl.DateTimeFormat("en-CA", { timeZone: "Europe/London", year: "numeric", month: "2-digit", day: "2-digit" }).format(new Date());

function button(parent, text, action, cls = "") {
  const el = parent.createEl("button", { text, cls });
  el.addEventListener("click", e => { e.stopPropagation(); Promise.resolve().then(() => action(e)).catch(error => new Notice(error.message)); });
  return el;
}
function external(parent, text, url) {
  if (!/^https?:\/\/[^\s]+$/.test(url || "")) return;
  return parent.createEl("a", { text, cls: "cm-source", attr: { href: url, target: "_blank", rel: "noopener noreferrer" } });
}
function select(parent, label, entries, value, changed) {
  const field = parent.createEl("label", { cls: "cm-select" });
  field.createSpan({ text: label });
  const el = field.createEl("select", { attr: { "aria-label": label } });
  entries.forEach(([key, title]) => el.createEl("option", { value: key, text: title }));
  el.value = value;
  el.addEventListener("change", () => changed(el.value));
  return el;
}

class CourseMap {
  constructor(owner) {
    this.owner = owner; this.plugin = owner.plugin; this.app = owner.app;
    this.state = { mode: "map", role: "all", track: "both", domain: "all", query: "", course: null, comparison: "personal", comparisonSource: null, studyComparison: false, tab: "overview", overviewMode: "route", unit: null, module: null, focus: false, expandedMap: false, knowledgeFilter: "all", knowledge: null, knowledgeUnit: null, trace: null, coverageFilter: "all", ...owner.courseMapState };
    this.records = new Map(); this.definitions = new Map(catalog.outcomes.map(o => [o.id, o]));
    this.history = []; this.providers = new Map([...knowledgeIndex(catalog.courses, this.definitions).values()].map(entry => [entry.id, entry.units]));
    this.expanded = new Set(); this.loaded = false; this.mtime = -1;
    this.panePositions = new Map(); this.directoryPositions = new Map(); this.inspectorPositions = new Map(); this.matrixPositions = new Map();
  }
  getState() { return { ...this.state }; }
  readingKey(state = this.state) {
    return JSON.stringify([state.course, state.tab, state.tab === "units" ? state.unit : state.tab === "overview" ? state.module : null,
      state.comparison, state.tab === "coverage" ? state.coverageFilter : null, state.tab === "overview" ? state.overviewMode : null]);
  }
  inspectorKey(state = this.state) {
    return JSON.stringify([state.course, state.knowledge, state.knowledgeUnit, state.comparison]);
  }
  captureViewPosition() {
    if (!this.host || this.host.isConnected === false) return null;
    const state = this.renderedState || this.state;
    const directory = this.host.querySelector(".cm-directory-list");
    const pane = this.host.querySelector(".cm-coverage-chapters") || this.host.querySelector(".cm-lesson-pane") || this.host.querySelector(".cm-route-scroll") || this.host.querySelector(".cm-page-scroll");
    const inspector = this.host.querySelector(".cm-inspector-body");
    const matrix = this.host.querySelector(".cm-matrix-scroll");
    const graph = this.host.querySelector(".cm-graph-scroll");
    const reading = { pageTop: this.owner.contentEl?.scrollTop, directory: directory?.scrollTop,
      pane: pane?.scrollTop, inspector: inspector?.scrollTop, matrixLeft: matrix?.scrollLeft, matrixTop: matrix?.scrollTop };
    if (directory) this.directoryPositions.set(state.course, reading.directory);
    if (pane) this.panePositions.set(this.readingKey(state), reading.pane);
    if (inspector) this.inspectorPositions.set(this.inspectorKey(state), reading.inspector);
    if (matrix) this.matrixPositions.set(JSON.stringify([state.course, matrix.dataset.moduleId || state.module]), { left: reading.matrixLeft, top: reading.matrixTop });
    if (graph) this.scroll = { x: graph.scrollLeft, y: graph.scrollTop };
    return reading;
  }
  restoreViewPosition(reading = {}) {
    const directory = this.host.querySelector(".cm-directory-list");
    const pane = this.host.querySelector(".cm-coverage-chapters") || this.host.querySelector(".cm-lesson-pane") || this.host.querySelector(".cm-route-scroll") || this.host.querySelector(".cm-page-scroll");
    const inspector = this.host.querySelector(".cm-inspector-body");
    const matrix = this.host.querySelector(".cm-matrix-scroll");
    const directoryTop = reading.directory ?? this.directoryPositions.get(this.state.course);
    const paneTop = reading.pane ?? this.panePositions.get(this.readingKey());
    const inspectorTop = reading.inspector ?? this.inspectorPositions.get(this.inspectorKey());
    const matrixPosition = this.matrixPositions.get(JSON.stringify([this.state.course, matrix?.dataset.moduleId || this.state.module]));
    const matrixLeft = reading.matrixLeft ?? matrixPosition?.left;
    const matrixTop = reading.matrixTop ?? matrixPosition?.top;
    if (directory && Number.isFinite(directoryTop)) directory.scrollTop = directoryTop;
    else if (directory) this.revealSelectedUnit();
    if (pane && Number.isFinite(paneTop)) pane.scrollTop = paneTop;
    if (inspector && Number.isFinite(inspectorTop)) inspector.scrollTop = inspectorTop;
    if (matrix && Number.isFinite(matrixLeft)) matrix.scrollLeft = matrixLeft;
    if (matrix && Number.isFinite(matrixTop)) matrix.scrollTop = matrixTop;
  }
  navigationSnapshot() {
    return { state: this.getState(), pageTop: this.owner.contentEl?.scrollTop, reading: this.captureViewPosition() };
  }
  restoreNavigation(previous) {
    this.change(previous.state);
    this.restoreViewPosition(previous.reading || {});
    if (Number.isFinite(previous.pageTop)) this.owner.contentEl.scrollTop = previous.pageTop;
  }
  change(patch, detail = false) {
    this.captureViewPosition();
    if ("knowledge" in patch && !patch.knowledge) patch.knowledgeUnit = null;
    if (["domain", "track", "role", "query", "knowledgeFilter"].some(k => k in patch && patch[k] !== this.state[k])) {
      patch = { ...patch, course: null, unit: null, comparison: "personal", comparisonSource: null, studyComparison: false, knowledge: null, knowledgeUnit: null, trace: null };
    }
    if (["domain", "track", "role", "query", "focus"].some(k => k in patch && patch[k] !== this.state[k])) this.resetGraphScroll = true;
    if ("course" in patch && !patch.course) { this.history = []; this.overviewReturn = null; patch.comparisonSource = null; patch.studyComparison = false; patch.knowledge = null; patch.knowledgeUnit = null; patch.trace = null; }
    Object.assign(this.state, patch); this.owner.courseMapState = this.getState(); this.owner.saveWorkspace(); this.draw();
    if (detail && this.owner.contentEl) this.owner.contentEl.scrollTop = 0;
  }
  async readRecords() {
    if (this.loading) return;
    const file = this.app.vault.getAbstractFileByPath(PRIVATE_RECORD);
    const mtime = file?.stat?.mtime || 0;
    if (this.loaded && this.mtime === mtime) return;
    this.loading = true;
    try {
      const text = file ? await this.app.vault.read(file) : "";
      this.records = parseRecords(text, catalog); this.mtime = mtime; this.loaded = true; this.error = "";
    } catch (error) { this.error = "学习记录未能读取：" + error.message; }
    finally { this.loading = false; if (this.host?.isConnected) this.draw(); }
  }
  render(parent, curriculum) {
    this.captureViewPosition();
    this.curriculum = curriculum;
    this.host = parent.createEl("section", { cls: "sos-course-map", attr: { "aria-label": "自主学习课程地图" } });
    this.host.addEventListener("keydown", e => { if (e.key === "Escape" && this.state.knowledge) { e.stopPropagation(); this.closeKnowledge(); } });
    this.draw(); void this.readRecords();
  }
  draw() {
    if (!this.host) return;
    const pageTop = this.owner.contentEl?.scrollTop;
    this.captureViewPosition();
    if (this.resetGraphScroll) { this.scroll = { x: 0, y: 0 }; this.resetGraphScroll = false; }
    const taskHost = this.owner.taskDetail?.host;
    if (taskHost && this.host.contains(taskHost)) this.owner.taskDock.appendChild(taskHost);
    this.host.empty(); this.detailEl = null;
    this.learned = learnedOutcomes(catalog, this.curriculum, this.records, this.state.comparison);
    this.personal = learnedOutcomes(catalog, this.curriculum, this.records);
    const head = this.host.createDiv({ cls: "cm-heading" + (this.state.course ? " is-course-open" : "") });
    const title = head.createDiv(); title.createEl("h3", { text: "公开课与知识地图" });
    title.createEl("p", { text: "从课程出发，连接你已经学过的知识。" });
    const tools = head.createDiv({ cls: "cm-heading-tools" });
    tools.createSpan({ cls: "cm-catalog-count", text: `${catalog.courses.length} 门课程 · ${catalog.domains.length} 个领域` });
    const shortcuts = tools.createDiv({ cls: "cm-shortcuts" });
    for (const [domain, title] of [["all", "完整地图"], ["probability-bayes", "概率与贝叶斯"]]) {
      const b = button(shortcuts, title, () => this.change({ mode: "map", domain, track: "both", role: "all", query: "", focus: false, course: null, comparison: "personal", knowledgeFilter: "all" }), "cm-quiet");
      b.setAttribute("aria-pressed", String(this.state.mode === "map" && this.state.domain === domain));
    }
    if (this.error) this.host.createEl("p", { cls: "sos-error", text: this.error });
    const selected = catalog.courses.find(c => c.id === this.state.course);
    this.owner.contentEl?.classList.toggle("is-course-workspace", !!selected);
    this.host.classList.toggle("is-course-workspace", !!selected);
    if (selected) this.state.unit = this.selectedUnit(selected)?.id || null;
    this.renderContext(selected);
    if (selected) {
      this.renderDetail(selected);
      this.owner.placeTaskDetail?.();
      this.restoreViewPosition();
      this.renderedState = this.getState();
      if (Number.isFinite(pageTop)) this.owner.contentEl.scrollTop = pageTop;
      return;
    }
    this.renderResume();
    const toolbar = this.host.createDiv({ cls: "cm-toolbar" });
    const modes = toolbar.createDiv({ cls: "cm-modes" });
    for (const [id, label] of [["map", "课程路线"], ["list", "课程清单"], ["knowledge", "知识覆盖"]]) {
      const b = button(modes, label, () => this.change({ mode: id, course: null, comparison: "personal" }), this.state.mode === id ? "is-active" : "");
      b.setAttribute("aria-pressed", String(this.state.mode === id));
    }
    select(toolbar, "学习路径", [["both", "研究与就业"], ["research", "研究 / 博士"], ["engineering", "工程就业"], ["data", "数据科学"]], this.state.track, track => this.change({ track }));
    select(toolbar, "课程作用", [["all", "全部课程"], ...Object.entries(ROLES)], this.state.role, role => this.change({ role }));
    select(toolbar, "领域", [["all", "全部领域"], ["probability-bayes", "概率与贝叶斯专题"], ...catalog.domains.map(d => [d.id, d.title])], this.state.domain, domain => this.change({ domain }));
    const search = toolbar.createEl("input", { cls: "cm-search", attr: { type: "search", placeholder: "课程、学校、主题…", "aria-label": "搜索公开课程" } });
    search.value = this.state.query;
    search.addEventListener("input", () => {
      clearTimeout(this.searchTimer);
      this.searchTimer = setTimeout(() => {
        const position = search.selectionStart; this.change({ query: search.value });
        const replacement = this.host.querySelector(".cm-search"); replacement?.focus();
        try { replacement?.setSelectionRange(position, position); } catch {}
      }, 220);
    });
    if (this.state.domain === "probability-bayes" && this.state.mode === "map") this.renderProbabilityRoute();
    const courses = this.visibleCourses();
    if (this.state.mode === "map") this.renderGraph(courses);
    else if (this.state.mode === "knowledge") this.renderKnowledge(courses);
    else this.renderList(courses);
    const foot = this.host.createDiv({ cls: "cm-map-foot" });
    foot.createSpan({ text: this.state.mode === "knowledge" ? "亮点来自本人确认的学习记录；相同知识范围跨课程共用。" : "实线：官方先修 · 虚线：根据课程内容推断的衔接" });
    foot.createSpan({ text: "目录收录不等于全部必修；替代课程可择一。" });
    this.renderOtherTracks();
    this.owner.placeTaskDetail?.();
    this.renderedState = this.getState();
    if (Number.isFinite(pageTop)) this.owner.contentEl.scrollTop = pageTop;
  }
  visibleCourses() {
    const q = this.state.query.trim().toLowerCase();
    let visible = catalog.courses.filter(c =>
      (this.state.role === "all" || c.role === this.state.role) &&
      (this.state.track === "both" || (c.tracks || []).includes(this.state.track)) &&
      inDomain(c, this.state.domain) &&
      (!q || [c.code, c.title, c.titleZh, c.school, c.purpose, ...(c.summary || []),
        ...(c.units || []).flatMap(u => [u.title, u.titleZh, ...unitOutcomes(u).flatMap(id => { const o = this.definitions.get(id); return [o?.title, o?.scope]; })])].join(" ").toLowerCase().includes(q)));
    if (this.state.focus && this.state.course) {
      const family = new Set([this.state.course]);
      for (const c of catalog.courses) for (const p of c.prerequisites || []) {
        if (c.id === this.state.course) family.add(p.id);
        if (p.id === this.state.course) family.add(c.id);
      }
      visible = visible.filter(c => family.has(c.id));
    }
    return visible;
  }
  renderProbabilityRoute() {
    const route = this.host.createEl("details", { cls: "cm-probability-route" });
    route.createEl("summary", { text: "概率 → 统计推断 → 贝叶斯建模 · 查看选课说明" });
    route.open = this.probabilityRouteOpen || false;
    route.addEventListener("toggle", () => { this.probabilityRouteOpen = route.open; });
    const steps = route.createDiv({ cls: "cm-route-steps" });
    for (const [id, label] of [["mit-6041sc", "概率主干 · MIT 6.041SC"], ["mit-1805", "推断入门 · MIT 18.05"], ["aalto-bda", "建模与计算 · Aalto BDA"]]) {
      if (catalog.courses.some(c => c.id === id)) button(steps, label, () => this.openCourse(id), "cm-quiet");
    }
    route.createEl("p", { text: "按能力选章节衔接：条件概率 → 随机变量、期望与联合分布 → 似然、先验与后验 → 计算推断、模型检验。入门概率可选 Stat110；已具备推断基础时可直接核对 Aalto 的先修要求。" });
    route.createEl("p", { text: "统计理论接 18.650 / STATS200；STA601 还要求数理统计与 R。18.675 属于有实分析基础后的研究分支。课程按钮提供可选学习来源，无须整条路线逐门修完。" });
  }
  renderKnowledge(courses) {
    const controls = this.host.createDiv({ cls: "cm-knowledge-controls" });
    select(controls, "知识范围", [["all", "全部知识点"], ["learned", "已有学习记录"], ["unrecorded", "尚无学习记录"], ["shared", "跨课程共用"]], this.state.knowledgeFilter, knowledgeFilter => this.change({ knowledgeFilter }));
    const entries = knowledgeSelection(catalog, this.state, this.personal);
    controls.createSpan({ text: `${entries.length} 个已核知识范围 · ${entries.filter(e => this.personal.has(e.id)).length} 个有学习记录` });
    const unresolved = courses.reduce((n, c) => n + c.units.filter(u => u.mapping === "syllabus").length, 0);
    if (unresolved) controls.createSpan({ text: `当前课程另有 ${unresolved} 个单元的范围待核对` });
    if (!entries.length) this.host.createEl("p", { cls: "cm-empty", text: "此筛选下没有已映射知识点。尚无学习记录不等于尚未掌握。" });
    for (const domain of catalog.domains) {
      const own = entries.filter(e => e.domain === domain.id); if (!own.length) continue;
      const section = this.host.createEl("section", { cls: "cm-knowledge-domain" });
      section.createEl("h4", { text: `${domain.title} · ${own.length}` });
      const grid = section.createDiv({ cls: "cm-knowledge-grid" });
      for (const entry of own) {
        const outcome = this.definitions.get(entry.id), learned = this.personal.has(entry.id), key = "knowledge/" + entry.id;
        const item = grid.createEl("details", { cls: "cm-knowledge-card" + (learned ? " is-lit" : ""), attr: { "data-knowledge-id": entry.id } });
        const summary = item.createEl("summary");
        summary.createSpan({ cls: "cm-light", text: learned ? "●" : "○" });
        summary.createEl("strong", { text: outcome.title });
        summary.createSpan({ cls: "cm-knowledge-count", text: `${entry.courses.length} 门课` });
        const fill = () => {
          if (item.querySelector(".cm-knowledge-body")) return;
          const body = item.createDiv({ cls: "cm-knowledge-body" });
          this.renderOutcome(body, entry.id, this.personal, false);
          body.createEl("small", { text: "在哪些课程学习" });
          for (const ref of entry.units) {
            const c = catalog.courses.find(c => c.id === ref.courseId);
            button(body, `${c.school} ${c.code} · ${ref.title}`, () => this.openUnit(ref.courseId, ref.unitId, entry.id), "cm-inline-link");
          }
          for (const id of outcome.extends || []) {
            const base = this.definitions.get(id);
            if (base) body.createEl("p", { cls: "cm-hint", text: `在「${base.title}」基础上深化；基础知识点亮不代表此范围已学。` });
          }
        };
        item.open = this.expanded.has(key); if (item.open) fill();
        item.addEventListener("toggle", () => { if (item.open) { this.expanded.add(key); fill(); } else this.expanded.delete(key); });
      }
    }
  }
  openUnit(courseId, unitId, knowledgeId) {
    const course = catalog.courses.find(c => c.id === courseId), unit = course?.units.find(u => u.id === unitId);
    if (!unit) return;
    const origin = knowledgeId || this.host.querySelector(".cm-knowledge-card:focus-within")?.dataset.knowledgeId;
    if (origin) this.knowledgeReturn = origin;
    this.expanded.add(`${courseId}|${unitId}`); this.expanded.add(courseId + "/" + (unit.module || "课程内容"));
    this.openCourse(courseId, unitId, { preserveComparison: courseId === this.state.course && this.state.tab === "coverage" });
  }
  progress(course) {
    const registered = this.curriculum?.courses?.find(c => c.id === BINDINGS[course.id]);
    if (!["mit-1806sc", "mit-6041sc"].includes(course.id) && registered?.unit_progress?.total) return { done: registered.unit_progress.completed, total: registered.unit_progress.total, original: true, registered };
    return { done: (course.units || []).filter(u => completionFor(course, u, this.curriculum, this.records).done).length, total: (course.units || []).length, registered };
  }
  renderResume() {
    const active = catalog.courses.filter(c => this.progress(c).registered?.active || this.progress(c).done > 0);
    const strip = this.host.createDiv({ cls: "cm-resume" });
    strip.createSpan({ cls: "cm-resume-label", text: "我的接续点" });
    for (const course of active) {
      const p = this.progress(course), next = (course.units || []).find(u => !completionFor(course, u, this.curriculum, this.records).done);
      const kind = course.unitLabel || (course.coverage === "full-outline" ? "节" : "模块");
      const complete = p.total > 0 && p.done === p.total;
      const original = p.original || (next && completionFor(course, next, this.curriculum, this.records).unavailable);
      let label, resume;
      if (complete) {
        label = `${course.code} · ${p.original ? "原计划" : kind === "模块" ? "所列模块" : "课程单元"}已完成 · 查看全景`;
        resume = () => this.openCourse(course.id);
      } else if (original && p.registered?.plan) {
        label = `${course.code} · 继续原学习计划 · ${p.done}/${p.total}`;
        resume = () => this.owner.openOriginal(p.registered.plan);
      } else if (next && !original) {
        label = `${course.code} · 下一${kind} ${this.unitLabel(course, next)} ${this.unitTitle(next)}`;
        resume = () => this.openCourse(course.id, next.id);
      } else {
        label = `${course.code} · 查看课程全景`;
        resume = () => this.openCourse(course.id);
      }
      const entry = button(strip, label, resume, "cm-resume-course");
      entry.dataset.resumeCourse = course.id;
    }
    if (!active.length) strip.createSpan({ text: "选一门课，查看内容与已有基础。" });
    strip.createSpan({ cls: "cm-learned-total", text: `${this.personal.size} 个知识主题已有学习记录` });
  }
  openCourse(id, unit = null, options = {}) {
    const course = catalog.courses.find(c => c.id === id); if (!course) return;
    const explicitUnit = unit && course.units.find(candidate => candidate.id === unit);
    if (unit && !explicitUnit) return;
    const tab = options.tab || (unit ? "units" : "overview");
    const preserveComparison = !!options.preserveComparison && id === this.state.course;
    const comparison = preserveComparison ? this.state.comparison : "personal";
    const comparisonSource = preserveComparison ? this.state.comparisonSource : options.comparisonSource || null;
    unit = unit || (id === this.state.course ? this.state.unit : null)
      || course.units.find(u => !completionFor(course, u, this.curriculum, this.records).done)?.id || course.units[0]?.id || null;
    if (id !== this.state.course || unit !== this.state.unit || this.state.tab !== tab || comparison !== this.state.comparison) {
      const previous = this.navigationSnapshot();
      if (!this.state.course) this.overviewReturn = previous;
      this.history.push(previous); if (this.history.length > 30) this.history.shift();
    }
    this.change({ course: id, unit, tab, module: explicitUnit ? explicitUnit.module || "课程内容" : id === this.state.course ? this.state.module : null, comparison, comparisonSource, studyComparison: preserveComparison, knowledge: null, knowledgeUnit: null, trace: null,
      coverageFilter: preserveComparison ? this.state.coverageFilter : "all" }, true);
  }
  openComparison(targetCourseId, sourceCourseId = this.state.course) {
    const source = catalog.courses.find(c => c.id === sourceCourseId && c.id !== targetCourseId);
    this.openCourse(targetCourseId, null, { tab: "coverage", comparisonSource: source?.id || null });
  }
  get comparisonReturn() {
    if (!this.state.studyComparison) return null;
    return this.history.findLast(entry => entry.state.course === this.state.course && entry.state.tab === "coverage")
      || (this.state.tab === "units" && (this.state.comparison !== "personal" || this.state.comparisonSource) ? { state: { ...this.state, tab: "coverage" } } : null);
  }
  returnToComparison() {
    const index = this.history.findLastIndex(entry => entry.state.course === this.state.course && entry.state.tab === "coverage");
    if (index < 0) { this.change({ tab: "coverage", studyComparison: false, knowledge: null, knowledgeUnit: null, trace: null }); return; }
    const previous = this.history[index]; this.history.splice(index);
    this.restoreNavigation(previous);
  }
  backCourse() {
    const previous = this.history.pop();
    if (!previous) return this.returnToOverview();
    this.restoreNavigation(previous);
  }
  returnToOverview() {
    const previous = this.overviewReturn;
    this.change({ ...(previous?.state || {}), course: null, unit: null, comparison: "personal" });
    if (Number.isFinite(previous?.pageTop)) this.owner.contentEl.scrollTop = previous.pageTop;
    else this.host.scrollIntoView({ block: "start", behavior: "instant" });
  }
  renderContext(course) {
    const context = this.host.createDiv({ cls: "sos-learning-context cm-context", attr: { "aria-label": "校外学习当前位置" } });
    context.createSpan({ text: "校外学习" }); context.createSpan({ text: "›", cls: "cm-context-separator" });
    const label = this.state.mode === "knowledge" ? "知识覆盖" : this.state.mode === "list" ? "课程清单" : "课程地图";
    if (course) {
      button(context, "返回" + label, () => this.state.mode === "knowledge" ? this.returnToKnowledge() : this.returnToOverview(), "cm-inline-link");
      context.createSpan({ text: "›", cls: "cm-context-separator" });
      context.createEl("strong", { text: `${course.school} ${course.code}` });
      const viewLabel = { overview: "课程全景", coverage: "跨课对照", routes: "先修与去向" };
      context.createSpan({ text: `› ${this.state.tab === "units" ? this.unitLabel(course, this.selectedUnit(course)) : viewLabel[this.state.tab] || "课程全景"}` });
      if (this.history.length && this.history[this.history.length - 1].state.course) {
        const state = this.history[this.history.length - 1].state, previous = catalog.courses.find(c => c.id === state.course);
        const location = state.tab === "units" ? state.unit : viewLabel[state.tab] || "课程全景";
        button(context, `← 返回 ${previous?.code || "上一课程"} · ${location}`, () => this.backCourse(), "cm-quiet cm-history-back");
      }
    } else context.createEl("strong", { text: label });
    const editingKey = this.owner.taskDetail?.rowKey;
    if (editingKey?.startsWith("map-")) {
      const [courseId, unitId] = editingKey.slice(4).split("|");
      button(context, "返回正在填写的记录", () => this.openUnit(courseId, unitId), "cm-quiet");
    }
  }
  metricValues(course, learned = this.personal) {
    const progress = this.progress(course), coverage = courseCoverage(course, learned, this.definitions);
    const label = course.unitLabel || (course.coverage === "full-outline" ? "节" : "模块");
    return { progress, coverage, label };
  }
  renderMetrics(parent, course, compact = false, learned = this.personal) {
    const { progress: p, coverage, label } = this.metricValues(course, learned);
    const metrics = parent.createDiv({ cls: compact ? "cm-metrics is-compact" : "cm-metrics" });
    for (const [kind, title, count, total, suffix] of [
      ["course", label === "模块" ? "已学模块" : "本课已学", p.done, p.total, label],
      ["knowledge", learned === this.personal ? "知识已学" : "对照覆盖", coverage.known.length, coverage.total, "知识点"]
    ]) {
      const metric = metrics.createDiv({ cls: "cm-metric is-" + kind + (count ? " has-progress" : "") });
      metric.createSpan({ text: title });
      metric.createEl("strong", { text: `${count} / ${total}${compact ? "" : " " + suffix}` });
      if (!compact) {
        const meter = metric.createEl("progress", { attr: { max: Math.max(total, 1), value: count, "aria-label": `${title} ${count}/${total} ${suffix}` } });
        meter.value = count;
      }
    }
    return metrics;
  }
  returnToKnowledge() {
    this.returnToOverview();
    const origin = [...this.host.querySelectorAll("[data-knowledge-id]")].find(el => el.dataset.knowledgeId === this.knowledgeReturn);
    (origin || this.host).scrollIntoView({ block: origin ? "center" : "start", behavior: "instant" });
  }
  renderGraph(courses) {
    const controls = this.host.createDiv({ cls: "cm-graph-controls" });
    controls.createSpan({ text: `${courses.length} 门显示 · 横向按先修展开，纵向按领域分组` });
    const expand = button(controls, this.state.expandedMap ? "收起为大视窗" : "展开全图", () => this.change({ expandedMap: !this.state.expandedMap }), "cm-inline-link");
    expand.setAttribute("aria-expanded", String(this.state.expandedMap));
    if (!courses.length) { this.host.createDiv({ cls: "cm-empty", text: "没有匹配课程，调整筛选即可查看完整目录。" }); return; }
    const full = graphLayout(catalog.courses), ids = new Set(courses.map(c => c.id));
    const stages = unique(courses.map(c => full.ranks.get(c.id))).sort((a, b) => a - b);
    const columns = new Map(stages.map((s, i) => [s, i]));
    const width = Math.max(760, stages.length * 202 + 100), pos = new Map(), lanes = [];
    let y = 43;
    for (const domain of catalog.domains) {
      const items = courses.filter(c => c.domain === domain.id); if (!items.length) continue;
      const used = new Map();
      for (const course of items) {
        const x = columns.get(full.ranks.get(course.id)), n = used.get(x) || 0;
        pos.set(course.id, { x: 97 + x * 202, y: y + 23 + n * 122 }); used.set(x, n + 1);
      }
      const height = Math.max(...used.values()) * 122 + 29;
      lanes.push({ domain, y, height }); y += height;
    }
    const scroller = this.host.createDiv({ cls: "cm-graph-scroll" + (this.state.expandedMap ? " is-expanded" : ""), attr: { tabindex: "0", "aria-label": this.state.expandedMap ? "课程依赖全图，随页面浏览" : "课程依赖图，可横向与纵向滚动" } });
    const world = scroller.createDiv({ cls: "cm-graph-world" }); world.style.width = width + "px"; world.style.height = y + "px";
    stages.forEach((stage, i) => { const label = world.createDiv({ cls: "cm-stage", text: stage === 0 ? "起步与独立入口" : `先修展开 ${stage}` }); label.style.left = (97 + i * 202) + "px"; });
    for (const lane of lanes) {
      const row = world.createDiv({ cls: "cm-lane" }); row.style.top = lane.y + "px"; row.style.height = lane.height + "px";
      row.style.setProperty("--cm-domain", COLORS[catalog.domains.indexOf(lane.domain) % COLORS.length]);
      row.createSpan({ cls: "cm-lane-label", text: lane.domain.title });
    }
    const ns = "http://www.w3.org/2000/svg", svg = document.createElementNS(ns, "svg");
    svg.setAttribute("width", width); svg.setAttribute("height", y); svg.setAttribute("class", "cm-edges"); svg.setAttribute("aria-label", "课程衔接关系"); world.appendChild(svg);
    const defs = document.createElementNS(ns, "defs"), marker = document.createElementNS(ns, "marker");
    marker.setAttribute("id", "cm-arrow"); marker.setAttribute("viewBox", "0 0 10 10"); marker.setAttribute("refX", "9"); marker.setAttribute("refY", "5"); marker.setAttribute("markerWidth", "5"); marker.setAttribute("markerHeight", "5"); marker.setAttribute("orient", "auto-start-reverse");
    const arrow = document.createElementNS(ns, "path"); arrow.setAttribute("d", "M 0 0 L 10 5 L 0 10 z"); arrow.setAttribute("fill", "var(--text-muted)"); marker.appendChild(arrow); defs.appendChild(marker); svg.appendChild(defs);
    for (const edge of full.edges.filter(e => ids.has(e.from) && ids.has(e.to))) {
      const a = pos.get(edge.from), b = pos.get(edge.to); if (!a || !b) continue;
      const x1 = a.x + 178, y1 = a.y + 51, x2 = b.x, y2 = b.y + 51, bend = Math.max(24, (x2 - x1) * .5);
      const path = document.createElementNS(ns, "path");
      path.setAttribute("d", `M${x1} ${y1} C${x1 + bend} ${y1},${x2 - bend} ${y2},${x2} ${y2}`);
      path.setAttribute("class", "cm-edge" + (edge.kind === "official" ? "" : " is-inferred") + ([edge.from, edge.to].includes(this.state.course) ? " is-selected" : ""));
      path.setAttribute("marker-end", "url(#cm-arrow)");
      const title = document.createElementNS(ns, "title"); title.textContent = edge.reason; path.appendChild(title); svg.appendChild(path);
    }
    for (const course of courses) {
      const point = pos.get(course.id); if (!point) continue;
      const p = this.progress(course), coverage = courseCoverage(course, this.personal, this.definitions);
      const node = button(world, "", () => this.openCourse(course.id), "cm-course-node" + (course.id === this.state.course ? " is-selected" : "") + (p.done ? " has-learning" : ""));
      node.dataset.courseMapId = course.id; node.style.left = point.x + "px"; node.style.top = point.y + "px";
      node.setAttribute("aria-label", `${course.school} ${course.code} ${course.titleZh || course.title}，本课已学 ${p.done}/${p.total} ${this.metricValues(course).label}，知识已学 ${coverage.known.length}/${coverage.total}`);
      node.createSpan({ cls: "cm-node-code", text: `${course.school} · ${course.code}` });
      node.createEl("strong", { text: course.titleZh || course.title });
      this.renderMetrics(node, course, true);
    }
    if (this.scroll) { scroller.scrollLeft = this.scroll.x; scroller.scrollTop = this.scroll.y; }
    if (full.cycles.length) this.host.createEl("p", { cls: "sos-error", text: "部分课程关系存在循环，顺序待核对。" });
  }
  renderList(courses) {
    const list = this.host.createDiv({ cls: "cm-course-list" });
    for (const course of courses) {
      const p = this.progress(course), card = button(list, "", () => this.openCourse(course.id), "cm-list-card");
      card.createSpan({ cls: "cm-node-code", text: `${course.school} · ${course.code} · ${ROLES[course.role] || "方向课程"}` });
      card.createEl("strong", { text: course.titleZh || course.title });
      card.createEl("p", { text: course.purpose });
      this.renderMetrics(card, course, true);
    }
  }
  unitLabel(course, unit) {
    if (unit.label !== undefined) return String(unit.label);
    return unit.id === "final-review" ? "复习" : /^ch\d+$/.test(unit.id) ? unit.id.slice(2) : unit.id.length > 7 ? String(course.units.indexOf(unit) + 1) : unit.id;
  }
  referenceLabel(ref) {
    const course = catalog.courses.find(course => course.id === ref.courseId);
    return course?.units.find(unit => unit.id === ref.unitId)?.label ?? ref.unitId;
  }
  compactUnitLabel(course, unit) {
    return this.unitLabel(course, unit).replace(/^(?:Lecture|Session|Class|Reading|Video)\s+(?=\d)/i, "");
  }
  outlineDescription(course) {
    if (!course.outlineVerified) return DEPTH[course.coverage] || "课程大纲";
    const count = `${course.units.length} ${course.unitLabel || "个单元"}`;
    return `${count} · ${course.outlineComplete ? "官方目录已展开" : "目录尚待补全"}`;
  }
  unitTitle(unit) {
    const title = (unit.titleZh || unit.title).replace(/^Ch\.\s*\d+\s*·\s*/i, "");
    const kind = { lab: "实验", recitation: "辅导", supplement: "补充", guest: "嘉宾讲座", review: "复习", exam: "考试", studio: "实践", workshop: "工作坊", project: "项目", qa: "答疑" }[unit.kind];
    const labels = [unit.optional ? "可选" : null, kind].filter(label => label && !title.includes(label));
    return [...labels, title].join(" · ");
  }
  selectedUnit(course) {
    return course.units.find(u => u.id === this.state.unit)
      || course.units.find(u => !completionFor(course, u, this.curriculum, this.records).done) || course.units[0];
  }
  selectUnit(id, { personal = false } = {}) {
    const course = catalog.courses.find(c => c.id === this.state.course);
    const unit = course?.units.find(candidate => candidate.id === id); if (!unit) return;
    this.change({ unit: id, module: unit.module || "课程内容", tab: "units", knowledge: null, knowledgeUnit: null, trace: null,
      ...(personal ? { comparison: "personal", comparisonSource: null, studyComparison: false } : {}) });
    const selected = [...this.host.querySelectorAll("[data-unit-select]")].find(el => el.dataset.unitSelect === id);
    selected?.focus({ preventScroll: true });
    this.revealSelectedUnit();
  }
  revealSelectedUnit() {
    const reveal = () => {
      const list = this.host.querySelector(".cm-directory-list");
      const item = [...this.host.querySelectorAll("[data-unit-select]")].find(el => el.dataset.unitSelect === this.state.unit);
      if (!list?.getBoundingClientRect || !item) return;
      const outer = list.getBoundingClientRect(), inner = item.getBoundingClientRect();
      if (inner.top < outer.top) list.scrollTop += inner.top - outer.top - 8;
      else if (inner.bottom > outer.bottom) list.scrollTop += inner.bottom - outer.bottom + 8;
    };
    if (typeof requestAnimationFrame === "function") requestAnimationFrame(reveal); else reveal();
  }
  rememberDisclosure(element, key) {
    element.open = this.expanded.has(key);
    element.addEventListener("toggle", () => { if (element.open) this.expanded.add(key); else this.expanded.delete(key); });
  }
  renderDetail(course) {
    if (!course) return;
    const selected = this.selectedUnit(course);
    if (selected) { this.state.unit = selected.id; this.owner.courseMapState = this.getState(); }
    const panel = this.detailEl = this.host.createDiv({ cls: "cm-detail", attr: { "data-course-detail": course.id } });
    const hero = panel.createDiv({ cls: "cm-course-hero" });
    const head = hero.createDiv({ cls: "cm-detail-head" });
    head.createEl("h3", { text: course.titleZh || course.title });
    if (course.titleZh) head.createDiv({ cls: "cm-en-title", text: course.title });
    const progress = hero.createDiv({ cls: "cm-hero-progress" });
    this.renderMetrics(progress, course);
    const p = this.progress(course), next = course.units.find(u => !completionFor(course, u, this.curriculum, this.records).done);
    let continueButton;
    if (p.original && p.registered?.plan) continueButton = button(progress, "继续原学习计划", () => this.owner.openOriginal(p.registered.plan), "cm-continue");
    else if (next && !p.original) continueButton = button(progress, `继续学习 · ${this.unitLabel(course, next)}`, () => this.selectUnit(next.id, { personal: true }), "cm-continue");
    const tabs = panel.createDiv({ cls: "cm-detail-tabs", attr: { "aria-label": "课程内容" } });
    for (const [id, text] of [["overview", "课程全景"], ["units", course.coverage === "full-outline" ? "逐节学习" : "模块学习"], ["coverage", "跨课对照"], ["routes", "先修与去向"]]) {
      const b = button(tabs, text, () => this.change({ tab: id, knowledge: null, ...(["overview", "units"].includes(id) ? { comparison: "personal", comparisonSource: null, studyComparison: false } : {}) }), this.state.tab === id ? "is-active" : "");
      b.dataset.courseTab = id;
      b.setAttribute("aria-pressed", String(this.state.tab === id));
    }
    if (continueButton) tabs.appendChild(continueButton);
    const content = panel.createDiv({ cls: "cm-tab-content" + (this.state.knowledge ? " has-inspector" : "") });
    if (this.state.tab === "units") this.renderUnits(content, course);
    else {
      const scroll = content.createDiv({ cls: "cm-page-scroll" }); scroll.dataset.scrollKey = course.id + "/" + this.state.tab;
      if (this.state.tab === "overview") this.renderCourseOverview(scroll, course);
      else if (this.state.tab === "coverage") this.renderCourseCoverage(scroll, course);
      else {
        this.renderRoutes(scroll, course);
        scroll.createEl("h4", { text: "课程与材料" });
        scroll.createEl("p", { cls: "cm-purpose", text: course.purpose });
        const sources = scroll.createDiv({ cls: "cm-materials" });
        external(sources, "课程主页 ↗", course.url); external(sources, "官方目录 ↗", course.outlineSource || course.syllabus);
        sources.createSpan({ text: `${course.version || "公开课"} · ${this.outlineDescription(course)} · 核对 ${course.outlineVerified || course.checked}` });
        if (course.coverage === "topic-outline" && !course.outlineNotes) scroll.createEl("p", { cls: "cm-hint", text: "这里按官方大纲的主题模块组织，一个模块可能包含多节视频。逐讲安排见官方目录。" });
        for (const note of unique([course.outlineNotes, course.progressCountNote, course.materialNote, course.note].filter(Boolean))) scroll.createEl("p", { cls: "cm-hint", text: note });
        if (p.original) scroll.createEl("p", { cls: "cm-hint", text: "进度来自原学习计划；大纲分组尚未与原计划逐节对齐，请在原计划记录。" });
        if (p.registered?.plan) button(scroll, "打开原学习计划", () => this.owner.openOriginal(p.registered.plan), "cm-quiet");
      }
    }
    if (this.state.knowledge) this.renderKnowledgeInspector(content, this.state.knowledge);
  }
  renderCourseOverview(panel, course) {
    panel.classList.add("cm-overview-page");
    const matrix = courseMatrix(course, this.personal);
    const currentModule = matrix.modules.find(m => m.id === this.state.module)
      || matrix.modules.find(m => m.units.some(u => u.id === this.state.unit)) || matrix.modules[0];
    const intro = panel.createDiv({ cls: "cm-overview-intro" });
    const routeMode = this.state.overviewMode !== "matrix";
    const modes = intro.createDiv({ cls: "cm-overview-toggle", attr: { "aria-label": "课程全景展示方式" } });
    for (const [mode, label] of [["route", "学习路线"], ["matrix", "知识矩阵"]]) {
      const selected = mode === "route" ? routeMode : !routeMode;
      const choice = button(modes, label, () => this.change({ overviewMode: mode, knowledge: null, knowledgeUnit: null }), selected ? "is-active" : "");
      choice.dataset.overviewMode = mode;
      choice.setAttribute("aria-pressed", String(selected));
    }
    const meta = intro.createDiv({ cls: "cm-overview-meta" });
    meta.createSpan({ text: `${course.school} ${course.code} · ${this.outlineDescription(course)}` });
    external(meta, "官方目录 ↗", course.outlineSource || course.syllabus);
    if (course.coverage !== "full-outline") meta.createSpan({ cls: "cm-outline-limit", text: course.outlineNotes || "按大纲模块组织；一个模块可能包含多节课。" });
    const pending = course.units.filter(unit => unit.mapping !== "reviewed-outline").length;
    if (pending) meta.createSpan({ cls: "cm-outline-limit", text: `${pending} 个单元的知识范围待核对；课次完成与知识点亮分开计算。` });
    const modules = panel.createDiv({ cls: "cm-module-selector", attr: { "aria-label": "选择课程分部" } });
    matrix.modules.forEach((module, index) => {
      const card = button(modules, "", () => this.change({ module: module.id, knowledge: null }), "cm-module-card" + (module.id === currentModule?.id ? " is-selected" : ""));
      card.dataset.moduleSelect = module.id; card.setAttribute("aria-pressed", String(module.id === currentModule?.id));
      card.createSpan({ cls: "cm-module-number", text: String(index + 1).padStart(2, "0") });
      card.createEl("strong", { text: module.title });
      const done = module.units.filter(u => completionFor(course, u, this.curriculum, this.records).done).length;
      const unavailable = module.units.some(u => completionFor(course, u, this.curriculum, this.records).unavailable);
      card.createSpan({ cls: "cm-module-progress", text: `${unavailable ? "章节进度待对齐" : `${done}/${module.units.length} ${this.metricValues(course).label}已学`} · ${module.rows.length ? `${module.rows.filter(r => r.known).length}/${module.rows.length} 知识已学` : module.counts.unmapped ? "知识范围待核对" : "概览 / 复习"}` });
      const strip = card.createDiv({ cls: "cm-module-strip", attr: { "aria-hidden": "true" } });
      module.units.forEach(unit => strip.createSpan({ cls: completionFor(course, unit, this.curriculum, this.records).done ? "is-done" : "" }));
    });
    if (!currentModule) return;
    if (routeMode) {
      this.renderLearningRoute(panel, course, currentModule);
      return;
    }
    const section = panel.createEl("section", { cls: "cm-matrix-section" });
    const heading = section.createDiv({ cls: "cm-section-heading" });
    heading.createEl("h5", { text: currentModule.title });
    heading.createSpan({ text: "章节 × 知识" });
    const legend = section.createDiv({ cls: "cm-matrix-legend" });
    legend.createSpan({ cls: "is-lit", text: "● 相同知识已有学习记录" });
    legend.createSpan({ text: "○ 本章包含，尚无记录" });
    legend.createSpan({ text: "✓ 本课这一节已学" });
    const viewport = section.createDiv({ cls: "cm-matrix-scroll", attr: { tabindex: "0", "aria-label": "章节与知识对应表，可横向滚动" } });
    viewport.dataset.moduleId = currentModule.id;
    const table = viewport.createEl("table", { cls: "cm-knowledge-matrix" });
    const tr = table.createEl("thead").createEl("tr");
    tr.createEl("th", { text: "知识范围", attr: { scope: "col" } });
    for (const unit of currentModule.units) {
      const completion = completionFor(course, unit, this.curriculum, this.records), done = completion.done;
      const cell = tr.createEl("th", { attr: { scope: "col" } });
      const link = button(cell, this.compactUnitLabel(course, unit), () => this.openUnit(course.id, unit.id), "cm-matrix-unit");
      link.title = this.unitTitle(unit); link.setAttribute("aria-label", `${this.unitLabel(course, unit)} ${this.unitTitle(unit)}${done ? "，本节已学" : completion.unavailable ? "，章节进度待对齐" : ""}`);
      cell.createSpan({ cls: "cm-matrix-completion" + (done ? " is-lit" : ""), text: done ? "✓" : completion.unavailable ? "?" : "·" });
    }
    const body = table.createEl("tbody");
    for (const row of currentModule.rows) {
      const tr = body.createEl("tr", { cls: this.state.knowledge === row.id ? "is-selected" : "" });
      const label = tr.createEl("th", { attr: { scope: "row" } });
      const link = button(label, this.definitions.get(row.id)?.title || row.id, () => this.openKnowledge(row.id, row.unitIds[0]), "cm-matrix-label" + (row.known ? " is-lit" : ""));
      link.dataset.knowledgeOpen = row.id; link.dataset.knowledgeUnit = row.unitIds[0];
      for (const unit of currentModule.units) {
        const cell = tr.createEl("td", { cls: unit.id === (this.state.knowledgeUnit || this.state.unit) ? "is-current" : "" });
        if (!row.unitIds.includes(unit.id)) { cell.createSpan({ cls: "cm-matrix-absent", text: "—" }); continue; }
        const dot = button(cell, row.known ? "●" : "○", () => this.openKnowledge(row.id, unit.id), "cm-matrix-dot" + (row.known ? " is-lit" : ""));
        dot.dataset.knowledgeOpen = row.id; dot.dataset.knowledgeUnit = unit.id;
        dot.setAttribute("aria-label", `${this.unitLabel(course, unit)} · ${this.definitions.get(row.id)?.title} · ${row.known ? "相同范围已学" : "暂无学习记录"}`);
      }
    }
    if (!currentModule.rows.length) section.createEl("p", { cls: "cm-hint", text: "这一部分尚无可比较的知识范围；可从上方编号进入课程材料。" });
    section.createEl("p", { cls: "cm-matrix-help", text: "点章节编号进入学习；点知识或圆点查看来源、连接其他课程。" });
  }
  renderLearningRoute(parent, course, module) {
    const route = parent.createDiv({ cls: "cm-route-scroll", attr: { "aria-label": `${module.title}的章节与知识` } });
    route.dataset.moduleId = module.id;
    const heading = route.createDiv({ cls: "cm-section-heading" });
    heading.createEl("h5", { text: module.title });
    heading.createSpan({ text: "按官方目录顺序" });
    const entries = module.units.map(unit => ({ unit, completion: completionFor(course, unit, this.curriculum, this.records) }));
    const pending = course.units.find(unit => {
      const completion = completionFor(course, unit, this.curriculum, this.records);
      return !completion.done && !completion.unavailable;
    });
    let prefixLength = 0;
    while (prefixLength < entries.length && entries[prefixLength].completion.done && !entries[prefixLength].completion.unavailable) prefixLength++;
    const renderEntry = (parent, { unit, completion }) => {
      const current = unit.id === this.state.unit;
      const next = unit.id === pending?.id;
      const row = parent.createEl("article", { cls: "cm-learning-route-row" + (completion.done ? " is-done" : "")
        + (current ? " is-current" : "") + (next ? " is-next" : "") + (completion.unavailable ? " is-unavailable" : ""),
        attr: { "data-route-unit": unit.id } });
      const head = row.createDiv({ cls: "cm-route-unit-head" });
      head.createSpan({ cls: "cm-route-unit-number", text: this.compactUnitLabel(course, unit) });
      const title = button(head, this.unitTitle(unit), () => this.openUnit(course.id, unit.id), "cm-route-unit-title");
      title.setAttribute("aria-label", `${course.code} ${this.unitLabel(course, unit)} ${this.unitTitle(unit)}，打开章节`);
      if (current) title.setAttribute("aria-current", "step");
      if (next) button(head, "接着学 →", () => this.openUnit(course.id, unit.id), "cm-route-unit-status cm-route-resume");
      else head.createSpan({ cls: "cm-route-unit-status", text: completion.unavailable ? "章节进度待对齐" : completion.done ? "✓ 本课已学" : "待记录" });
      const knowledge = row.createDiv({ cls: "cm-route-unit-knowledge" });
      const ids = unitOutcomes(unit);
      for (const id of ids) {
        const outcome = this.definitions.get(id), known = this.personal.has(id);
        const otherCount = new Set((this.providers.get(id) || []).filter(ref => ref.courseId !== course.id).map(ref => ref.courseId)).size;
        const chip = button(knowledge, outcome?.title || id, () => this.openKnowledge(id, unit.id), "cm-knowledge-chip" + (known ? " is-lit" : ""));
        chip.dataset.knowledgeOpen = id; chip.dataset.knowledgeUnit = unit.id;
        chip.setAttribute("aria-expanded", String(this.state.knowledge === id && this.state.knowledgeUnit === unit.id));
        chip.setAttribute("aria-label", `${outcome?.title || id}，${known ? "相同范围已有学习记录" : "暂无学习记录"}${otherCount ? `，也出现在${otherCount}门课程` : ""}，查看知识连接`);
        chip.title = `${outcome?.scope || outcome?.title || id}${otherCount ? ` · 也出现在 ${otherCount} 门课程` : ""}`;
      }
      if (!ids.length) knowledge.createSpan({ cls: "cm-route-scope-note", text: unit.mapping === "reviewed-outline" ? "概览 / 复习 · 不新增知识标记" : "知识范围待核对" });
    };
    if (prefixLength) {
      const unitName = course.unitLabel || (course.coverage === "full-outline" ? "节" : "个模块");
      const completed = route.createEl("details", { cls: "cm-route-completed" });
      const prefix = entries.slice(0, prefixLength);
      const known = unique(prefix.flatMap(({ unit }) => unitOutcomes(unit))).filter(id => this.personal.has(id)).length;
      completed.createEl("summary", { text: `前 ${prefixLength} ${unitName}已学${known ? ` · ${known} 个知识点已点亮` : ""} · 展开回顾` });
      this.rememberDisclosure(completed, `route-completed/${course.id}/${module.id}`);
      if (prefix.some(({ unit }) => unit.id === this.state.unit || (this.state.knowledge && unit.id === this.state.knowledgeUnit))) completed.open = true;
      const list = completed.createDiv({ cls: "cm-route-completed-list" });
      prefix.forEach(entry => renderEntry(list, entry));
    }
    const list = route.createDiv({ cls: "cm-route-pending-list" });
    entries.slice(prefixLength).forEach(entry => renderEntry(list, entry));
    route.createEl("p", { cls: "cm-matrix-help", text: "绿色知识可来自其他课程；本课章节完成单独记录。点章节进入学习，点知识查看连接。" });
  }
  renderComparison(panel, course, compact = false) {
    const bar = panel.createDiv({ cls: "cm-compare-bar" });
    const compare = catalog.courses.find(c => c.id === this.state.comparison);
    const source = catalog.courses.find(c => c.id === this.state.comparisonSource);
    if (source && !compact) {
      const choices = bar.createDiv({ cls: "cm-baseline-choices" });
      for (const [value, label] of [["personal", "我现在已经学过的"], [source.id, `${source.code} 全部教学范围`]]) {
        const b = button(choices, label, () => this.change({ comparison: value, knowledge: null }), this.state.comparison === value ? "is-active" : "");
        b.setAttribute("aria-pressed", String(this.state.comparison === value));
      }
    }
    let picker = bar;
    if (source) { picker = bar.createEl("details", { cls: "cm-other-baseline" }); picker.createEl("summary", { text: "其他比较课程" }); }
    select(picker, "比较起点", [["personal", "我的实际已学"], ...comparisonCandidates(course, catalog.courses).filter(candidate => candidate.overlap > 0).map(({course: c}) => [c.id, `${c.school} ${c.code} · 完整课程范围`])], this.state.comparison, comparison => this.change({ comparison, knowledge: null }));
    if (compare) panel.createDiv({ cls: "cm-comparison-notice", text: `比较依据：${compare.school} ${compare.code} 已映射的全部教学范围。本人学习进度另记。` });
    return compare;
  }
  renderUnits(panel, course) {
    panel = panel.createDiv({ cls: "cm-unit-workspace" });
    const unit = this.selectedUnit(course);
    if (!unit) { panel.createDiv({ cls: "cm-empty", text: "课程已收录，章节目录仍待整理。" }); return; }
    const compare = catalog.courses.find(c => c.id === this.state.comparison);
    if (compare || this.comparisonReturn) {
      const context = panel.createDiv({ cls: "cm-lesson-compare-context" });
      context.createSpan({ text: compare ? `正在对照 ${compare.school} ${compare.code} 全部教学范围` : "正在对照我的实际已学" });
      button(context, "← 返回全课对照", () => this.returnToComparison(), "cm-inline-link");
      if (compare) button(context, "切回我的已学", () => this.change({ comparison: "personal" }), "cm-inline-link");
    }
    const layout = panel.createDiv({ cls: "cm-study-layout" });
    const directory = layout.createEl("nav", { cls: "cm-directory", attr: { "aria-label": "课程章节目录" } });
    const directoryHead = directory.createDiv({ cls: "cm-directory-head" });
    directoryHead.createEl("strong", { text: "课程目录" });
    directoryHead.createSpan({ text: `${course.units.length} ${course.unitLabel || (course.coverage === "full-outline" ? "节" : "个模块")}` });
    const list = directory.createDiv({ cls: "cm-directory-list" });
    for (const name of unique(course.units.map(u => u.module || "课程内容"))) {
      const group = list.createEl("section", { cls: "cm-directory-group" });
      group.createEl("h5", { text: name });
      for (const item of course.units.filter(u => (u.module || "课程内容") === name)) {
        const completion = completionFor(course, item, this.curriculum, this.records), done = completion.done;
        const coverage = coverageFor(item, this.learned, this.definitions);
        const choice = button(group, "", () => this.selectUnit(item.id), "cm-directory-item" + (item.id === unit.id ? " is-selected" : "") + (done ? " is-done" : ""));
        choice.dataset.unitSelect = item.id;
        choice.title = `${this.unitLabel(course, item)} · ${this.unitTitle(item)}`;
        choice.setAttribute("aria-current", item.id === unit.id ? "step" : "false");
        choice.createSpan({ cls: "cm-directory-number", text: this.compactUnitLabel(course, item) });
        choice.createSpan({ cls: "cm-directory-title", text: this.unitTitle(item) });
        choice.createSpan({ cls: "cm-directory-status", text: `${done ? "✓ 已学" : completion.unavailable ? "进度待对齐" : "待记录"}${coverage.total ? ` · ${compare ? "重合" : "知识"} ${coverage.known.length}/${coverage.total}` : ""}` });
      }
    }
    const pane = layout.createEl("section", { cls: "cm-lesson-pane", attr: { "aria-label": "当前章节" } });
    pane.dataset.scrollKey = course.id + "/" + unit.id;
    this.renderUnit(pane, course, unit, compare);
    const navigation = pane.createDiv({ cls: "cm-lesson-navigation" });
    const browsingGaps = this.state.studyComparison && this.state.coverageFilter === "new";
    const sequence = browsingGaps ? course.units.filter(item => coverageFor(item, this.learned, this.definitions).missing.length) : course.units;
    const order = course.units.indexOf(unit);
    const earlier = sequence.filter(item => course.units.indexOf(item) < order).at(-1);
    const later = sequence.find(item => course.units.indexOf(item) > order);
    const gapLabel = compare ? "含额外范围" : "含未记录范围";
    const previous = button(navigation, browsingGaps ? `← 上一处${gapLabel}` : "← 上一节", () => this.selectUnit(earlier?.id), "cm-quiet");
    previous.disabled = !earlier;
    const index = sequence.indexOf(unit);
    const status = coverageFor(unit, this.learned, this.definitions).status;
    const outsideSequence = status === "unknown" ? "本章知识范围待核对" : status === "overview" ? "概览/复习，不参与差异比较" : `所列范围已${compare ? "重合" : "覆盖"}`;
    navigation.createSpan({ text: browsingGaps ? (index < 0 ? outsideSequence : `差异章节 ${index + 1} / ${sequence.length}`) : `${order + 1} / ${course.units.length}` });
    const next = button(navigation, browsingGaps ? `下一处${gapLabel} →` : "下一节 →", () => this.selectUnit(later?.id), "cm-quiet");
    next.disabled = !later;
  }
  renderUnit(parent, course, unit, compare) {
    const completed = completionFor(course, unit, this.curriculum, this.records), coverage = coverageFor(unit, this.learned, this.definitions);
    const row = parent.createDiv({ cls: "cm-unit is-" + coverage.status + (completed.done ? " is-done" : ""), attr: { "data-course-unit": `${course.id}|${unit.id}` } });
    const detail = row.createDiv({ cls: "cm-unit-body" });
    if (this.state.trace) {
      const trace = this.state.trace, origin = catalog.courses.find(c => c.id === trace.courseId);
      const banner = detail.createDiv({ cls: "cm-trace" });
      banner.createSpan({ text: `循着「${this.definitions.get(trace.knowledgeId)?.title || "知识点"}」来到这里` });
      button(banner, `← ${origin?.code || "来源课程"} ${this.referenceLabel(trace)}`, () => this.returnToTrace(), "cm-inline-link");
    }
    const heading = detail.createDiv({ cls: "cm-unit-heading" });
    heading.createSpan({ cls: "cm-eyebrow cm-unit-number", text: `当前章节 · ${this.unitLabel(course, unit)}` });
    button(heading, "↑ 课程目录", () => this.host.querySelector(".cm-directory")?.scrollIntoView({ block: "start", behavior: "instant" }), "cm-quiet cm-open-directory");
    heading.createEl("h4", { cls: "cm-unit-title", text: this.unitTitle(unit) });
    if (unit.titleZh) heading.createDiv({ cls: "cm-en-title", text: unit.title });
    const actions = detail.createDiv({ cls: "cm-unit-actions" });
    const directoryOnly = unit.sourceType === "outline" || [course.url, course.syllabus, course.outlineSource].filter(Boolean).includes(unit.source);
    const primary = external(actions, directoryOnly ? "查看官方课次 ↗" : "打开本节材料 ↗", unit.source || course.syllabus);
    primary?.classList.add("cm-primary-action");
    for (const resource of unit.resources || []) if (resource.url !== unit.source) external(actions, `${resource.label} ↗`, resource.url);
    const complete = actions.createEl("label", { cls: "cm-completion-control" });
    const check = complete.createEl("input", { cls: "cm-check", attr: { type: "checkbox", "aria-label": `${completed.done ? "取消已学" : "标记已学"}：${course.code} ${this.unitLabel(course, unit)} ${unit.title}` } });
    check.checked = completed.done; check.disabled = !!compare || !!completed.unavailable || this.loading || !!this.error || !!this.saving;
    complete.createSpan({ cls: "cm-unit-completion" + (completed.done ? " is-done" : ""), text: completed.done ? "本课已学" : "标记本节已学" });
    if (completed.unavailable) complete.title = "此课程已有原计划；需要先完成章节对齐，请在原计划记录。";
    check.addEventListener("change", async () => {
      check.disabled = true;
      try { await this.setCompleted(course, unit, check.checked); }
      catch (error) { check.checked = completed.done; check.disabled = false; new Notice(error.message); }
    });
    if (completed.task) {
      button(actions, "本节笔记 / 材料", () => this.owner.openStudyMaterial(completed.task), "cm-quiet");
      const rowKey = "map-" + course.id + "|" + unit.id;
      row.dataset.taskId = completed.task.id; row.dataset.key = rowKey;
      button(actions, "学习记录", () => this.owner.openTask(completed.task, "", null, rowKey), "cm-quiet");
    }
    const body = detail.createDiv({ cls: "cm-unit-content" });
    const sectionHeading = body.createDiv({ cls: "cm-section-heading" });
    sectionHeading.createEl("h5", { text: "这一节的知识" });
    sectionHeading.createSpan({ cls: "cm-coverage-badge is-" + coverage.status, text: coverage.total ? `${coverage.known.length} / ${coverage.total} ${compare ? "重合" : "已学"}` : STATUS[coverage.status] });
    body.createEl("p", { cls: "cm-knowledge-guide", text: "点选知识点，查看已学来源和其他课程中的位置。" });

    if (!coverage.total) body.createEl("p", { cls: "cm-hint", text: coverage.status === "overview" ? "本节用于概览或复习，知识覆盖依据具体教学单元计算。" : "本节目录已核对，知识范围待细分；学完后仍可记录本节进度。" });
    for (const id of unitOutcomes(unit)) this.renderKnowledgeRow(body, id, this.learned);
    if (unit.mappingEvidence?.length || unit.note) {
      const evidence = body.createEl("details", { cls: "cm-unit-evidence" });
      evidence.createEl("summary", { text: "知识范围的依据" });
      this.rememberDisclosure(evidence, `evidence/${course.id}/${unit.id}`);
      if (unit.note) evidence.createEl("p", { cls: "cm-hint", text: unit.note });
      unique(unit.mappingEvidence || []).forEach((url, i) => external(evidence, `官方材料 ${i + 1} ↗`, url));
    }
    if (completed.task) body.createDiv({ cls: "sos-task-detail" });
  }
  renderOutcome(parent, id, learned, showHeading = true) {
    const outcome = this.definitions.get(id); if (!outcome) return;
    const sources = learned.get(id) || [], item = parent.createDiv({ cls: "cm-outcome" + (sources.length ? " is-lit" : "") });
    if (showHeading) {
      const heading = item.createDiv({ cls: "cm-outcome-name" }); heading.createSpan({ cls: "cm-light", text: sources.length ? "●" : "○" }); heading.createEl("strong", { text: outcome.title });
      heading.createSpan({ cls: "cm-knowledge-state", text: sources.length ? (sources[0].simulated ? "对照包含" : "已学") : "暂无记录" });
    }
    item.createEl("p", { text: outcome.scope || "范围以课程说明为准。" });
    if (sources.length) {
      const line = item.createDiv({ cls: "cm-learning-source" }); line.createSpan({ text: sources[0].simulated ? "对照课程包含：" : "已学来源：" });
      for (const source of sources) button(line, `${source.courseTitle} ${this.referenceLabel(source)}`, () => this.openUnit(source.courseId, source.unitId), "cm-inline-link");
    }
    if (showHeading && this.state.course) {
      const providers = (this.providers.get(id) || []).filter(ref => ref.courseId !== this.state.course);
      if (providers.length) {
        const shared = item.createEl("details", { cls: "cm-shared-courses" });
        shared.createEl("summary", { text: `同一知识点也在 ${new Set(providers.map(ref => ref.courseId)).size} 门课程中` });
        const sharedKey = `shared/${this.state.course}/${id}`;
        shared.open = this.expanded.has(sharedKey);
        shared.addEventListener("toggle", () => { if (shared.open) this.expanded.add(sharedKey); else this.expanded.delete(sharedKey); });
        for (const ref of providers) {
          const course = catalog.courses.find(c => c.id === ref.courseId);
          button(shared, `${course.school} ${course.code} · ${ref.title}`, () => this.openUnit(ref.courseId, ref.unitId), "cm-inline-link");
        }
      }
    }
    const notes = outcome.notes || [];
    if (notes.length) button(item, "打开已有知识笔记", event => this.openNotes(notes, event), "cm-inline-link");
  }
  async openNotes(ids, event) {
    const files = this.app.vault.getMarkdownFiles().filter(f => f.path.startsWith("00_Knowledge/"));
    const matches = files.filter(f => ids.includes(this.app.metadataCache.getFileCache(f)?.frontmatter?.atom_id));
    if (!matches.length) { new Notice("当前知识库中尚未匹配到这项笔记。"); return; }
    if (matches.length === 1) { await this.owner.openOriginal(matches[0].path); return; }
    const menu = new Menu();
    for (const file of matches) menu.addItem(item => item.setTitle(file.basename).onClick(() => this.owner.openOriginal(file.path)));
    menu.showAtMouseEvent(event);
  }
  renderCourseNeighborhood(panel, course) {
    const route = panel.createDiv({ cls: "cm-neighborhood", attr: { "aria-label": "当前课程的先修与后续" } });
    const before = (course.prerequisites || []).map(relation => ({ course: catalog.courses.find(c => c.id === relation.id), relation })).filter(ref => ref.course);
    const after = catalog.courses.flatMap(c => (c.prerequisites || []).filter(relation => relation.id === course.id).map(relation => ({ course: c, relation })));
    const priorities = ["ml", "statistics", "deep-learning", "math"];
    after.sort((a, b) => (priorities.indexOf(a.course.domain) < 0 ? 99 : priorities.indexOf(a.course.domain)) - (priorities.indexOf(b.course.domain) < 0 ? 99 : priorities.indexOf(b.course.domain)));
    for (const [kind, title, refs] of [["before", "先修准备", before], ["current", "当前课程", []], ["after", "学完可以去", after]]) {
      const column = route.createDiv({ cls: "cm-neighbor-column is-" + kind });
      column.createSpan({ cls: "cm-neighbor-label", text: title });
      const links = column.createDiv({ cls: "cm-neighbor-links" });
      if (kind === "current") { links.createEl("strong", { text: course.code }); continue; }
      for (const ref of refs.slice(0, 1)) {
        const b = button(links, `${ref.course.code} · ${ref.course.titleZh || ref.course.title}`, () => this.openCourse(ref.course.id), "cm-route-link");
        b.title = `${ref.course.titleZh || ref.course.title} · ${ref.relation.kind === "official" ? "官方先修" : "内容衔接 · 推断"}：${ref.relation.reason}`;
        b.setAttribute("aria-label", b.title); b.dataset.relationKind = ref.relation.kind;
      }
      if (refs.length > 1) button(links, `+${refs.length - 1}`, () => this.change({ tab: "routes", knowledge: null }), "cm-route-link");
      if (!refs.length) button(links, kind === "before" ? "查看准备要求" : "查看课程说明", () => this.change({ tab: "routes", knowledge: null }), "cm-route-link");
    }
  }
  renderCourseCoverage(panel, course) {
    panel.classList.add("cm-comparison-page");
    const compare = this.renderComparison(panel, course), coverage = courseCoverage(course, this.learned, this.definitions);
    const projection = comparisonSummary(course, this.learned);
    const result = panel.createDiv({ cls: "cm-compare-result" });
    for (const [count, label, lit] of [[coverage.known.length, compare ? "个知识点重合" : "个知识点已学", true], [coverage.missing.length, compare ? "个额外知识点" : "个知识点暂无记录", false]]) {
      const metric = result.createDiv({ cls: lit ? "is-lit" : "" }); metric.createEl("strong", { text: String(count) }); metric.createSpan({ text: label });
    }
    const firstGap = projection.chapters.find(chapter => chapter.missing.length);
    if (firstGap) {
      const start = button(result, "", () => {
        this.change({ coverageFilter: "new", knowledge: null });
        this.openUnit(course.id, firstGap.unit.id);
      }, "cm-gap-start");
      start.dataset.gapStart = firstGap.unit.id;
      start.createSpan({ text: `从 ${this.unitLabel(course, firstGap.unit)} 查看差异 →` });
      start.createEl("small", { text: this.unitTitle(firstGap.unit) });
      start.title = firstGap.missing.map(id => this.definitions.get(id)?.title || id).join("、");
    }
    const unitName = course.coverage === "full-outline" ? (course.unitLabel || "节") : "个模块";
    panel.createEl("p", { cls: "cm-compare-conclusion", text: `${projection.counts.complete} ${unitName}的所列知识${compare ? "完全重合" : "均已有记录"}，${projection.counts.withExtra} ${unitName}含${compare ? "额外" : "尚未记录的"}内容。${projection.counts.unmapped ? ` 另有 ${projection.counts.unmapped} ${unitName}的知识范围待核对。` : ""}${projection.counts.overview ? ` ${projection.counts.overview} ${unitName}概览或复习不计入知识范围比较。` : ""}` });
    if (course.coverage !== "full-outline") panel.createEl("p", { cls: "cm-hint", text: "本课目前按大纲模块比较，不代表逐讲或逐个练习的完整范围。" });
    const filters = panel.createDiv({ cls: "cm-coverage-filters", attr: { "aria-label": "按章节知识覆盖筛选" } });
    const matching = course.units.map(unit => ({ unit, coverage: coverageFor(unit, this.learned, this.definitions) }));
    for (const [id, label, count] of [["all", course.coverage === "full-outline" ? "全部章节" : "全部模块", matching.length], ["new", compare ? "含额外内容" : "含待学范围", matching.filter(x => x.coverage.missing.length).length], ["covered", compare ? "所列范围重合" : "所列范围已学", matching.filter(x => x.coverage.status === "covered").length]]) {
      const b = button(filters, `${label} ${count}`, () => this.change({ coverageFilter: id, knowledge: null }), this.state.coverageFilter === id ? "is-active" : "");
      b.dataset.coverageFilter = id; b.setAttribute("aria-pressed", String(this.state.coverageFilter === id));
    }
    const list = panel.createDiv({ cls: "cm-coverage-chapters" });
    const labels = list.createDiv({ cls: "cm-coverage-table-head" });
    labels.createSpan({ text: course.coverage === "full-outline" ? "章节与材料" : "大纲模块" }); labels.createSpan({ text: compare ? "两课重合" : "已经学过" }); labels.createSpan({ text: compare ? "基准映射中未包含" : "暂无学习记录" });
    const visible = matching.filter(x => this.state.coverageFilter === "new" ? x.coverage.missing.length : this.state.coverageFilter === "covered" ? x.coverage.status === "covered" : true);
    if (!visible.length) list.createDiv({ cls: "cm-empty", text: "没有符合这一条件的章节。" });
    for (const {unit, coverage: cov} of visible) {
      const row = list.createEl("section", { cls: "cm-coverage-chapter", attr: { "data-coverage-unit": unit.id } });
      const heading = row.createDiv({ cls: "cm-chapter-heading" });
      heading.createSpan({ cls: "cm-chapter-number", text: this.unitLabel(course, unit) });
      button(heading, this.unitTitle(unit), () => this.openUnit(course.id, unit.id), "cm-inline-link");
      heading.createSpan({ cls: "cm-chapter-count" + (cov.status === "covered" ? " is-lit" : ""), text: cov.total ? `${cov.known.length}/${cov.total} ${compare ? "重合" : "已学"}` : STATUS[cov.status] });
      for (const [label, ids, lit] of [[compare ? "重合" : "已学", cov.known, true], [compare ? "额外" : "未记录", cov.missing, false]]) {
        const group = row.createDiv({ cls: "cm-chapter-knowledge" + (lit ? " is-lit" : "") });
        group.createSpan({ cls: "cm-mobile-column-label", text: label });
        if (!ids.length) group.createSpan({ cls: "cm-empty-cell", text: cov.total ? "—" : STATUS[cov.status] });
        for (const id of ids) {
          const b = button(group, this.definitions.get(id)?.title || id, () => this.openKnowledge(id, unit.id), "cm-knowledge-chip"); b.dataset.knowledgeOpen = id; b.dataset.knowledgeUnit = unit.id;
        }
      }
    }
    panel.createEl("p", { cls: "cm-hint", text: compare ? "按已核对的映射比较，未映射内容可能仍有重合。绿色不表示本人学过；章节进度单独记录。" : "按已核对的知识范围比较。绿色表示学过相同范围，独立掌握与本课章节完成另记。" });
  }
  renderKnowledgeRow(parent, id, learned) {
    const outcome = this.definitions.get(id); if (!outcome) return;
    const sources = learned.get(id) || [], others = new Set((this.providers.get(id) || []).filter(r => r.courseId !== this.state.course).map(r => r.courseId));
    const row = button(parent, "", () => this.openKnowledge(id), "cm-knowledge-row" + (sources.length ? " is-lit" : "") + (this.state.knowledge === id ? " is-selected" : ""));
    row.dataset.knowledgeOpen = id; row.dataset.knowledgeUnit = this.state.unit; row.setAttribute("aria-expanded", String(this.state.knowledge === id));
    row.createSpan({ cls: "cm-light", text: sources.length ? "●" : "○" });
    const text = row.createDiv({ cls: "cm-knowledge-row-text" });
    text.createEl("strong", { text: outcome.title });
    text.createSpan({ text: others.size ? `也出现在 ${others.size} 门课程中` : "查看知识范围与笔记" });
    row.createSpan({ cls: "cm-knowledge-state", text: sources.length ? (sources[0].simulated ? "重合" : "已学") : this.state.comparison !== "personal" ? "本课额外" : "未记录" });
    row.createSpan({ cls: "cm-row-arrow", text: "›" });
  }
  openKnowledge(id, unitId = this.state.unit) {
    if (!this.definitions.has(id)) return;
    const unit = catalog.courses.find(c => c.id === this.state.course)?.units.find(u => u.id === unitId && unitOutcomes(u).includes(id));
    this.change({ knowledge: id, knowledgeUnit: unit?.id || null });
    this.host.querySelector(".cm-inspector-close")?.focus({ preventScroll: true });
  }
  closeKnowledge() {
    const id = this.state.knowledge, unitId = this.state.knowledgeUnit;
    this.change({ knowledge: null, knowledgeUnit: null });
    const matches = [...this.host.querySelectorAll("[data-knowledge-open]")].filter(el => el.dataset.knowledgeOpen === id);
    const origin = matches.find(el => el.dataset.knowledgeUnit === unitId) || (matches.length === 1 ? matches[0] : null);
    origin?.focus({ preventScroll: true });
  }
  jumpKnowledge(courseId, unitId, id) {
    const target = catalog.courses.find(c => c.id === courseId)?.units.find(u => u.id === unitId);
    if (!target || !unitOutcomes(target).includes(id)) return;
    if (courseId === this.state.course && unitId === this.state.unit && this.state.tab === "units") return;
    const trace = { knowledgeId: id, courseId: this.state.course, unitId: this.state.knowledgeUnit || this.state.unit };
    this.openCourse(courseId, unitId);
    this.change({ knowledge: id, knowledgeUnit: unitId, trace });
  }
  returnToTrace() {
    const trace = this.state.trace; if (!trace) return;
    const previous = this.history[this.history.length - 1]?.state;
    if (previous?.course === trace.courseId && (previous?.knowledgeUnit || previous?.unit) === trace.unitId) return this.backCourse();
    this.openCourse(trace.courseId, trace.unitId);
    this.change({ knowledge: trace.knowledgeId, knowledgeUnit: trace.unitId, trace: null });
  }
  renderKnowledgeInspector(parent, id) {
    const outcome = this.definitions.get(id); if (!outcome) return;
    const panel = parent.createEl("aside", { cls: "cm-inspector", attr: { "aria-label": "知识点详情", "data-knowledge-inspector": id } });
    const head = panel.createDiv({ cls: "cm-inspector-head" });
    const current = catalog.courses.find(c => c.id === this.state.course);
    const trace = this.state.trace, origin = trace && catalog.courses.find(c => c.id === trace.courseId);
    const sourceUnit = current?.units.find(u => u.id === this.state.knowledgeUnit) || (current && this.selectedUnit(current));
    head.createSpan({ text: `${origin ? origin.code + " " + this.referenceLabel(trace) + " → " : ""}${current?.code || "当前课程"} › ${sourceUnit ? this.unitLabel(current, sourceUnit) : ""} › 知识` });
    button(head, "关闭 ×", () => this.closeKnowledge(), "cm-inspector-close");
    const body = panel.createDiv({ cls: "cm-inspector-body" });
    body.createEl("h4", { text: outcome.title });
    const sources = this.personal.get(id) || [];
    body.createSpan({ cls: "cm-inspector-state" + (sources.length ? " is-lit" : ""), text: sources.length ? "● 已学过相同范围" : "○ 暂无本人的学习记录" });
    body.createEl("p", { cls: "cm-scope", text: outcome.scope });
    const compare = catalog.courses.find(c => c.id === this.state.comparison);
    if (compare) {
      const comparison = body.createDiv({ cls: "cm-inspector-comparison" });
      const refs = (this.learned.get(id) || []).filter(s => s.courseId === compare.id);
      comparison.createEl("strong", { text: refs.length ? `与 ${compare.code} 的教学范围重合` : `${compare.code} 已映射范围中未包含` });
      for (const ref of refs) button(comparison, `${compare.code} ${this.referenceLabel(ref)} · ${ref.unitTitle}`, () => this.jumpKnowledge(ref.courseId, ref.unitId, id), "cm-inline-link");
      comparison.createSpan({ text: "整课对照与上方本人的学习状态分开计算。" });
    }
    if (sources.length) {
      const section = body.createDiv({ cls: "cm-inspector-sources" });
      section.createSpan({ text: "已学来源" });
      for (const source of sources) {
        const sourceLink = button(section, `${source.courseTitle} · ${this.referenceLabel(source)}`, () => this.jumpKnowledge(source.courseId, source.unitId, id), "cm-inspector-source");
        sourceLink.title = source.unitTitle;
      }
    }
    const refs = (this.providers.get(id) || []).filter(r => r.courseId !== this.state.course);
    const links = body.createEl("section", { cls: "cm-inspector-section" });
    links.createEl("h5", { text: `其他课程中的位置 · ${new Set(refs.map(r => r.courseId)).size} 门` });
    if (!refs.length) links.createEl("p", { cls: "cm-hint", text: "当前目录中还没有其他课程映射到相同范围。" });
    for (const courseId of unique(refs.map(r => r.courseId))) {
      const course = catalog.courses.find(c => c.id === courseId);
      const group = links.createDiv({ cls: "cm-provider-course" });
      group.createSpan({ cls: "cm-location-course", text: `${course.school} · ${course.code} · ${course.coverage === "full-outline" ? "逐节目录" : "大纲模块"}` });
      group.createEl("strong", { text: course.titleZh || course.title });
      const compareButton = button(group, `对照 ${current?.code || "当前课"} 与 ${course.code}`, () => this.openComparison(course.id, current?.id), "cm-provider-compare");
      compareButton.dataset.compareCourse = course.id;
      for (const ref of refs.filter(r => r.courseId === course.id)) {
        const unit = course.units.find(u => u.id === ref.unitId), isModule = course.coverage !== "full-outline";
        const card = group.createDiv({ cls: "cm-inspector-location", attr: { "data-knowledge-location": `${course.id}|${unit.id}` } });
        card.createEl("strong", { text: `${this.unitLabel(course, unit)} · ${this.unitTitle(unit)}` });
        const extra = unitOutcomes(unit).filter(key => key !== id && !this.personal.has(key));
        if (extra.length) card.createEl("p", { text: "此外尚无本人学习记录：" + extra.map(key => this.definitions.get(key)?.title || key).join("、") });
        else card.createEl("p", { text: this.personal.has(id) ? "所列知识范围均已有学习记录。" : "聚焦当前知识点，尚无本人的学习记录。" });
        const jump = button(card, isModule ? "进入这个模块 →" : "进入这一节 →", () => this.jumpKnowledge(course.id, unit.id, id), "cm-inspector-jump");
        jump.dataset.jumpCourse = course.id; jump.dataset.jumpUnit = unit.id;
      }
    }
    if (outcome.notes?.length) button(body, "打开已有知识笔记 ↗", event => this.openNotes(outcome.notes, event), "cm-note-link");
    body.createEl("p", { cls: "cm-inspector-footnote", text: "共享的是知识范围。另一门课的章节进度不会因此变成已完成。" });
  }
  renderRoutes(panel, course) {
    const related = relatedCourses(catalog, course);
    panel.createEl("h4", { text: "从这里出发，接向哪些方向？" });
    panel.createEl("p", { cls: "cm-hint", text: "每条关系说明这门课能提供哪一部分准备。是否可以开始下一门课，还需要同时检查它的其他要求。" });
    const directions = panel.createDiv({ cls: "cm-route-columns" });
    const before = directions.createEl("section", { cls: "cm-route-section" });
    before.createEl("h5", { text: `${course.code} 的准备要求` });
    for (const requirement of course.requirements || []) {
      const block = before.createDiv({ cls: "cm-requirement" }); block.createEl("strong", { text: requirement.title });
      const ids = requirement.outcomes || [], known = ids.filter(id => this.personal.has(id));
      block.createSpan({ text: ids.length ? ` · ${known.length}/${ids.length} 个已映射主题有学习记录` : " · 按课程要求核对" });
      external(block, "官方说明 ↗", requirement.source);
      for (const id of ids) this.renderOutcome(block, id, this.personal);
    }
    if (!course.requirements?.length) before.createEl("p", { cls: "cm-hint", text: "具体入学准备请核对课程官方说明。图中没有连线不表示没有先修要求。" });
    if (related.prerequisites.length) before.createEl("h5", { text: "可以从这些课程补充" });
    for (const ref of related.prerequisites) this.renderRelation(before, ref.course, ref.edge);
    const after = directions.createEl("section", { cls: "cm-route-section" });
    after.createEl("h5", { text: `后续方向 · ${related.downstream.length}` });
    for (const ref of related.downstream) this.renderRelation(after, ref.course, ref.edge);
    if (!related.downstream.length) after.createEl("p", { cls: "cm-hint", text: "目录内尚未整理直接后续课程。" });
  }
  renderRelation(parent, course, relation) {
    const card = parent.createDiv({ cls: "cm-relation" });
    if (course) button(card, `${course.school} ${course.code} · ${course.titleZh || course.title}`, () => this.openCourse(course.id), "cm-inline-link");
    else card.createEl("strong", { text: relation.id });
    card.createSpan({ cls: "cm-relation-kind", text: relation.kind === "official" ? "官方先修" : "内容衔接 · 推断" });
    card.createEl("p", { text: relation.reason }); external(card, "依据 ↗", relation.source);
    if (course) {
      const prep = unique((course.requirements || []).flatMap(r => r.outcomes || []));
      if (prep.length) card.createDiv({ cls: "cm-route-readiness", text: `已映射准备范围：${prep.filter(id => this.personal.has(id)).length}/${prep.length} 项有学习记录` });
      if (course.requirements?.length) card.createDiv({ cls: "cm-route-requirements", text: "还需核对：" + course.requirements.map(r => r.title).join("；") });
    }
  }
  async setCompleted(course, unit, done) {
    if (this.saving || this.error || !this.loaded) throw new Error("请等待学习记录读取完成。");
    const completion = completionFor(course, unit, this.curriculum, this.records);
    if (completion.unavailable) throw new Error("请在此课程已有的学习计划中记录。");
    this.saving = true;
    try {
      if (completion.task) await this.plugin.setTaskCompletion(completion.task, done);
      else {
        let file = this.app.vault.getAbstractFileByPath(PRIVATE_RECORD);
        if (!file) {
          const folder = PRIVATE_RECORD.slice(0, PRIVATE_RECORD.lastIndexOf("/"));
          if (!this.app.vault.getAbstractFileByPath(folder)) await this.app.vault.createFolder(folder);
          const header = "# Course learning record · 课程学习记录\n\n<!-- bilingual-en:start -->\nThese checkboxes record user-confirmed course units. Knowledge coverage is calculated from them; independent mastery is recorded separately.\n<!-- bilingual-en:end -->\n\n以下复选框记录本人确认学完的课程单元。知识覆盖据此计算；独立掌握另行记录。\n\n";
          try { file = await this.app.vault.create(PRIVATE_RECORD, header); }
          catch (error) { file = this.app.vault.getAbstractFileByPath(PRIVATE_RECORD); if (!file) throw error; }
        }
        await this.app.vault.process(file, text => updateRecord(text, course, unit, done, day()));
        this.loaded = false; await this.readRecords();
      }
      this.curriculum = this.plugin.steward.lastState?.curriculum || this.curriculum;
      new Notice(done ? "已记录本节学习；相关课程的知识覆盖已同步。" : "已取消本节完成；其他学习来源仍保留。");
    } finally { this.saving = false; this.draw(); }
  }
  renderOtherTracks() {
    const others = (this.curriculum?.courses || []).filter(c => !isSchoolCourse(c) && ["foundation", "other", "queued"].includes(c.group) && c.active && !Object.values(BINDINGS).includes(c.id));
    if (!others.length) return;
    const details = this.host.createEl("details", { cls: "cm-other-tracks" });
    details.createEl("summary", { text: `其他已登记学习 · ${others.length} 项` });
    for (const track of others) {
      const row = details.createDiv(); row.createSpan({ text: `${track.id} · ${track.title}` });
      button(row, "原计划与记录", () => track.plan ? this.owner.openOriginal(track.plan) : this.owner.selectTasks(track.id, "all"), "cm-inline-link");
    }
  }
}

module.exports = { CourseMap };
