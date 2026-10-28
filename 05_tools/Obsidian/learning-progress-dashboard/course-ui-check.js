const assert = require("assert/strict");
const Module = require("module");
const notices = [], originalLoad = Module._load;
Module._load = function load(request, parent, isMain) {
  if (request === "obsidian") return { Notice: class { constructor(message) { notices.push(message); } }, Menu: class {} };
  return originalLoad(request, parent, isMain);
};
const { CourseMap } = require("./course-map");
Module._load = originalLoad;
const catalog = require("./course-catalog");

class Element {
  constructor(tag = "section", options = {}) {
    this.tag = tag; this.tagName = tag.toUpperCase(); this.text = options.text || "";
    this.children = []; this.events = {}; this.attributes = {};
    this.dataset = new Proxy({}, { set: (target, key, value) => {
      target[key] = String(value); this.attributes["data-" + key.replace(/[A-Z]/g, letter => "-" + letter.toLowerCase())] = String(value); return true;
    } });
    this.style = { setProperty() {} };
    this.scrollTop = 0; this.scrollLeft = 0; this.classes = new Set((options.cls || "").split(/\s+/).filter(Boolean));
    this.classList = { contains: name => this.classes.has(name), add: (...names) => names.forEach(name => this.classes.add(name)),
      remove: (...names) => names.forEach(name => this.classes.delete(name)), toggle: (name, enabled) => {
        const add = enabled === undefined ? !this.classes.has(name) : enabled;
        if (add) this.classes.add(name); else this.classes.delete(name);
        return add;
      } };
    for (const [name, value] of Object.entries(options.attr || {})) this.setAttribute(name, value);
  }
  createEl(tag, options = {}) { return this.appendChild(new Element(tag, options)); }
  createDiv(options) { return this.createEl("div", options); }
  createSpan(options) { return this.createEl("span", options); }
  appendChild(child) { child.remove(); this.children.push(child); child.parentNode = this; return child; }
  append(...children) { children.forEach(child => this.appendChild(child)); }
  appendText(text) { this.text += String(text); }
  remove() {
    if (this.parentNode) this.parentNode.children.splice(this.parentNode.children.indexOf(this), 1);
    this.parentNode = null;
  }
  contains(target) { return this === target || this.children.some(child => child.contains(target)); }
  setAttribute(name, value) {
    this.attributes[name] = String(value);
    if (name === "class") this.classes = new Set(String(value).split(/\s+/).filter(Boolean));
    if (name.startsWith("data-")) this.dataset[name.slice(5).replace(/-([a-z])/g, (_, letter) => letter.toUpperCase())] = String(value);
  }
  getAttribute(name) { return this.attributes[name]; }
  addEventListener(name, callback) { this.events[name] = callback; }
  empty() { for (const child of this.children) child.parentNode = null; this.children = []; }
  matches(selector) {
    if (selector.includes(",")) return selector.split(",").some(part => this.matches(part.trim()));
    if (selector.startsWith(".")) return this.classes.has(selector.slice(1));
    const attr = selector.match(/^\[([^=\]]+)(?:=["']?([^"'\]]+)["']?)?\]$/);
    if (attr) return attr[1] in this.attributes && (attr[2] === undefined || this.attributes[attr[1]] === attr[2]);
    return this.tag === selector.toLowerCase();
  }
  querySelectorAll(selector) { return this.children.flatMap(child => [...(child.matches(selector) ? [child] : []), ...child.querySelectorAll(selector)]); }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
  closest(selector) { return this.matches(selector) ? this : this.parentNode?.closest(selector); }
  scrollIntoView(options) { this.lastScroll = options; }
  focus() { this.focused = true; }
  get textContent() { return this.text + this.children.map(child => child.textContent).join(" "); }
  set textContent(value) { this.empty(); this.text = value; }
  findButton(text) { return this.querySelectorAll("button").find(child => child.text === text); }
}

function fixture(state) {
  const owner = { plugin: {}, app: {}, saves: 0, saveWorkspace() { this.saves++; }, courseMapState: state };
  const map = new CourseMap(owner);
  map.host = new Element(); map.curriculum = { courses: [] }; map.draws = [];
  // Keep the real toolbar and callbacks; graph layout and private records are outside this check.
  for (const [method, mode] of [["renderGraph", "map"], ["renderKnowledge", "knowledge"], ["renderList", "list"]]) {
    map[method] = courses => { map.rendered = { mode, courses }; map.draws.push(mode); };
  }
  map.renderDetail = course => { map.rendered = { mode: "detail", course }; map.draws.push("detail"); };
  map.draw();
  return map;
}

async function checkShortcut(label, domain) {
  const map = fixture({ mode: "knowledge", knowledgeFilter: "learned", domain: "ml", role: "branch", track: "research",
    query: "restrictive search", focus: true, course: "mit-1806sc", comparison: "stanford-engr108" });
  assert.equal(map.state.mode, "knowledge", "Fixture must start in the persisted non-map mode");
  const shortcut = map.host.findButton(label);
  assert(shortcut, `Missing shortcut: ${label}`);
  shortcut.events.click({ stopPropagation() {} });
  await new Promise(resolve => setImmediate(resolve));
  assert.deepEqual(notices, [], "Shortcut must not fail behind a Notice");
  assert.equal(map.rendered.mode, "map", `${label} must render the course map`);
  for (const [key, value] of Object.entries({ mode: "map", knowledgeFilter: "all", domain,
    role: "all", track: "both", query: "", focus: false, course: null, comparison: "personal" })) {
    assert.equal(map.state[key], value, `${label} must reset ${key}`);
    assert.equal(map.owner.courseMapState[key], value, `${label} must persist ${key}`);
  }
  const expected = catalog.courses.filter(course => domain === "all" || ["probability", "statistics", "bayesian"].includes(course.domain));
  assert.deepEqual(map.rendered.courses.map(c => c.id).sort(), expected.map(c => c.id).sort(), `${label} must show its full course scope`);
  assert.equal(map.owner.saves, 1, "A shortcut click should save its new view once");
}

async function checkKnowledgeNavigation() {
  const selected = { course: "mit-1806sc", comparison: "stanford-engr108" };
  const map = fixture({ mode: "map", ...selected });
  await click(map.host.findButton("返回课程地图"));
  map.draws = [];
  map.host.findButton("知识覆盖").events.click({ stopPropagation() {} });
  await new Promise(resolve => setImmediate(resolve));
  assert.deepEqual(map.draws, ["knowledge"], "Entering knowledge mode must show the index, not the selected course");
  assert.equal(map.state.course, null); assert.equal(map.state.comparison, "personal");
  for (const patch of [{ domain: "probability-bayes" }, { query: "Bayes" }, { knowledgeFilter: "shared" }]) {
    map.draws = []; map.change(selected);
    assert.deepEqual(map.draws, ["detail"], "Knowledge detail must replace the index rather than follow it");
    map.draws = []; map.change(patch);
    assert.deepEqual(map.draws, ["knowledge"], "Changing a knowledge filter must return to the index");
    assert.equal(map.state.course, null); assert.equal(map.state.comparison, "personal");
    for (const [key, value] of Object.entries(patch)) assert.equal(map.state[key], value, `Keep the requested ${key} filter`);
  }
  assert.equal(map.state.domain, "probability-bayes"); assert.equal(map.state.query, "Bayes");
  assert.deepEqual(notices, [], "Navigation must not fail behind a Notice");
}

function checkPagePosition() {
  const map = fixture({ mode: "map" });
  map.owner.contentEl = Object.assign(new Element(), { scrollTop: 1234 });
  const empty = map.host.empty.bind(map.host);
  // Clearing the long map temporarily clamps the outer page's scroll position.
  map.host.empty = () => { empty(); map.owner.contentEl.scrollTop = 0; };
  map.change({ domain: "probability-bayes" });
  assert.equal(map.owner.contentEl.scrollTop, 1234, "Filtering must preserve the outer Hub position");
}

async function click(button) {
  assert(button, "Expected navigation button");
  button.events.click({ stopPropagation() {} });
  await new Promise(resolve => setImmediate(resolve));
  assert.deepEqual(notices, [], "The interaction must not fail behind a Notice");
}

function realFixture(state = {}) {
  const mit = catalog.courses.find(course => course.id === "mit-1806sc");
  const curriculum = { courses: [{ id: "MIT 18.06", plan: "canonical.md" }], weekly: {
    tasks: mit.units.map((unit, index) => ({ id: "canonical-" + unit.id, course_id: "MIT 18.06",
      title: `MIT 18.06｜${unit.id} ${unit.title}`, state: index < 8 ? "done" : "unknown",
      status: index < 8 ? "recorded_complete" : "completion_unknown", path: "canonical.md", raw: "original" }))
  } };
  const owner = { plugin: {}, app: {}, saves: 0, saveWorkspace() { this.saves++; }, courseMapState: state,
    contentEl: Object.assign(new Element(), { scrollTop: 250 }), taskDock: new Element(), board: new Element() };
  const map = new CourseMap(owner);
  map.host = owner.board.createEl("section"); map.curriculum = curriculum;
  map.draw();
  return map;
}

async function checkCourseNavigation() {
  for (const mode of ["map", "list", "knowledge"]) {
    const map = fixture({ mode });
    map.draws = []; map.openCourse("mit-1806sc");
    assert.deepEqual(map.draws, ["detail"], `${mode}: opening a course must replace its overview, not append below it`);
    assert.equal(map.host.querySelector(".cm-toolbar"), null, "Course browsing must keep its own context instead of overview filters");
    await click(map.host.findButton("返回" + ({ map: "课程地图", list: "课程清单", knowledge: "知识覆盖" }[mode])));
    assert.equal(map.state.course, null); assert.equal(map.state.mode, mode);
  }
  const map = realFixture({ mode: "map", domain: "math" });
  map.owner.contentEl.scrollTop = 725;
  const graph = map.host.querySelector(".cm-graph-scroll"); graph.scrollLeft = 135; graph.scrollTop = 420;
  map.openCourse("mit-1806sc");
  await click(map.host.findButton("先修与去向"));
  map.change({ comparison: "stanford-engr108" });
  map.owner.contentEl.scrollTop = 1125;
  map.openCourse("stanford-engr108");
  assert.equal(map.state.comparison, "personal", "The newly opened course starts from actual learning");
  map.owner.contentEl.scrollTop = 1775;
  await click(map.host.querySelector(".cm-history-back"));
  assert.equal(map.state.course, "mit-1806sc");
  assert.equal(map.state.tab, "routes", "Back must restore the tab being read");
  assert.equal(map.state.comparison, "stanford-engr108", "Back must restore the previous comparison");
  assert.equal(map.owner.contentEl.scrollTop, 1125, "Back must restore the prior course's reading position");
  await click(map.host.findButton("返回课程地图"));
  assert.equal(map.state.course, null); assert.equal(map.history.length, 0);
  assert.equal(map.state.domain, "math", "Returning must preserve the original overview filter");
  assert.equal(map.owner.contentEl.scrollTop, 725, "Returning must restore the Hub position from before course browsing");
  assert.equal(map.host.querySelector(".cm-graph-scroll").scrollTop, 420);
  assert.equal(map.host.querySelector(".cm-graph-scroll").scrollLeft, 135);
  map.openCourse("mit-1806sc"); map.openCourse("stanford-engr108");
  await click(map.host.findButton("完整地图"));
  assert.equal(map.history.length, 0, "A fresh map shortcut must not retain a stale back stack");
}

async function checkKnowledgeReturn() {
  const map = realFixture({ mode: "knowledge", domain: "math", knowledgeFilter: "learned" });
  const originId = "la.linear-combinations";
  const card = map.host.querySelector(`[data-knowledge-id="${originId}"]`);
  assert(card, "The learned shared knowledge must appear in the index");
  card.open = true; card.events.toggle();
  const provider = card.querySelectorAll("button").find(button => button.text.startsWith("Stanford ENGR108"));
  await click(provider);
  assert.equal(map.state.course, "stanford-engr108"); assert(map.state.unit);
  await click(map.host.querySelector(`[data-knowledge-open="${originId}"]`));
  const source = map.host.querySelectorAll(".cm-inspector-source").find(button => button.text === "18.06SC · 1.1");
  await click(source);
  assert.equal(map.state.course, "mit-1806sc"); assert.equal(map.state.unit, "1.1");
  assert.equal(map.host.querySelectorAll("[data-course-unit]").length, 1,
    "Following a learning source must show only the exact source lesson");
  assert(map.host.querySelector('[data-course-unit="mit-1806sc|1.1"]').querySelector(".cm-unit-body"),
    "Following a learning source must expose its content directly");
  await click(map.host.querySelector(".cm-history-back"));
  assert.equal(map.state.course, "stanford-engr108", "Source inspection must have a return route to the target course");
  await click(map.host.findButton("返回知识覆盖"));
  assert.equal(map.state.course, null); assert.equal(map.state.mode, "knowledge");
  assert.equal(map.state.knowledgeFilter, "learned");
  const returned = map.host.querySelector(`[data-knowledge-id="${originId}"]`);
  assert(returned.open, "The knowledge card must remain expanded on return");
  assert.equal(returned.lastScroll.block, "center", "Return must locate the original knowledge card");
}

async function checkLessonSelection() {
  const map = realFixture({ mode: "map" });
  const before = JSON.stringify(map.curriculum), records = [...map.records.entries()];
  map.openCourse("mit-1806sc");
  assert.equal(map.state.tab, "overview", "Opening a course without a specific unit must show the course panorama");
  await click(map.host.findButton("继续学习 · 1.9"));
  const selectedLesson = () => {
    const lessons = map.host.querySelectorAll("[data-course-unit]");
    assert.equal(lessons.length, 1, "Only the selected lesson should occupy the reading pane");
    return lessons[0];
  };
  assert.equal(map.state.unit, "1.9", "Opening the course should resume at the first unrecorded lesson");
  assert.equal(selectedLesson().dataset.courseUnit, "mit-1806sc|1.9");
  assert.equal(map.host.querySelectorAll("[data-unit-select]").length, 36,
    "The entire course directory must remain available beside the selected lesson");
  const historyLength = map.history.length;
  await click(map.host.querySelector('[data-unit-select="1.4"]'));
  assert.equal(map.state.unit, "1.4");
  assert.equal(selectedLesson().dataset.courseUnit, "mit-1806sc|1.4");
  assert(map.host.querySelector('[data-unit-select="1.4"]').classList.contains("is-selected"),
    "The directory must visibly identify the lesson being read");
  assert(map.host.querySelector('[data-unit-select="1.4"]').focused,
    "Keyboard focus must follow the newly rendered directory selection");
  assert.equal(map.history.length, historyLength, "Ordinary chapter selection must not bury the cross-course return route");
  map.draw();
  assert.equal(map.state.unit, "1.4", "A background redraw must retain the chosen lesson");
  assert.equal(selectedLesson().dataset.courseUnit, "mit-1806sc|1.4");
  await click(map.host.findButton("下一节 →"));
  assert.equal(map.state.unit, "1.5");
  await click(map.host.findButton("← 上一节"));
  assert.equal(map.state.unit, "1.4");
  await click(map.host.findButton("继续学习 · 1.9"));
  assert.equal(map.state.unit, "1.9", "Continue must return to the next unrecorded lesson, not mark it complete");
  assert.equal(map.owner.courseMapState.unit, "1.9", "The selected lesson must survive workspace restoration");
  assert.equal(JSON.stringify(map.curriculum), before, "Browsing lessons must not alter actual course completion");
  assert.deepEqual([...map.records.entries()], records, "Browsing must not write personal learning evidence");
}

async function checkOriginalPlanResume() {
  const map = realFixture({ mode: "map" });
  const courseId = "mit-1801sc", path = "original-calculus-plan.md";
  map.curriculum.courses.push({ id: "MIT 18.01", plan: path, unit_progress: { completed: 12, total: 35 } });
  const opened = [];
  map.owner.openOriginal = selectedPath => opened.push(selectedPath);
  map.openCourse(courseId);
  assert.match(map.host.querySelector(".is-course").textContent, /12 \/ 35/,
    "An unaligned course must retain completion from its original learning plan");
  assert(!map.host.querySelectorAll("button").some(button => button.text.startsWith("继续学习 ·")),
    "Unaligned syllabus modules cannot identify the next chapter in the original plan");
  await click(map.host.findButton("继续原学习计划"));
  assert.deepEqual(opened, [path], "The resume action must open the authoritative original plan");
  map.openCourse(courseId, catalog.courses.find(course => course.id === courseId).units[0].id);
  assert(map.host.querySelectorAll(".cm-check").every(check => check.disabled),
    "Unaligned modules must remain unavailable for a second completion record");
}

async function checkReadingContextPersistence() {
  const map = realFixture({ mode: "map", course: "mit-1806sc", unit: "1.4", tab: "units" });
  const evidence = map.host.querySelector(".cm-unit-evidence");
  assert(evidence, "The fixture must include expandable scope evidence");
  evidence.open = true; evidence.events.toggle();
  map.host.querySelector(".cm-directory-list").scrollTop = 360;
  map.host.querySelector(".cm-lesson-pane").scrollTop = 180;
  map.draw();
  assert.equal(map.host.querySelector(".cm-directory-list").scrollTop, 360,
    "Refreshing data must not move the reader's course directory");
  assert.equal(map.host.querySelector(".cm-lesson-pane").scrollTop, 180,
    "Refreshing data must retain the current lesson's bounded reading position");
  assert(map.host.querySelector(".cm-unit-evidence").open);
  const restoredEvidence = map.host.querySelector(".cm-unit-evidence");
  restoredEvidence.open = false; restoredEvidence.events.toggle();
  map.draw();
  assert.equal(map.host.querySelector(".cm-unit-evidence").open, false,
    "A reader's explicitly closed scope evidence must stay closed");
  await click(map.host.querySelector('[data-unit-select="1.5"]'));
  assert.equal(map.host.querySelector(".cm-lesson-pane").scrollTop, 0,
    "A newly chosen chapter must start at its own title, not the previous chapter's scroll position");
  await click(map.host.findButton("跨课对照"));
  map.host.querySelector(".cm-coverage-chapters").scrollTop = 240;
  map.draw();
  assert.equal(map.host.querySelector(".cm-coverage-chapters").scrollTop, 240,
    "Refreshing data must retain the chapter comparison reading position");
}

function checkDualMetrics() {
  const map = realFixture({ mode: "map" });
  const before = JSON.stringify(map.curriculum), records = [...map.records.entries()];
  for (const [id, lessons, knowledge, label] of [["mit-1806sc", [8, 36], [12, 58], "节"], ["stanford-engr108", [0, 16], [4, 33], "章"]]) {
    const course = catalog.courses.find(course => course.id === id);
    const node = map.host.querySelector(`[data-course-map-id="${id}"]`);
    assert(node, "Every course node must remain selectable");
    assert.match(node.querySelector(".is-course").textContent, new RegExp(`${lessons[0]} / ${lessons[1]}`));
    assert.match(node.querySelector(".is-knowledge").textContent, new RegExp(`${knowledge[0]} / ${knowledge[1]}`));
    const panel = new Element(); map.renderMetrics(panel, course);
    assert.equal(panel.querySelector(".is-course").querySelector("progress").getAttribute("aria-label"), `本课已学 ${lessons[0]}/${lessons[1]} ${label}`);
    assert.equal(panel.querySelector(".is-knowledge").querySelector("progress").getAttribute("aria-label"), `知识已学 ${knowledge[0]}/${knowledge[1]} 知识点`);
  }
  map.change({ mode: "list" });
  const stanfordCard = map.host.querySelectorAll(".cm-list-card").find(card => card.textContent.includes("ENGR108"));
  assert.match(stanfordCard.querySelector(".is-course").textContent, /0 \/ 16/);
  assert.match(stanfordCard.querySelector(".is-knowledge").textContent, /4 \/ 33/, "The list must expose both dimensions before opening a course");
  map.openCourse("stanford-engr108", "ch1");
  const rows = map.host.querySelectorAll("[data-course-unit]");
  assert(rows.some(row => row.querySelector(".is-lit")), "Shared known knowledge must light in the unstarted course");
  assert(rows.every(row => !row.querySelector(".cm-check").checked), "Shared coverage must not complete any target lesson");
  assert(rows.every(row => row.querySelector(".cm-unit-completion").textContent === "标记本节已学"),
    "Shared coverage must leave the target lesson's completion action available");
  map.change({ comparison: "mit-1806sc" });
  assert(rows.length > 0);
  assert(map.host.querySelectorAll(".cm-check").every(check => check.disabled), "Hypothetical full-course comparison must not allow lesson completion");
  assert.match(map.host.querySelector(".cm-detail").querySelector(".is-knowledge").textContent, /4 \/ 33/,
    "The course header must continue to describe personal learning during hypothetical comparison");
  assert.equal(JSON.stringify(map.curriculum), before); assert.deepEqual([...map.records.entries()], records);
  const topic = { ...catalog.courses.find(course => course.id === "cambridge-ml-bayes"), coverage: "topic-outline", unitLabel: undefined };
  const panel = new Element(); map.renderMetrics(panel, topic);
  assert.match(panel.querySelector(".is-course").textContent, /已学模块/);
  assert.match(panel.querySelector(".is-course").querySelector("progress").getAttribute("aria-label"), /模块$/,
    "Unaligned syllabus modules must not be presented as counted lectures");
}

async function checkKnowledgeInspector() {
  const map = realFixture({ mode: "map", course: "mit-1806sc", unit: "1.4", tab: "units" });
  const id = "la.matrix-multiplication", before = JSON.stringify(map.curriculum), records = [...map.records.entries()];
  const origin = { course: map.state.course, unit: map.state.unit, tab: map.state.tab };
  await click(map.host.querySelector(`[data-knowledge-open="${id}"]`));
  assert.equal(map.state.knowledge, id);
  for (const [key, value] of Object.entries(origin)) assert.equal(map.state[key], value,
    "Inspecting a knowledge point must retain the current course, chapter, and tab");
  const inspector = () => map.host.querySelector(`[data-knowledge-inspector="${id}"]`);
  assert(inspector(), "Knowledge must open in the local inspector");
  assert(inspector().textContent.includes(map.definitions.get(id).scope), "The inspector must explain the precise shared scope");
  assert.match(inspector().textContent, /18\.06SC[\s\S]*1\.4/, "The inspector must identify the actual learned source lesson");
  const targetLocation = inspector().querySelector('[data-knowledge-location="stanford-engr108|ch10"]');
  assert(targetLocation, "The inspector must expose the exact alternative course chapter");
  for (const extra of ["la.qr", "la.householder", "la.matrix-powers-basic"]) {
    assert(targetLocation.textContent.includes(map.definitions.get(extra).title),
      "The alternative chapter's additional scopes must be visible before choosing it");
  }
  inspector().querySelector(".cm-inspector-body").scrollTop = 410;
  map.draw();
  assert(inspector(), "A background refresh must preserve the inspected knowledge point");
  assert.equal(inspector().querySelector(".cm-inspector-body").scrollTop, 410,
    "A background refresh must retain the inspector's alternative-course reading position");
  await click(inspector().querySelectorAll(".cm-inspector-jump").find(button =>
    button.dataset.jumpCourse === "stanford-engr108" && button.dataset.jumpUnit === "ch10"));
  assert.equal(map.state.course, "stanford-engr108"); assert.equal(map.state.unit, "ch10");
  assert.equal(map.state.knowledge, id);
  assert.deepEqual(map.state.trace, { knowledgeId: id, courseId: "mit-1806sc", unitId: "1.4" },
    "A deliberate cross-course jump must retain the knowledge bridge and originating chapter");
  assert(inspector(), "The same knowledge point must remain open at the destination");
  const target = map.host.querySelector('[data-course-unit="stanford-engr108|ch10"]');
  assert(target, "A knowledge jump must target the exact alternative chapter");
  assert.equal(target.querySelector(".cm-check").checked, false, "Knowledge overlap cannot complete the destination chapter");
  assert.match(target.querySelector(".cm-coverage-badge").textContent, /1 \/ 4/,
    "Only matrix multiplication is learned; the other three chapter scopes remain new");
  const resumed = realFixture(JSON.parse(JSON.stringify(map.getState())));
  assert.equal(resumed.history.length, 0, "Restoring saved workspace state must exercise a fresh session without in-memory history");
  await click(resumed.host.querySelector(".cm-trace").querySelector("button"));
  assert.equal(resumed.state.course, "mit-1806sc"); assert.equal(resumed.state.unit, "1.4");
  assert.equal(resumed.state.knowledge, id);
  assert(resumed.host.querySelector(`[data-knowledge-inspector="${id}"]`),
    "A saved knowledge bridge must return to its named source even after Obsidian restarts");
  await click(map.host.querySelector(".cm-history-back"));
  assert.equal(map.state.course, origin.course); assert.equal(map.state.unit, origin.unit);
  assert.equal(map.state.knowledge, id); assert(inspector(), "Back must restore the originating inspector as well as its chapter");
  await click(map.host.querySelector(".cm-inspector-close"));
  assert.equal(map.state.knowledge, null);
  assert.equal(map.host.querySelector(".cm-inspector"), null);
  assert(map.host.querySelector(`[data-knowledge-open="${id}"]`).focused,
    "Closing the inspector must return keyboard focus to the knowledge point that opened it");
  map.openKnowledge(id); map.jumpKnowledge("stanford-engr108", "ch10", id);
  await click(map.host.querySelector('[data-unit-select="ch11"]'));
  assert.equal(map.state.knowledge, null); assert.equal(map.state.trace, null,
    "Ordinary chapter navigation must clear a stale cross-course inspection trail");
  assert.equal(JSON.stringify(map.curriculum), before); assert.deepEqual([...map.records.entries()], records);
}

async function checkInspectorSourceBoundaries() {
  const id = "la.matrix-multiplication";
  const map = realFixture({ mode: "map", course: "mit-1806sc", unit: "1.4", tab: "units" });
  map.openKnowledge(id);
  const ownSource = () => map.host.querySelectorAll(".cm-inspector-source").find(button => button.text === "18.06SC · 1.4");
  await click(ownSource());
  assert.equal(map.state.course, "mit-1806sc"); assert.equal(map.state.unit, "1.4");
  assert.equal(map.state.knowledge, id);
  assert.equal(map.state.trace, null, "Following a source already in view must not create a misleading return trail to itself");
  await click(map.host.findButton("跨课对照"));
  await click(map.host.querySelector(`[data-knowledge-open="${id}"]`));
  await click(ownSource());
  assert.equal(map.state.tab, "units", "A source link from comparison should open the exact lesson");
  await click(map.host.querySelector(".cm-history-back"));
  assert.equal(map.state.tab, "coverage", "Returning from the same course's source lesson must restore its originating comparison tab");
  assert.equal(map.state.knowledge, id);
  assert(map.host.querySelector(`[data-knowledge-inspector="${id}"]`));
  const eigenvalueLesson = catalog.courses.find(course => course.id === "mit-18065").units.find(unit => unit.outcomes.includes("la.eigenvalues"));
  assert(eigenvalueLesson, "The official matrix-methods outline must retain its eigenvalue lesson");
  map.openCourse("mit-18065", eigenvalueLesson.id); map.openKnowledge("la.eigenvalues");
  const target = map.host.querySelector('[data-knowledge-location="mit-1806sc|2.8"]');
  assert(target, "The fixture needs an unlearned chapter containing just the focused knowledge point");
  assert(!target.textContent.includes("均已有学习记录"),
    "Having no additional scopes must not mislabel the focused, unlearned scope as already learned");
}

async function checkChapterCoverage() {
  const map = realFixture({ mode: "map", course: "stanford-engr108", unit: "ch10", tab: "units" });
  const before = JSON.stringify(map.curriculum), records = [...map.records.entries()];
  for (const [tab, title] of [["overview", "课程全景"], ["units", "逐节学习"], ["coverage", "跨课对照"], ["routes", "先修与去向"]]) {
    await click(map.host.findButton(title));
    assert.equal(map.state.tab, tab);
    assert.match(map.host.querySelector(".cm-context").textContent, /ENGR108/,
      "Every course tab must retain the current course context");
    assert.equal(map.host.findButton(title).getAttribute("aria-pressed"), "true", "The active course section must stay identified");
  }
  await click(map.host.findButton("跨课对照"));
  const rows = () => map.host.querySelectorAll("[data-coverage-unit]");
  assert.equal(rows().length, 16, "Coverage should be organized around every actual course chapter");
  assert(map.host.querySelector('[data-coverage-unit="ch10"]'), "A partially covered chapter must remain visible");
  await click(map.host.querySelector('[data-coverage-filter="covered"]'));
  assert.equal(map.state.coverageFilter, "covered");
  assert.deepEqual(rows().map(row => row.dataset.coverageUnit), ["ch6"],
    "Only ENGR108 chapter 6 has every mapped scope covered by MIT 1.1–1.8");
  await click(map.host.querySelector('[data-coverage-filter="new"]'));
  assert.equal(map.state.coverageFilter, "new");
  assert.equal(rows().length, 15, "New-content filtering must include partial overlaps as well as entirely new chapters");
  assert(map.host.querySelector('[data-coverage-unit="ch10"]'));
  assert.equal(map.host.querySelector('[data-coverage-unit="ch6"]'), null);
  map.draw();
  assert.equal(map.state.coverageFilter, "new"); assert.equal(rows().length, 15);
  await click(map.host.querySelector('[data-coverage-filter="all"]'));
  assert.equal(rows().length, 16);
  const unknownScope = map.host.querySelector('[data-coverage-unit="ch10"]').querySelector('[data-knowledge-open="la.qr"]');
  assert.match(unknownScope.closest(".cm-chapter-knowledge").textContent, /^未记录 /,
    "Missing personal evidence should be described as unrecorded, not as proven lack of knowledge");
  map.change({ comparison: "mit-1806sc" });
  assert(map.host.querySelectorAll(".cm-hint").some(hint => hint.textContent.includes("不表示本人学过")),
    "Whole-course comparison must explain that green means scope overlap, not personal completion");
  const overlap = map.host.querySelector('[data-coverage-unit="ch10"]').querySelector('[data-knowledge-open="la.qr"]');
  assert(overlap.closest(".cm-chapter-knowledge").classList.contains("is-lit"),
    "QR must appear as a scope overlap when the full MIT course is compared");
  await click(overlap);
  assert.equal(map.state.comparison, "mit-1806sc", "Knowledge inspection must retain the chosen comparison");
  const personalState = map.host.querySelector(".cm-inspector-state");
  assert.equal(personalState.classList.contains("is-lit"), false);
  assert.match(personalState.textContent, /暂无本人的学习记录/,
    "The inspector must still distinguish actual personal learning from simulated full-course overlap");
  assert.equal(JSON.stringify(map.curriculum), before); assert.deepEqual([...map.records.entries()], records);
}

async function checkComparisonJourney() {
  const map = realFixture({ course: "mit-1806sc", unit: "1.4", tab: "units" });
  const before = JSON.stringify(map.curriculum);
  map.openKnowledge("la.matrix-multiplication", "1.4");
  await click(map.host.querySelector('[data-compare-course="stanford-engr108"]'));
  assert.equal(map.state.tab, "coverage"); assert.equal(map.state.comparison, "personal");
  assert.equal(map.state.comparisonSource, "mit-1806sc", "Opening a course comparison must retain its source as an explicit baseline choice");
  await click(map.host.findButton("18.06SC 全部教学范围"));
  await click(map.host.querySelector('[data-coverage-filter="new"]'));
  map.host.querySelector(".cm-coverage-chapters").scrollTop = 260;
  await click(map.host.querySelector('[data-coverage-unit="ch10"]').querySelector("button"));
  assert.equal(map.state.tab, "units"); assert.equal(map.state.unit, "ch10");
  assert.equal(map.state.comparison, "mit-1806sc", "Inspecting a comparison chapter must not silently switch back to personal learning");
  assert.equal(map.state.studyComparison, true);
  assert.match(map.host.querySelector(".cm-coverage-badge").textContent, /3 \/ 4/);
  assert(map.host.querySelector(".cm-check").disabled, "Simulated comparison must remain read-only for completion");
  await click(map.host.querySelector('[data-unit-select="ch11"]'));
  assert.equal(map.state.comparison, "mit-1806sc", "Directory navigation must keep the active comparison task");
  await click(map.host.findButton("← 返回全课对照"));
  assert.equal(map.state.tab, "coverage"); assert.equal(map.state.coverageFilter, "new");
  assert.equal(map.state.comparison, "mit-1806sc");
  assert.equal(map.host.querySelector(".cm-coverage-chapters").scrollTop, 260);
  await click(map.host.findButton("课程全景"));
  assert.equal(map.state.comparison, "personal");
  map.openUnit("stanford-engr108", "ch10");
  assert.equal(map.state.comparison, "personal"); assert.equal(map.state.studyComparison, false);
  assert.equal(map.comparisonReturn, null, "Ordinary learning must not resurrect an unrelated older comparison return route");
  assert.match(map.host.querySelector(".cm-coverage-badge").textContent, /1 \/ 4/);
  assert.equal(JSON.stringify(map.curriculum), before);
}

async function checkExactKnowledgeOrigin() {
  const map = realFixture({ course: "stanford-engr108", unit: "ch1", tab: "coverage" });
  const chip = () => map.host.querySelector('[data-coverage-unit="ch6"]').querySelector('[data-knowledge-open="la.linear-combinations"]');
  await click(chip());
  assert.equal(map.state.unit, "ch1", "Inspecting comparison content should not replace the selected learning chapter");
  assert.equal(map.state.knowledgeUnit, "ch6", "The clicked comparison row, not the selected learning chapter, owns the knowledge origin");
  assert.match(map.host.querySelector(".cm-inspector-head").textContent, /ENGR108 › 6 ›/);
  await click(map.host.querySelector(".cm-inspector-close"));
  assert.equal(map.state.knowledgeUnit, null); assert(chip().focused);
  assert(!map.host.querySelector('[data-coverage-unit="ch1"]').querySelector('[data-knowledge-open="la.linear-combinations"]').focused,
    "Closing must not focus the first duplicate knowledge chip in a different chapter");
  await click(chip());
  await click(map.host.querySelectorAll(".cm-inspector-source").find(button => button.text === "18.06SC · 1.1"));
  assert.equal(map.state.trace.unitId, "ch6");
  await click(map.host.querySelector(".cm-history-back"));
  assert.equal(map.state.tab, "coverage"); assert.equal(map.state.knowledgeUnit, "ch6");
}

async function checkFullHostReadingPersistence() {
  const map = realFixture({ course: "mit-1806sc", unit: "1.4", tab: "units" });
  map.openKnowledge("la.matrix-multiplication", "1.4");
  const position = () => [map.host.querySelector(".cm-directory-list").scrollTop,
    map.host.querySelector(".cm-lesson-pane").scrollTop, map.host.querySelector(".cm-inspector-body").scrollTop];
  map.host.querySelector(".cm-directory-list").scrollTop = 360;
  map.host.querySelector(".cm-lesson-pane").scrollTop = 180;
  map.host.querySelector(".cm-inspector-body").scrollTop = 410;
  map.jumpKnowledge("stanford-engr108", "ch10", "la.matrix-multiplication");
  map.backCourse();
  assert.deepEqual(position(), [360, 180, 410], "Cross-course Back must restore every bounded reading region, not just outer pageTop");
  // Student OS refresh captures before clearing its board; the old host is detached before render builds its replacement.
  map.captureViewPosition();
  const oldHost = map.host; oldHost.isConnected = false;
  map.owner.board.empty(); map.readRecords = async () => {};
  map.render(map.owner.board, map.curriculum);
  assert.notEqual(map.host, oldHost);
  assert.deepEqual(position(), [360, 180, 410], "A full parent remount must recover reading positions after the old DOM is detached");
}

async function checkCoursePanorama() {
  const map = realFixture({ mode: "map", overviewMode: "matrix" });
  const before = JSON.stringify(map.curriculum), id = "la.matrix-multiplication";
  map.openCourse("mit-1806sc");
  assert.equal(map.state.tab, "overview"); assert(map.host.querySelector(".cm-knowledge-matrix"));
  const dot = (scope, unit) => map.host.querySelectorAll(".cm-matrix-dot").find(el => el.dataset.knowledgeOpen === scope && el.dataset.knowledgeUnit === unit);
  assert(dot(id, "1.4").classList.contains("is-lit"), "The overview matrix must reflect canonical learned scopes");
  map.host.querySelector(".cm-matrix-scroll").scrollLeft = 320;
  map.host.querySelector(".cm-matrix-scroll").scrollTop = 300;
  await click(dot(id, "1.4"));
  assert.equal(map.state.knowledgeUnit, "1.4");
  assert.equal(map.host.querySelector(".cm-matrix-scroll").scrollLeft, 320, "Opening a matrix knowledge point must not move its chapter columns");
  assert.equal(map.host.querySelector(".cm-matrix-scroll").scrollTop, 300, "Opening a matrix knowledge point must retain its knowledge row position");
  await click(map.host.querySelector(".cm-inspector-close"));
  assert.equal(map.host.querySelector(".cm-matrix-scroll").scrollLeft, 320);
  assert.equal(map.host.querySelector(".cm-matrix-scroll").scrollTop, 300);
  map.openCourse("stanford-engr108");
  await click(map.host.querySelector('[data-module-select="矩阵与线性系统"]'));
  assert(dot(id, "ch10").classList.contains("is-lit"), "The same canonical scope must light in another course's matrix");
  assert(!dot("la.qr", "ch10").classList.contains("is-lit"), "Extra scopes must remain unrecorded in the target course");
  assert.match(map.host.querySelector(".is-course").textContent, /0 \/ 16/);
  map.host.querySelector(".cm-matrix-scroll").scrollLeft = 240;
  map.host.querySelector(".cm-matrix-scroll").scrollTop = 150;
  await click(map.host.querySelectorAll(".cm-matrix-unit").find(button => button.text === "10"));
  assert.equal(map.state.unit, "ch10"); assert.equal(map.state.tab, "units");
  map.backCourse();
  assert.equal(map.state.tab, "overview"); assert.equal(map.state.module, "矩阵与线性系统");
  assert.equal(map.host.querySelector(".cm-matrix-scroll").scrollLeft, 240);
  assert.equal(map.host.querySelector(".cm-matrix-scroll").scrollTop, 150,
    "Returning from a chapter must restore both axes of its course matrix");
  assert.equal(JSON.stringify(map.curriculum), before, "Exploring matrices must not modify attendance or personal learning evidence");
}

async function checkResumeEntry() {
  const map = realFixture({ mode: "map" }), before = JSON.stringify(map.curriculum);
  const entry = map.host.querySelector('[data-resume-course="mit-1806sc"]');
  const next = catalog.courses.find(course => course.id === "mit-1806sc").units.find(unit => unit.id === "1.9");
  assert(entry.textContent.includes(next.title), "A resume entry must name the actual upcoming lesson, not just the course");
  await click(entry);
  assert.equal(map.state.course, "mit-1806sc"); assert.equal(map.state.unit, "1.9"); assert.equal(map.state.tab, "units");
  assert(map.host.querySelector('[data-course-unit="mit-1806sc|1.9"]'), "Resume must arrive directly at the advertised lesson");
  assert.equal(JSON.stringify(map.curriculum), before, "Resuming must not complete the lesson");
  map.returnToOverview();
  for (const task of map.curriculum.weekly.tasks) { task.state = "done"; task.status = "recorded_complete"; }
  map.draw();
  const completed = map.host.querySelector('[data-resume-course="mit-1806sc"]');
  assert.match(completed.textContent, /已完成.*查看全景/);
  await click(completed);
  assert.equal(map.state.tab, "overview", "A fully recorded course should offer review without inventing another pending unit");

  const original = realFixture({ mode: "map" }), opened = [];
  original.curriculum.courses.push({ id: "MIT 18.01", plan: "original-calculus.md", unit_progress: { completed: 12, total: 35 } });
  original.owner.openOriginal = path => opened.push(path); original.draw();
  await click(original.host.querySelector('[data-resume-course="mit-1801sc"]'));
  assert.deepEqual(opened, ["original-calculus.md"], "An unaligned course must resume its authoritative plan");
  assert.equal(original.state.course, null, "An original-plan resume must not pretend that the first catalog module is next");
}

async function checkLearningRoute() {
  const map = realFixture({ mode: "map" }), before = JSON.stringify(map.curriculum), records = [...map.records.entries()];
  map.openCourse("mit-1806sc");
  assert.equal(map.state.overviewMode, "route");
  assert(map.host.querySelector(".cm-route-scroll")); assert.equal(map.host.querySelector(".cm-knowledge-matrix"), null);
  const prefix = map.host.querySelector(".cm-route-completed");
  assert.equal(prefix.open, false, "The completed prefix should be collapsed when resuming the next lesson");
  assert.equal(prefix.querySelectorAll("[data-route-unit]").length, 8);
  const pending = map.host.querySelector(".cm-route-pending-list").querySelectorAll("[data-route-unit]");
  assert.equal(pending[0].dataset.routeUnit, "1.9"); assert(pending[0].classList.contains("is-next"));
  map.host.querySelector(".cm-route-scroll").scrollTop = 310;
  await click(map.host.querySelector('[data-overview-mode="matrix"]'));
  map.host.querySelector(".cm-matrix-scroll").scrollTop = 270;
  map.host.querySelector(".cm-matrix-scroll").scrollLeft = 210;
  await click(map.host.querySelector('[data-overview-mode="route"]'));
  assert.equal(map.host.querySelector(".cm-route-scroll").scrollTop, 310, "Route and matrix must keep separate reading positions");
  await click(map.host.querySelector('[data-overview-mode="matrix"]'));
  assert.equal(map.host.querySelector(".cm-matrix-scroll").scrollTop, 270);
  assert.equal(map.host.querySelector(".cm-matrix-scroll").scrollLeft, 210);
  await click(map.host.querySelector('[data-overview-mode="route"]'));
  const completed = map.host.querySelector(".cm-route-completed"); completed.open = true; completed.events.toggle();
  await click(map.host.querySelector('[data-route-unit="1.4"]').querySelector('[data-knowledge-open="la.matrix-multiplication"]'));
  assert.equal(map.state.knowledgeUnit, "1.4"); assert.equal(map.state.unit, "1.9");
  assert(map.host.querySelector(".cm-route-completed").open, "Inspecting an earlier learned scope must retain its visible route location");
  map.openCourse("stanford-engr108");
  await click(map.host.querySelector('[data-module-select="矩阵与线性系统"]'));
  const row = map.host.querySelector('[data-route-unit="ch10"]');
  assert(row.querySelector('[data-knowledge-open="la.matrix-multiplication"]').classList.contains("is-lit"));
  assert(!row.classList.contains("is-done"), "Shared scope coverage must not mark the alternative course's chapter as attended");
  assert(!row.querySelector('[data-knowledge-open="la.qr"]').classList.contains("is-lit"));
  await click(row.querySelector(".cm-route-unit-title"));
  await click(map.host.querySelector('[data-unit-select="ch13"]'));
  await click(map.host.findButton("课程全景"));
  assert.equal(map.state.module, "最小二乘及应用");
  assert.equal(map.host.querySelector(".cm-route-scroll").dataset.moduleId, "最小二乘及应用",
    "Returning to the panorama after chapter navigation must show the current chapter's module");
  map.backCourse();
  assert.equal(map.state.module, "矩阵与线性系统", "Explicit Back must still recover the original overview browsing context");
  assert.equal(JSON.stringify(map.curriculum), before); assert.deepEqual([...map.records.entries()], records);
}

async function checkGapNavigation() {
  const map = realFixture({ course: "mit-1806sc", unit: "1.4", tab: "units" });
  const before = JSON.stringify(map.curriculum), records = [...map.records.entries()];
  map.openComparison("stanford-engr108", "mit-1806sc");
  await click(map.host.findButton("18.06SC 全部教学范围"));
  await click(map.host.querySelector('[data-gap-start="ch1"]'));
  assert.equal(map.state.unit, "ch1"); assert.equal(map.state.coverageFilter, "new");
  assert.equal(map.state.comparison, "mit-1806sc"); assert.equal(map.state.studyComparison, true);
  await click(map.host.querySelector('[data-unit-select="ch4"]'));
  await click(map.host.findButton("下一处含额外范围 →"));
  assert.equal(map.state.unit, "ch7", "Gap navigation must skip chapters 5 and 6 whose mapped scopes fully overlap");
  await click(map.host.findButton("← 上一处含额外范围"));
  assert.equal(map.state.unit, "ch4");
  await click(map.host.querySelector('[data-unit-select="ch14"]'));
  await click(map.host.findButton("下一处含额外范围 →"));
  assert.equal(map.state.unit, "ch16", "Navigation must use official unit IDs even when chapter numbering is nonconsecutive");
  await click(map.host.findButton("← 返回全课对照"));
  assert.equal(map.state.tab, "coverage"); assert.equal(map.state.coverageFilter, "new");
  assert.equal(map.state.comparison, "mit-1806sc"); assert.equal(map.state.comparisonSource, "mit-1806sc");
  assert.equal(JSON.stringify(map.curriculum), before); assert.deepEqual([...map.records.entries()], records);
}

async function checkGapScopeBoundaries() {
  for (const [courseId, unitId, message] of [
    ["duke-sta601", "continuation-hmc", "本章知识范围待核对"],
    ["mit-1806sc", "1.2", "概览/复习，不参与差异比较"]
  ]) {
    const map = realFixture({ mode: "map" }), before = JSON.stringify(map.curriculum), records = [...map.records.entries()];
    map.openComparison(courseId, null);
    await click(map.host.querySelector(".cm-gap-start"));
    assert.equal(map.state.studyComparison, true); assert.equal(map.state.coverageFilter, "new");
    await click(map.host.querySelector(`[data-unit-select="${unitId}"]`));
    assert.equal(map.state.unit, unitId);
    const status = map.host.querySelector(".cm-lesson-navigation").querySelector("span").textContent;
    assert.equal(status, message, "An out-of-sequence chapter must explain why it is excluded from gap comparison");
    assert(!/已覆盖|已重合/.test(status), "Unmapped or overview-only content must not be presented as confirmed covered scope");
    assert.equal(JSON.stringify(map.curriculum), before); assert.deepEqual([...map.records.entries()], records);
  }
}

function checkExpandedDirectories() {
  for (const course of catalog.courses.filter(course => course.outlineVerified)) {
    const first = course.units[0], last = course.units.at(-1);
    const map = realFixture({ mode: "map", course: course.id, unit: first.id, tab: "units" });
    assert(map.host.querySelector(`[data-course-unit="${course.id}|${first.id}"]`), course.id + ' first unit must render');
    assert.match(map.host.querySelector('.cm-unit-number').textContent, new RegExp(String(first.label).replace(/[.*+?^${}()|[\]\\]/g, '\\$&')));
    map.selectUnit(last.id);
    assert(map.host.querySelector(`[data-course-unit="${course.id}|${last.id}"]`), course.id + ' last unit must remain reachable');
    assert.equal(map.records.size, 0, 'Browsing expanded outlines must not write learning progress');
    map.change({ tab: 'overview' });
    assert(map.host.querySelector('.cm-overview-meta').textContent.includes(`${course.units.length} ${course.unitLabel}`));
    if (course.outlineComplete === false) assert(map.host.querySelector('.cm-overview-meta').textContent.includes('目录尚待补全'));
  }
}

async function checkTaskRecordMount() {
  const map = realFixture({ mode: "map", course: "mit-1806sc", unit: "1.8", tab: "units" });
  let placements = 0;
  map.owner.placeTaskDetail = () => {
    const detail = map.owner.taskDetail; if (!detail) return;
    const row = map.host.querySelectorAll("[data-task-id]").find(row => row.dataset.taskId === detail.controller.task.id && row.dataset.key === detail.rowKey);
    if (!row) { map.owner.taskDock.appendChild(detail.host); map.owner.taskDock.hidden = true; return; }
    const target = row.querySelector(".sos-task-detail"); assert(target, "A unit needs an inline task-form mount");
    target.appendChild(detail.host); placements++;
  };
  map.owner.openTask = (task, report, mode, rowKey) => {
    const host = new Element("form"); host.createEl("textarea").value = "Unsubmitted personal reflection";
    map.owner.taskDetail = { host, controller: { task }, rowKey };
    map.owner.placeTaskDetail();
  };
  const row = map.host.querySelector('[data-course-unit="mit-1806sc|1.8"]');
  assert.equal(row.dataset.taskId, "canonical-1.8");
  assert.notEqual(row.tagName, "DETAILS", "The selected lesson's form must be visible without opening another accordion");
  await click(row.findButton("学习记录"));
  assert(row.querySelector(".cm-unit-body"));
  const form = map.owner.taskDetail.host, key = map.owner.taskDetail.rowKey;
  assert.equal(form.parentNode, row.querySelector(".sos-task-detail"), "Opening a record must stay in its map unit");
  const empty = map.host.empty.bind(map.host);
  map.host.empty = () => {
    assert(map.owner.taskDock.contains(form), "Redraw must detach the live form before destroying its old unit");
    empty();
  };
  map.draw();
  const replacement = map.host.querySelector('[data-course-unit="mit-1806sc|1.8"]');
  assert.equal(map.owner.taskDetail.host, form, "Redraw must reuse the existing editor instance");
  assert.equal(map.owner.taskDetail.rowKey, key);
  assert.equal(form.parentNode, replacement.querySelector(".sos-task-detail"));
  assert.equal(form.querySelector("textarea").value, "Unsubmitted personal reflection", "Redraw must preserve unsaved user text");
  assert(replacement.querySelector(".cm-unit-body"));
  assert.equal(placements, 2);
  await click(map.host.querySelector('[data-knowledge-open="la.nullspace.solution-basis"]'));
  assert.equal(map.owner.taskDetail.host, form, "Opening the knowledge inspector must preserve the live task editor");
  assert.equal(form.querySelector("textarea").value, "Unsubmitted personal reflection");
  await click(map.host.querySelector(".cm-inspector-close"));
  assert.equal(form.parentNode, map.host.querySelector('[data-course-unit="mit-1806sc|1.8"]').querySelector(".sos-task-detail"));
  await click(map.host.querySelector('[data-unit-select="1.9"]'));
  assert(map.owner.taskDock.contains(form), "Changing the selected lesson must preserve the draft from the previous lesson");
  assert.equal(map.state.unit, "1.9");
  await click(map.host.findButton("返回正在填写的记录"));
  assert.equal(map.state.unit, "1.8");
  assert.equal(form.parentNode, map.host.querySelector('[data-course-unit="mit-1806sc|1.8"]').querySelector(".sos-task-detail"));
  assert.equal(form.querySelector("textarea").value, "Unsubmitted personal reflection");
  map.openCourse("stanford-engr108");
  assert(map.owner.taskDock.contains(form), "Leaving the source course must preserve its active form");
  await click(map.host.findButton("返回正在填写的记录"));
  assert.equal(map.state.course, "mit-1806sc"); assert.equal(map.state.unit, "1.8");
  assert.equal(map.owner.taskDetail.host, form);
  assert.equal(form.parentNode, map.host.querySelector('[data-course-unit="mit-1806sc|1.8"]').querySelector(".sos-task-detail"));
  assert.equal(form.querySelector("textarea").value, "Unsubmitted personal reflection", "The visible return route must recover the same unsaved draft");
  assert.equal(placements, 6);
}

(async () => {
  const failures = [];
  for (const [label, domain] of [["完整地图", "all"], ["概率与贝叶斯", "probability-bayes"]]) {
    try { await checkShortcut(label, domain); }
    catch (error) { failures.push(`${label}: ${error.message}`); }
  }
  try { await checkKnowledgeNavigation(); }
  catch (error) { failures.push(`Knowledge navigation: ${error.message}`); }
  try { checkPagePosition(); }
  catch (error) { failures.push(`Page position: ${error.message}`); }
  const originalDocument = global.document;
  global.document = { createElementNS: (_namespace, tag) => new Element(tag) };
  try {
    for (const [label, check] of [["Course navigation", checkCourseNavigation], ["Knowledge return", checkKnowledgeReturn], ["Lesson selection", checkLessonSelection],
      ["Original plan resume", checkOriginalPlanResume], ["Reading context", checkReadingContextPersistence],
      ["Dual metrics", checkDualMetrics], ["Knowledge inspector", checkKnowledgeInspector], ["Inspector source boundaries", checkInspectorSourceBoundaries],
      ["Chapter coverage", checkChapterCoverage], ["Comparison journey", checkComparisonJourney], ["Exact knowledge origin", checkExactKnowledgeOrigin],
      ["Full host reading position", checkFullHostReadingPersistence], ["Course panorama", checkCoursePanorama], ["Resume entry", checkResumeEntry],
      ["Learning route", checkLearningRoute], ["Gap navigation", checkGapNavigation], ["Gap scope boundaries", checkGapScopeBoundaries],
      ["Expanded directories", checkExpandedDirectories], ["Task record mount", checkTaskRecordMount]]) {
      try { await check(); } catch (error) { failures.push(`${label}: ${error.message}`); }
    }
  } finally { if (originalDocument === undefined) delete global.document; else global.document = originalDocument; }
  assert.deepEqual(failures, [], failures.join("\n"));
  console.log("course-ui-check: shortcuts, course/knowledge return, single-lesson navigation, dual progress, knowledge inspector, chapter coverage and live task-form preservation passed");
})().catch(error => { console.error(error.message); process.exitCode = 1; });
