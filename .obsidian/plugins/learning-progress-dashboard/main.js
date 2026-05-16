const { ItemView, Notice, Plugin, TFile, normalizePath, setIcon } = require("obsidian");

const VIEW_TYPE = "learning-progress-dashboard-view";
const DATA_PATH = "98_attachment/vault-home/learning-board.md";

const COURSE_ROOTS = {
  "01_Math": "Math",
  "02_Economy": "Economy",
  "03_Computer_Science": "Computer Science",
};

const ROOT_ORDER = ["Math", "Economy", "Computer Science"];
const ROOT_FOLDER_BY_LABEL = {
  Math: "01_Math",
  Economy: "02_Economy",
  "Computer Science": "03_Computer_Science",
};
const LESSON_RE = /^(\d{1,2})[_-](.+)$/;
const SECTION_RE = /^(\d+)[_-]/;
const EXCLUDE_KEYWORDS = [
  "作业",
  "考试",
  "划重点",
  "补充",
  "course map",
  "exam",
  "review",
  "roadmap",
  "index",
  "main",
  "零散",
];

const STATE_META = {
  raw: {
    mark: "○",
    label: "未处理",
    description: "还没有开始加工",
  },
  active: {
    mark: "◐",
    label: "加工中",
    description: "听过/有材料/有笔记，但还没到可复习",
  },
  reviewable: {
    mark: "●",
    label: "可复习",
    description: "已经整理到可回顾、可输出的程度",
  },
};

function hashText(value) {
  let hash = 0;
  for (let index = 0; index < value.length; index += 1) {
    hash = (hash << 5) - hash + value.charCodeAt(index);
    hash |= 0;
  }
  return Math.abs(hash).toString(36);
}

function makeId(value) {
  const safe = value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return `${safe || "item"}-${hashText(value)}`;
}

function readJsonBlock(markdown) {
  const match = markdown.match(/```learning-board-json\s*\n([\s\S]*?)\n```/);
  if (!match) return null;
  return JSON.parse(match[1]);
}

function courseSortKey(course) {
  const rootIndex = ROOT_ORDER.indexOf(course.root);
  return [
    rootIndex === -1 ? 99 : rootIndex,
    Number.isFinite(Number(course.order)) ? Number(course.order) : 9999,
    course.title || "",
  ];
}

function compareSortKey(a, b) {
  const left = courseSortKey(a);
  const right = courseSortKey(b);
  for (let index = 0; index < left.length; index += 1) {
    if (left[index] < right[index]) return -1;
    if (left[index] > right[index]) return 1;
  }
  return 0;
}

function normalizeRoot(value) {
  const input = String(value || "").trim().toLowerCase();
  if (["math", "01_math", "01 math"].includes(input)) return "Math";
  if (["economy", "econ", "02_economy", "02 economy"].includes(input)) return "Economy";
  if (["computer science", "cs", "03_computer_science", "03 computer science"].includes(input)) {
    return "Computer Science";
  }
  return null;
}

function nextCourseOrder(courses) {
  return courses.reduce((max, course) => Math.max(max, Number(course.order) || 0), 0) + 1;
}

function nextLessonLabel(course) {
  const labels = (course.lessons || [])
    .map((lesson) => String(lesson.label || "").match(/\d+$/))
    .filter(Boolean)
    .map((match) => Number(match[0]));
  const next = labels.length > 0 ? Math.max(...labels) + 1 : 1;
  return String(next).padStart(2, "0");
}

function serializeData(data) {
  const payload = {
    ...data,
    version: 1,
    updatedAt: new Date().toISOString(),
    courses: [...data.courses].sort(compareSortKey),
  };
  return [
    "---",
    "learningBoard: true",
    "version: 1",
    `updated: "${payload.updatedAt}"`,
    "---",
    "",
    "# Learning Progress Board",
    "",
    "This file is the single source of truth for the Learning Progress Dashboard plugin.",
    "You can hand-edit it, but keep the fenced JSON block valid.",
    "",
    "```learning-board-json",
    JSON.stringify(payload, null, 2),
    "```",
    "",
  ].join("\n");
}

function isLessonFile(file) {
  const basename = file.basename.toLowerCase();
  if (!LESSON_RE.test(file.basename)) return false;
  return !EXCLUDE_KEYWORDS.some((keyword) => basename.includes(keyword.toLowerCase()));
}

function lessonTitle(file) {
  const match = file.basename.match(LESSON_RE);
  if (!match) return file.basename;
  return match[2].replace(/[_-]+/g, " ").trim();
}

function lessonLabelAndSort(file, coursePath) {
  const match = file.basename.match(LESSON_RE);
  const lessonNumber = match ? Number(match[1]) : 999;
  const lessonLabel = match ? match[1].padStart(2, "0") : "?";
  const relative = file.path.slice(coursePath.length + 1);
  const parentParts = relative.split("/").slice(0, -1);
  if (parentParts.length > 0) {
    const sectionMatch = parentParts[0].match(SECTION_RE);
    if (sectionMatch) {
      const sectionNumber = Number(sectionMatch[1]);
      return {
        label: `${sectionNumber}.${lessonLabel}`,
        sort: sectionNumber + lessonNumber / 100,
      };
    }
  }
  return { label: lessonLabel, sort: lessonNumber };
}

class LearningBoardStore {
  constructor(app) {
    this.app = app;
  }

  async ensureFolder(path) {
    const parts = path.split("/");
    let current = "";
    for (let index = 0; index < parts.length - 1; index += 1) {
      current = current ? `${current}/${parts[index]}` : parts[index];
      if (!this.app.vault.getAbstractFileByPath(current)) {
        await this.app.vault.createFolder(current);
      }
    }
  }

  async ensureDataFile() {
    const normalized = normalizePath(DATA_PATH);
    let file = this.app.vault.getFileByPath(normalized);
    if (file) return file;
    await this.ensureFolder(normalized);
    const data = await this.scanData(null);
    file = await this.app.vault.create(normalized, serializeData(data));
    return file;
  }

  async read() {
    const file = await this.ensureDataFile();
    const markdown = await this.app.vault.cachedRead(file);
    const parsed = readJsonBlock(markdown);
    if (!parsed || !Array.isArray(parsed.courses)) {
      const data = await this.scanData(null);
      await this.write(data);
      return data;
    }
    return {
      version: 1,
      updatedAt: parsed.updatedAt || new Date().toISOString(),
      courses: parsed.courses,
    };
  }

  async write(data) {
    const file = await this.ensureDataFile();
    await this.app.vault.modify(file, serializeData(data));
  }

  async rescan(existingData) {
    const data = await this.scanData(existingData);
    await this.write(data);
    return data;
  }

  async scanData(existingData) {
    const existing = existingData && Array.isArray(existingData.courses) ? existingData : { courses: [] };
    const existingByPath = new Map(existing.courses.map((course) => [course.path, course]));
    const scannedCourses = new Map();

    for (const file of this.app.vault.getMarkdownFiles()) {
      const parts = file.path.split("/");
      if (parts.length < 3) continue;
      const rootName = parts[0];
      const rootLabel = COURSE_ROOTS[rootName];
      if (!rootLabel || !isLessonFile(file)) continue;

      const courseTitle = parts[1];
      const coursePath = `${rootName}/${courseTitle}`;
      const labelData = lessonLabelAndSort(file, coursePath);
      const course =
        scannedCourses.get(coursePath) ||
        {
          id: makeId(coursePath),
          title: courseTitle,
          root: rootLabel,
          path: coursePath,
          visible: true,
          order: scannedCourses.size + 1,
          lessons: [],
        };
      course.lessons.push({
        id: makeId(file.path),
        label: labelData.label,
        title: lessonTitle(file),
        notePath: file.path,
        state: "active",
        remark: "",
        sort: labelData.sort,
      });
      scannedCourses.set(coursePath, course);
    }

    const resultByPath = new Map();
    let maxOrder = 0;
    for (const existingCourse of existing.courses) {
      const cloned = {
        ...existingCourse,
        lessons: Array.isArray(existingCourse.lessons) ? existingCourse.lessons.map((lesson) => ({ ...lesson })) : [],
      };
      maxOrder = Math.max(maxOrder, Number(cloned.order) || 0);
      resultByPath.set(cloned.path, cloned);
    }

    const sortedScanned = [...scannedCourses.values()].sort(compareSortKey);
    for (const scanned of sortedScanned) {
      const current = resultByPath.get(scanned.path);
      if (!current) {
        maxOrder += 1;
        scanned.order = maxOrder;
        scanned.lessons.sort((a, b) => a.sort - b.sort || a.title.localeCompare(b.title));
        scanned.lessons.forEach((lesson) => delete lesson.sort);
        resultByPath.set(scanned.path, scanned);
        continue;
      }

      const existingLessons = new Map();
      for (const lesson of current.lessons) {
        existingLessons.set(lesson.notePath || lesson.id, lesson);
      }
      for (const lesson of scanned.lessons.sort((a, b) => a.sort - b.sort || a.title.localeCompare(b.title))) {
        const key = lesson.notePath || lesson.id;
        if (!existingLessons.has(key)) {
          delete lesson.sort;
          current.lessons.push(lesson);
        }
      }
    }

    return {
      version: 1,
      updatedAt: new Date().toISOString(),
      courses: [...resultByPath.values()].sort(compareSortKey),
    };
  }
}

class LearningProgressDashboardView extends ItemView {
  constructor(leaf, plugin) {
    super(leaf);
    this.plugin = plugin;
    this.data = null;
    this.query = "";
    this.rootFilter = "All";
    this.showHidden = false;
    this.selected = null;
  }

  getViewType() {
    return VIEW_TYPE;
  }

  getDisplayText() {
    return "Learning Progress Dashboard";
  }

  getIcon() {
    return "map";
  }

  async onOpen() {
    await this.refresh();
  }

  async refresh() {
    this.data = await this.plugin.store.read();
    this.render();
  }

  render() {
    const container = this.containerEl.children[1];
    container.empty();
    container.addClass("lpd-root");

    const shell = container.createDiv({ cls: "lpd-shell" });
    this.renderSidebar(shell.createDiv({ cls: "lpd-sidebar" }));
    this.renderMain(shell.createDiv({ cls: "lpd-main" }));
    this.renderDetail(shell.createDiv({ cls: "lpd-detail" }));
  }

  visibleCourses() {
    if (!this.data) return [];
    const query = this.query.trim().toLowerCase();
    return [...this.data.courses]
      .sort(compareSortKey)
      .filter((course) => this.showHidden || course.visible !== false)
      .filter((course) => this.rootFilter === "All" || course.root === this.rootFilter)
      .filter((course) => {
        if (!query) return true;
        return [course.title, course.path, course.root].some((value) =>
          String(value || "").toLowerCase().includes(query)
        );
      });
  }

  renderSidebar(sidebar) {
    sidebar.createDiv({ cls: "lpd-kicker", text: "Academic Vault" });
    sidebar.createEl("h1", { text: "Learning Progress" });
    sidebar.createDiv({
      cls: "lpd-muted",
      text: "一门课一条轨道，一节课一个点。",
    });

    const search = sidebar.createEl("input", {
      cls: "lpd-search",
      attr: { type: "search", placeholder: "Filter courses..." },
    });
    search.value = this.query;
    search.addEventListener("input", () => {
      this.query = search.value;
      this.render();
    });

    const roots = sidebar.createDiv({ cls: "lpd-filter-list" });
    ["All", ...ROOT_ORDER].forEach((root) => {
      const button = roots.createEl("button", {
        cls: `lpd-filter ${this.rootFilter === root ? "is-active" : ""}`,
        text: root,
      });
      button.addEventListener("click", () => {
        this.rootFilter = root;
        this.render();
      });
    });

    const showHidden = sidebar.createEl("label", { cls: "lpd-checkbox-line" });
    const hiddenInput = showHidden.createEl("input", { attr: { type: "checkbox" } });
    hiddenInput.checked = this.showHidden;
    showHidden.createSpan({ text: "Show hidden courses" });
    hiddenInput.addEventListener("change", () => {
      this.showHidden = hiddenInput.checked;
      this.render();
    });

    const actions = sidebar.createDiv({ cls: "lpd-sidebar-actions" });
    const rescan = actions.createEl("button", { cls: "lpd-primary", text: "Rescan courses" });
    rescan.addEventListener("click", async () => {
      await this.plugin.rescanCourses();
    });
    const addCourse = actions.createEl("button", { cls: "lpd-ghost", text: "Add course" });
    addCourse.addEventListener("click", () => {
      const defaultRoot = this.rootFilter === "All" ? "Math" : this.rootFilter;
      this.plugin.addManualCourse(defaultRoot);
    });
    const openFile = actions.createEl("button", { cls: "lpd-ghost", text: "Open data file" });
    openFile.addEventListener("click", () => this.plugin.openPath(DATA_PATH));

    const legend = sidebar.createDiv({ cls: "lpd-legend" });
    Object.entries(STATE_META).forEach(([state, meta]) => {
      const row = legend.createDiv({ cls: "lpd-legend-row" });
      row.createSpan({ cls: `lpd-node lpd-state-${state}`, text: meta.mark });
      row.createSpan({ text: `${meta.label}：${meta.description}` });
    });
  }

  renderMain(main) {
    const top = main.createDiv({ cls: "lpd-main-head" });
    top.createEl("h2", { text: "Course Rails" });
    const stats = this.computeStats();
    top.createDiv({
      cls: "lpd-muted",
      text: `${stats.courses} courses · ${stats.lessons} lessons · ${stats.reviewable} reviewable`,
    });

    const list = main.createDiv({ cls: "lpd-rail-list" });
    const courses = this.visibleCourses();
    if (courses.length === 0) {
      list.createDiv({ cls: "lpd-empty", text: "No courses match the current filter." });
      return;
    }

    courses.forEach((course) => {
      const row = list.createDiv({ cls: `lpd-course-row ${course.visible === false ? "is-hidden" : ""}` });
      const courseInfo = row.createDiv({ cls: "lpd-course-info" });
      courseInfo.createDiv({ cls: "lpd-course-title", text: course.title });
      courseInfo.createDiv({ cls: "lpd-course-path", text: `${course.root} · ${course.path}` });

      const track = row.createDiv({ cls: "lpd-track" });
      const lessons = [...(course.lessons || [])].sort((a, b) => String(a.label).localeCompare(String(b.label)));
      lessons.forEach((lesson) => {
        const meta = STATE_META[lesson.state] || STATE_META.raw;
        const node = track.createEl("button", {
          cls: `lpd-lesson-node lpd-state-${lesson.state || "raw"} ${
            this.selected && this.selected.courseId === course.id && this.selected.lessonId === lesson.id ? "is-selected" : ""
          }`,
          attr: {
            title: `${lesson.label} ${lesson.title} · ${meta.label}`,
          },
        });
        node.createSpan({ cls: "lpd-node-mark", text: meta.mark });
        node.createSpan({ cls: "lpd-node-label", text: lesson.label });
        node.addEventListener("click", () => {
          this.selected = { courseId: course.id, lessonId: lesson.id };
          this.render();
        });
      });

      const courseActions = row.createDiv({ cls: "lpd-course-actions" });
      const progress = this.courseProgress(course);
      courseActions.createDiv({
        cls: "lpd-course-progress",
        text: `${progress.active + progress.reviewable}/${progress.total} touched · ${progress.reviewable} reviewable`,
      });
      const moveUp = courseActions.createEl("button", { cls: "lpd-mini", text: "↑" });
      moveUp.addEventListener("click", () => this.plugin.moveCourse(course.id, -1));
      const moveDown = courseActions.createEl("button", { cls: "lpd-mini", text: "↓" });
      moveDown.addEventListener("click", () => this.plugin.moveCourse(course.id, 1));
      const addLesson = courseActions.createEl("button", { cls: "lpd-mini", text: "+ Lesson" });
      addLesson.addEventListener("click", () => this.plugin.addManualLesson(course.id));
      const toggle = courseActions.createEl("button", {
        cls: "lpd-mini",
        text: course.visible === false ? "Show" : "Hide",
      });
      toggle.addEventListener("click", () => this.plugin.toggleCourseVisibility(course.id));
    });
  }

  renderDetail(detail) {
    detail.createDiv({ cls: "lpd-kicker", text: "Lesson Detail" });
    const selected = this.getSelectedLesson();
    if (!selected) {
      detail.createEl("h2", { text: "Select a lesson" });
      detail.createDiv({
        cls: "lpd-muted",
        text: "点击轨道上的课节点，在这里编辑状态、链接和备注。",
      });
      return;
    }

    const { course, lesson } = selected;
    detail.createEl("h2", { text: `${lesson.label} ${lesson.title}` });
    detail.createDiv({ cls: "lpd-detail-course", text: course.title });

    const stateField = detail.createEl("label", { cls: "lpd-field" });
    stateField.createSpan({ text: "状态" });
    const stateSelect = stateField.createEl("select");
    Object.entries(STATE_META).forEach(([state, meta]) => {
      const option = stateSelect.createEl("option", {
        text: `${meta.mark} ${meta.label}`,
        attr: { value: state },
      });
      option.selected = (lesson.state || "raw") === state;
    });

    const pathField = detail.createEl("label", { cls: "lpd-field" });
    pathField.createSpan({ text: "课节笔记链接" });
    const notePath = pathField.createEl("input", {
      attr: { type: "text", placeholder: "01_Math/..." },
    });
    notePath.value = lesson.notePath || "";

    const remarkField = detail.createEl("label", { cls: "lpd-field" });
    remarkField.createSpan({ text: "人工备注" });
    const remark = remarkField.createEl("textarea", {
      attr: { placeholder: "哪里卡住、下一步要回看什么、Big Picture 里怎么定位..." },
    });
    remark.value = lesson.remark || "";

    const actions = detail.createDiv({ cls: "lpd-detail-actions" });
    const save = actions.createEl("button", { cls: "lpd-primary", text: "Save" });
    save.addEventListener("click", async () => {
      await this.plugin.updateLesson(course.id, lesson.id, {
        state: stateSelect.value,
        notePath: notePath.value.trim(),
        remark: remark.value.trim(),
      });
      new Notice("Learning progress saved.");
    });

    const open = actions.createEl("button", { cls: "lpd-ghost", text: "Open note" });
    open.addEventListener("click", () => this.plugin.openPath(notePath.value.trim()));
  }

  getSelectedLesson() {
    if (!this.selected || !this.data) return null;
    const course = this.data.courses.find((item) => item.id === this.selected.courseId);
    if (!course) return null;
    const lesson = (course.lessons || []).find((item) => item.id === this.selected.lessonId);
    if (!lesson) return null;
    return { course, lesson };
  }

  computeStats() {
    const courses = this.data ? this.data.courses.filter((course) => course.visible !== false) : [];
    const lessons = courses.flatMap((course) => course.lessons || []);
    return {
      courses: courses.length,
      lessons: lessons.length,
      reviewable: lessons.filter((lesson) => lesson.state === "reviewable").length,
    };
  }

  courseProgress(course) {
    const lessons = course.lessons || [];
    return {
      total: lessons.length,
      active: lessons.filter((lesson) => lesson.state === "active").length,
      reviewable: lessons.filter((lesson) => lesson.state === "reviewable").length,
    };
  }
}

module.exports = class LearningProgressDashboardPlugin extends Plugin {
  async onload() {
    this.store = new LearningBoardStore(this.app);
    this.registerView(VIEW_TYPE, (leaf) => new LearningProgressDashboardView(leaf, this));

    this.addRibbonIcon("map", "Learning Progress Dashboard", () => this.openDashboard());
    this.addCommand({
      id: "open-learning-progress-dashboard",
      name: "Open Learning Progress Dashboard",
      callback: () => this.openDashboard(),
    });
    this.addCommand({
      id: "rescan-learning-progress-courses",
      name: "Rescan learning progress courses",
      callback: () => this.rescanCourses(),
    });

    this.registerEvent(
      this.app.vault.on("modify", (file) => {
        if (file instanceof TFile && file.path === DATA_PATH) {
          this.refreshViews();
        }
      })
    );

    this.app.workspace.onLayoutReady(async () => {
      await this.store.ensureDataFile();
      await this.openDashboard();
    });
  }

  onunload() {
    this.app.workspace.detachLeavesOfType(VIEW_TYPE);
  }

  async openDashboard() {
    const existing = this.app.workspace.getLeavesOfType(VIEW_TYPE);
    if (existing.length > 0) {
      this.app.workspace.revealLeaf(existing[0]);
      return;
    }
    await this.app.workspace.getLeaf("tab").setViewState({
      type: VIEW_TYPE,
      active: true,
    });
  }

  async refreshViews() {
    for (const leaf of this.app.workspace.getLeavesOfType(VIEW_TYPE)) {
      if (leaf.view instanceof LearningProgressDashboardView) {
        await leaf.view.refresh();
      }
    }
  }

  async rescanCourses() {
    const existing = await this.store.read();
    await this.store.rescan(existing);
    await this.refreshViews();
    new Notice("Learning courses rescanned.");
  }

  async updateLesson(courseId, lessonId, patch) {
    const data = await this.store.read();
    const course = data.courses.find((item) => item.id === courseId);
    if (!course) return;
    const lesson = (course.lessons || []).find((item) => item.id === lessonId);
    if (!lesson) return;
    Object.assign(lesson, patch);
    await this.store.write(data);
    await this.refreshViews();
  }

  async addManualCourse(defaultRoot) {
    const title = window.prompt("Course title");
    if (!title || !title.trim()) return;

    const rootInput = window.prompt("Course root: Math, Economy, or Computer Science", defaultRoot || "Math");
    const root = normalizeRoot(rootInput);
    if (!root) {
      new Notice("Course root must be Math, Economy, or Computer Science.");
      return;
    }

    const data = await this.store.read();
    const path = `${ROOT_FOLDER_BY_LABEL[root]}/${title.trim()}`;
    if (data.courses.some((course) => course.path === path)) {
      new Notice("Course already exists.");
      return;
    }

    data.courses.push({
      id: makeId(path),
      title: title.trim(),
      root,
      path,
      visible: true,
      order: nextCourseOrder(data.courses),
      lessons: [],
    });
    await this.store.write(data);
    await this.refreshViews();
    new Notice("Course added.");
  }

  async addManualLesson(courseId) {
    const data = await this.store.read();
    const course = data.courses.find((item) => item.id === courseId);
    if (!course) return;

    const label = window.prompt("Lesson label", nextLessonLabel(course));
    if (!label || !label.trim()) return;
    const title = window.prompt("Lesson title");
    if (!title || !title.trim()) return;
    const notePath = window.prompt("Lesson note path, optional", "") || "";

    course.lessons = Array.isArray(course.lessons) ? course.lessons : [];
    course.lessons.push({
      id: makeId(`${course.path}/${label.trim()}-${title.trim()}-${Date.now()}`),
      label: label.trim(),
      title: title.trim(),
      notePath: notePath.trim(),
      state: "raw",
      remark: "",
    });
    await this.store.write(data);
    await this.refreshViews();
    new Notice("Lesson added.");
  }

  async toggleCourseVisibility(courseId) {
    const data = await this.store.read();
    const course = data.courses.find((item) => item.id === courseId);
    if (!course) return;
    course.visible = course.visible === false ? true : false;
    await this.store.write(data);
    await this.refreshViews();
  }

  async moveCourse(courseId, delta) {
    const data = await this.store.read();
    const courses = [...data.courses].sort(compareSortKey);
    const index = courses.findIndex((course) => course.id === courseId);
    const targetIndex = index + delta;
    if (index < 0 || targetIndex < 0 || targetIndex >= courses.length) return;
    const current = courses[index];
    const target = courses[targetIndex];
    const currentOrder = Number(current.order) || index + 1;
    current.order = Number(target.order) || targetIndex + 1;
    target.order = currentOrder;
    await this.store.write(data);
    await this.refreshViews();
  }

  async openPath(path) {
    if (!path) {
      new Notice("No note path set.");
      return;
    }
    let file = this.app.vault.getFileByPath(path);
    if (!file && !path.endsWith(".md")) {
      file = this.app.vault.getFileByPath(`${path}.md`);
    }
    if (!file) {
      new Notice(`Note not found: ${path}`);
      return;
    }
    await this.app.workspace.getLeaf(false).openFile(file);
  }
};
