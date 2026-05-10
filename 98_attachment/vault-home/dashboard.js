const data = window.__VAULT_HOME_DATA__;

const $ = (selector) => document.querySelector(selector);

function textDate(value) {
  if (!value) return "No timestamp";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function linkCard(item, className = "note-link") {
  const anchor = document.createElement("a");
  anchor.className = className;
  anchor.href = item.uri;
  anchor.innerHTML = `
    <strong>${escapeHtml(item.title)}</strong>
    <div class="meta">
      ${item.path ? `<span class="pill">${escapeHtml(item.path)}</span>` : ""}
      ${item.mtime ? `<span class="pill">${textDate(item.mtime)}</span>` : ""}
    </div>
  `;
  return anchor;
}

function metric(value, label) {
  const node = document.createElement("div");
  node.className = "metric";
  node.innerHTML = `<strong>${escapeHtml(value)}</strong><span>${escapeHtml(label)}</span>`;
  return node;
}

function renderTop() {
  $("#today-title").textContent = `${data.today.date} ${data.today.weekday}`;
  $("#today-subtitle").textContent = `Generated ${textDate(data.generatedAt)}`;

  const strip = $("#metric-strip");
  strip.replaceChildren(
    metric(data.counts.notes, "Markdown notes"),
    metric(data.counts.factaseCards, "Factase cards"),
    metric(data.counts.canvases, "Canvas maps"),
    metric(`${data.counts.coursesWithCanvas}/${data.counts.courseFolders}`, "Courses with maps"),
  );

  const actions = $("#quick-actions");
  const links = data.quickLinks.slice(0, 5).map((item) => {
    const anchor = document.createElement("a");
    anchor.className = "action-link";
    anchor.href = item.uri;
    anchor.textContent = item.title;
    return anchor;
  });
  actions.replaceChildren(...links);
}

function renderToday() {
  const panel = $("#today-panel");
  const today = data.today.dailyNote;
  const todayCard = linkCard({
    title: today.exists ? "Open today's daily note" : "Create today's daily note",
    path: today.path,
    uri: today.dailyUri,
    mtime: today.mtime,
  });
  const inboxCard = linkCard({
    title: `Inbox · ${data.today.inbox.count} items`,
    path: data.today.inbox.path,
    uri: data.today.inbox.uri,
  });
  panel.replaceChildren(todayCard, inboxCard);
}

function renderCourses() {
  const roots = ["Math", "Economy", "Computer Science"];
  const map = $("#course-map");
  const lanes = roots.map((root) => {
    const lane = document.createElement("section");
    lane.className = "course-lane";
    const courses = data.courses.filter((course) => course.root === root);
    const title = document.createElement("div");
    title.className = "lane-title";
    title.innerHTML = `<span>${escapeHtml(root)}</span><span>${courses.length}</span>`;
    lane.append(title);
    courses
      .sort((a, b) => Number(b.hasBigPicture) - Number(a.hasBigPicture) || b.noteCount - a.noteCount)
      .slice(0, 7)
      .forEach((course) => {
        const card = document.createElement("div");
        card.className = `course-card ${course.hasBigPicture ? "has-canvas" : ""}`;
        const recent = course.recent
          .slice(0, 2)
          .map((item) => `<a href="${item.uri}">${escapeHtml(item.title)}</a>`)
          .join("");
        card.innerHTML = `
          <a class="course-title" href="${course.uri}"><strong>${escapeHtml(course.title)}</strong></a>
          <div class="meta">
            <span class="pill">${course.noteCount} notes</span>
            <span class="pill">${course.canvasCount} canvas</span>
            ${course.pdfCount ? `<span class="pill">${course.pdfCount} pdf</span>` : ""}
          </div>
          <div class="recent-mini">${recent}</div>
        `;
        lane.append(card);
      });
    return lane;
  });
  map.replaceChildren(...lanes);
}

function renderFactase() {
  const panel = $("#factase-panel");
  const cards = data.factase.categories.map((category) => {
    const anchor = document.createElement("a");
    anchor.className = "fact-card";
    anchor.href = category.uri;
    anchor.innerHTML = `<strong>${category.count}</strong><span>${escapeHtml(category.title)}</span>`;
    return anchor;
  });
  panel.replaceChildren(...cards);
}

function renderCanvases() {
  const panel = $("#canvas-panel");
  const items = data.canvases.map((canvas) => {
    const anchor = linkCard(canvas, "canvas-item");
    const meta = anchor.querySelector(".meta");
    meta.insertAdjacentHTML(
      "afterbegin",
      `<span class="pill">${canvas.nodeCount ?? "?"} nodes</span><span class="pill">${canvas.edgeCount ?? "?"} edges</span>`,
    );
    return anchor;
  });
  panel.replaceChildren(...items);
}

function renderRecent() {
  const panel = $("#recent-panel");
  panel.replaceChildren(...data.recentNotes.slice(0, 10).map((item) => linkCard(item)));
}

function renderHealth() {
  const panel = $("#health-panel");
  const rows = [
    ["Git changed files", data.health.gitDirty],
    ["Tracked notes", data.health.notes],
    ["Tracked canvases", data.health.canvases],
    ["Generated", textDate(data.generatedAt)],
  ];
  const nodes = rows.map(([label, value]) => {
    const row = document.createElement("div");
    row.className = "health-row";
    row.innerHTML = `<span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong>`;
    return row;
  });
  if (data.health.gitLines.length > 0) {
    const dirty = document.createElement("div");
    dirty.className = "note-link";
    dirty.innerHTML = `
      <strong>Current git surface</strong>
      <div class="meta">${data.health.gitLines
        .slice(0, 8)
        .map((line) => `<span class="pill">${escapeHtml(line)}</span>`)
        .join("")}</div>
    `;
    nodes.push(dirty);
  }
  panel.replaceChildren(...nodes);
}

function render() {
  if (!data) {
    document.body.innerHTML = '<main class="vault-shell"><div class="empty-state">dashboard-data.js is missing.</div></main>';
    return;
  }
  renderTop();
  renderToday();
  renderCourses();
  renderFactase();
  renderCanvases();
  renderRecent();
  renderHealth();
}

render();
