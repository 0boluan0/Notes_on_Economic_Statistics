const assert = require("assert/strict");
const catalog = require("./course-catalog");
const { learnedOutcomes, unitOutcomes } = require("./course-model");
const { courseMatrix, comparisonSummary, relatedCourses } = require("./course-projections");

const course = id => catalog.courses.find(item => item.id === id);
const mit = course("mit-1806sc"), engr = course("stanford-engr108");
const before = JSON.stringify(catalog);
const fullMit = learnedOutcomes(catalog, null, new Map(), mit.id);
const comparison = comparisonSummary(engr, fullMit);
assert.equal(comparison.counts.knowledge, 33);
assert.equal(comparison.known.length, 19);
assert.equal(comparison.missing.length, 14);
assert.equal(comparison.counts.complete, 4);
assert.equal(comparison.counts.withExtra, 12);
assert.equal(comparison.counts.chapters, 16);
assert.deepEqual(comparison.unknown, []);
assert.deepEqual(comparison.chapters.filter(chapter => chapter.status === "complete").map(chapter => chapter.id), ["ch5", "ch6", "ch8", "ch11"]);
assert.deepEqual(comparison.chapters.find(chapter => chapter.id === "ch10").missing, ["la.householder"]);
assert.equal(comparison.counts.complete + comparison.counts.mixed + comparison.counts.new + comparison.counts.unmapped + comparison.counts.overview, engr.units.length);

// Eight confirmed MIT chapters light shared scope without completing another course.
const curriculum = { weekly: { tasks: mit.units.slice(0, 8).map(unit => ({
  course_id: "MIT 18.06", title: `MIT 18.06｜${unit.id} ${unit.title}`, state: "done",
})) } };
const personal = learnedOutcomes(catalog, curriculum, new Map());
const mitMatrix = courseMatrix(mit, personal), engrMatrix = courseMatrix(engr, personal);
assert.equal(mitMatrix.counts.chapters, 36);
assert.equal(mitMatrix.counts.knowledge, 58);
assert.equal(mitMatrix.counts.known, 12);
assert.equal(engrMatrix.counts.known, 4);
assert.deepEqual(mitMatrix.modules.map(group => group.units.length), [14, 12, 9, 1]);
for (const matrix of [mitMatrix, engrMatrix]) {
  const rows = matrix.modules.flatMap(group => group.rows);
  assert(rows.find(row => row.id === "la.matrix-multiplication").known);
  for (const group of matrix.modules) {
    assert.equal(group.rows.length, group.counts.knowledge);
    for (const row of group.rows) {
      assert.equal(row.known, personal.has(row.id));
      assert.deepEqual(row.unitIds, group.units.filter(unit => unitOutcomes(unit).includes(row.id)).map(unit => unit.id));
    }
  }
}

// A syllabus can mention an outcome without establishing verified coverage.
const fixture = { id: "scope-boundaries", units: [
  { id: "pending", module: "One", mapping: "syllabus", outcomes: ["unverified-topic"] },
  { id: "recap", module: "One", mapping: "reviewed-outline", outcomes: [] },
  { id: "first", module: "One", mapping: "reviewed-outline", outcomes: ["known-topic", "new-topic", "known-topic"] },
  { id: "later", module: "Two", mapping: "reviewed-outline", outcomes: ["known-topic"] },
] };
const learned = new Map([["known-topic", []], ["unverified-topic", []]]);
const summary = comparisonSummary(fixture, learned), matrix = courseMatrix(fixture, learned);
assert.deepEqual(summary.known, ["known-topic"]);
assert.deepEqual(summary.missing, ["new-topic"]);
assert.deepEqual(summary.unknown, ["pending"]);
assert.equal(summary.counts.unmapped, 1);
assert.equal(summary.counts.overview, 1);
assert.equal(summary.counts.new, 0, "Unmapped and recap chapters are not new content");
assert.equal(summary.counts.withExtra, 1);
assert.equal(matrix.counts.knowledge, 2, "Whole-course counts deduplicate knowledge shared by modules");
assert.equal(matrix.modules[0].counts.knowledge, 2);
assert.equal(matrix.modules[1].counts.knowledge, 1);
assert(!matrix.modules.flatMap(group => group.rows).some(row => row.id === "unverified-topic"));
assert.deepEqual(matrix.modules[0].rows.find(row => row.id === "known-topic").unitIds, ["first"]);
assert.deepEqual(comparisonSummary({ units: [] }).chapters, []);
assert.deepEqual(courseMatrix({ units: [] }).modules, []);
for (const candidate of catalog.courses) {
  const actual = comparisonSummary(candidate, personal);
  const unmapped = candidate.units.filter(unit => unit.mapping === "syllabus");
  assert.deepEqual(actual.unknown, unmapped.map(unit => unit.id));
  for (const unit of unmapped) {
    const chapter = actual.chapters.find(item => item.id === unit.id);
    assert.equal(chapter.status, "unmapped");
    assert.deepEqual(chapter.missing, [], "An unmapped real catalog chapter cannot claim new knowledge IDs");
  }
}

const ml = course("stanford-cs229"), relations = relatedCourses(catalog, ml);
assert.deepEqual(relations.prerequisites.map(ref => ref.course.id), ml.prerequisites.map(edge => edge.id));
assert.equal(relations.prerequisites[0].edge, ml.prerequisites[0], "The original evidence and relation type are preserved");
const downstream = relatedCourses(catalog, mit).downstream;
assert(downstream.some(ref => ref.course.id === ml.id && ref.edge.kind === "inferred"));
assert(downstream.some(ref => ref.course.code === "18.065" && ref.edge.kind === "official"));
assert(!downstream.some(ref => ref.course.id === engr.id), "Shared knowledge alone must not create a prerequisite edge");
assert.equal(JSON.stringify(catalog), before, "Projection must leave all catalog facts unchanged");
assert.equal(curriculum.weekly.tasks.length, 8);
assert(curriculum.weekly.tasks.every(task => task.state === "done"));

console.log("course-projections-check: ok");
