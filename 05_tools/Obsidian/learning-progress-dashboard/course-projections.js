const { unitOutcomes, coverageFor, courseCoverage } = require("./course-model");

// These projections describe mapped scope only. They never record completion or
// infer prerequisite links, mastery, or knowledge inside unmapped chapters.
// Chapter status "complete" means its mapped knowledge is fully covered.
function comparisonSummary(course, learned = new Map()) {
  const coverage = courseCoverage(course, learned);
  const chapters = (course.units || []).map(unit => {
    const scope = coverageFor(unit, learned);
    const status = scope.status === "unknown" ? "unmapped"
      : scope.status === "overview" ? "overview"
      : scope.known.length === scope.total ? "complete"
      : scope.known.length ? "mixed" : "new";
    return { id: unit.id, unit, status, known: scope.known, missing: scope.missing, total: scope.total };
  });
  const counts = {
    knowledge: coverage.total, known: coverage.known.length, missing: coverage.missing.length,
    chapters: chapters.length, complete: 0, mixed: 0, new: 0, unmapped: 0, overview: 0,
  };
  for (const chapter of chapters) counts[chapter.status]++;
  counts.withExtra = counts.mixed + counts.new;
  return {
    known: coverage.known,
    missing: coverage.missing,
    // Unknown knowledge has no verified IDs. These are chapter IDs, not outcomes.
    unknown: chapters.filter(chapter => chapter.status === "unmapped").map(chapter => chapter.id),
    counts,
    chapters,
  };
}

function courseMatrix(course, learned = new Map()) {
  const modules = new Map();
  for (const unit of course.units || []) {
    const title = unit.module || "课程内容";
    if (!modules.has(title)) modules.set(title, { id: title, title, units: [], rows: new Map() });
    const group = modules.get(title);
    group.units.push(unit);
    for (const id of unitOutcomes(unit)) {
      if (!group.rows.has(id)) group.rows.set(id, { id, unitIds: [], known: learned.has(id) });
      group.rows.get(id).unitIds.push(unit.id);
    }
  }
  return {
    modules: [...modules.values()].map(group => ({
      id: group.id, title: group.title, units: group.units, rows: [...group.rows.values()],
      counts: comparisonSummary({ units: group.units }, learned).counts,
    })),
    counts: comparisonSummary(course, learned).counts,
  };
}

function relatedCourses(catalog, course) {
  const byId = new Map(catalog.courses.map(item => [item.id, item]));
  return {
    prerequisites: (course.prerequisites || [])
      .filter(edge => byId.has(edge.id))
      .map(edge => ({ course: byId.get(edge.id), edge })),
    downstream: catalog.courses.flatMap(target => (target.prerequisites || [])
      .filter(edge => edge.id === course.id)
      .map(edge => ({ course: target, edge }))),
  };
}

module.exports = { courseMatrix, comparisonSummary, relatedCourses };
