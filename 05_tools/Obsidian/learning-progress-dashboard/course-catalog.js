const { catalogFrom } = require("./course-model");
const catalog = catalogFrom([
  require("./course-data/math.json"),
  require("./course-data/probability.json"),
  require("./course-data/cs.json"),
  require("./course-data/ai.json"),
  require("./course-data/bayesian.json"),
]);
const domainOrder = ["math", "probability", "statistics", "bayesian", "causal", "optimization", "programming", "algorithms", "systems", "data-engineering", "ml", "deep-learning", "ai-systems"];
catalog.domains.sort((a, b) => domainOrder.indexOf(a.id) - domainOrder.indexOf(b.id));
module.exports = catalog;
