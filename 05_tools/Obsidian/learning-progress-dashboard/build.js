const fs = require("fs");
const path = require("path");

// Bundle local modules and the reviewed catalog; keep Obsidian/Node imports native.
const modules = new Map();
function bundle(file) {
  const id = path.relative(__dirname, file);
  if (modules.has(id)) return id;
  modules.set(id, "");
  let source = fs.readFileSync(file, "utf8");
  if (file.endsWith(".json")) source = "module.exports = " + JSON.stringify(JSON.parse(source)) + ";";
  else source = source.replace(/require\(["'](\.\.?\/[^"']+)["']\)/g, (_, request) => {
    let target = path.resolve(path.dirname(file), request);
    if (!path.extname(target)) target += ".js";
    if (!target.startsWith(__dirname + path.sep)) throw new Error("Local import outside plugin: " + request);
    return `__localRequire(${JSON.stringify(bundle(target))})`;
  });
  modules.set(id, source);
  return id;
}
const entry = bundle(path.join(__dirname, "main.js"));
process.stdout.write(`const __modules = {\n${[...modules].map(([id, source]) => `${JSON.stringify(id)}: (module, exports) => {\n${source}\n}`).join(",\n")}\n};\nconst __cache = {};\nfunction __localRequire(id) {\n if (!__cache[id]) { const mod = __cache[id] = { exports: {} }; __modules[id](mod, mod.exports); }\n return __cache[id].exports;\n}\nmodule.exports = __localRequire(${JSON.stringify(entry)});\n`);
