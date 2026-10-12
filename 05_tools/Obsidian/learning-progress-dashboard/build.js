const fs = require("fs");
const path = require("path");

const main = fs.readFileSync(path.join(__dirname, "main.js"), "utf8");
const steward = fs.readFileSync(path.join(__dirname, "steward-view.js"), "utf8");
const marker = 'require("./steward-view")';
if (main.split(marker).length !== 2) throw new Error("Steward module must have exactly one entrypoint");
// Obsidian supplies its own require('obsidian'); keep that binding in one plugin file.
process.stdout.write(main.replace(marker, `(() => { const module = { exports: {} };\n${steward}\nreturn module.exports; })()`));
