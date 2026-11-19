// Match the vault's explicitly marked English passages to neighbouring Chinese text.
// Unmatched material is shared. A note with incomplete markers stays in its native layout.
const han = /\p{Script=Han}/u;
const marker = /^<!--\s*bilingual-en:(start|end)\s*-->$/;
const possibleMarker = /^(?:[ \t]*>[ \t]?)*[ \t]*<!--\s*bilingual-en:start\s*-->[ \t]*\r?$/m;

function visibleWords(text) {
  return text.replace(/!?\[\[([^\]]+)\]\]/g, (_, link) => link.split('|').pop())
    .replace(/!?\[([^\]]+)\]\([^)]*\)/g, '$1')
    .replace(/<[^>]+>|&nbsp;/g, ' ');
}

function key(text) {
  return text.toLowerCase().replace(/[^\p{L}\p{N}]/gu, '');
}

function shape(body) {
  if (/^#{1,6}\s/.test(body)) return 'heading';
  if (/^\[![\w-]+\]/.test(body)) return 'callout';
  if (/^\|.*\|\s*$/.test(body)) return 'table';
  if (/^[-*+]\s|^\d+[.)]\s/.test(body)) return 'list';
  if (/^(?:!\[\[|!\[[^\]]*\]\(|<img\b|<iframe\b)/i.test(body)) return 'media';
  if (/^(?:\[\[[^\]]+\]\]|\[[^\]]+\]\([^)]*\)|[\s·|/,，、：:–—-])+$/.test(body)) return 'navigation';
  if (/^(?:-{3,}|\*{3,}|_{3,}|\^[\w-]+|&nbsp;)$/.test(body)) return 'boundary';
  return 'prose';
}

function scan(source) {
  const lines = source.split(/\r?\n/);
  const blocks = [], groups = [], byLine = new Array(lines.length);
  let region = null, current = null, fence = null, math = false, comment = false;
  let frontmatter = lines[0]?.trim() === '---';
  function append(i, kind, body, depth, merge) {
    if (!merge || !current || current.kind !== kind || current.depth !== depth || current.region !== region) {
      current = { kind, depth, start: i, end: i, region, body, index: blocks.length };
      blocks.push(current);
      if (region) region.blocks.push(current);
    } else {
      current.body += '\n' + body;
      current.end = i;
    }
    byLine[i] = current;
  }
  for (let i = 0; i < lines.length; i++) {
    const raw = lines[i];
    if (frontmatter) {
      if (i && /^(?:---|\.\.\.)\s*$/.test(raw)) frontmatter = false;
      continue;
    }
    const quote = raw.match(/^(?:[ \t]*>[ \t]?)*/)[0];
    const depth = (quote.match(/>/g) || []).length;
    const unquoted = raw.slice(quote.length), body = unquoted.trim();
    if (fence) {
      append(i, 'code', body, fence.depth, true);
      if (depth === fence.depth && body[0] === fence.char && body.length >= fence.length &&
          [...body].every(ch => ch === fence.char)) { fence = null; current = null; }
      continue;
    }
    if (math) {
      append(i, 'math', body, depth, true);
      if (body.endsWith('$$')) { math = false; current = null; }
      continue;
    }
    if (comment) { if (body.includes('-->')) comment = false; continue; }
    const m = !/^(?: {4}|\t)/.test(unquoted) && marker.exec(body);
    if (m) {
      current = null;
      if (m[1] === 'start') {
        if (region) return null;
        region = { start: i, end: i, blocks: [], depth };
        groups.push(region);
      } else {
        if (!region || region.depth !== depth) return null;
        region.end = i; region = null;
      }
      continue;
    }
    if (body.startsWith('<!--')) {
      comment = !body.includes('-->'); current = null; continue;
    }
    const open = /^(`{3,}|~{3,})/.exec(body);
    if (open) {
      fence = { char: open[1][0], length: open[1].length, depth };
      append(i, 'code', body, depth, false); continue;
    }
    if (body.startsWith('$$')) {
      math = body.length === 2 || !body.endsWith('$$');
      append(i, 'math', body, depth, false); if (!math) current = null;
      continue;
    }
    if (!body) { current = null; continue; }
    const kind = shape(body);
    append(i, kind, body, depth, ['prose', 'list', 'table'].includes(kind));
  }
  if (region || !groups.length) return null;
  for (const block of blocks) {
    block.words = visibleWords(block.body);
    block.key = key(block.words);
  }
  return { blocks, groups, byLine };
}

function isCaption(block) {
  return block.kind === 'prose' && !block.body.includes('\n') &&
    block.words.length < 170 && !/[。！？.!?]\s*[*_]*$/.test(block.words);
}

function compatible(a, b) {
  if (a.kind === b.kind) return ['heading','prose','list','table'].includes(a.kind);
  return (a.kind === 'heading' && isCaption(b)) || (b.kind === 'heading' && isCaption(a));
}

function linkPair(english, chinese) {
  const id = english[0].start;
  for (const block of english) Object.assign(block, { lang: 'en', pair: id, peer: chinese.start });
  Object.assign(chinese, { lang: 'zh', pair: id, peer: id });
}

function readingPlan(source) {
  if (!source || !possibleMarker.test(source)) return null;
  const plan = scan(source);
  if (!plan) return null;
  const { groups, blocks } = plan;
  const title = blocks.find(b => b.kind === 'heading');
  const preferred = title && !title.region && han.test(title.words) ? -1 : 1;
  const offers = [];
  for (const group of groups) {
    if (!group.blocks.length) continue;
    for (const direction of [-1, 1]) {
      const edge = direction < 0 ? group.blocks[0] : group.blocks.at(-1);
      const adjacent = blocks[edge.index + direction];
      if (!adjacent || adjacent.region || !han.test(adjacent.words) || adjacent.depth !== group.depth) continue;
      const matches = group.blocks.filter(b => b.depth === adjacent.depth && compatible(b, adjacent));
      if (!matches.length) continue;
      const distance = direction < 0 ? group.start - adjacent.end : adjacent.start - group.end;
      const match = matches.find(b => b.kind === adjacent.kind) || matches[0];
      const heading = adjacent.kind === 'heading' || match.kind === 'heading';
      const score = (heading ? 12 : 0) + (direction === preferred ? 4 : 0) - Math.min(distance, 8);
      offers.push({ group, adjacent, match, score });
    }
  }
  for (const { group, adjacent, match } of offers.sort((a,b) => b.score - a.score)) {
    if (match.lang || adjacent.lang) continue;
    // An English explanation often has several paragraphs and a single Chinese summary.
    // A table, heading, code sample or equation in the same marker can have its own audience.
    const english = group.blocks.filter(b => !b.lang && b.depth === adjacent.depth && b.kind === match.kind);
    linkPair(english, adjacent);
    // When an author supplied the same asset in each version, show it once per
    // language. A code block or display equation supplied only once stays shared.
    const direction = adjacent.start < group.start ? -1 : 1;
    const edge = direction < 0 ? group.blocks[0] : group.blocks.at(-1);
    for (let i = edge.index + direction; i >= 0 && i < blocks.length; i += direction) {
      const other = blocks[i];
      if (other.region || other.depth !== adjacent.depth || ['heading','callout','boundary'].includes(other.kind)) break;
      if (other.lang || !['code','math','media'].includes(other.kind)) continue;
      const same = group.blocks.find(b => !b.lang && b.kind === other.kind && b.body === other.body);
      if (same) linkPair([same],other);
    }
  }
  // Some course notes leave the English section heading and its short Chinese subtitle unmarked.
  for (let i = 0; i + 1 < blocks.length; i++) {
    const a = blocks[i], b = blocks[i + 1];
    if (!a.region && !b.region && !a.lang && !b.lang && a.kind === 'heading' && !han.test(a.words) &&
        /[A-Za-z]/.test(a.words) && isCaption(b) && han.test(b.words) && a.depth === b.depth && b.start - a.end <= 2) {
      linkPair([a], b);
    }
  }
  return plan;
}

module.exports = { readingPlan, visibleWords, key, possibleMarker };
