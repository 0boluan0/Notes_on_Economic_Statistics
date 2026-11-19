const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readingPlan } = require('./language');

const begin = '<!-- bilingual-en:start -->', end = '<!-- bilingual-en:end -->';
const marked = (...lines) => [begin,...lines,end].join('\n');
function languages(source) {
  return readingPlan(source)?.blocks.map(b => b.lang || 'shared');
}

test('English-first notes pair questions at the same quote depth', () => {
  const lines = [marked('# Lesson'), '### 第一讲', '> [!question]- Test / 自测',
    ...marked('Which value?').split('\n').map(l => '> ' + l), '> 哪个数值？',
    '> > [!success]- Answer / 答案', '> > 只有中文写了这条补充。'];
  assert.deepEqual(languages(lines.join('\n\n')), ['en','zh','shared','en','zh','shared','shared']);
});

test('Chinese-first notes retain a later untranslated paragraph', () => {
  const note = ['# 什么是投影', marked('*Projection*'), '中文解释。', marked('The explanation.'),
    '旧笔记里的补充。', '## 来源', '- [[资料]]'].join('\n\n');
  assert.deepEqual(languages(note), ['zh','en','zh','en','shared','shared','shared']);
});

test('keeps only copy of a formula, worked code, table and navigation', () => {
  const note = [marked('Explain it.', '', '$$x=2$$','','```py','x = 2','```','','| x | y |','|---|---|','| 1 | 2 |'),
    '读这个例子。','![[图.png]]','[[课件.pdf#page=2|课件]]'].join('\n\n');
  assert.deepEqual(languages(note),['en','shared','shared','shared','zh','shared','shared']);
});

test('pairs duplicated fenced examples, leaving an unrelated example visible', () => {
  const code='```py\nx = 2\n```';
  const note=[marked('Example.', '', code),'例子。',code,'```py\nx = 3\n```'].join('\n\n');
  assert.deepEqual(languages(note),['en','en','zh','zh','shared']);
});

test('code literals, frontmatter, inline markers and broken pairs leave the original readable', () => {
  for (const source of ['plain text', '`' + begin + '`','```md\n'+marked('example')+'\n```',
    '---\n'+marked('example')+'\n---\nBody',begin+'\nno closing tag',marked(marked('nested'))]) {
    assert.equal(readingPlan(source),null);
  }
});

test('an English-only block and a Chinese-only callout are both readable', () => {
  const note = [marked('Read this sentence.'),'> [!tip] 补充','> 这段只有中文。'].join('\n\n');
  assert.deepEqual(languages(note),['shared','shared','shared']);
});

test('multiple matched paragraphs have separate scroll targets', () => {
  const source = [marked('First paragraph.', '', 'Second paragraph.'),'第一段。','第二段。'].join('\n\n');
  const b = readingPlan(source).blocks;
  assert.deepEqual(b.map(x=>x.lang), ['en','en','zh','zh']);
  assert.equal(b[0].pair,b[2].pair);
  assert.equal(b[1].pair,b[3].pair);
  assert.notEqual(b[0].pair,b[1].pair);
});

test('legacy math punctuation does not swallow the next translation', () => {
  for (const equation of ['$$x=2$$，这是公式。', '$$\n公式末尾漏了定界符。']) {
    const source = [equation, marked('An English sentence.'), '这是译文。'].join('\n\n');
    assert.deepEqual(languages(source).slice(-2),['en','zh']);
  }
});
