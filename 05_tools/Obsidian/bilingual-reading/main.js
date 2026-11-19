const { Plugin, MarkdownView, setTooltip } = require('obsidian');
const { readingPlan, key } = require('./language');

function tag(element, block, path) {
  if (!block.lang) return;
  element.dataset.brLang = block.lang;
  element.dataset.brPair = path + '#' + block.pair;
}

function localizeTitle(el) {
  if (el.childNodes.length !== 1 || el.firstChild.nodeType !== 3) return;
  const title = el.textContent;
  for (const separator of [' / ', ' · ', ' | ', ' ｜ ']) {
    const index = title.lastIndexOf(separator);
    if (index < 1) continue;
    const en = title.slice(0,index), zh = title.slice(index + separator.length);
    if (/\p{Script=Han}/u.test(en) || !/\p{Script=Han}/u.test(zh) || !/[a-z]/i.test(en)) continue;
    const span = (value, lang) => {
      const s = el.ownerDocument.createElement('span');
      s.textContent = value; s.dataset.brLang = lang; return s;
    };
    el.replaceChildren(span(en,'en'), span(separator,'separator'), span(zh,'zh'));
    return;
  }
}

function decorate(el, ctx, info, plan) {
  const blocks = [...new Set(plan.byLine.slice(info.lineStart,info.lineEnd + 1).filter(Boolean))];
  for (const title of el.querySelectorAll('.callout-title-inner')) {
    if (!title.closest('.internal-embed')) localizeTitle(title);
  }
  if (!blocks.some(b => b.lang)) return;
  // Leave Obsidian's section wrapper in the document flow for its virtual scroller.
  if (blocks.every(b => b.lang && b.lang === blocks[0].lang && b.pair === blocks[0].pair)) {
    for (const child of el.children) {
      if (!child.matches('.internal-embed, img') && !child.querySelector('.internal-embed,img')) {
        tag(child,blocks[0],ctx.sourcePath);
      }
    }
    return;
  }
  // A complete question with nested answers is often one native Markdown section.
  // Match its rendered blocks to the source, keeping native folding, links and math.
  const nodes = el.querySelectorAll('p, h1, h2, h3, h4, h5, h6, li, table');
  let cursor = 0;
  for (const node of nodes) {
    if (node.closest('.internal-embed, .callout-title, td, th, pre, [data-br-lang]') ||
        node.querySelector('.internal-embed, img, ul, ol, .callout')) continue;
    const kind = node.matches('h1,h2,h3,h4,h5,h6') ? 'heading'
      : node.matches('table') ? 'table' : node.matches('li') ? 'list' : 'prose';
    const words = key(node.textContent);
    if (!words) continue;
    const candidates = blocks.filter(b => b.kind === kind);
    // Inline Markdown, especially MathJax, can change the visible string. Match a
    // substantial native text node in that case. Ambiguous matches stay visible.
    const walker = node.ownerDocument.createTreeWalker(node,4);
    const runs = [];
    for (let n = walker.nextNode(); n; n = walker.nextNode()) {
      if (!n.parentElement?.closest('mjx-container, .math, svg')) {
        const run = key(n.textContent);
        if (run.length >= 5) runs.push(run);
      }
    }
    const exact = candidates.filter(b => b.key === words);
    const matches = exact.length ? exact : candidates.filter(b =>
      (words.length >= 4 && b.key.includes(words)) || runs.some(r => b.key.includes(r)));
    const options = matches.filter(b => b.index >= cursor);
    const first = options[0];
    if (!first || (options.length > 1 && !exact.length && options.some(b => b.lang !== first.lang))) continue;
    cursor = first.index;
    tag(node,first,ctx.sourcePath);
  }
}

function viewportMark(view) {
  const port = view.previewMode.containerEl.querySelector('.markdown-preview-view');
  if (!port || port.scrollTop < 3 || !port.clientHeight) return null;
  const top = port.getBoundingClientRect().top;
  for (const node of port.querySelectorAll('[data-br-pair]')) {
    const rect = node.getBoundingClientRect();
    if (rect.height && rect.bottom > top + 3 && rect.top < top + port.clientHeight) {
      return { port, pair: node.dataset.brPair, offset: Math.max(0, rect.top - top),
        fraction: Math.min(1, Math.max(0,(top - rect.top) / rect.height)),
        scope: node.closest('.markdown-embed') || port };
    }
  }
  return null;
}

function restoreMark(mark, language) {
  if (!mark || !mark.port.isConnected) return;
  for (const node of mark.scope.querySelectorAll('[data-br-pair]')) {
    if (node.dataset.brPair !== mark.pair || node.dataset.brLang !== language) continue;
    const rect = node.getBoundingClientRect();
    if (!rect.height) continue;
    mark.port.scrollTop += rect.top - mark.port.getBoundingClientRect().top - mark.offset + rect.height * mark.fraction;
    return;
  }
}

module.exports = class BilingualReading extends Plugin {
  async onload() {
    const settings = await this.loadData();
    this.language = settings?.language === 'zh' ? 'zh' : 'en';
    this.buttons = new Map();
    this.cache = new Map();
    this.saves = Promise.resolve();
    this.registerMarkdownPostProcessor((el, ctx) => {
      const info = ctx.getSectionInfo(el);
      if (!info) return;
      const plan = this.plan(ctx.sourcePath,info.text);
      if (plan) decorate(el,ctx,info,plan);
    });
    const refresh = () => this.refresh();
    this.app.workspace.onLayoutReady(() => {
      if (!this.buttons) return;
      refresh();
      // Already open reading views were rendered before this processor registered.
      for (const view of this.buttons.keys()) {
        if (view.getMode() === 'preview') view.previewMode.rerender(true);
      }
    });
    for (const event of ['layout-change', 'active-leaf-change', 'file-open']) {
      this.registerEvent(this.app.workspace.on(event,refresh));
    }
    this.registerEvent(this.app.metadataCache.on('changed',file => {
      for (const view of this.buttons.keys()) {
        if (view.file === file) this.updateButton(view);
      }
    }));
    this.addCommand({ id: 'toggle-language', name: '切换中文 / English', checkCallback: checking => {
      const view = this.app.workspace.getActiveViewOfType(MarkdownView);
      const button = this.buttons.get(view);
      if (!button || button.hidden) return false;
      if (!checking) void this.toggle(view);
      return true;
    }});
  }

  plan(path, text) {
    const cached = this.cache.get(path);
    if (cached && cached.text === text) return cached.plan;
    if (this.cache.size >= 32) this.cache.delete(this.cache.keys().next().value);
    const plan = readingPlan(text);
    this.cache.set(path,{text,plan});
    return plan;
  }

  refresh() {
    const open = new Set(this.app.workspace.getLeavesOfType('markdown').map(l => l.view).filter(v => v instanceof MarkdownView));
    for (const [view,button] of this.buttons) {
      if (!open.has(view)) { button.remove(); this.buttons.delete(view); }
    }
    for (const view of open) {
      if (!this.buttons.has(view)) {
        const button = view.addAction('languages','切换阅读语言',() => this.toggle(view));
        button.classList.add('bilingual-reading-action');
        this.buttons.set(view,button);
      }
      this.updateButton(view);
    }
  }

  updateButton(view) {
    const button = this.buttons.get(view);
    const preview = view.previewMode.containerEl;
    preview.dataset.brReading = this.language;
    const file = view.file;
    button.hidden = !file || !this.plan(file.path,view.getViewData());
    const next = this.language === 'en' ? '中文' : 'English';
    // In the editor both versions are present. One click opens the selected reading language.
    button.textContent = next;
    setTooltip(button, '切换到' + (next === '中文' ? '中文' : '英文') + '阅读');
  }

  async toggle(view) {
    const editor = view.getMode() !== 'preview';
    const marks = [...this.buttons.keys()].filter(v => v.getMode() === 'preview').map(viewportMark);
    const language = this.language === 'en' ? 'zh' : 'en';
    this.language = language;
    this.refresh();
    for (const mark of marks) restoreMark(mark,language);
    this.saves = this.saves.catch(() => {}).then(() => this.saveData({language}));
    if (editor) {
      const state = view.leaf.getViewState();
      await view.leaf.setViewState({...state,state:{...state.state,mode:'preview'}});
    }
    await this.saves;
  }

  onunload() {
    for (const [view,button] of this.buttons) {
      button.remove();
      delete view.previewMode.containerEl.dataset.brReading;
    }
    this.buttons.clear();
    this.cache.clear();
  }
};
