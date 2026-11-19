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
  const embed = el.closest('.internal-embed');
  for (const node of el.querySelectorAll('[data-br-pair]')) {
    if (node.closest('.internal-embed') === embed || node.parentElement === el) {
      delete node.dataset.brPair;
      delete node.dataset.brLang;
    }
  }
  const blocks = [...new Set(plan.byLine.slice(info.lineStart,info.lineEnd + 1).filter(Boolean))];
  for (const title of el.querySelectorAll('.callout-title-inner')) {
    if (title.closest('.internal-embed') === embed) localizeTitle(title);
  }
  if (!blocks.some(b => b.lang)) return;
  // Leave Obsidian's section wrapper in the document flow for its virtual scroller.
  if (blocks.every(b => b.lang && b.lang === blocks[0].lang && b.pair === blocks[0].pair)) {
    for (const child of el.children) {
      if (blocks[0].kind === 'media' || (!child.matches('.internal-embed, img') && !child.querySelector('.internal-embed,img'))) {
        tag(child,blocks[0],ctx.sourcePath);
      }
    }
    return;
  }
  // A complete question with nested answers is often one native Markdown section.
  // Match its rendered blocks to the source, keeping native folding, links and math.
  const nodes = el.querySelectorAll('p, h1, h2, h3, h4, h5, h6, li, table, pre, .math-block');
  let cursor = 0;
  for (const node of nodes) {
    if (node.closest('.internal-embed') !== embed || node.closest('.callout-title, td, th, [data-br-lang]') ||
        node.querySelector('.internal-embed, img, ul, ol, .callout')) continue;
    const kind = node.matches('h1,h2,h3,h4,h5,h6') ? 'heading'
      : node.matches('table') ? 'table' : node.matches('li') ? 'list'
      : node.matches('pre') ? 'code' : node.matches('.math-block') ? 'math' : 'prose';
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
    this.observers = new Map();
    this.decorated = new WeakMap();
    this.cache = new Map();
    this.original = new WeakMap();
    this.embedded = new WeakMap();
    this.saves = Promise.resolve();
    this.active = true;
    this.registerMarkdownPostProcessor((el, ctx) => {
      const info = ctx.getSectionInfo(el);
      if (!info) return;
      const plan = this.plan(ctx.sourcePath,info.text);
      if (plan) this.decorateSection(el,ctx,info,plan);
    });
    const refresh = () => this.refresh();
    this.app.workspace.onLayoutReady(() => {
      if (!this.active) return;
      refresh();
      for (const view of this.buttons.keys()) {
        if (view.getMode() === 'preview') this.decorateOpenView(view);
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
    this.addCommand({ id: 'show-original', name: '显示双语原文', checkCallback: checking => {
      const view = this.app.workspace.getActiveViewOfType(MarkdownView);
      if (!view || view.getMode() !== 'preview' || this.buttons.get(view)?.hidden !== false) return false;
      if (!checking) {
        this.original.set(view,view.file.path);
        this.updateButton(view);
      }
      return true;
    }});
  }

  decorateSection(el, ctx, info, plan) {
    const prior = this.decorated.get(el);
    if (prior?.plan === plan && prior.first === el.firstElementChild) return;
    decorate(el,ctx,info,plan);
    this.decorated.set(el,{plan,first:el.firstElementChild});
  }

  decorateOpenView(view) {
    if (!view.file) return;
    const plan = this.plan(view.file.path,view.getViewData());
    if (!plan) return;
    // On plugin enable, Obsidian may recycle already-rendered sections without
    // invoking a newly registered postprocessor. Decorate those original nodes.
    const preview = view.previewMode, renderer = preview.renderer;
    if (typeof renderer?.getSectionInfo !== 'function') return;
    for (const el of preview.containerEl.querySelectorAll(':scope > .markdown-preview-view > .markdown-preview-sizer > div')) {
      if (!el.childElementCount) continue;
      const prior = this.decorated.get(el);
      if (prior?.plan === plan && prior.first === el.firstElementChild) continue;
      const info = renderer.getSectionInfo(el);
      if (info) this.decorateSection(el,{sourcePath:view.file.path},info,plan);
    }
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
      if (!open.has(view)) {
        this.observers.get(view)?.disconnect(); this.observers.delete(view);
        button.remove(); this.buttons.delete(view);
      }
    }
    for (const view of open) {
      if (!this.buttons.has(view)) {
        const button = view.addAction('languages','切换阅读语言',() => this.toggle(view));
        button.classList.add('bilingual-reading-action');
        this.buttons.set(view,button);
        const observer = new MutationObserver(changes => {
          if (changes.some(c => c.addedNodes.length && (c.target.matches?.('.markdown-preview-sizer') ||
              c.target.parentElement?.matches('.markdown-preview-sizer')))) {
            this.decorateOpenView(view);
            if (button.hidden) this.updateButton(view);
          }
        });
        observer.observe(view.previewMode.containerEl,{childList:true,subtree:true});
        this.observers.set(view,observer);
      }
      this.updateButton(view);
    }
  }

  updateButton(view) {
    const button = this.buttons.get(view);
    const preview = view.previewMode.containerEl;
    const file = view.file;
    const source = file && view.getViewData();
    let bilingual = !!file && !!this.plan(file.path,source);
    if (!bilingual && file && this.app.metadataCache.getFileCache(file)?.embeds?.length) {
      const prior = this.embedded.get(view);
      bilingual = (prior?.file === file && prior.source === source) ||
        !!preview.querySelector('.markdown-embed [data-br-pair]');
      if (bilingual) this.embedded.set(view,{file,source});
    }
    button.hidden = !bilingual;
    if (bilingual && this.original.get(view) !== file.path) preview.dataset.brReading = this.language;
    else delete preview.dataset.brReading;
    const next = this.language === 'en' ? '中文' : 'English';
    // In the editor both versions are present. One click opens the selected reading language.
    if (button.textContent !== next) {
      button.textContent = next;
      setTooltip(button, '切换到' + (next === '中文' ? '中文' : '英文') + '阅读');
    }
  }

  async toggle(view) {
    const editor = view.getMode() !== 'preview';
    const marks = [...this.buttons.keys()].filter(v => v.getMode() === 'preview').map(viewportMark);
    const language = this.language === 'en' ? 'zh' : 'en';
    this.language = language;
    this.original.delete(view);
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
    this.active = false;
    for (const observer of this.observers.values()) observer.disconnect();
    this.observers.clear();
    for (const [view,button] of this.buttons) {
      button.remove();
      delete view.previewMode.containerEl.dataset.brReading;
    }
    this.buttons.clear();
    this.cache.clear();
  }
};
