/* Local visual explanations. No network requests, telemetry or learning records. */
(() => {
  'use strict';
  const $ = (s, root = document) => root.querySelector(s);
  const $$ = (s, root = document) => [...root.querySelectorAll(s)];
  const escape = x => String(x ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const palette = ['#17614d','#ba5729','#2e6eaa','#995074','#697627','#6c56a0','#a3aaa0','#38878b'];
  const root = $('#application');
  const content = globalThis.LabContent;
  const math = globalThis.LabMath;
  if (!content || !math) { root.innerHTML = '<p class="error" role="alert">实验资源没有完整载入。请从实验库入口重新打开，或检查 assets 文件夹是否与页面保存在一起。</p>'; return; }
  const groups = [...new Set(Object.values(content).map(c=>c.group))];
  const ids = groups.flatMap(group=>Object.keys(content).filter(id=>content[id].group===group));
  let activeId, params = {}, result, activePanel = 'observe', resizeTimer, computeTimer, toastTimer, playTimer;
  const stateCache = new Map();
  let storyStep=0, showAllControls=false, filmExample=null;
  function story(){return globalThis.LabStories?.[activeId] || globalThis.LabGuides?.[activeId] || {steps:content[activeId].steps.slice(0,3).map((caption,i)=>({title:['先观察','改变条件','解释变化'][i],caption,action:caption,focus:content[activeId].controls.filter(c=>c.key!=='seed').slice(0,2).map(c=>c.key),plot:Math.min(i,1)}))};}
  const number = (value, digits = 3) => {
    if (value === null || value === undefined) return '未定义';
    if (typeof value !== 'number') return String(value);
    if (!Number.isFinite(value)) return value > 0 ? '+∞' : value < 0 ? '−∞' : '未定义';
    if (value === 0 || Math.abs(value) < 1e-14) return '0';
    if (Math.abs(value) >= 1e5 || Math.abs(value) < 10 ** -Math.max(digits, 4)) return value.toExponential(Math.min(digits, 3)).replace('e-', 'e−');
    return value.toLocaleString('en-US', { maximumFractionDigits: digits, useGrouping: false }).replace('-', '−');
  };
  const prose = text => {
    // Content is authored locally. Escape prose before rendering explicit math delimiters.
    return escape(text).replace(/\$\$([\s\S]+?)\$\$/g, (_, s) => renderFormula(s, true)).replace(/\$([^$\n]+?)\$/g, (_, s) => renderFormula(s, false));
  };
  function renderFormula(source, display = true) {
    const decoded = String(source).replace(/&amp;/g,'&').replace(/&lt;/g,'<').replace(/&gt;/g,'>').replace(/&#39;/g,"'").replace(/&quot;/g,'"');
    if (!globalThis.katex) return `<code>${escape(decoded)}</code>`;
    try { return katex.renderToString(decoded, { displayMode:display, throwOnError:true, trust:false, strict:'warn', output:'htmlAndMathml' }); }
    catch (error) { console.error('Formula rendering failed:', decoded, error); return `<code class="formula-error">${escape(decoded)}</code>`; }
  }
  function announce(text) {
    const el = $('#toast'); el.textContent = text; el.classList.add('visible');
    clearTimeout(toastTimer); toastTimer = setTimeout(() => el.classList.remove('visible'), 2800);
  }
  function sendParent(message) { if (window.parent !== window) window.parent.postMessage({ source:'visual-knowledge-lab', ...message }, '*'); }
  function defaultParams(id) { return Object.fromEntries(content[id].controls.map(c => [c.key, c.value])); }
  function readRoute() {
    let hash = location.hash.slice(1); if (!hash && globalThis.LAB_INITIAL_EXPERIMENT) hash = globalThis.LAB_INITIAL_EXPERIMENT;
    const [id, query = ''] = hash.split('?');
    if (id === 'library') return { id:'library' };
    if (!ids.includes(id)) return { id:ids[0] };
    const p = defaultParams(id);
    new URLSearchParams(query).forEach((value, key) => {
      const c = content[id].controls.find(x => x.key === key);
      if (!c) return;
      if (c.type === 'select') {
        const option = c.options.find(o => String(o.value) === value); if (option) p[key] = option.value;
      } else {
        const numeric = Number(value); if (Number.isFinite(numeric) && numeric >= c.min && numeric <= c.max) p[key] = numeric;
      }
    });
    return { id, params:query?p:undefined };
  }
  function routeString() { return activeId === 'library' ? '#library' : `#${activeId}?${new URLSearchParams(params)}`; }
  function updateRoute() {
    try { history.replaceState(null, '', routeString()); } catch (_) { /* Sandboxed srcdoc has no navigable origin. */ }
    sendParent({ type:'state', id:activeId, params });
  }
  function navigate(id, routeParams) {
    globalThis.LabStories?.[activeId]?.dispose?.($('#story-scene'));
    clearInterval(playTimer);playTimer=null;
    if (activeId && activeId !== 'library') stateCache.set(activeId, { ...params });
    activeId = ids.includes(id) ? id : 'library';
    params = activeId === 'library' ? {} : { ...defaultParams(activeId), ...(routeParams || stateCache.get(activeId) || (globalThis.LabStories?.[activeId]||globalThis.LabGuides?.[activeId])?.steps[0]?.params || {}) };
    $$('.nav-link').forEach(a => a.setAttribute('aria-current', a.dataset.id === activeId ? 'page' : 'false'));
    $('#mobile-nav').value = activeId;
    activePanel = 'observe'; storyStep=0;showAllControls=false;filmExample=null; render(); updateRoute();window.scrollTo({top:0,behavior:'instant'});
    document.title = activeId === 'library' ? '可视化实验室' : `${content[activeId].title} · 可视化实验室`;
    sendParent({ type:'ready', id:activeId });
  }
  function navigation() {
    const directory=$('#toggle-directory');
    const setDirectory=open=>{document.body.classList.toggle('directory-open',open);directory.setAttribute('aria-expanded',String(open));};
    directory.addEventListener('click',()=>setDirectory(!document.body.classList.contains('directory-open')));
    document.addEventListener('keydown',e=>{if(e.key==='Escape'&&document.body.classList.contains('directory-open')){setDirectory(false);directory.focus();}});
    $('#lab-count').textContent = String(ids.length);
    let lastGroup;
    $('#navigation').innerHTML = '<a class="nav-link" href="#library" data-id="library"><span class="nav-index">⌂</span>全部实验</a>' + ids.map((id, i) => {
      const c = content[id]; const group = c.group !== lastGroup ? `<p class="nav-group">${escape(c.group)}</p>` : ''; lastGroup = c.group;
      return group + `<a class="nav-link" href="#${id}" data-id="${id}"><span class="nav-index">${String(i+1).padStart(2,'0')}</span>${escape(c.title)}</a>`;
    }).join('');
    $('#mobile-nav').innerHTML = '<option value="library">全部实验</option>' + ids.map(id => `<option value="${id}">${escape(content[id].title)}</option>`).join('');
    $('#mobile-nav').addEventListener('change', e => navigate(e.target.value));
    document.addEventListener('click', e => {
      const route = e.target.closest('a[data-id]'); if (route) { e.preventDefault(); navigate(route.dataset.id);setDirectory(false); window.scrollTo({top:0,behavior:'instant'}); }
      const source = e.target.closest('a[data-source]'); if(source && window.parent !== window){ e.preventDefault(); sendParent({type:'open-source',id:activeId,url:source.href}); }
      const atom = e.target.closest('a[data-atom]'); if (atom && window.parent !== window) { e.preventDefault(); sendParent({type:'open-note', path:atom.dataset.atom}); }
    });
    $('.brand').addEventListener('click', e => { e.preventDefault(); navigate('library'); });
  }
  const routes=[
    ['数学与几何',['tangent','linear','projection','svd','jacobian']],
    ['概率与推断',['sampling','confidence','testing','bayes','bootstrap','likelihood','biasvariance','entropy','uncertainty','causal']],
    ['机器学习',['regression','overfit','knn','logistic','svm','tree','ensemble','boosting','kde','modelselection','metrics','calibration','kernel','double-descent','fairness']],
    ['表示与聚类',['pca','kmeans','gmm','embeddings','autoencoder','contrastive']],
    ['网络与训练',['gradient','activations','neuron','mlp','backprop','initialization','optimizers','regularization','normalization','softmax','residual']],
    ['结构与序列',['convolution','pooling','attention','positional','gnn','rnn','lstm']],
    ['生成与决策',['vae','gan','flow','diffusion','rl-bellman','rl-policy']]
  ];
  function miniature(id){
    const r=math.compute(id,defaultParams(id));let marks='',label='';
    if(id==='tangent'){
      const pts=r.plots[0].series[0].points,sx=x=>35+(x-.25)/1.5*250,sy=y=>135-(y+.5)/4.2*100;
      marks=`<path d="${pts.filter(q=>q[0]>=.25&&q[0]<=1.75).map((q,i)=>`${i?'L':'M'}${sx(q[0])},${sy(q[1])}`).join(' ')}" fill="none" stroke="#45685a" stroke-width="2.5"/><path d="M${sx(.6)},${sy(r.data.forward*(-.4))}L${sx(1.6)},${sy(r.data.forward*.6)}" stroke="#3477a7" stroke-dasharray="6 4" stroke-width="2"/>`+[[1,0,'P'],[1.5,1.875,'Q']].map(([x,y,name])=>`<circle cx="${sx(x)}" cy="${sy(y)}" r="4" fill="#3477a7"/><text x="${sx(x)+10}" y="${sy(y)-10}">${name}</text>`).join('');label='在三次函数上，蓝色割线连接 P 和 Q';
    }else if(id==='backprop'){
      const vals=[['x',1],['z',r.data.forward.z],['h',r.data.forward.h],['ŷ',r.data.forward.output],['L',r.data.forward.loss]];
      marks=vals.map(([name,value],i)=>`<rect x="${16+i*60}" y="50" width="48" height="54" rx="8" fill="${i===4?'#faeadf':'#eaf1e9'}"/><text x="${40+i*60}" y="71" text-anchor="middle">${name}</text><text x="${40+i*60}" y="91" text-anchor="middle" class="mini-value">${number(value,2)}</text>${i<4?`<path d="M${65+i*60},77h9l-3,-3m3,3l-3,3" fill="none" stroke="#8da797"/>`:''}`).join('')+`<path d="M280,116H100l6,-4m-6,4l6,4" stroke="#b96940" fill="none"/><text x="195" y="138" text-anchor="middle" class="mini-value">从损失往回看</text>`;label='同一条计算链，前向算值，反向传梯度';
    }else{
      const weights=r.data.weights[2];marks=weights.map((v,i)=>`<text x="25" y="${34+i*32}">${['A','B','C','D'][i]}</text><rect x="50" y="${22+i*32}" width="210" height="16" rx="3" fill="#edf2ee"/><rect x="50" y="${22+i*32}" width="${v*210}" height="16" rx="3" fill="#367b65"/><text x="275" y="${34+i*32}" class="mini-value">${number(v,2)}</text>`).join('');label='Query C 读取四个位置的实际权重';
    }
    return `<svg class="library-miniature" viewBox="0 0 320 160" role="img" aria-label="${escape(label)}">${marks}</svg>`;
  }
  function renderLibrary() {
    root.innerHTML = `<div class="library-intro"><span class="eyebrow">数学 · 统计 · 机器学习 · 深度学习</span><h1>看见变化，理解原理。</h1><p class="question">从一个具体问题出发。移动一个参数，沿着图形找到原因，再回到公式与知识笔记。</p></div><div class="featured-stories">${[['tangent','两个测量点，怎样接近瞬时变化？','用点、线与一个小三角理解导数'],['backprop','一个权重，怎样改变最终损失？','沿同一张计算图，前算与倒传'],['attention','一次读取，究竟混合了什么？','看清匹配、分配与内容的加权']].map(([id,title,sub])=>`<a href="#${id}" data-id="${id}" class="featured-story">${miniature(id)}<div><small>跟着图，一步一步</small><h2>${escape(title)}</h2><p>${escape(sub)} <span>↗</span></p></div></a>`).join('')}</div>
    <div class="library-directory"><div><h2>探索全部 ${ids.length} 个实验</h2><p>选择方向，或用概念名称寻找。</p></div><label class="library-search"><span class="sr-only">搜索实验</span><input id="library-search" type="search" placeholder="搜索概念，如 PCA、方差、梯度…" aria-label="搜索实验"></label></div><div class="library-filters" aria-label="按学习方向筛选"><button data-route="all" aria-pressed="true">全部 <span>${ids.length}</span></button>${routes.map(([name,items],i)=>`<button data-route="${i}" aria-pressed="false">${escape(name)} <span>${items.length}</span></button>`).join('')}</div><p id="search-status" aria-live="polite"></p><div class="library-grid">${ids.map((id,i)=>{const c=content[id];return `<a class="library-card" href="#${id}" data-id="${id}"><small>${String(i+1).padStart(2,'0')} / ${escape(c.group)}</small><h2>${escape(c.title)}</h2><p>${escape(c.question)}</p><span>${(globalThis.LabStories?.[id]||globalThis.LabGuides?.[id])?.steps.length||3} 步探索 →</span></a>`;}).join('')}</div>`;
    let selected='all';const filter=()=>{const q=$('#library-search').value.trim().toLowerCase(),allowed=selected==='all'?ids:routes[Number(selected)][1];let visible=0;$$('.library-card').forEach(card=>{const c=content[card.dataset.id],match=allowed.includes(card.dataset.id)&&[card.dataset.id,c.title,c.en,c.summary,c.question].join(' ').toLowerCase().includes(q);card.hidden=!match;if(match)visible++;});$('#search-status').textContent=visible?`${visible} 个实验`:'没有匹配的实验。试试中文概念、英文名称或更短的关键词。';};
    $('#library-search').addEventListener('input',filter);$$('[data-route]').forEach(b=>b.addEventListener('click',()=>{selected=b.dataset.route;$$('[data-route]').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));filter();}));
  }
  function render() {
    clearTimeout(computeTimer);
    root.dataset.experiment=activeId;
    if (activeId === 'library') { renderLibrary(); return; }
    const c = content[activeId], index = ids.indexOf(activeId) + 1;
    const stages=story().steps;
    root.innerHTML = `<section class="experiment-heading"><div><span class="eyebrow">${escape(c.group)} <span aria-hidden="true">/</span> ${String(index).padStart(2,'0')}</span><h1>${escape(c.title)}</h1><p class="english-title">${escape(c.en)}</p></div><div class="heading-actions">${globalThis.LAB_EMBEDDED?`<a class="note-action" href="${atomURI("07_Programme/可视化实验室/实验/"+activeId+".md")}" data-atom="07_Programme/可视化实验室/实验/${activeId}.md">实验笔记 ↗</a>`:""}<button id="copy-link">复制双链</button><details class="export-menu"><summary>导出</summary><div class="export-options"><button id="export-csv">图形数据 · CSV</button><button id="export-json">计算结果与参数 · JSON</button>${globalThis.LAB_EMBEDDED?'':'<button id="print-experiment">打印当前实验</button>'}</div></details></div></section>
      <p class="opening-question">${prose(({tangent:'只知道两个测量点，怎样读出一个点的变化速度？',backprop:'改一个权重，怎么预先知道损失往哪边变？',attention:'一次注意力读取，究竟把哪些信息加在了一起？'})[activeId]||c.question)}</p>
      <nav class="story-steps" aria-label="分步探索">${stages.map((s,i)=>`<button class="story-step" data-story-step="${i}" aria-current="${i===0?'step':'false'}"><span>${String(i+1).padStart(2,'0')}</span>${escape(s.title)}</button>`).join('')}</nav>
      <section class="story-layout" aria-label="交互讲解"><div class="story-visual"><div id="story-scene"></div><div id="story-readout" aria-live="polite"></div></div><div class="story-coach"><p id="story-count" class="story-kicker"></p><h2 id="story-title"></h2><p id="story-caption"></p><div class="story-action"><span>动手试一试</span><p id="story-action"></p><div id="stage-example"></div></div><aside class="controls-panel" aria-label="实验参数"><div class="controls-grid">${c.controls.map(controlHTML).join('')}</div><button id="toggle-controls" aria-expanded="false">展开全部参数 +</button>${c.animation?'<div class="play-controls"><button id="play-steps">逐步播放 ▶</button><button id="next-step">前进一步 →</button></div>':''}</aside><div class="story-pagination"><button id="story-previous" aria-label="上一步">←</button><button id="story-next" class="primary">下一步 →</button></div></div></section>
      <p class="model-context"><span>本实验的模型</span>${prose(c.assumptions[0])}</p>
      <div class="toolbar"><div class="presets"><span class="presets-label">换个情境，检验刚才的解释</span>${c.presets.map((p,i)=>`<button class="preset" data-preset="${i}">${escape(p.label)}</button>`).join('')}</div><button class="reset" id="reset-params">恢复初始参数 ↺</button></div>
      <details class="advanced-results"><summary><span>完整图表与数学细节</span><small>展开数值、误差图、公式与适用边界</small></summary><div class="visual-area" aria-label="实验结果"><div id="metrics" class="metrics" aria-live="polite" aria-atomic="true"></div><div id="dynamic-notes"></div><div id="plots"></div><div class="formula-box"><p class="formula-label">数学表达 / THE MATHEMATICS</p>${renderFormula(c.formula)}</div></div></details>
      <section class="learning-area" aria-label="实验解释与检验"><div class="learning-tabs" role="tablist" aria-label="阅读实验说明">${[['observe','操作路线'],['explain','原理与边界'],['challenge','检验理解'],['source','来源与双链']].map(([id,label])=>`<button class="learning-tab" id="tab-${id}" role="tab" aria-selected="${id===activePanel}" aria-controls="panel-${id}" data-panel="${id}" tabindex="${id===activePanel?0:-1}">${label}</button>`).join('')}</div>${learningPanels(c)}</section><input id="import-snapshot" type="file" accept="application/json" hidden>`;
    $$('.story-step').forEach(b=>b.addEventListener('click',()=>setStoryStep(Number(b.dataset.storyStep))));
    const examples=globalThis.LabFilmExamples?.[activeId];
    if(examples){
      const panel=document.createElement('details');panel.className='film-examples';
      panel.innerHTML=`<summary>影片参数与延伸对照</summary><p>用“载入本片的模型与数据”复现实例；其他项目提供延伸对照。点击上方步骤可回到常规探索。</p>${examples.map((s,i)=>`<div><span>${escape(s.title)}</span><button data-film-example="${i}" data-endpoint="from">${s.single?'载入':activeId==='bootstrap'?'影片参数':'起点'}</button>${s.single?'':`<button data-film-example="${i}" data-endpoint="to">${activeId==='bootstrap'?'只增加 B':'终点'}</button>`}</div>`).join('')}`;
      $('.story-steps').after(panel);
      panel.addEventListener('click',e=>{const b=e.target.closest('[data-film-example]');if(!b)return;clearTimeout(computeTimer);clearInterval(playTimer);playTimer=null;if($('#play-steps'))$('#play-steps').textContent='逐步播放 ▶';globalThis.LabStories?.[activeId]?.dispose?.($('#story-scene'));filmExample=examples[Number(b.dataset.filmExample)];params={...filmExample[b.dataset.endpoint]};showAllControls=false;syncControls();compute();announce(filmExample.single?'已载入影片中的模型与数据':'已载入对照'+(b.dataset.endpoint==='from'?'起点':'终点'));});
    }
    $('#story-next').addEventListener('click',()=>{if(storyStep<stages.length-1)setStoryStep(storyStep+1);else{switchPanel('challenge');$('.learning-area').scrollIntoView({behavior:'smooth',block:'start'});}});
    $('#story-previous').addEventListener('click',()=>setStoryStep(storyStep-1));
    $('#toggle-controls').addEventListener('click',()=>{showAllControls=!showAllControls;syncStory();});
    $('#stage-example').addEventListener('click',e=>{if(e.target.closest('button'))setStoryStep(storyStep);});
    $('.advanced-results').addEventListener('toggle',()=>{if($('.advanced-results').open)renderPlots();});
    for (const control of c.controls) {
      const input=$(`#control-${control.key}`); input.value=String(params[control.key]);
      input.addEventListener(control.type==='select'?'change':'input', () => {
        if(playTimer){clearInterval(playTimer);playTimer=null;$('#play-steps').textContent='逐步播放 ▶';}
        const value = control.type==='select' ? control.options.find(o=>String(o.value)===input.value).value : Number(input.value);
        if (input.type==='number' && (input.value==='' || value<control.min || value>control.max)) { input.setCustomValidity(`请输入 ${control.min} 至 ${control.max} 之间的整数`); return; }
        input.setCustomValidity(''); params[control.key]=value;
        const valueLabel=$(`#value-${control.key}`); if(valueLabel)valueLabel.textContent=number(value,control.step<.01?3:2);
        $$('.preset').forEach(b=>b.classList.remove('is-selected'));
        clearTimeout(computeTimer); computeTimer=setTimeout(compute,30);
      });
    }
    $$('.preset').forEach(button=>button.addEventListener('click',()=>{
      clearInterval(playTimer);playTimer=null;filmExample=null;if($('#play-steps'))$('#play-steps').textContent='逐步播放 ▶';params={...defaultParams(activeId),...c.presets[Number(button.dataset.preset)].params}; syncControls(); compute();
      $$('.preset').forEach(b=>b.classList.toggle('is-selected',b===button));
    }));
    $('#reset-params').addEventListener('click',()=>{ clearInterval(playTimer);playTimer=null;filmExample=null;if($('#play-steps'))$('#play-steps').textContent='逐步播放 ▶';params=defaultParams(activeId); syncControls(); compute(); $$('.preset').forEach(b=>b.classList.remove('is-selected')); announce('已恢复初始参数'); });
    $('#copy-link').addEventListener('click',copyLink);
    $('#export-csv').addEventListener('click',exportCSV);
    $('#export-json').addEventListener('click',()=>download(`${activeId}-${params.seed||'fixed'}.json`,JSON.stringify({schema:'visual-knowledge-lab/v2',nonFiniteNumbers:'Strings Infinity, -Infinity and NaN preserve non-finite numeric values; null remains null.',experiment:activeId,params,result},(_,v)=>typeof v==='number'&&!Number.isFinite(v)?String(v):v,2),'application/json'));
    $('#print-experiment')?.addEventListener('click',()=>window.print());
    const reseed=$('#reseed');if(reseed)reseed.addEventListener('click',()=>{params.seed=(Number(params.seed)*1664525+1013904223)%99999+1;syncControls();compute();});
    $$('.learning-tab').forEach(tab=>{tab.addEventListener('click',()=>switchPanel(tab.dataset.panel));tab.addEventListener('keydown', e=>{
      if(!['ArrowLeft','ArrowRight','Home','End'].includes(e.key))return;e.preventDefault();const tabs=$$('.learning-tab'),i=tabs.indexOf(tab);const next=e.key==='Home'?0:e.key==='End'?tabs.length-1:(i+(e.key==='ArrowRight'?1:-1)+tabs.length)%tabs.length; switchPanel(tabs[next].dataset.panel);tabs[next].focus();
    });});
    if(c.animation){const key=c.animation.key,ctl=c.controls.find(x=>x.key===key);$('#play-steps').addEventListener('click',()=>{if(playTimer){clearInterval(playTimer);playTimer=null;$('#play-steps').textContent='逐步播放 ▶';return;}const end=Number(params[key])>ctl.min?Number(params[key]):ctl.max;params[key]=ctl.min;syncControls();compute();$('#play-steps').textContent='暂停 ❚❚';playTimer=setInterval(()=>{params[key]=Math.min(end,Number(params[key])+ctl.step);syncControls();compute();if(params[key]>=end){clearInterval(playTimer);playTimer=null;$('#play-steps').textContent='逐步播放 ▶';}},450);});$('#next-step').addEventListener('click',()=>{clearInterval(playTimer);playTimer=null;$('#play-steps').textContent='逐步播放 ▶';params[key]=Math.min(ctl.max,Number(params[key])+ctl.step);syncControls();compute();});}
    compute();
  }
  function controlHTML(c) {
    const value=params[c.key];
    // Every integer accepted by the model must also survive the native range input.
    const inputStep=math.schemas[activeId]?.[c.key]?.[3]?1:c.step;
    if(c.type==='select')return `<label class="control"><span class="control-line"><span class="control-name">${escape(c.label)}</span></span><select id="control-${c.key}" data-key="${c.key}">${c.options.map(o=>`<option value="${escape(o.value)}" ${String(o.value)===String(value)?'selected':''}>${escape(o.label)}</option>`).join('')}</select></label>`;
    if(c.key==='seed')return `<div class="control"><label class="control-line" for="control-seed"><span class="control-name">${escape(c.label)}</span></label><div class="seed-row"><input type="number" id="control-seed" data-key="seed" min="${c.min}" max="${c.max}" step="1" value="${value}"><button id="reseed" aria-label="更换随机种子，生成另一组样本">换一组</button></div><p class="seed-help">保持种子不变，比较参数的影响。</p></div>`;
    return `<label class="control"><span class="control-line"><span class="control-name">${escape(c.label)}</span><output class="control-value" id="value-${c.key}" for="control-${c.key}">${number(value,c.step<.01?3:2)}</output></span><input type="range" id="control-${c.key}" data-key="${c.key}" min="${c.min}" max="${c.max}" step="${inputStep}" value="${value}"><span class="range-endpoints" aria-hidden="true"><span>${c.min}</span><span>${c.max}</span></span></label>`;
  }
  function syncControls(){ for(const c of content[activeId].controls){$(`#control-${c.key}`).value=String(params[c.key]);const output=$(`#value-${c.key}`);if(output)output.textContent=number(params[c.key],c.step<.01?3:2);}}
  function learningPanels(c){
    const list=items=>`<ul>${items.map(x=>`<li>${prose(x)}</li>`).join('')}</ul>`;
    return `<div class="learning-panel" id="panel-observe" role="tabpanel" aria-labelledby="tab-observe"><p>${prose(c.summary)}</p><ol>${c.steps.map(x=>`<li>${prose(x)}</li>`).join('')}</ol><div class="assumption-box"><h3>开始前，明确这些条件</h3>${list(c.assumptions)}</div></div>
      <div class="learning-panel" id="panel-explain" role="tabpanel" aria-labelledby="tab-explain" hidden><p>${prose(c.explanation)}</p><h3>容易误解的地方</h3>${list(c.pitfalls)}<h3>继续连接</h3><div class="related-links">${c.related.filter(id=>content[id]).map(id=>`<a href="#${id}" data-id="${id}">${escape(content[id].title)} →</a>`).join('')}</div></div>
      <div class="learning-panel" id="panel-challenge" role="tabpanel" aria-labelledby="tab-challenge" hidden><p>先写下预测或完成推导，再展开提示与解释。</p>${c.challenges.map((q,i)=>`<article class="challenge"><p>${i+1}. ${prose(q.question)}</p><details><summary>一点提示</summary><div class="challenge-answer">${prose(q.hint)}</div></details><details><summary>展开解释与推导</summary><div class="challenge-answer">${prose(q.answer)}</div></details></article>`).join('')}</div>
      <div class="learning-panel" id="panel-source" role="tabpanel" aria-labelledby="tab-source" hidden><ul class="source-list">${c.sources.map(s=>`<li><a href="${escape(s.url)}" data-source="true" target="_blank" rel="noopener noreferrer">${escape(s.label)} ↗</a><span class="source-section">${escape(s.section)}</span><p class="source-support">支持：${prose(s.support)}</p></li>`).join('')}</ul><h3>回到知识原子</h3><div class="related-links">${(c.atoms||[]).map(a=>`<a href="${atomURI(a.path)}" data-atom="${escape(a.path)}">${escape(a.title)} ↗</a>`).join('') || '<p>对应知识点见实验的 Obsidian 入口。</p>'}</div><p class="source-section">本实验使用明确给定的数学模型或合成样本；样本结果随参数和种子改变。</p></div>`;
  }
  function atomURI(path){return `obsidian://open?vault=Academic&file=${encodeURIComponent(path)}`;}
  function switchPanel(id){activePanel=id;$$('.learning-tab').forEach(b=>{b.setAttribute('aria-selected',String(b.dataset.panel===id));b.tabIndex=b.dataset.panel===id?0:-1;});$$('.learning-panel').forEach(p=>p.hidden=p.id!==`panel-${id}`);}
  function compute(){
    try{
      result=math.compute(activeId,params);if(result.params)params={...result.params};
      $('#metrics').innerHTML=result.metrics.map(m=>`<div class="metric"><span class="metric-label">${escape(m.label)}</span><span class="metric-value ${typeof m.value==='string'?'is-text':''}">${escape(number(m.value,m.digits??4))}${m.suffix?`<span class="metric-unit">${escape(m.suffix)}</span>`:''}</span></div>`).join('');
      $('#dynamic-notes').innerHTML=result.notes?.length?`<div class="dynamic-notes">${result.notes.map(note=>`<p>${escape(note)}</p>`).join('')}</div>`:'';
      renderPlots();renderStory();updateRoute();
      root.dataset.experiment=activeId;root.dataset.ready='true';
    }catch(error){console.error(error);$('#dynamic-notes').innerHTML=`<p class="error" role="alert">无法计算这个参数组合：${escape(error.message)}</p>`;root.dataset.ready='error';}
  }
  function setStoryStep(index){
    filmExample=null;
    clearInterval(playTimer);playTimer=null;if($('#play-steps'))$('#play-steps').textContent='逐步播放 ▶';
    storyStep=Math.max(0,Math.min(story().steps.length-1,index));showAllControls=false;
    const changes=story().steps[storyStep].params;
    if(changes){params={...params,...changes};syncControls();}
    compute();
  }
  function syncStory(){
    const stages=story().steps,s=filmExample||stages[storyStep];
    $$('.story-step').forEach((b,i)=>b.setAttribute('aria-current',!filmExample&&i===storyStep?'step':'false'));
    $('#story-count').textContent=filmExample?(filmExample.single?'影片实例':'延伸对照'):`第 ${storyStep+1} 步 / ${stages.length}`;
    $('#story-title').textContent=s.title;$('#story-caption').innerHTML=prose(s.caption);$('#story-action').innerHTML=prose(s.action||s.caption);
    const mismatch=s.params&&Object.entries(s.params).some(([key,value])=>params[key]!==value);$('#stage-example').innerHTML=mismatch?'<small>当前为自定义条件</small><button type="button">载入本步示例 ↺</button>':'';
    const focus=s.focus||[];let more=0;
    for(const c of content[activeId].controls){const el=$(`#control-${c.key}`).closest('.control');const hide=!focus.includes(c.key);if(hide)more++;el.hidden=hide&&!showAllControls;}
    $('#toggle-controls').hidden=!more;$('#toggle-controls').textContent=showAllControls?'收起更多参数 −':`全部参数（另 ${more} 项） +`;$('#toggle-controls').setAttribute('aria-expanded',String(showAllControls));
    $('#story-previous').disabled=storyStep===0;$('#story-next').textContent=storyStep===stages.length-1?'检验一下理解 →':'下一步 →';
  }
  function renderStory(){
    if(!result||activeId==='library'||!$('#story-scene'))return;
    syncStory();const custom=filmExample?null:globalThis.LabStories?.[activeId],step=filmExample||story().steps[storyStep];
    if(globalThis.LabMechanisms?.ids.includes(activeId)&&globalThis.LabMechanisms.scene(activeId,params,result,step.plot||0)){globalThis.LabMechanisms.render({el:$('#story-scene'),params:{...params},result,id:activeId,plot:step.plot||0});$('#story-readout').innerHTML='';}
    else if(custom?.render){custom.render({el:$('#story-scene'),params:{...params},result,step:storyStep,onParam:(key,value)=>{const c=content[activeId].controls.find(c=>c.key===key);if(!c)return;params[key]=value;syncControls();compute();},format:number});$('#story-readout').innerHTML='';}
    else{
      const index=Math.max(0,Math.min(result.plots.length-1,step.plot||0));let p=result.plots[index];
      if(activeId==='confidence'){const shown=Math.min(40,params.reps);p={...p,title:`前 ${shown} 条置信区间 · 覆盖率按全部 ${params.reps} 次计算`,height:370,yDomain:[0,shown+1],series:p.series.map(s=>s.type==='segments'?{...s,points:s.points.filter(q=>q[1]<=shown)}:s)};}
      $('#story-scene').innerHTML=`<div class="hero-plot-heading"><span class="story-kicker">观察 ${String(index+1).padStart(2,'0')}</span><h2>${escape(p.title)}</h2></div><div class="chart-wrap" id="hero-chart"></div><div class="plot-legend">${legendHTML(p)}</div>`;
      drawChart($('#hero-chart'),p,'hero');
      const picks=step.metrics||[0,1];$('#story-readout').innerHTML=picks.map(i=>result.metrics[i]).filter(Boolean).slice(0,3).map(m=>`<div><span>${escape(m.label)}</span><strong>${escape(number(m.value,m.digits??4))}${escape(m.suffix||'')}</strong></div>`).join('');
    }
  }
  function colorScale(s){return s.scale||((s.zDomain?.[0]??-1)>=0?'sequential':'diverging');}
  function legendHTML(p){return p.series.map((s,j)=>!s.label?'':`<span class="legend-item"><i class="legend-swatch ${s.type} ${s.type==='heatmap'?colorScale(s):''} ${s.dashed?'dashed':''}" style="--series-color:${palette[(s.color??j)%palette.length]}" aria-hidden="true"></i>${escape(s.label)}${s.type==='heatmap'&&s.zDomain?' · '+number(s.zDomain[0])+' → '+number(s.zDomain[1]):''}</span>`).join('');}
  function renderPlots(){
    if(!result || activeId==='library')return;
    $('#plots').innerHTML=result.plots.map((p,i)=>`<figure class="plot" style="margin-left:0;margin-right:0"><div class="plot-header"><h2 class="plot-title" id="plot-title-${i}">${escape(p.title)}</h2><button class="plot-save" data-save="${i}" aria-label="保存${escape(p.title)}的矢量图">保存 SVG ↗</button></div><div class="chart-wrap" id="chart-${i}"></div><figcaption><div class="plot-legend">${legendHTML(p)}</div></figcaption></figure>`).join('');
    result.plots.forEach((p,i)=>drawChart($(`#chart-${i}`),p,i));
    $$('[data-save]').forEach(b=>b.addEventListener('click',()=>exportSVG(Number(b.dataset.save))));
  }
  function domain(values, supplied){
    if(supplied && supplied.length===2 && supplied.every(Number.isFinite) && supplied[0]<supplied[1])return [...supplied];
    const finite=values.filter(Number.isFinite);let lo=Math.min(...finite),hi=Math.max(...finite);if(!finite.length)return [-1,1];
    if(hi-lo<=64*Number.EPSILON*Math.max(1,Math.abs(lo),Math.abs(hi))){const pad=Math.max(1,Math.abs(lo)*.1);return [lo-pad,hi+pad];}const pad=(hi-lo)*.075;return [lo-pad,hi+pad];
  }
  function ticks(lo,hi,count){
    const raw=(hi-lo)/Math.max(1,count),power=10**Math.floor(Math.log10(raw)),fraction=raw/power;
    const step=(fraction<=1?1:fraction<=2?2:fraction<=5?5:10)*power;
    let values=[];for(let v=Math.ceil(lo/step)*step;v<=hi+step*1e-8 && values.length<20;v+=step)values.push(Math.abs(v)<step*1e-8?0:v);return values;
  }
  function drawDiagram(el,p,index){
    const width=Math.max(250,Math.floor(el.getBoundingClientRect().width||600)),nodes=p.diagram.nodes,vertical=width<480,n=nodes.length;
    const gap=vertical?28:16,bw=vertical?Math.min(250,width-36):(width-32-gap*(n-1))/n,bh=90;
    const height=vertical?32+n*bh+(n-1)*gap:180;
    const position=i=>vertical?[(width-bw)/2,16+i*(bh+gap)]:[16+i*(bw+gap),40];
    let edges='',marks='';
    nodes.forEach((node,i)=>{const [x,y]=position(i);if(i<n-1){const [nx,ny]=position(i+1);edges+=`<path d="${vertical?`M${x+bw/2},${y+bh}L${nx+bw/2},${ny-3}`:`M${x+bw},${y+bh/2}L${nx-3},${ny+bh/2}`}" stroke="#608a73" stroke-width="2" marker-end="url(#diagram-arrow-${index})"/>`;}
      marks+=`<g><rect x="${x}" y="${y}" width="${bw}" height="${bh}" rx="9" fill="#eaf1e9" stroke="#92b5a0"/><text x="${x+bw/2}" y="${y+23}" text-anchor="middle" style="font-size:14px;fill:#17614d;font-weight:600">${escape(node.label)}</text><text x="${x+bw/2}" y="${y+48}" text-anchor="middle" style="font-size:17px;fill:#243831">${escape(number(node.value,5))}</text><text x="${x+bw/2}" y="${y+73}" text-anchor="middle" style="font-size:${Math.min(12,bw/Math.max(5,[...(node.operation||'')].length))}px">${escape(node.operation||'')}</text></g>`;
    });
    el.innerHTML=`<svg class="chart" xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="diagram-title-${index}"><title id="diagram-title-${index}">${escape(p.title+'。'+nodes.map(n=>n.label+'='+number(n.value)+': '+n.operation).join(' → '))}</title><defs><marker id="diagram-arrow-${index}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0L10,5L0,10Z" fill="#608a73"/></marker></defs>${edges}${marks}</svg>`;
  }
  function drawChart(el,p,index){
    if(p.diagram){drawDiagram(el,p,index);return;}
    const width=Math.max(250,Math.floor(el.getBoundingClientRect().width||600));
    const height=p.height|| (p.equalAspect?Math.min(430,Math.max(280,width*.66)):Math.min(360,Math.max(260,width*.5)));
    const margin={left:width<430?54:67,right:18,top:20,bottom:57},pw=width-margin.left-margin.right,ph=height-margin.top-margin.bottom;
    const xs=[],ys=[];
    for(const s of p.series)for(const point of s.points){if(!point)continue;xs.push(point[0]);ys.push(point[1]);if(s.type==='segments'){xs.push(point[2]);ys.push(point[3]);}if(s.type==='bars'){const half=(point[2]||0)/2;xs.push(point[0]-half,point[0]+half);ys.push(0);}if(s.type==='heatmap'){xs.push(point[0]-(s.cellWidth||1)/2,point[0]+(s.cellWidth||1)/2);ys.push(point[1]-(s.cellHeight||1)/2,point[1]+(s.cellHeight||1)/2);}}
    let xd=domain(xs,p.xDomain),yd=domain(ys,p.yDomain);
    if(p.equalAspect){const unit=Math.max((xd[1]-xd[0])/pw,(yd[1]-yd[0])/ph),mx=(xd[0]+xd[1])/2,my=(yd[0]+yd[1])/2;xd=[mx-unit*pw/2,mx+unit*pw/2];yd=[my-unit*ph/2,my+unit*ph/2];}
    const sx=x=>margin.left+(x-xd[0])/(xd[1]-xd[0])*pw,sy=y=>margin.top+(p.reverseY?(y-yd[0])/(yd[1]-yd[0])*ph:ph-(y-yd[0])/(yd[1]-yd[0])*ph);
    const xt=p.xTicks||(p.xTickLabels?p.xTickLabels.map((_,i)=>i).filter(i=>i>=xd[0]&&i<=xd[1]):ticks(...xd,width<430?4:6)),yt=p.yTicks||(p.yTickLabels?p.yTickLabels.map((_,i)=>i).filter(i=>i>=yd[0]&&i<=yd[1]):ticks(...yd,5));
    const text=(x,y,value,cls,anchor='middle')=>`<text x="${x}" y="${y}" text-anchor="${anchor}" class="${cls}">${escape(value)}</text>`;
    let marks='';
    const candidates=[],onHeatmap=p.series.some(s=>s.type==='heatmap');
    p.series.forEach((s,j)=>{
      const color=palette[(s.color??j)%palette.length],dash=s.dashed?' stroke-dasharray="6 4"':'';
      if(s.type==='line'){
        let pen=false;const path=s.points.map(point=>{if(!point||!point.every(Number.isFinite)){pen=false;return '';}const command=pen?'L':'M';pen=true;return `${command}${sx(point[0]).toFixed(2)},${sy(point[1]).toFixed(2)}`;}).join(' ');
        if(s.halo)marks+=`<path d="${path}" fill="none" stroke="#fffefa" stroke-width="${(s.width||2.1)+3}" stroke-linejoin="round"/>`;
        marks+=`<path d="${path}" fill="${s.fill?color:'none'}" fill-opacity=".17" stroke="${color}" stroke-width="${s.width??(s.color===6?1.1:2.1)}"${dash} stroke-linejoin="round"/>`;
      }else if(s.type==='scatter'){
        s.points.forEach((point,k)=>{if(!point||!point.every(Number.isFinite))return;const x=sx(point[0]),y=sy(point[1]);const radius=s.radius??(s.points.length>150?2.3:3.3);if(onHeatmap&&j%2&&s.shape!=='diamond')marks+=`<path d="M${x-radius},${y-radius}L${x+radius},${y+radius}M${x-radius},${y+radius}L${x+radius},${y-radius}" stroke="#fffefa" stroke-width="3.6"/>`;marks+=s.shape==='diamond'?`<path d="M${x},${y-radius}L${x+radius},${y}L${x},${y+radius}L${x-radius},${y}Z" fill="${color}" stroke="#fffefa" stroke-width="1.5"/>`:j%2?`<path d="M${x-radius},${y-radius}L${x+radius},${y+radius}M${x-radius},${y+radius}L${x+radius},${y-radius}" stroke="${color}" stroke-width="1.6"/>`:`<circle cx="${x}" cy="${y}" r="${radius}" fill="${color}" fill-opacity=".9" ${onHeatmap?'stroke="#fffefa" stroke-width="1"':''}/>`;});
      }else if(s.type==='segments'){
        s.points.forEach(point=>{if(!point||!point.every(Number.isFinite))return;if(s.halo)marks+=`<line x1="${sx(point[0])}" y1="${sy(point[1])}" x2="${sx(point[2])}" y2="${sy(point[3])}" stroke="#fffefa" stroke-width="${(s.width||1.8)+3}"/>`;marks+=`<line x1="${sx(point[0])}" y1="${sy(point[1])}" x2="${sx(point[2])}" y2="${sy(point[3])}" stroke="${color}" stroke-width="${s.width??(s.color===6?1:1.8)}"${dash}${s.arrow?` marker-end="url(#arrow-${index}-${j})"`:""}/>`;});
      }else if(s.type==='heatmap'){
        const zs=s.points.map(q=>q[2]).filter(Number.isFinite),zd=s.zDomain||[Math.min(...zs),Math.max(...zs)];
        s.points.forEach(point=>{if(!point||!point.every(Number.isFinite))return;const [x,y,z]=point,cw=s.cellWidth||1,ch=s.cellHeight||1;
          const t=zd[1]===zd[0]?.5:Math.max(0,Math.min(1,(z-zd[0])/(zd[1]-zd[0])));
          const left=sx(x-cw/2),top=Math.min(sy(y-ch/2),sy(y+ch/2)),w=Math.abs(sx(x+cw/2)-left),h=Math.abs(sy(y-ch/2)-sy(y+ch/2));
          const color=colorScale({...s,zDomain:zd})==='sequential'?`rgb(${[238,246,244].map((c,i)=>Math.round(c+([23,97,77][i]-c)*t)).join(',')})`:t<.5?`rgb(${[49,108,165].map((c,i)=>Math.round(c+([248,247,239][i]-c)*2*t)).join(',')})`:`rgb(${[248,247,239].map((c,i)=>Math.round(c+([170,71,42][i]-c)*(2*t-1))).join(',')})`;
          marks+=`<rect x="${left}" y="${top}" width="${w}" height="${h}" fill="${color}" stroke="#fffefa" stroke-opacity=".35" stroke-width=".5"><title>${escape(`${s.label}: (${x}, ${y}) = ${number(z,6)}`)}</title></rect>`;
          if(s.showValues && w>=25 && h>=20)marks+=`<text x="${sx(x)}" y="${sy(y)+4}" text-anchor="middle" style="font-size:${Math.min(13,w/3)}px;fill:${colorScale({...s,zDomain:zd})==='sequential'?(t>.62?'white':'#243831'):(t<.18||t>.82?'white':'#243831')}">${number(z,2)}</text>`;
        });
      }else if(s.type==='bars'){
        s.points.forEach((point,k)=>{if(!point||!point.every(Number.isFinite))return;const bw=point[2]||Math.abs((s.points[k+1]?.[0]??s.points[k-1]?.[0]??point[0]+1)-point[0]);const x=sx(point[0]-bw/2),y=sy(Math.max(0,point[1])),barwidth=Math.max(.5,sx(point[0]+bw/2)-x-1);marks+=`<rect x="${x}" y="${y}" width="${barwidth}" height="${Math.abs(sy(point[1])-sy(0))}" fill="${color}" fill-opacity=".55" stroke="${color}" stroke-opacity=".5" stroke-width=".4"/>`;});
      }
      for(const point of s.points){if(!point)continue;if(Number.isFinite(point[0])&&Number.isFinite(point[1]))candidates.push({x:point[0],y:point[1],label:s.label,color,z:s.type==='heatmap'?point[2]:undefined});if(s.type==='segments')candidates.push({x:point[2],y:point[3],label:s.label,color});}
    });
    const grid=xt.map(v=>`<line x1="${sx(v)}" y1="${margin.top}" x2="${sx(v)}" y2="${margin.top+ph}" stroke="${v===0?'#bcc8bb':'#e8ece4'}"/>${text(sx(v),margin.top+ph+22,p.xTickLabels?.[v]??number(v,3),'tick')}`).join('')+yt.map(v=>`<line x1="${margin.left}" y1="${sy(v)}" x2="${margin.left+pw}" y2="${sy(v)}" stroke="${v===0?'#bcc8bb':'#e8ece4'}"/>${text(margin.left-10,sy(v)+4,p.yTickLabels?.[v]??number(v,3),'tick','end')}`).join('');
    const annotations=(p.annotations||[]).map(a=>text(sx(a.x),sy(a.y)-8,a.text,'annotation')).join('');
    const description=`${p.title}。横轴：${p.xLabel}，范围 ${number(xd[0])} 至 ${number(xd[1])}；纵轴：${p.yLabel}，范围 ${number(yd[0])} 至 ${number(yd[1])}。${p.series.map(s=>s.label).filter(Boolean).join('；')}。精确数据可由导出菜单获取。`;
    el.innerHTML=`<svg class="chart" xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="chart-title-${index} chart-desc-${index}"><title id="chart-title-${index}">${escape(p.title)}</title><desc id="chart-desc-${index}">${escape(description)}</desc><defs>${p.series.map((s,j)=>s.arrow?`<marker id="arrow-${index}-${j}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="${palette[(s.color??j)%palette.length]}"/></marker>`:'').join('')}<clipPath id="clip-${index}"><rect x="${margin.left}" y="${margin.top}" width="${pw}" height="${ph}"/></clipPath></defs>${grid}<g clip-path="url(#clip-${index})">${marks}${annotations}</g>${text(margin.left+pw/2,height-10,p.xLabel,'axis-title')}<text transform="translate(14 ${margin.top+ph/2}) rotate(-90)" text-anchor="middle" class="axis-title">${escape(p.yLabel)}</text><circle class="hover-point" r="5" fill="none" stroke="#263b32" stroke-width="2" visibility="hidden"/><rect class="chart-hit" x="${margin.left}" y="${margin.top}" width="${pw}" height="${ph}" fill="transparent"/></svg><div class="chart-tooltip" hidden></div>`;
    const tooltip=$('.chart-tooltip',el),marker=$('.hover-point',el),hit=$('.chart-hit',el);let pinned=false;
    function hover(e){const rect=el.getBoundingClientRect(),x=(e.clientX-rect.left)*width/rect.width,y=(e.clientY-rect.top)*width/rect.width;let best,distance=Infinity;for(const candidate of candidates){const d=(sx(candidate.x)-x)**2+(sy(candidate.y)-y)**2;if(d<distance){distance=d;best=candidate;}}if(!best)return;tooltip.hidden=false;tooltip.textContent=`${best.label}\n${p.xLabel}: ${p.xTickLabels?.[best.x]??number(best.x,5)}\n${p.yLabel}: ${p.yTickLabels?.[best.y]??number(best.y,5)}${best.z===undefined?'':'\n数值: '+number(best.z,6)}`;tooltip.style.left=Math.max(0,Math.min(rect.width-220,e.clientX-rect.left+15))+'px';tooltip.style.top=Math.max(0,e.clientY-rect.top-65)+'px';marker.setAttribute('cx',sx(best.x));marker.setAttribute('cy',sy(best.y));marker.setAttribute('visibility','visible');}
    hit.addEventListener('pointermove',e=>{if(!pinned)hover(e);});hit.addEventListener('pointerleave',()=>{if(!pinned){tooltip.hidden=true;marker.setAttribute('visibility','hidden');}});hit.addEventListener('click',e=>{pinned=!pinned;hover(e);if(!pinned){tooltip.hidden=true;marker.setAttribute('visibility','hidden');}});
  }
  function download(filename,data,type){const url=URL.createObjectURL(new Blob([data],{type}));const a=document.createElement('a');a.href=url;a.download=filename;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);announce('已生成导出文件');}
  function exportCSV(){
    const cell=value=>`"${String(value??'').replaceAll('"','""')}"`;
    const rows=[['experiment','parameters','plot','series','type','x','y','x_end','y_end','bin_width','value']];
    result.plots.forEach(p=>p.series.forEach(s=>s.points.forEach(point=>{if(!point)return;rows.push([activeId,JSON.stringify(params),p.title,s.label,s.type,point[0],point[1],s.type==='segments'?point[2]:'',s.type==='segments'?point[3]:'',s.type==='bars'?point[2]:'',s.type==='heatmap'?point[2]:'']);})));
    download(`${activeId}-plot-data.csv`,'\uFEFF'+rows.map(row=>row.map(cell).join(',')).join('\r\n'),'text/csv;charset=utf-8');
  }
  function exportSVG(index){
    const original=$(`#chart-${index} svg`),svg=original.cloneNode(true),style=document.createElementNS('http://www.w3.org/2000/svg','style');
    style.textContent='.chart text,text{font-family:Arial,"PingFang SC",sans-serif;fill:#526257}.tick{font-size:12px}.axis-title{font-size:13px}';svg.prepend(style);$('.chart-hit',svg)?.remove();$('.hover-point',svg)?.remove();svg.setAttribute('style','background:#fffefa');
    const oldHeight=Number(svg.getAttribute('height')),width=Number(svg.getAttribute('width'));const series=result.plots[index].series,legend=series.map(s=>s.label).filter(Boolean).join(' / ');
    const svgText=(text,x,y,size=12)=>{const el=document.createElementNS('http://www.w3.org/2000/svg','text');el.setAttribute('x',x);el.setAttribute('y',y);el.setAttribute('font-size',size);el.textContent=text;svg.append(el);};
    let y=oldHeight+24;
    function wrapped(text,prefix=''){const max=Math.max(16,Math.floor((width-30)/12));const chars=[...text];while(chars.length){svgText(prefix+chars.splice(0,max).join(''),12,y);y+=20;}}
    wrapped(content[activeId].title+' · '+result.plots[index].title);
    series.forEach((s,j)=>{if(!s.label)return;const mark=document.createElementNS('http://www.w3.org/2000/svg','line');mark.setAttribute('x1','14');mark.setAttribute('x2','30');mark.setAttribute('y1',y-4);mark.setAttribute('y2',y-4);mark.setAttribute('stroke',palette[(s.color??j)%palette.length]);mark.setAttribute('stroke-width','3');if(s.dashed)mark.setAttribute('stroke-dasharray','4 3');svg.append(mark);const chars=[...s.label],max=Math.max(12,Math.floor((width-55)/12));while(chars.length){svgText(chars.splice(0,max).join(''),40,y);y+=21;}});
    wrapped('参数：'+JSON.stringify(params));
    svg.setAttribute('height',String(y+10));svg.setAttribute('viewBox',`0 0 ${width} ${y+10}`);
    const metadata=document.createElementNS('http://www.w3.org/2000/svg','metadata');metadata.textContent=JSON.stringify({experiment:activeId,params,series:legend,sources:content[activeId].sources});svg.prepend(metadata);
    download(`${activeId}-figure-${index+1}.svg`,new XMLSerializer().serializeToString(svg),'image/svg+xml;charset=utf-8');
  }
  async function copyLink(){
    let link;
    if(globalThis.LAB_EMBEDDED){sendParent({type:'copy-link',id:activeId,params});return;}
    if(globalThis.LAB_OBSIDIAN_NOTE){link=atomURI(globalThis.LAB_OBSIDIAN_NOTE);}
    else{link=location.href.split('#')[0]+routeString();}
    try{await navigator.clipboard.writeText(link);announce('实验链接已复制');}
    catch(_){const input=document.createElement('textarea');input.value=link;input.style.position='fixed';input.style.opacity='0';document.body.append(input);input.select();let copied=false;try{copied=document.execCommand('copy');}catch(_){}input.remove();if(copied)announce('实验链接已复制');else {sendParent({type:'copy-link',id:activeId,params});announce('当前容器限制剪贴板；可从 Obsidian 实验入口复制双链。');}}
  }
  window.addEventListener('hashchange',()=>{const route=readRoute();navigate(route.id,route.params);});
  window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>{renderPlots();renderStory();},100);});
  window.addEventListener('message',e=>{if(e.source!==window.parent)return;if(e.data?.type==='visual-lab:copied')announce('实验双链已复制');if(e.data?.type==='visual-lab:navigate' && ids.includes(e.data.id))navigate(e.data.id);});
  navigation();const initial=readRoute();navigate(initial.id,initial.params);
  // Observable deterministic state for the separate in-browser verification page.
  globalThis.VisualLab={navigate,drawChart,getState:()=>({id:activeId,params:{...params},result}),number,version:'2.0.0',getStoryState:()=>({step:storyStep,steps:activeId==='library'?[]:story().steps})};
})();
