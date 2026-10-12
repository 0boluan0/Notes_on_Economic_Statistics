/* One scalar network, shown as a connected forward/backward calculation. */
(function (root) {
  'use strict';
  const escape = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const states = new WeakMap();
  let diagramId = 0;
  const steps = [
    {title:'前向：从输入算到预测',caption:'每个节点接收相连的值，做一次运算。先看 w、b 怎样改变隐藏单元，再得到预测。',action:'改变 w，沿着 z → h → ŷ 找到变化；也可载入容易手算的数值。',focus:['w','bias'],params:{x:1,w:0,bias:0,v:2,target:1}},
    {title:'损失：预测离目标多远？',caption:'损失是残差平方的一半；它对预测的导数就是带符号的残差。',action:'改变目标 t，区分非负的损失与可以为负的导数。',focus:['target','v']},
    {title:'反向：沿每条依赖相乘',caption:'橙色沿原来的依赖返回。每到一条边，就把收到的梯度乘上该运算的局部导数。',action:'追到 w、b、v 三个分支，再看 v=0 的对照，哪些梯度会停住？',focus:['v','x']},
    {title:'核对：只把 w 改一点',caption:'梯度预测小改动会让损失怎样变。真的改一次 w，再比较局部预测和重新计算的结果。',action:'先按梯度符号预测增减，再点击 w ± 0.1；其他参数保持。',focus:['w']}
  ];
  function render({el,params,result,step=0,onParam,format}) {
    const d = result && result.data;
    if (!d || !d.forward || !d.gradients) { el.textContent='等待当前网络的计算结果…'; return; }
    const stage = Math.max(0,Math.min(3,Number.isInteger(step)?step:0)), reverse=stage>=2, f=d.forward;
    const fmt = value => {
      if (!Number.isFinite(value)) return '未定义';
      if (value === 0) return '0';
      if (Math.abs(value)<1e-4) return value.toExponential(2).replace('-', '−');
      return String(typeof format==='function'?format(value,4):Number(value.toFixed(4))).replace(/^-/, '−');
    };
    const n = value => escape(fmt(value));
    let state=states.get(el);
    if(!state){state={comparison:null,observer:null,frame:null};states.set(el,state);}
    if(state.observer)state.observer.disconnect();
    if(state.frame!==null && typeof root.cancelAnimationFrame==='function')root.cancelAnimationFrame(state.frame);
    const priorFocus=el.ownerDocument.activeElement;
    const focusKey=el.contains(priorFocus)?priorFocus.getAttribute('data-bp-action'):null;
    if(state.comparison){
      const c=state.comparison;
      if(['x','bias','v','target'].some(k=>params[k]!==c.params[k]) || Math.abs(params.w-c.afterW)>1e-10)state.comparison=null;
    }
    const derivative=(label,value)=>`<span class="bp-gradient${value===0?' is-zero':''}">${label} <b>${n(value)}</b></span>`;
    const card=(key,label,value,operation,gradient,soft=false)=>`<div class="bp-node bp-node-${key}${soft?' is-soft':''}" data-bp-node="${key}"><span class="bp-node-label">${label}</span><strong class="bp-node-value">${n(value)}</strong><span class="bp-operation">${operation}</span>${reverse?derivative(key==='loss'?'∂L/∂L =':`∂L/∂${key==='output'?'ŷ':key} =`,gradient):''}</div>`;
    const parameter=(key,symbol,value,local,gradient,fixed=false)=>`<div class="bp-parameter${fixed?' is-target':''}" data-bp-node="${key}"><span class="bp-parameter-label">${fixed?'目标':'参数'} <b>${symbol}</b></span><strong>${n(value)}</strong>${reverse&&!fixed?`<span class="bp-local">局部乘数 ${local}</span>${derivative(`∂L/∂${symbol} =`,gradient)}`:''}${fixed?'<span class="bp-fixed">用于比较</span>':''}</div>`;
    const zeroNote=params.v===0
      ? `v=0，所以传给 h 的梯度为 0。v 自己的分支乘的是 h=${n(f.h)}，其梯度是 ${n(d.gradients.v)}。`
      : params.x===0
        ? `x=0，使 w 的分支乘上 0；b 的分支乘 1，所以 ∂L/∂b=${n(d.gradients.bias)}。`
        : `到 w 乘输入 x，到 b 乘 1，到 v 乘隐藏值 h。同一个损失，沿不同分支得到不同梯度。`;
    const product = [
      {name:'损失对预测',symbol:'ŷ − t',value:d.dOutput},
      {name:'输出对隐藏值',symbol:'v',value:params.v},
      {name:'tanh 的局部导数',symbol:'1 − h²',value:1-f.h*f.h},
      {name:'z 对 w',symbol:'x',value:params.x}
    ];
    let detail;
    if(stage===0){
      detail=`<div class="bp-forward-equations"><p><span>先合成输入</span><b>z = ${n(params.w)} × ${n(params.x)} + (${n(params.bias)}) = ${n(f.z)}</b></p><p><span>再经过非线性与输出权重</span><b>h = tanh(${n(f.z)}) ≈ ${n(f.h)}<br>ŷ = ${n(params.v)} × ${n(f.h)} ≈ ${n(f.output)}</b></p></div><p class="bp-footnote">括号保留正负号；中间显示值经过取整，计算使用完整精度。</p>`;
    }else if(stage===1){
      detail=`<div class="bp-loss-readout"><div><span>带符号的残差</span><strong>ŷ − t = ${n(d.dOutput)}</strong><p>预测 ${n(f.output)}，目标 ${n(params.target)}</p></div><div><span>非负的平方损失</span><strong>L = ${n(f.loss)}</strong><p>½ × (ŷ − t)²</p></div><div class="bp-loss-slope"><span>传回预测的梯度</span><strong>∂L/∂ŷ = ${n(d.dOutput)}</strong><p>${d.dOutput<0?'局部增加预测，损失会下降。':d.dOutput>0?'局部减小预测，损失会下降。':'预测正好等于目标，一阶变化为 0。'}</p></div></div>`;
    }else if(stage===2){
      detail=`<p class="bp-branch-note">${zeroNote}</p><details class="bp-chain-details"><summary>展开 w 梯度的四个因子</summary><div class="bp-chain" aria-label="w 梯度的四个因子">${product.map((p,i)=>`${i?'<span class="bp-times" aria-hidden="true">×</span>':''}<div class="bp-factor"><span>${p.name}</span><b>${p.symbol}</b><strong>${n(p.value)}</strong></div>`).join('')}<span class="bp-times" aria-hidden="true">≈</span><div class="bp-factor bp-factor-result"><span>w 的梯度</span><b>∂L/∂w</b><strong>${n(d.gradients.w)}</strong></div></div></details><p class="bp-footnote">对照按钮载入新情境；v=0 使用 x=w=t=1、b=0；x=0 使用 w=v=t=1、b=0.5。</p>`;
    }else{
      const c=state.comparison;
      const comparison=c?`<div class="bp-comparison" role="status"><div class="bp-comparison-title"><span>只改 w：${n(c.beforeW)} → ${n(params.w)}</span><button type="button" data-bp-action="undo">撤回这一步</button></div><div class="bp-comparison-grid"><div><span>原来的损失</span><strong>${n(c.beforeLoss)}</strong></div><div><span>重新计算的损失</span><strong>${n(f.loss)}</strong></div><div><span>局部预测 ΔL</span><strong>${n(c.predicted)}</strong><small>原梯度 × Δw</small></div><div><span>实际变化 ΔL</span><strong>${n(f.loss-c.beforeLoss)}</strong><small>新损失 − 原损失</small></div></div><p>用改动前的梯度 ${n(c.beforeGradient)} 预测；两者不必完全相等，因为 ΔL≈梯度×Δw 是局部线性近似。</p></div>`:`<p class="bp-before-prediction">当前 ∂L/∂w = <b>${n(d.gradients.w)}</b>。${d.gradients.w<0?'增加一点 w，局部预测损失下降。':d.gradients.w>0?'减小一点 w，局部预测损失下降。':'一阶预测为 0；有限改动仍需实际重算。'}</p>`;
      detail=`<div class="bp-perturb"><div class="bp-probe-heading"><div><span class="bp-kicker">先预测，再动手</span><h4>ΔL ≈ (∂L/∂w) × Δw</h4></div><div class="bp-probe-actions"><button type="button" data-bp-action="minus" ${params.w<=-3?'disabled':''}>w − 0.1</button><button type="button" data-bp-action="plus" ${params.w>=3?'disabled':''}>w + 0.1</button></div></div>${comparison}<p class="bp-footnote">每次只改变 w；按钮在 −3 与 3 的边界截断步长。这是单步数值核对，不是自动训练。</p></div>`;
    }
    const examples=stage===2?'<div class="bp-examples bp-examples-top"><button type="button" data-bp-action="zero-v">v=0：梯度全停了吗？</button><button type="button" data-bp-action="zero-x">x=0：w 与 b 有何不同？</button></div>':'';
    el.innerHTML=`<section class="bp-story" data-stage="${stage}" aria-label="反向传播的逐步计算图"><header class="bp-scene-heading"><span class="bp-kicker">一个输入 · 一个隐藏单元</span><button type="button" data-bp-action="simple">用可手算数值</button></header>${examples}<div class="bp-flow-key"><span><i></i>前向值</span>${reverse?'<span class="bp-key-gradient"><i></i>反向梯度 · 边上是局部乘数</span>':'<span>沿箭头读取运算</span>'}</div><div class="bp-graph${reverse?' has-gradients':''}"><svg class="bp-wires" aria-hidden="true"></svg><div class="bp-params-z">${parameter('w','w',params.w,'x',d.gradients.w)}${parameter('bias','b',params.bias,'1',d.gradients.bias)}</div><div class="bp-params-v">${parameter('v','v',params.v,'h',d.gradients.v)}</div><div class="bp-params-target">${parameter('target','t',params.target,'',0,true)}</div>${card('x','输入 x',params.x,'给定的输入',d.gradients.x,stage===1)}${card('z','线性组合 z',f.z,'w × x + b',d.dZ,stage===1)}${card('h','隐藏值 h',f.h,'tanh(z)',d.dH,stage===1)}${card('output','预测 ŷ',f.output,'v × h',d.dOutput)}${card('loss','损失 L',f.loss,'½(ŷ − t)²',1,stage===0)}</div><div class="bp-detail">${detail}</div><p class="bp-sr">前向依赖：x、w、b 一起生成 z；z 生成 h；h 与 v 生成预测 ŷ；预测与目标 t 生成损失。反向沿这些依赖使用链式法则，目标保持固定。</p></section>`;
    const apply=(values)=>{state.comparison=null;for(const [key,value]of Object.entries(values))if(typeof onParam==='function')onParam(key,value);};
    el.querySelectorAll('[data-bp-action]').forEach(button=>button.addEventListener('click',()=>{
      const action=button.getAttribute('data-bp-action');
      if(action==='simple')apply({x:1,w:0,bias:0,v:2,target:1});
      else if(action==='zero-v')apply({x:1,w:1,bias:0,v:0,target:1});
      else if(action==='zero-x')apply({x:0,w:1,bias:.5,v:1,target:1});
      else if(action==='undo') {const before=state.comparison?.beforeW;state.comparison=null;if(before!==undefined&&typeof onParam==='function')onParam('w',before);}
      else if(action==='plus'||action==='minus'){
        const next=Math.max(-3,Math.min(3,Number((params.w+(action==='plus'?.1:-.1)).toFixed(10)))),delta=next-params.w;
        state.comparison={params:{...params},beforeW:params.w,afterW:next,beforeLoss:f.loss,beforeGradient:d.gradients.w,predicted:d.gradients.w*delta};
        if(typeof onParam==='function')onParam('w',next);
      }
    }));
    if(focusKey)el.querySelector(`[data-bp-action="${focusKey}"]`)?.focus({preventScroll:true});

    const graph=el.querySelector('.bp-graph'),svg=el.querySelector('.bp-wires'),uid=++diagramId;
    const draw=()=>{
      if(!graph.isConnected)return;
      const bounds=graph.getBoundingClientRect(),wide=bounds.width>=600;
      const box=key=>{const r=graph.querySelector(`[data-bp-node="${key}"]`).getBoundingClientRect();return{x:r.left-bounds.left,y:r.top-bounds.top,w:r.width,h:r.height};};
      const boxes=Object.fromEntries(['x','z','h','output','loss','w','bias','v','target'].map(k=>[k,box(k)]));
      const stroke=reverse?'#ac502a':'#397863';
      let lines=`<defs><marker id="bp-arrow-${uid}" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0L7 3.5L0 7Z" fill="${stroke}"/></marker><marker id="bp-muted-${uid}" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0L7 3.5L0 7Z" fill="#8a9d91"/></marker></defs>`;
      const path=(geometry,muted=false)=>`<path d="${geometry}" fill="none" stroke="${muted?'#8a9d91':stroke}" stroke-width="${reverse&&!muted?2:1.6}" marker-end="url(#bp-${muted?'muted':'arrow'}-${uid})"/>`;
      const tag=(x,y,value,symbol)=>`<g class="bp-edge-label"><rect x="${x-44}" y="${y-13}" width="88" height="34" rx="5"/><text x="${x}" y="${y}" text-anchor="middle">× ${n(value)}</text><text class="bp-edge-symbol" x="${x}" y="${y+14}" text-anchor="middle">${escape(symbol)}</text></g>`;
      const chain=[['x','z',params.w,'w'],['z','h',1-f.h*f.h,'1 − h²'],['h','output',params.v,'v'],['output','loss',d.dOutput,'ŷ − t']];
      for(const [from,to,local,symbol]of chain){
        const a=boxes[from],b=boxes[to];
        if(wide){
          if(reverse){const sx=b.x+b.w*.25,sy=b.y+b.h,tx=a.x+a.w*.75,ty=a.y+a.h,level=Math.max(sy,ty)+27;lines+=path(`M${sx},${sy}C${sx},${level} ${tx},${level} ${tx},${ty+3}`);lines+=tag((sx+tx)/2,level+19,local,symbol);}
          else lines+=path(`M${a.x+a.w},${a.y+a.h/2}L${b.x-3},${b.y+b.h/2}`);
        }else{
          const x=a.x+a.w/2,ay=a.y+a.h,by=b.y;
          lines+=reverse?path(`M${x},${by}L${x},${ay+3}`):path(`M${x},${ay}L${x},${by-3}`);
          if(reverse)lines+=tag(x-50,(ay+by)/2-4,local,symbol);
        }
      }
      for(const [from,to,port]of [['w','z',.27],['bias','z',.73],['v','output',.5],['target','loss',.5]]){
        const a=boxes[from],b=boxes[to],back=reverse&&from!=='target';
        let sx,sy,tx,ty;
        if(wide){sx=a.x+a.w/2;sy=a.y+a.h;tx=b.x+b.w*port;ty=b.y;}
        else{sx=a.x+a.w;sy=a.y+a.h/2;tx=b.x;ty=b.y+b.h*port;}
        const middle=wide?(sy+ty)/2:(sx+tx)/2;
        const forward=wide?`M${sx},${sy}C${sx},${middle} ${tx},${middle} ${tx},${ty-3}`:`M${sx},${sy}C${middle},${sy} ${middle},${ty} ${tx-3},${ty}`;
        const backward=wide?`M${tx},${ty}C${tx},${middle} ${sx},${middle} ${sx},${sy+3}`:`M${tx},${ty}C${middle},${ty} ${middle},${sy} ${sx+3},${sy}`;
        lines+=path(back?backward:forward,reverse&&from==='target');
      }
      svg.setAttribute('viewBox',`0 0 ${bounds.width} ${bounds.height}`);svg.innerHTML=lines;
    };
    draw();
    if(typeof root.requestAnimationFrame==='function')state.frame=root.requestAnimationFrame(draw);
    if(typeof root.ResizeObserver==='function'){state.observer=new root.ResizeObserver(draw);state.observer.observe(graph);}
  }
  function dispose(el){
    const state=states.get(el);if(!state)return;
    state.observer?.disconnect();
    if(state.frame!==null&&typeof root.cancelAnimationFrame==='function')root.cancelAnimationFrame(state.frame);
    states.delete(el);
  }
  root.LabStories ||= {};
  root.LabStories.backprop={steps,render,dispose};
})(typeof globalThis!=='undefined'?globalThis:this);
