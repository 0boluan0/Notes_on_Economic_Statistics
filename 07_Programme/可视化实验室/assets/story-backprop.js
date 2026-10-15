(function(root){
  'use strict';
  const states=new WeakMap(),nodes=['x','z','h','output','loss'];
  const symbols={x:'x',z:'z',h:'h',output:'ŷ',loss:'L',w:'w',bias:'b',v:'v'};
  const names={x:'输入',z:'乘加',h:'激活',output:'预测',loss:'损失'};
  const lesson=()=>root.BackpropLesson;
  const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const n=x=>!Number.isFinite(x)?'—':x===0?'0':Math.abs(x)<.0001?x.toExponential(2).replace('-','−'):String(Number(x.toFixed(5))).replace(/^-/,'−');
  const formula=s=>root.katex?root.katex.renderToString(s,{throwOnError:false,displayMode:true}):esc(s);
  const fnum=x=>x<0?`(${n(x)})`:n(x);
  const modelKeys=['x','w','bias','v','target'];
  const sig=p=>JSON.stringify(['x','w','bias','v','target'].map(k=>p[k]));
  const defaults={x:2,w:.25,bias:0,v:2,target:1.5};
  const presets=[
    ['导览起点',defaults,'普通链式法则：每个中间数都可追踪。'],
    ['手算起点',{x:1,w:0,bias:0,v:2,target:1},'h=0，输出为0；第一层梯度却是 −2。'],
    ['输出权重为零',{x:1,w:1,bias:0,v:0,target:1},'v 截断了通向 w、b 的梯度；v 自己仍可能有梯度。'],
    ['激活饱和',{x:2,w:3,bias:0,v:1,target:0},'h 接近1，但 tanh 的本地斜率接近0。'],
    ['输入为零',{x:0,w:1,bias:.5,v:1,target:1},'w 的影响乘上 x=0；偏置 b 的影响仍保留。'],
    ['步幅过大',{x:1,w:0,bias:0,v:2,target:.2},'先把学习率设为1，比较更新前后的真实损失。']
  ];
  const steps=[
    {title:'前向 · 保存中间值',caption:'逐个执行运算，保留反向时需要的数。',action:'按“算下一步”，或选择一个运算盒。',focus:[],params:defaults},
    {title:'反向 · 追踪敏感度',caption:'上游梯度乘本地导数。',action:'沿同一张图反向走，参数暂时保持。',focus:[]},
    {title:'干预 · 检验梯度',caption:'先预测，再真正改一个参数重算。',action:'比较局部线性预测和实际变化。',focus:[]},
    {title:'更新 · 再算一次',caption:'用同一组旧梯度，同时更新三个参数。',action:'调整学习率，执行一次更新。',focus:[]}
  ];
  function forwardInfo(p,d,cursor){
    const f=d.forward;
    return [
      ['从给定的数开始','输入和目标由这一个样本给定；w、b、v 是可调整的参数。',`x=${n(p.x)},\\quad t=${n(p.target)}`,'Input and target belong to the example; weights are adjustable.','输入不在这次参数更新中改变。'],
      ['先乘权重，再加偏置','这一步只用 x、w、b，后面的激活函数还没参与。',`z=wx+b=${fnum(p.w)}\\times ${fnum(p.x)}+${fnum(p.bias)}=${n(f.z)}`,'Multiply the input by its weight, then add the bias.','反向时要用到 x，所以这次输入被缓存。'],
      ['把 z 送进 tanh','tanh 把数压到 −1 与 1 之间。输出大小和当前位置的斜率是两个不同的量。',`h=\\tanh(z)=\\tanh(${fnum(f.z)})=${n(f.h)}`,'Activation size and local slope are different quantities.',`保存 h=${n(f.h)}；本地导数为 1−h²=${n(1-f.h*f.h)}。`],
      ['隐藏值乘输出权重','输出权重 v 决定隐藏值如何影响预测。此处是线性读出，没有再加激活函数。',`\\hat y=vh=${fnum(p.v)}\\times ${fnum(f.h)}=${n(f.output)}`,'The output weight turns the hidden value into a prediction.','反向对 v 求导需要 h；对 h 求导需要 v。'],
      ['比较预测与目标','平方损失非负；残差保留方向，反向时会用到它。',`L=\\tfrac12(\\hat y-t)^2=\\tfrac12(${fnum(f.output)}-${fnum(p.target)})^2=${n(f.loss)}`,'The loss measures error; the signed residual carries direction.',`残差 ŷ−t=${n(d.dOutput)}。这次前向计算完成，参数还没有更新。`]
    ][cursor];
  }
  function reverseInfo(p,d,cursor){
    const f=d.forward;
    return [
      ['从损失自身的导数 1 开始','反向要问：每个量微小改变时，最终损失改变多快？',`\\frac{\\partial L}{\\partial L}=1`,'Backpropagation starts with the derivative of loss with respect to itself.','保留刚才的前向值；现在逐步求导，w、b、v 仍保持不变。'],
      ['经过平方损失','收到的梯度为1；平方损失对预测的本地导数是残差。',`\\frac{\\partial L}{\\partial\\hat y}=1\\times(\\hat y-t)=1\\times${fnum(d.dOutput)}=${n(d.dOutput)}`,'Upstream gradient × local derivative.',`损失 L=${n(f.loss)}；对预测的梯度是 ${n(d.dOutput)}。它们含义不同。`],
      ['经过输出乘法','ŷ=vh，因此预测对隐藏值的本地导数是 v。',`\\frac{\\partial L}{\\partial h}=\\frac{\\partial L}{\\partial\\hat y}\\,v=${fnum(d.dOutput)}\\times${fnum(p.v)}=${n(d.dH)}`,'The gradient arriving at the hidden value is scaled by v.',p.v===0?'v=0 使这条路径的梯度为0。稍后仍要单独检查 ∂L/∂v。':`这里复用了前向计算时的 v=${n(p.v)}。`],
      ['经过激活的本地斜率','tanh 在当前 z 的斜率是 1−h²；复用前向保存的 h。',`\\frac{\\partial L}{\\partial z}=\\frac{\\partial L}{\\partial h}(1-h^2)=${fnum(d.dH)}\\times${fnum(1-f.h*f.h)}=${n(d.dZ)}`,'A saturated activation can carry a large value and a tiny derivative.',`h=${n(f.h)}，本地斜率=${n(1-f.h*f.h)}。看这两个数如何分别变化。`],
      ['把梯度交给各个参数','z=wx+b 对 w 的本地导数是 x，对 b 是1；ŷ=vh 对 v 的本地导数是 h。',`\\frac{\\partial L}{\\partial w}=${fnum(d.dZ)}\\times${fnum(p.x)}=${n(d.gradients.w)}`,'Each parameter receives the derivative along its own dependency.',`w 的梯度=${n(d.gradients.w)}；b 的梯度=${n(d.gradients.bias)}；v 的梯度=${n(d.gradients.v)}。下面分别展开它们的来源。`]
    ][cursor];
  }
  function calculation(p,d,s){
    const info=s.stage===0?forwardInfo(p,d,s.cursor):reverseInfo(p,d,s.cursor);
    const local=s.stage===1&&s.cursor>0?[[1,d.dOutput,d.dOutput,'∂L/∂ŷ'],[d.dOutput,p.v,d.dH,'∂L/∂h'],[d.dH,1-d.forward.h*d.forward.h,d.dZ,'∂L/∂z'],[d.dZ,p.x,d.gradients.w,'∂L/∂w']][s.cursor-1]:null;
    const f=d.forward;
    const extra=s.stage===1&&s.cursor===4?`<div class="bp-parameter-gradients">${[
      ['w','∂L/∂z × x',d.dZ,p.x,d.gradients.w],['b','∂L/∂z × 1',d.dZ,1,d.gradients.bias],['v','∂L/∂ŷ × h',d.dOutput,f.h,d.gradients.v]
    ].map(([key,label,a,b,g])=>`<div><b>∂L/∂${key}</b><span>${label}</span><strong>${n(a)} × ${fnum(b)} = ${n(g)}</strong></div>`).join('')}</div>`:'';
    return `<div class="bp-calculation" aria-live="polite"><span class="bp-overline">${s.stage===0?'FORWARD / 前向计算':'BACKWARD / 反向求导'} · ${s.cursor+1} / 5</span><h2>${info[0]}</h2><p>${info[1]}</p>${local?`<div class="bp-multiply"><div><span>收到的上游梯度</span><b>${n(local[0])}</b></div><i>×</i><div><span>当前运算的局部导数</span><b>${n(local[1])}</b></div><i>=</i><div><span>关于 ${local[3].split('/')[1].slice(1)} 的梯度</span><b>${n(local[2])}</b></div></div>`:''}<div class="bp-equation">${formula(info[2])}</div><p class="bp-en">${info[3]}</p><div class="bp-reason">${info[4]}</div>${extra}</div>`;
  }
  function parameterControls(p){return `<div class="bp-controls"><h3>当前这一组数</h3>${[
    ['x','给定输入 x',-2,2],['target','给定目标 t',-2,2],['w','第一层权重 w',-3,3],['bias','隐藏偏置 b',-3,3],['v','输出权重 v',-3,3]
  ].map(([key,label,min,max])=>`<label><span>${label}</span><input data-bp-param="${key}" type="range" min="${min}" max="${max}" step="any" value="${p[key]}" aria-label="${label}"><output>${n(p[key])}</output></label>`).join('')}<p>改数后，所有读数和推导立即重新计算。</p></div>`;}
  function lossPlot(p,key,delta){
    const q=p[key],lo=Math.max(-3,q-Math.max(.25,Math.abs(delta)*3)),hi=Math.min(3,q+Math.max(.25,Math.abs(delta)*3));
    const d=lesson().evaluate(p).data,pts=Array.from({length:81},(_,i)=>{const x=lo+(hi-lo)*i/80;return [x,lesson().evaluate({...p,[key]:x}).data.forward.loss];});
    const tangent=x=>d.forward.loss+d.gradients[key]*(x-q),values=[...pts.map(x=>x[1]),tangent(lo),tangent(hi)],y0=Math.min(...values),y1=Math.max(...values),pad=Math.max((y1-y0)*.12,.001);
    const X=x=>48+420*(x-lo)/(hi-lo),Y=y=>174-132*(y-y0+pad)/(y1-y0+2*pad);
    const after=lesson().evaluate({...p,[key]:q+delta}).data.forward.loss;
    return `<svg class="bp-loss-plot" viewBox="0 0 510 220" role="img" aria-label="只改变${symbols[key]}时的真实损失曲线，及当前位置梯度给出的切线预测"><path d="M48 30V174H468" class="bp-axis"/><text x="12" y="25">L</text><text x="461" y="213">${symbols[key]}</text><text x="48" y="194">${n(lo)}</text><text x="442" y="194">${n(hi)}</text><text x="2" y="${Y(y1)+4}">${n(y1)}</text><text x="2" y="${Y(y0)+4}">${n(y0)}</text><path class="bp-loss-line" d="${pts.map(([x,y],i)=>(i?'L':'M')+X(x)+','+Y(y)).join(' ')}"/><path class="bp-tangent-line" d="M${X(lo)},${Y(tangent(lo))}L${X(hi)},${Y(tangent(hi))}"/><circle cx="${X(q)}" cy="${Y(d.forward.loss)}" r="5" class="bp-dot-before"/><circle cx="${X(q+delta)}" cy="${Y(after)}" r="5" class="bp-dot-after"/></svg>`;
  }
  function intervention(p,d,s){
    const key=s.key,delta=s.delta,g=d.gradients[key],allowed=p[key]+delta>=-3&&p[key]+delta<=3;
    const q=s.revealed&&allowed?lesson().probe(p,key,delta):null;
    let feedback='';
    if(q){const direction=Math.abs(q.predicted)<1e-14?'flat':q.predicted>0?'up':'down';feedback=`<p class="bp-feedback">${s.prediction===direction?'你的方向预测与当前梯度一致。':'当前梯度给出的方向是'+({up:'上升',down:'下降',flat:'一阶变化为零'})[direction]+'。'}有限改动还要看真实重算值。</p>`;}
    return `<div class="bp-calculation"><span class="bp-overline">INTERVENTION / 改一个量，查清因果</span><h2>梯度的预测，经得起重算吗？</h2><p>其余参数保持不变。先选一个参数和改变量，再预测损失的局部变化方向。</p><div class="bp-probe-tools"><label>参数<select data-bp-key aria-label="选择要扰动的参数">${['w','bias','v'].map(k=>`<option value="${k}" ${k===key?'selected':''}>${symbols[k]}</option>`).join('')}</select></label><label>改变量 Δ<select data-bp-delta aria-label="参数的改变量">${[-.5,-.1,-.01,-.001,.001,.01,.1,.5].map(x=>`<option value="${x}" ${x===delta?'selected':''}>${x>0?'+':''}${x}</option>`).join('')}</select></label><b>∂L/∂${symbols[key]} = ${n(g)}</b></div><div class="bp-equation">${formula(`\\Delta L\\approx\\frac{\\partial L}{\\partial ${symbols[key]}}\\Delta ${symbols[key]}=${fnum(g)}\\times${fnum(delta)}`)}</div><div class="bp-predictions" role="group" aria-label="先预测损失的局部变化">${[['down','下降'],['up','上升'],['flat','一阶不变']].map(([k,v])=>`<button data-bp-predict="${k}" aria-pressed="${s.prediction===k}">${v}</button>`).join('')}<button data-bp-action="reveal" class="bp-primary" ${!s.prediction||!allowed?'disabled':''}>重算并比较</button></div>${!allowed?'<p>改动会超出本实验 −3 至 3 的参数范围，请减小改变量。</p>':''}${q?`${feedback}<div class="bp-compare"><div><span>原损失</span><strong>${n(q.before.data.forward.loss)}</strong></div><div><span>重算后的损失</span><strong>${n(q.after.data.forward.loss)}</strong></div><div><span>梯度预测 ΔL</span><strong>${n(q.predicted)}</strong></div><div><span>真实变化 ΔL</span><strong>${n(q.actual)}</strong></div></div>${lossPlot(p,key,delta)}<p class="bp-plot-key">━━ 真实损失曲线　┄┄ 当前梯度的切线　● 原位置　○ 改动后</p><div class="bp-reason">差值 ${n(Math.abs(q.actual-q.predicted))} 来自局部线性近似。把改变量缩小，再比较。当前实验台仍保留原参数，便于重复对照。</div>`:'<div class="bp-probe-placeholder">先做一个方向预测，计算结果随后揭示。</div>'}</div>`;
  }
  function training(p,d,s){
    const u=lesson().update(p,s.rate),r=s.receipt,shown=r||u;
    return `<div class="bp-calculation"><span class="bp-overline">UPDATE / 梯度算完以后</span><h2>同时更新，再重新前向计算</h2><p>三个梯度都来自同一组旧参数。输入 x 与目标 t 保持固定。</p><label class="bp-rate">学习率 η <input data-bp-rate type="range" min="0.01" max="1" step="0.01" value="${s.rate}" aria-label="学习率"><output>${n(s.rate)}</output></label><div class="bp-equation">${formula('\\theta_{\\mathrm{new}}=\\theta_{\\mathrm{old}}-\\eta\\,\\nabla_\\theta L')}</div><p class="bp-table-title">${r?'刚才已执行的参数更新':'即将执行的参数更新'}</p><table class="bp-update-table"><thead><tr><th>参数</th><th>旧值</th><th>旧梯度</th><th>改变量</th><th>新值</th></tr></thead><tbody>${shown.deltas.map(x=>`<tr><th>${symbols[x.key]}</th><td>${n(x.old)}</td><td>${n(x.gradient)}</td><td>${n(x.delta)}</td><td>${n(x.next)}</td></tr>`).join('')}</tbody></table><button data-bp-action="${r?'prepare-update':'update'}" class="bp-primary" ${u.allowed||r?'':'disabled'}>${r?'查看下一次更新':'执行这一次更新'}</button>${u.allowed?'': '<p>这一步超出实验参数范围，请降低学习率。</p>'}${r?`<div class="bp-update-receipt" role="status"><span>刚才这一轮的实际结果</span><strong>${n(r.before.data.forward.loss)} → ${n(r.after.data.forward.loss)}</strong><p>损失${r.after.data.forward.loss<r.before.data.forward.loss?'下降':r.after.data.forward.loss>r.before.data.forward.loss?'上升':'不变'}。表格记录刚才这一步；实验台显示更新后重新计算的结果。</p><button data-bp-action="undo">撤回刚才这一轮</button></div>`:'<p class="bp-reason">反向传播负责计算梯度。这一个按钮才改变参数；更新后会重新计算全部中间值。</p>'}<p class="bp-en">All parameters use the same old gradient. A large step can increase loss.</p></div>`;
  }
  function forkPanel(p,s){
    const d=lesson().branches(p,1,s.branch),v=d.viaA+d.viaB;
    return `<details class="bp-fork" ${s.fork?'open':''}><summary>深入一层：一个变量走两条路径，为什么梯度相加？</summary><p>这里单独考察共享隐藏值 h 的两条读出路径：ŷ=ah+ch，a=1；移动 c，观察两条敏感度如何共同影响 h。</p><label>第二条路径的系数 c <input type="range" data-bp-branch min="-2" max="2" step=".1" value="${s.branch}" aria-label="第二条路径的系数"><output>${n(s.branch)}</output></label><div class="bp-fork-paths"><div><span>路径 h → ah → ŷ → L</span><strong>${n(d.upstream)} × 1 = ${n(d.viaA)}</strong></div><b>＋</b><div><span>路径 h → ch → ŷ → L</span><strong>${n(d.upstream)} × ${fnum(s.branch)} = ${n(d.viaB)}</strong></div><b>＝</b><div><span>总梯度 ∂L/∂h</span><strong>${n(v)}</strong></div></div><p>该扩展的 ŷ=${n(d.y)}，L=${n(d.L)}。${s.branch===-1?'两条路径各自有影响，却恰好相互抵消；总梯度为0。':'共享变量改变时，两条路径同时改变，所以要把贡献相加。'}</p><p class="bp-en">This separate two-path example shares h. Add the contributions at the shared variable.</p></details>`;
  }
  function render({el,params:p,result,step=0,onParams,onParam,onStep}){
    let s=states.get(el);
    if(!s){s={stage:step,cursor:0,key:'w',delta:.01,prediction:null,revealed:false,rate:.1,receipt:null,branch:-1,fork:false,playing:false,timer:0,signature:sig(p),change:null,answer:'',checked:false,transferOpen:false,theme:'light'};states.set(el,s);}
    if(s.stage!==step){s.stage=step;s.cursor=0;s.playing=false;clearTimeout(s.timer);}
    if(s.signature!==sig(p)){s.signature=sig(p);s.prediction=null;s.revealed=false;if(s.receipt&&sig(s.receipt.after.params)!==sig(p))s.receipt=null;if(s.change&&sig(s.change.after)!==sig(p))s.change=null;}
    s.args={el,params:p,result,step,onParams,onParam,onStep};
    const active=el.ownerDocument.activeElement;
    const focus=active?.closest('.bp-shell')?['data-bp-param','data-bp-action','data-bp-key','data-bp-delta','data-bp-predict','data-bp-rate','data-bp-branch'].map(k=>active.hasAttribute(k)?[k,active.getAttribute(k)]:null).find(Boolean):null;
    const d=result.data,f=d.forward,values={x:p.x,z:f.z,h:f.h,output:f.output,loss:f.loss},grad={x:d.gradients.x,z:d.dZ,h:d.dH,output:d.dOutput,loss:1};
    const selected=step===0?nodes[s.cursor]:step===1?['loss','output','h','z','z'][s.cursor]:s.key==='v'?'output':'z';
    const cache=nodes.map((id,i)=>{const known=step>0||i<=s.cursor,gradientKnown=step>1||(step===1&&i>=4-s.cursor);return `<button class="bp-cache ${id===selected?'is-selected':''}" data-bp-inspect="${id}" aria-pressed="${id===selected}"><span>${names[id]} <b>${symbols[id]}</b></span><strong>${known?n(values[id]):'待计算'}</strong><small>${step===0?(known?'前向值已保存':'等待执行'):gradientKnown?'∂L/∂'+symbols[id]+' = '+n(grad[id]):'梯度待传到这里'}</small></button>`;}).join('');
    const instrument=root.BackpropInstrument?.svg||'';
    const caption=step===0?'前向保存数值；点击读数可以检查当前运算。':step===1?'上方保留前向值；下面逐步显示损失对各个量的梯度。':'实验台保留当前参数的读数，对照结果在下方单独显示。';
    const title=step===0?'一个数，怎样走到损失？':step===1?'损失怎样感受到每一个参数？':step===2?'用一次干预，检查你的解释。':'参数真正改变的时刻。';
    const markup=`<section class="bp-shell" data-stage="${step}" data-cursor="${s.cursor}"><div class="bp-lab-head"><div><span class="bp-overline">COMPUTATION INSTRUMENT / 01</span><h2>${title}</h2></div><label class="bp-preset-label">观察情境<select data-bp-preset aria-label="选择观察情境"><option value="">${presets.find(x=>sig(x[1])===sig(p))?.[0]||'当前自定义参数'}</option>${presets.map((x,i)=>`<option value="${i}">${x[0]}</option>`).join('')}</select></label></div>${s.change?`<div class="bp-change" role="status"><span>已载入情境：${modelKeys.filter(k=>s.change.before[k]!==p[k]).map(k=>`${symbols[k]||'t'} ${n(s.change.before[k])} → ${n(p[k])}`).join('；')}</span><button data-bp-action="undo-example">撤回换例</button></div>`:''}<div class="bp-chapters"><label>从影片章节继续 <select data-bp-chapter aria-label="从影片章节继续"><option value="">选择一个位置，载入片中参数</option>${(root.BackpropFilm?.chapters||[]).map(c=>`<option value="${c.index}">${Math.floor(c.start/60)}:${String(Math.floor(c.start%60)).padStart(2,'0')} · ${c.title}</option>`).join('')}</select></label></div><div class="bp-instrument-card iso-plate"><div class="bp-card-top"><span>输入 → 乘加 → 激活 → 预测 → 损失</span><span>一个隐藏单元 · tanh</span></div><div class="bp-anatomy iso" data-theme="${s.theme}">${instrument}</div><div class="bp-cache-grid">${cache}</div><p class="bp-machine-caption">${caption}<span>五个读数尺分别对应 x ±2、z ±9、h ±1、ŷ ±3、L 0–12.5；长度不能跨尺比较。</span></p></div>${step<2?`<div class="bp-transport"><button data-bp-action="previous" ${s.cursor===0?'disabled':''}>← 上一步</button><div class="bp-progress" aria-label="计算进度">${Array.from({length:5},(_,i)=>`<button data-bp-cursor="${i}" aria-label="第${i+1}个计算位置" aria-current="${s.cursor===i?'step':'false'}">${i+1}</button>`).join('')}</div><button data-bp-action="play" aria-pressed="${s.playing}">${s.playing?'暂停':'自动演示'}</button><button data-bp-action="next" class="bp-primary">${s.cursor===4?(step===0?'开始反向求导 →':'用干预检验 →'):'算下一步 →'}</button></div>`:''}<div class="bp-workbench">${step<2?calculation(p,d,s):step===2?intervention(p,d,s):training(p,d,s)}</div><aside class="bp-parameters">${parameterControls(p)}<div class="bp-checkpoint"><span>这一阶段要分清</span><p>${['预测 ŷ 与损失 L：一个是模型输出，一个是比较目标后的误差。','激活值 h 与梯度 ∂L/∂h：一个是数值，一个是对最终损失的敏感度。','梯度是局部预测。有限的改动，需要用新的前向计算核对。','求梯度与更新参数是两步。新的参数需要新的前向与反向计算。'][step]}</p></div></aside>${transferPanel(s)}${forkPanel(p,s)}${tensorPanel()}<p class="bp-boundary">教学模型：一个样本、一个 tanh 隐藏单元、平方损失。上面的装置表示计算依赖；全部数值使用未舍入的计算结果。</p><div class="bp-film-caption" aria-hidden="true"></div></section>`;
    const template=el.ownerDocument.createElement('template');template.innerHTML=markup;
    if(el.firstElementChild)patch(el.firstElementChild,template.content.firstElementChild);else el.append(template.content);
    for(const module of root.BackpropInstrument?.modules||[]){
      const id=module.id,i=nodes.indexOf(id),known=step>0||i<=s.cursor;
      const display=el.querySelector(`[data-bp-display="${id}"]`);if(display)display.textContent=known?n(values[id]):'—';
      const max={x:2,z:9,h:1,output:3,loss:12.5}[id],fraction=id==='loss'?values[id]/max*2-1:values[id]/max,amount=known?fraction*15:0;
      el.querySelector(`[data-bp-carriage="${id}"]`)?.setAttribute('style',`transform:translate(${module.vector[0]*amount}px,${module.vector[1]*amount}px)`);
      el.querySelector(`[data-bp-module="${id}"]`)?.classList.toggle('is-active',id===selected);
    }
    if(!s.bound){bind(el,s);s.bound=true;}
    if(!s.observer&&root.IntersectionObserver){s.visible=true;s.observer=new IntersectionObserver(entries=>{s.visible=entries[entries.length-1].isIntersecting;});s.observer.observe(el);}
    if(focus)el.querySelector(`[${focus[0]}="${focus[1]}"]`)?.focus({preventScroll:true});
    const refresh=()=>render(s.args),stop=()=>stopPlayback(s);
    root.BackpropPlayer={
      state:()=>({stage:s.stage,cursor:s.cursor,params:{...s.args.params},rate:s.rate,playing:s.playing}),
      set:config=>{
        stop();
        s.revealed=false;s.prediction=null;s.receipt=null;
        if(config.params)onParams(config.params);
        if(Number.isInteger(config.stage)&&config.stage!==s.stage)onStep(config.stage);
        if(Number.isInteger(config.cursor))s.cursor=Math.max(0,Math.min(4,config.cursor));
        if(config.rate)s.rate=config.rate;
        if(config.key)s.key=config.key;
        if(config.delta!==undefined)s.delta=config.delta;
        if(config.reveal){s.prediction=config.prediction||'down';s.revealed=true;}
        if(config.fork!==undefined)s.fork=config.fork;
        if(config.branch!==undefined)s.branch=config.branch;
        if(config.theme)s.theme=config.theme;
        if(config.update){const receipt=lesson().update(s.args.params,s.rate);if(receipt.allowed){s.receipt=receipt;onParams(receipt.next);}}
        refresh();
        if(config.caption){const c=el.querySelector('.bp-film-caption');c.innerHTML=`<strong>${esc(config.caption)}</strong><span>${esc(config.en||'')}</span>`;}
      }
    };
  }
  function dispose(el){const s=states.get(el);if(s){clearTimeout(s.timer);s.observer?.disconnect();states.delete(el);}root.BackpropPlayer=null;}
  function transferPanel(s){
    const p={x:2,w:0,bias:0,v:1,target:-.5},d=lesson().evaluate(p).data,correct=s.answer.trim()!==''&&Number.isFinite(Number(s.answer))&&Math.abs(Number(s.answer)-d.gradients.w)<.005;
    return `<details class="bp-transfer" ${s.transferOpen?'open':''}><summary>换一组数，独立算一次 w 的梯度</summary><p>给定 x=2，w=0，b=0，v=1，t=−0.5。先算 h 与 ŷ，再求 ∂L/∂w。此题使用独立例子，实验台参数保持。</p><div class="bp-answer-row"><label for="bp-answer">你的 ∂L/∂w</label><input id="bp-answer" data-bp-answer type="number" step="any" value="${esc(s.answer)}" placeholder="输入数值"><button data-bp-action="check">核对推导</button></div>${s.checked?`<div class="bp-reason" role="status">${correct?'数值正确。':'需要再检查。'}h=0，ŷ=0，残差为 +0.5；乘 v=1、tanh′(0)=1，最后乘输入 x=2，所以 ∂L/∂w=1。${!correct?'若得到 0.5，漏了最后的 x；若得到 −1，检查 ŷ−t 的符号。':'试着解释：改变目标的符号，为什么会翻转更新方向？'}</div>`:''}</details>`;
  }
  function tensorPanel(){return `<details class="bp-tensor"><summary>从一个隐藏单元，走到实际网络</summary><p>对一个样本，用列向量表示：x∈ℝᵈ，W∈ℝᵐˣᵈ，b,h,v∈ℝᵐ。原来的乘法会变成矩阵运算，局部规则仍相同。</p><div class="bp-equation">${formula('z=Wx+b,\\quad h=\\tanh z,\\quad \\hat y=v^Th')}</div><div class="bp-equation">${formula('g_h=(\\hat y-t)v,\\quad g_z=g_h\\odot(1-h^2),\\quad \\nabla_W L=g_zx^T')}</div><p>g_z 是 m×1，xᵀ 是 1×d，外积产生与 W 相同的 m×d 梯度。一个参数在多个位置或多个样本中使用时，汇总各处贡献；小批量平均损失再除以样本数。</p></details>`;}
  // Keep active range inputs attached while the numerical scene changes during a drag.
  function patch(node,fresh){
    if(node.nodeType!==fresh.nodeType||node.nodeName!==fresh.nodeName){node.replaceWith(fresh.cloneNode(true));return;}
    if(node.nodeType===3){if(node.data!==fresh.data)node.data=fresh.data;return;}
    if(node.nodeType!==1)return;
    for(const a of [...node.attributes])if(!fresh.hasAttribute(a.name)&&!(a.name==='style'&&node.hasAttribute('data-bp-carriage')))node.removeAttribute(a.name);
    for(const a of fresh.attributes)if(node.getAttribute(a.name)!==a.value)node.setAttribute(a.name,a.value);
    if(node.tagName==='INPUT'&&node.value!==fresh.value)node.value=fresh.value;
    const children=[...fresh.childNodes];
    children.forEach((child,i)=>{if(node.childNodes[i])patch(node.childNodes[i],child);else node.append(child.cloneNode(true));});
    while(node.childNodes.length>children.length)node.lastChild.remove();
    if(node.tagName==='SELECT')node.value=fresh.value;
  }
  function stopPlayback(s){s.playing=false;clearTimeout(s.timer);}
  function bind(el,s){
    const refresh=()=>render(s.args);
    const apply=(values,receipt=false)=>{stopPlayback(s);if(!receipt)s.receipt=null;s.prediction=null;s.revealed=false;s.args.onParams(values);};
    const next=()=>{if(s.cursor<4){s.cursor++;refresh();}else{stopPlayback(s);s.args.onStep(Math.min(3,s.stage+1));}};
    el.addEventListener('click',event=>{
      const b=event.target.closest('[data-bp-action],[data-bp-cursor],[data-bp-predict],[data-bp-inspect],[data-bp-node]');if(!b||!el.contains(b))return;
      const p=s.args.params,step=s.stage;
      if(b.hasAttribute('data-bp-cursor')){stopPlayback(s);s.cursor=+b.dataset.bpCursor;refresh();return;}
      if(b.hasAttribute('data-bp-predict')){s.prediction=b.dataset.bpPredict;s.revealed=false;refresh();return;}
      const id=b.dataset.bpInspect||b.dataset.bpNode;
      if(id){stopPlayback(s);if(step===0)s.cursor=nodes.indexOf(id);else if(step===1)s.cursor=({loss:0,output:1,h:2,z:3,x:4})[id];else return;refresh();return;}
      const action=b.dataset.bpAction;
      if(action==='previous'){stopPlayback(s);s.cursor=Math.max(0,s.cursor-1);refresh();}
      if(action==='next'){stopPlayback(s);next();}
      if(action==='play'){
        if(s.playing){stopPlayback(s);refresh();return;}
        if(root.matchMedia?.('(prefers-reduced-motion: reduce)').matches){next();return;}
        if(s.cursor===4)s.cursor=0;s.playing=true;refresh();
        const advance=()=>{if(!s.playing||!el.isConnected)return;if(document.hidden||!s.visible){s.timer=setTimeout(advance,500);return;}if(s.cursor<4){s.cursor++;refresh();s.timer=setTimeout(advance,6500);}else{stopPlayback(s);refresh();}};
        s.timer=setTimeout(advance,6500);
      }
      if(action==='reveal'){s.revealed=true;refresh();}
      if(action==='prepare-update'){s.receipt=null;refresh();}
      if(action==='update'){const receipt=lesson().update(p,s.rate);if(receipt.allowed){s.receipt=receipt;apply(receipt.next,true);}}
      if(action==='undo'&&s.receipt){const old=s.receipt.before.params;s.receipt=null;apply(old);}
      if(action==='undo-example'&&s.change){const old=s.change.before;s.change=null;apply(old);}
      if(action==='check'){s.transferOpen=true;s.checked=true;refresh();}
    });
    el.addEventListener('input',event=>{
      const b=event.target;
      if(b.hasAttribute('data-bp-param')){s.change=null;apply({[b.dataset.bpParam]:+b.value});}
      if(b.hasAttribute('data-bp-rate')){s.rate=+b.value;s.receipt=null;refresh();}
      if(b.hasAttribute('data-bp-branch')){s.branch=+b.value;s.fork=true;refresh();}
      if(b.hasAttribute('data-bp-answer')){s.answer=b.value;s.checked=false;}
    });
    el.addEventListener('change',event=>{
      const b=event.target;
      if(b.hasAttribute('data-bp-preset')&&b.value!==''){const index=+b.value;s.cursor=0;if(index===5)s.rate=1;s.change={before:{...s.args.params},after:{...presets[index][1]}};apply(presets[index][1]);}
      if(b.hasAttribute('data-bp-chapter')&&b.value!==''){const frame=root.BackpropFilm.frames[+b.value],before={...s.args.params};root.BackpropPlayer.set(frame);s.change={before,after:{...s.args.params}};refresh();}
      if(b.hasAttribute('data-bp-key')){s.key=b.value;s.prediction=null;s.revealed=false;refresh();}
      if(b.hasAttribute('data-bp-delta')){s.delta=+b.value;s.prediction=null;s.revealed=false;refresh();}
    });
    el.addEventListener('toggle',event=>{if(event.target.matches('.bp-fork'))s.fork=event.target.open;if(event.target.matches('.bp-transfer'))s.transferOpen=event.target.open;},true);
    el.addEventListener('keydown',event=>{const b=event.target;if(b.hasAttribute('data-bp-node')&&['Enter',' '].includes(event.key)){event.preventDefault();b.dispatchEvent(new MouseEvent('click',{bubbles:true}));}if(b.hasAttribute('data-bp-param')&&['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(event.key)){event.preventDefault();const direction=['ArrowRight','ArrowUp'].includes(event.key)?1:-1;apply({[b.dataset.bpParam]:Math.min(+b.max,Math.max(+b.min,Number(b.value)+direction*(event.shiftKey?.1:.01)))});}if(b.hasAttribute('data-bp-answer')&&event.key==='Enter'){event.preventDefault();s.transferOpen=true;s.checked=true;refresh();}});
  }
  root.LabStories||={};root.LabStories.backprop={steps,render,dispose,preserveParams:true,ownsControls:true};
})(globalThis);
