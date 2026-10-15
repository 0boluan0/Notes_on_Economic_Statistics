(function(root){
  'use strict';
  let current;
  const C=()=>root.BackpropCausal;
  const cases=[['标准情境',()=>({...C().base})],['激活进入饱和区',()=>({...C().base,w:2})],['输出权重 v = 0',()=>({...C().base,v:0})],['输出权重 v = −2',()=>({...C().base,v:-2})]];
  const same=(a,b)=>['x','w','bias','v','target'].every(k=>a[k]===b[k]);
  function dispose(){if(!current)return;cancelAnimationFrame(current.raf);current.observer?.disconnect();current=null;}
  function mount({el,params,onParams}){
    if(!el)return;
    if(current?.el===el){if(!same(current.params,params)){current.stop();current.progress=0;current.focus=null;}current.params={...params};current.onParams=onParams;current.draw();return;}
    dispose();
    const s=current={el,params:{...params},onParams,mode:0,progress:0,focus:null,delta:.05,playing:false,raf:0};
    const $=q=>el.querySelector(q);
    el.innerHTML=`<nav class="bc-modes" aria-label="理解传播的四个阶段">${['01 前向计算','02 追踪一个改动','03 反向追溯影响','04 更新参数'].map((t,i)=>`<button data-bc-mode="${i}" aria-pressed="${i===0}">${t}</button>`).join('')}</nav>
      <div class="bc-context"><label>情境 <select aria-label="因果演示情境" data-bc-case>${cases.map(([name],i)=>`<option value="${i}">${name}</option>`).join('')}<option value="custom" disabled>自定义参数</option></select></label><label>从影片继续 <select data-bc-chapter aria-label="从有声影片章节继续"><option value="">选择片中位置</option>${(root.BackpropFilm?.chapters||[]).map(c=>`<option value="${c.index}">${Math.floor(c.start/60)}:${String(Math.floor(c.start%60)).padStart(2,'0')} ${c.title}</option>`).join('')}</select></label></div>
      <div class="bc-scene" aria-label="传播过程图"></div>
      <div class="bc-local" aria-label="放大一个局部运算">${['w → z 乘加','z → h 激活','h → ŷ 预测','ŷ → L 损失'].map((t,i)=>`<button data-bc-focus="${i}" aria-pressed="false">${t}</button>`).join('')}<button data-bc-follow>跟随过程</button></div>
      <div class="bc-transport"><button data-bc-action="play" class="bc-primary">播放这一段</button><button data-bc-action="step">前进一步</button><button data-bc-action="replay">回到本段起点</button><label>过程进度 <input type="range" min="0" max="1" step=".001" value="0" aria-label="传播演示进度" data-bc-progress></label></div>
      <div class="bc-probe" hidden><label>只改变 w <input type="range" min="-.2" max=".2" step=".001" value=".05" data-bc-delta aria-label="权重 w 的改变量"><output data-bc-delta-value>+0.05</output></label><p>先判断损失会往哪边变，再播放。把改变量从 0.05 缩小到 0.005，观察真实变化与梯度预测怎样靠近。</p></div>
      <div class="bc-update" hidden><button data-bc-action="update" class="bc-primary">执行一次更新，再看前向</button><span>上方可预览变化；点击此按钮才采用新参数。学习率 0.1，同时更新 w、b、v。</span></div>
      <p class="bc-status" role="status" aria-live="polite"></p>
      <p class="bc-boundary">分段慢放：局部曲线将输入从旧值移到新值，依次重算每一段。图中追踪 w 到损失的一条路径；输入 x 与目标 t 给定，多分支需将各路径梯度相加。</p>`;
    const status=t=>{$('.bc-status').textContent=t;};
    s.draw=()=>{
      if(!el.isConnected)return s.stop();
      const bounded=Math.max(-3-s.params.w,Math.min(3-s.params.w,s.delta));
      if(bounded!==s.delta){s.delta=bounded;status('改变量已限制在当前参数可用范围内。');}
      $('.bc-scene').innerHTML=C().svg(s);
      el.querySelectorAll('[data-bc-mode]').forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.bcMode===s.mode)));
      el.querySelectorAll('[data-bc-focus]').forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.bcFocus===s.focus)));
      $('[data-bc-follow]').setAttribute('aria-pressed',String(s.focus===null));
      $('[data-bc-progress]').value=String(s.progress);
      $('[data-bc-action="play"]').textContent=s.playing?'暂停演示':s.progress===1?'重播这一段':'播放这一段';
      $('[data-bc-action="step"]').disabled=s.progress===1||s.mode===3;
      $('.bc-probe').hidden=s.mode!==1;$('.bc-update').hidden=s.mode!==3;
      $('[data-bc-delta]').value=String(s.delta);$('[data-bc-delta-value]').textContent=(s.delta>0?'+':'')+C().n(s.delta);
      const selected=cases.findIndex(([,get])=>same(get(),s.params));$('[data-bc-case]').value=selected<0?'custom':String(selected);
      $('[data-bc-action="update"]').disabled=!root.BackpropLesson.update(s.params,.1).allowed;
    };
    s.stop=()=>{cancelAnimationFrame(s.raf);s.playing=false;};
    const step=()=>{s.stop();s.progress=Math.min(1,(Math.floor(s.progress*4+1e-6)+1)/4);s.draw();status(`已到本段第 ${Math.round(s.progress*4)} 个运算位置。`);};
    const play=()=>{
      if(s.playing){s.stop();s.draw();return;}
      if(matchMedia('(prefers-reduced-motion: reduce)').matches){if(s.progress===1)s.progress=0;step();return;}
      if(s.progress===1)s.progress=0;
      const start=performance.now(),from=s.progress,duration=16000;
      s.playing=true;s.focus=null;
      const tick=now=>{if(!s.playing)return;s.progress=Math.min(1,from+(now-start)/duration);if(s.progress===1){s.stop();status('这一段已结束。选择一个局部运算，可以留在原处仔细看。');}s.draw();if(s.playing)s.raf=requestAnimationFrame(tick);};
      s.raf=requestAnimationFrame(tick);
    };
    el.addEventListener('click',e=>{
      const b=e.target.closest('button,[data-causal-node]');if(!b)return;
      if(b.hasAttribute('data-bc-mode')){s.stop();s.mode=+b.dataset.bcMode;s.progress=0;s.focus=null;status(['先看每一步怎样把输入变成输出。','现在只改 w，其他参数和输入保持。','沿原路回来，前向数值保持不变。','先读清更新方向，再执行一次更新。'][s.mode]);s.draw();return;}
      if(b.hasAttribute('data-bc-focus')||b.hasAttribute('data-causal-node')){s.stop();s.focus=+(b.dataset.bcFocus??b.dataset.causalNode);s.draw();return;}
      if(b.hasAttribute('data-bc-follow')){s.focus=null;s.draw();return;}
      const action=b.dataset.bcAction;
      if(action==='play')play();
      if(action==='step')step();
      if(action==='replay'){s.stop();s.progress=0;s.focus=null;s.draw();status('已回到本段起点，参数保持不变。');}
      if(action==='update'){
        const u=root.BackpropLesson.update(s.params,.1);if(!u.allowed)return;
        s.stop();s.mode=0;s.progress=0;s.focus=null;s.onParams(u.next);
        status(`已更新 w、b、v。损失由 ${C().n(u.before.data.forward.loss)} 变为 ${C().n(u.after.data.forward.loss)}。播放前向，查看新的计算。`);
      }
    });
    el.addEventListener('keydown',e=>{const node=e.target.closest('[data-causal-node]');if(node&&(e.key==='Enter'||e.key===' ')){e.preventDefault();s.stop();s.focus=+node.dataset.causalNode;s.draw();$(`[data-bc-focus="${s.focus}"]`).focus();}});
    el.addEventListener('input',e=>{
      if(e.target.matches('[data-bc-progress]')){s.stop();s.progress=+e.target.value;s.draw();}
      if(e.target.matches('[data-bc-delta]')){s.stop();s.delta=+e.target.value;s.progress=1;s.draw();}
    });
    el.addEventListener('change',e=>{
      if(e.target.matches('[data-bc-case]')){s.stop();s.progress=0;s.focus=null;s.onParams(cases[+e.target.value][1]());status(`已载入${cases[+e.target.value][0]}。用同样的改动比较曲线和梯度。`);}
      if(e.target.matches('[data-bc-chapter]')&&e.target.value!=='')root.BackpropConceptPlayer.set(root.BackpropFilm.frames[+e.target.value]);
    });
    s.observer=new IntersectionObserver(entries=>{if(!entries[0].isIntersecting&&s.playing){s.stop();s.draw();}},{threshold:.05});s.observer.observe(el);
    s.draw();status('先播放前向；再进入“追踪一个改动”，看蓝色位置怎样依次移动。');
  }
  root.BackpropConceptPlayer={mount,dispose,state:()=>current?{params:{...current.params},mode:current.mode,progress:current.progress,focus:current.focus,delta:current.delta,playing:current.playing}:null,set:config=>{
    if(!current)return;const s=current;s.stop();if(config.params)s.onParams(config.params);
    s.mode=config.mode??0;s.progress=config.to??config.progress??0;s.focus=config.focus??null;s.delta=config.delta??.05;s.draw();
  }};
})(globalThis);
