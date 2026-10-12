(function (root) {
  'use strict';
  root.LabStories ||= {};
  const cache = new Map();
  let serial = 0;
  const esc = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const expressions = {cubic:'x³ − x', sine:'sin x', exponential:'exp x', exp:'exp x'};
  const steps = [
    {title:'先看两个测量点', caption:'P 是现在的位置，Q 是向右隔 h 再测的一点。先看见两个点，再谈它们之间的变化。', action:'选择函数和观察位置，找到 P 与 Q。', focus:['function','x'], params:{function:'cubic',x:1,h:.5}},
    {title:'一段变化，除以一段间隔', caption:'横边是间隔 Δx，竖边是变化 Δy。连接 P、Q 的割线，把这段平均变化率画了出来。', action:'改变 h，看看三角形和割线怎样一起变化。', focus:['h']},
    {title:'让第二点靠近', caption:'P 不动，让 Q 靠近。割线的斜率在逼近 P 处的切线斜率；坐标范围始终不变。', action:'点击“间隔减半”，比较割线与切线。', focus:['h']},
    {title:'两侧都测一次', caption:'再取左侧点 R。中心差分连接 R 和 Q，这条弦通常不经过 P。三条线用同一坐标比较。', action:'看清 R–Q 中心弦，再缩小 h 比较三种斜率。', focus:['h'], params:{h:.5}}
  ];

  // Background samples are computed by the same model at its allowed maximum h.
  // Only the drawing window is cached; no second implementation of f or f′ exists here.
  function reference(params) {
    const key = `${params.function}:${params.x}`;
    if (cache.has(key)) return cache.get(key);
    const model = root.LabMath;
    if (!model || typeof model.compute !== 'function') throw Error('导数场景需要已加载的 LabMath。');
    const ref = model.compute('tangent', {...params,h:1});
    const source = ref.plots[0].series.find(s => s.type === 'line');
    if (!source) throw Error('导数模型缺少原函数采样曲线。');
    const xlo=params.x-1.12, xhi=params.x+1.12;
    const points=source.points.filter(p=>p&&p[0]>=xlo-.03&&p[0]<=xhi+.03);
    const values=points.map(p=>p[1]);
    let lo=Math.min(...values),hi=Math.max(...values),span=hi-lo;
    if (!(span>0)) span=1;
    const out={points,xlo,xhi,ylo:lo-span*.16,yhi:hi+span*.15};
    if (cache.size>=24) cache.delete(cache.keys().next().value);
    cache.set(key,out);return out;
  }

  function samplePoints(result,p) {
    for (const plot of result.plots||[]) for (const series of plot.series||[]) {
      const pts=series.points;
      if(series.type==='scatter'&&pts.length===3&&pts.every(q=>q&&q.length===2)&&
        Math.abs(pts[0][0]-(p.x-p.h))<1e-10&&Math.abs(pts[1][0]-p.x)<1e-10&&Math.abs(pts[2][0]-(p.x+p.h))<1e-10) return pts;
    }
    throw Error('导数模型缺少左、中、右三个真实采样点。');
  }
  function ticks(lo,hi,count) {
    const raw=(hi-lo)/count,power=10**Math.floor(Math.log10(raw)),f=raw/power;
    const step=(f<=1?1:f<=2?2:f<=5?5:10)*power,out=[];
    for(let v=Math.ceil(lo/step)*step;v<=hi+step*1e-7;v+=step)out.push(Math.abs(v)<step*1e-8?0:v);
    return out;
  }
  function render({el,params,result,step=0,onParam,format}) {
    const stage=clamp(Number(step)||0,0,3),p=params,d=result.data;
    const fmt=(v,digits=3)=>typeof format==='function'?format(v,digits):Number(v).toLocaleString('zh-CN',{maximumFractionDigits:digits});
    const num=(v,digits=3)=>esc(fmt(v,digits));
    const view=reference(p),[R,P,Q]=samplePoints(result,p);
    const measured=Math.round(el.getBoundingClientRect?.().width||el.clientWidth||760);
    const width=Math.max(280,measured),narrow=width<480,height=narrow?294:312;
    const box={l:narrow?52:60,r:width-19,t:35,b:height-40};
    const sx=x=>box.l+(x-view.xlo)/(view.xhi-view.xlo)*(box.r-box.l);
    const sy=y=>box.b-(y-view.ylo)/(view.yhi-view.ylo)*(box.b-box.t);
    const id=`tangent-story-${++serial}`;
    const line=(a,b,cls,extra='')=>`<line class="${cls}" x1="${sx(a[0])}" y1="${sy(a[1])}" x2="${sx(b[0])}" y2="${sy(b[1])}" ${extra}/>`;
    const text=(x,y,label,cls='',anchor='middle')=>`<text class="${cls}" x="${x}" y="${y}" text-anchor="${anchor}">${esc(label)}</text>`;
    const point=(point,name,cls,dx,dy)=>{
      const x=sx(point[0]),y=sy(point[1]);
      return `<g class="ts-point ${cls}"><circle cx="${x}" cy="${y}" r="5.1"/>${text(clamp(x+dx,box.l+8,box.r-8),clamp(y+dy,box.t+13,box.b-6),name,'ts-point-label')}</g>`;
    };
    let grid='',marks='',axes='';
    ticks(view.ylo,view.yhi,narrow?4:5).forEach(y=>{
      grid+=`<line class="ts-grid" x1="${box.l}" x2="${box.r}" y1="${sy(y)}" y2="${sy(y)}"/>`;
      axes+=text(box.l-9,sy(y)+4,fmt(y,2),'ts-tick','end');
    });
    const xticks=narrow?[p.x-1,p.x,p.x+1]:[p.x-1,p.x-.5,p.x,p.x+.5,p.x+1];
    xticks.forEach(x=>{axes+=`<line class="ts-tick-line" x1="${sx(x)}" x2="${sx(x)}" y1="${box.b}" y2="${box.b+5}"/>`+text(sx(x),box.b+21,fmt(x,2),'ts-tick');});
    axes+=`<path class="ts-axis" d="M${box.l},${box.t}V${box.b}H${box.r}"/>`;
    axes+=text(box.l,17,'函数值 f(x)','ts-axis-label','start')+text(box.r,height-1,'位置 x','ts-axis-label','end');
    if(view.ylo<0&&view.yhi>0)grid+=`<line class="ts-zero" x1="${box.l}" x2="${box.r}" y1="${sy(0)}" y2="${sy(0)}"/>`;
    const path=view.points.map((q,i)=>`${i?'L':'M'}${sx(q[0]).toFixed(3)},${sy(q[1]).toFixed(3)}`).join(' ');
    marks+=`<path class="ts-function" d="${path}"/>`;
    const A=stage===3?R:P,B=Q,C=[Q[0],A[1]];
    const dx=B[0]-A[0],dy=B[1]-A[1],denominator=stage===3?'2h':'h';
    if(stage>=1){
      marks+=`<path class="ts-triangle ${stage===3?'is-central':''}" d="M${sx(A[0])},${sy(A[1])}L${sx(C[0])},${sy(C[1])}L${sx(B[0])},${sy(B[1])}Z"/>`;
      marks+=line(A,C,'ts-dimension')+line(C,B,'ts-dimension');
      if(Math.abs(sx(B[0])-sx(A[0]))>58) {
        const ypos=clamp(sy(A[1])+23,box.t+18,box.b-7);
        marks+=text((sx(A[0])+sx(B[0]))/2,ypos,`Δx = ${denominator}`,'ts-delta');
      }
      if(Math.abs(sy(B[1])-sy(A[1]))>39) {
        const labelx=sx(B[0])+13,nearEdge=labelx>box.r-34;
        marks+=text(clamp(labelx,box.l+30,box.r-4),(sy(A[1])+sy(B[1]))/2+4,'Δy','ts-delta',nearEdge?'end':'start');
      }
      const rightSize=Math.min(8,Math.abs(sx(B[0])-sx(A[0]))/3,Math.abs(sy(B[1])-sy(A[1]))/3);
      if(rightSize>2){const dir=sy(B[1])<sy(A[1])?-1:1;marks+=`<path class="ts-right-angle" d="M${sx(C[0])-rightSize},${sy(C[1])}v${dir*rightSize}h${rightSize}"/>`;}
    }
    if(stage>=2){
      const a=[view.xlo,P[1]+d.exact*(view.xlo-p.x)],b=[view.xhi,P[1]+d.exact*(view.xhi-p.x)];
      marks+=line(a,b,'ts-tangent');
    }
    if(stage>=1){
      const a=[view.xlo,P[1]+d.forward*(view.xlo-p.x)],b=[view.xhi,P[1]+d.forward*(view.xhi-p.x)];
      marks+=line(a,b,`ts-forward ${stage===3?'is-muted':''}`);
      marks+=line(P,Q,'ts-forward-segment');
    }
    if(stage===3){
      // A central-difference secant is anchored at R, never at P.
      const a=[view.xlo,R[1]+d.central*(view.xlo-R[0])],b=[view.xhi,R[1]+d.central*(view.xhi-R[0])];
      marks+=line(a,b,'ts-central-extension');
      marks+=line(R,Q,'ts-central','data-role="central-chord"');
      const middleY=(R[1]+Q[1])/2;
      if(Math.abs(sy(middleY)-sy(P[1]))>23){
        marks+=line(P,[p.x,middleY],'ts-offset');
        marks+=`<circle class="ts-chord-mid" cx="${sx(p.x)}" cy="${sy(middleY)}" r="3.3"/>`;
        marks+=text(sx(p.x)-11,(sy(middleY)+sy(P[1]))/2+4,'不经过 P','ts-gap','end');
      }
      marks+=point(R,'R','is-r',-13,21);
    }
    const close=Math.hypot(sx(Q[0])-sx(P[0]),sy(Q[1])-sy(P[1]))<38;
    marks+=point(P,'P','is-p',-13,close?23:19);
    marks+=point(Q,'Q','is-q',close?14:13,-12);
    const legends=[`<span><i class="ts-key is-curve"></i>原函数</span>`];
    if(stage>=1)legends.push(`<span><i class="ts-key is-forward"></i>P–Q 割线</span>`);
    if(stage>=2)legends.push(`<span><i class="ts-key is-tangent"></i>P 处切线</span>`);
    if(stage===3)legends.push(`<span><i class="ts-key is-central"></i>R–Q 中心弦</span>`);
    let readout='';
    if(stage===0) readout=`<div class="ts-measure"><span><b>P</b> (${num(P[0])}, ${num(P[1])})</span><span class="ts-measure-arrow" aria-hidden="true">→</span><span><b>Q</b> (${num(Q[0])}, ${num(Q[1])})</span><span class="ts-interval">间隔 h = ${num(p.h,4)}</span></div>`;
    else if(stage===1)readout=`<div class="ts-ratio"><span><small>纵向变化 Δy</small><strong>${num(dy,4)}</strong></span><span aria-hidden="true">÷</span><span><small>横向间隔 Δx</small><strong>${num(dx,4)}</strong></span><span aria-hidden="true">=</span><span class="ts-answer"><small>这段的平均变化率</small><strong>${num(d.forward,4)}</strong></span></div>`;
    else if(stage===2)readout=`<div class="ts-comparison"><span class="is-forward"><small>P–Q 割线斜率</small><strong>${num(d.forward,5)}</strong></span><span class="is-tangent"><small>P 处切线斜率</small><strong>${num(d.exact,5)}</strong></span></div><p class="ts-local-note">h = ${num(p.h,4)} · 当前差距 ${num(Math.abs(d.forward-d.exact),5)}</p>`;
    else readout=`<div class="ts-comparison has-three"><span class="is-forward"><small>前向 · P–Q</small><strong>${num(d.forward,5)}</strong></span><span class="is-central"><small>中心 · R–Q</small><strong>${num(d.central,5)}</strong></span><span class="is-tangent"><small>切线 · P</small><strong>${num(d.exact,5)}</strong></span></div><p class="ts-local-note">中心弦：Δx = ${num(dx,4)}，Δy = ${num(dy,4)}${(R[1]+Q[1])/2===P[1]?' · 此处恰好经过 P':''}</p>`;
    const desc=`${steps[stage].title}。${steps[stage].caption}。P坐标${P.join(',')}，Q坐标${Q.join(',')}。${stage===3?'R坐标'+R.join(',')+'，中心差分斜率'+d.central+'。':''}${stage>=1?'前向割线斜率'+d.forward+'。':''}${stage>=2?'切线斜率'+d.exact+'。':''}`;
    el.innerHTML=`<section class="tangent-story" aria-label="导数的几何故事"><div class="ts-scene-head"><span class="ts-function-name">f(x) = ${esc(expressions[p.function]||p.function)}</span><span class="ts-window-note">固定视窗 · 改 h 不缩放</span></div><svg class="ts-chart" xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="${id}-title ${id}-desc"><title id="${id}-title">${esc(steps[stage].title)}</title><desc id="${id}-desc">${esc(desc)}</desc><defs><clipPath id="${id}-clip"><rect x="${box.l}" y="${box.t}" width="${box.r-box.l}" height="${box.b-box.t}"/></clipPath></defs>${grid}<g clip-path="url(#${id}-clip)">${marks}</g>${axes}</svg><div class="ts-legend" aria-label="图中线条">${legends.join('')}</div><div class="ts-readout" aria-live="polite">${readout}</div>${stage>=2?`<div class="ts-scene-actions"><button type="button" data-tangent-half ${p.h<=.001?'disabled':''}>间隔减半 <span>h → ${num(Math.max(.001,p.h/2),4)}</span></button>${stage===3?'<button type="button" class="ts-secondary" data-tangent-wide>放大间隔看弦 · h=0.5</button>':''}${p.h<=.001?'<span class="ts-limit">已到本实验下限</span>':''}</div>`:''}</section>`;
    el.querySelector('[data-tangent-half]')?.addEventListener('click',()=>onParam('h',Math.max(.001,p.h/2)));
    el.querySelector('[data-tangent-wide]')?.addEventListener('click',()=>onParam('h',.5));
  }
  root.LabStories.tangent={steps,render};
})(typeof globalThis!=='undefined'?globalThis:this);
