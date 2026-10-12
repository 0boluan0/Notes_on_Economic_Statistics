/* A visual trace of the existing computations. No second sampling algorithm. */
(() => {
  'use strict';
  const stories = globalThis.LabStories ||= {};
  const states = new WeakMap();
  const ink = '#eef3ff', muted = '#9aaac1', teal = '#5be0c5', gold = '#ffcc7b', blue = '#91acff';
  const esc = v => String(v).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const clamp = (x,a,b) => Math.max(a,Math.min(b,x));
  const text = (x,y,value,color=muted,size=14,anchor='start') => `<text x="${x}" y="${y}" fill="${color}" font-size="${size}" text-anchor="${anchor}">${esc(value)}</text>`;
  const line = (x1,y1,x2,y2,color='#304056',extra='') => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" ${extra}/>`;
  const dot = (x,y,r,color=teal,extra='') => `<circle cx="${x}" cy="${y}" r="${r}" fill="${color}" ${extra}/>`;
  const formula = latex => globalThis.katex.renderToString(latex,{throwOnError:true,trust:false,displayMode:true});
  function span(values) {
    let lo=Math.min(...values),hi=Math.max(...values);
    const pad=Math.max((hi-lo)*.1,Math.abs(lo)*.01,1e-3);
    return [lo-pad,hi+pad];
  }
  function axes(domain,maximum,w,fmt,{left=58,top=48,bottom=270,yLabel='次数',xLabel='样本均值'}={}) {
    const right=w-24,sx=x=>left+(x-domain[0])/(domain[1]-domain[0])*(right-left),sy=y=>bottom-y/maximum*(bottom-top);
    let svg=text(left,22,yLabel,muted,12);
    for(let i=0;i<=4;i++) {
      const y=maximum*i/4,x=domain[0]+(domain[1]-domain[0])*i/4;
      svg+=line(left,sy(y),right,sy(y),i?'#253246':'#56647a')+text(left-10,sy(y)+4,fmt(y,2),muted,11,'end');
      svg+=text(sx(x),bottom+23,fmt(x,2),muted,12,'middle');
    }
    svg+=text((left+right)/2,bottom+49,xLabel,muted,13,'middle');
    return {sx,sy,svg,bottom,top,left,right};
  }
  function counts(values,domain,bins=28) {
    const width=(domain[1]-domain[0])/bins, out=Array(bins).fill(0);
    for(const x of values) if(x>=domain[0]&&x<=domain[1])out[Math.min(bins-1,Math.floor((x-domain[0])/width))]++;
    return {out,width,omitted:values.length-out.reduce((a,b)=>a+b,0)};
  }
  const samplingSteps = [
    {title:'一份样本，一个均值',caption:'每个青色点是一条观测。把这 n 个数相加再除以 n，整份样本就对应一个金色的均值点。',action:'播放点的汇合，再换一份样本。留意：一个金色点代表整份样本。',focus:['n','distribution'],params:{n:6,distribution:'exponential',reps:500}},
    {title:'均值也有分布',caption:'每次都从同一总体独立抽取 n 项，算出一个均值。把 B 个均值放进直方图，就得到抽样分布的模拟。',action:'播放均值汇入直方图，再增加 B。每份样本的大小 n 并没有变。',focus:['reps','seed']},
    {title:'让每份样本更大',caption:'蓝线固定为 n=3，金线使用你选的 n；两条曲线共用固定横轴。对正态和指数总体，均值的标准误为 1/√n。',action:'把 n 从 3 拖到 100，再换成柯西分布。标准柯西均值的分布仍是标准柯西。',focus:['n','distribution'],params:{n:30}},
    {title:'一条路径，不是许多均值',caption:'现在追踪同一条观测序列：第 k 个点用前 k 项求平均。它与前面 B 份独立样本的直方图回答不同问题。',action:'播放累积均值的路径，再换总体或种子。一条路径不能证明大数定律。',focus:['n','distribution','seed']}
  ];
  const bootstrapSteps = [
    {title:'固定手里的样本',caption:'青色点就是手里已经观测到的 n 个数。Bootstrap 之后的每次抽取，都只能从这份样本里选。',action:'先认清原始观测和它们的均值。改变种子会换掉这整份样本。',focus:['n','seed'],params:{n:8,distribution:'exponential',reps:500}},
    {title:'有放回地重新抽取',caption:'下排每个位置都独立选择一个原始观测。编号相同表示同一条观测被再次抽到；有的观测也可能一次都没出现。',action:'播放一次重抽样，沿连线核对编号，再看另一轮。每轮仍然抽 n 次。',focus:['seed']},
    {title:'每轮留下一个均值',caption:'对每份重抽样数据求均值。B 次重抽样产生 B 个均值，描述给定这份原始样本后的变化。',action:'播放直方图逐渐形成，再增加 B。原样本没有增加任何一条新观测。',focus:['reps']},
    {title:'从分布读取区间',caption:'蓝色虚线是重抽样均值的 2.5% 和 97.5% 分位点，金线是原始样本均值。百分位区间的覆盖率并不保证恰好是 95%。',action:'换一份原样本，观察区间怎样移动。增加 B 不能补回原样本缺少的信息。',focus:['n','distribution','seed']}
  ];
  function dispose(el) { const s=states.get(el);if(s?.raf)cancelAnimationFrame(s.raf);states.delete(el); }
  function render(id,{el,params:p,result,step,onParam,format}) {
    const d=result.data, boot=id==='bootstrap', fmt=(x,n=3)=>format(x,n), means=boot?d.bootstrapMeans:d.means;
    const key=JSON.stringify([id,p,step]);let state=states.get(el);
    if(!state||state.key!==key){dispose(el);const example=boot&&step===1?Math.max(0,d.resampleExamples.findIndex(row=>new Set(row).size<p.n)):0;state={key,progress:step===1&&!boot||step>=2?1:0,example,raf:0};states.set(el,state);}
    if(state.raf){cancelAnimationFrame(state.raf);state.raf=0;}
    const width=Math.max(250,Math.round(el.getBoundingClientRect().width)-(matchMedia('(max-width:680px)').matches?28:52)),compact=width<520,w=width;
    const captions=boot?['这一份样本，包含了全部可用信息','同一个观测，可以被抽中多次','一轮重抽样，只贡献一个均值','区间的不确定性，来自这一份样本']:['从一组观测，到一个位置','许多份样本，留下许多个均值','固定横轴，才能看清“变窄”','沿着同一条观测序列向前走'];
    const canPlay=!(step===2&&!boot)&&!(step===3&&boot);
    el.innerHTML=`<div class="stat-film"><div class="sf-heading"><span>${boot?'BOOTSTRAP':'SAMPLING'} <i>/ ${String(step+1).padStart(2,'0')}</i></span><span>n = ${p.n} <i>·</i> B = ${p.reps}</span></div><h2>${captions[step]}</h2><div class="sf-transport">${canPlay?'<button type="button" class="sf-play" aria-label="播放图解">播放图解 ▷</button><label class="sf-time"><span class="sr-only">图解进度</span><input type="range" min="0" max="1000" value="0" aria-label="图解进度"></label>':'<span class="sf-still">拖动右侧参数，比较同一坐标下的变化</span>'}${step===0&&!boot||step===1&&boot?'<button type="button" class="sf-another">另一份样本 →</button>':''}</div><div class="sf-formula"></div><svg class="sf-stage" viewBox="0 0 ${w} 328" role="img" aria-label="${esc(captions[step])}"></svg><div class="sf-key" aria-live="polite"></div><div class="sf-note"></div></div>`;
    const svg=el.querySelector('svg'),eq=el.querySelector('.sf-formula'),keyEl=el.querySelector('.sf-key'),note=el.querySelector('.sf-note'),slider=el.querySelector('input'),play=el.querySelector('.sf-play');
    const smallSample=step===0;
    const distribution=boot||step===1;
    // The sample fixes Bootstrap's plotting window; n never silently changes a comparison's horizontal scale.
    const fixedDomain=boot?span(d.sample):d.densityDomain;
    const fullCounts=counts(means,fixedDomain),maxCount=Math.max(...fullCounts.out,1)*1.15;
    function draw() {
      const t=state.progress,ease=t*t*(3-2*t);let marks='',noteText='';
      if(slider)slider.value=String(Math.round(t*1000));
      if(smallSample) {
        const sample=boot?d.sample:d.sampleExamples[state.example],mean=boot?d.sampleMean:means[state.example];
        const dom=span(boot?sample:d.sampleExamples.flat()),sx=x=>48+(x-dom[0])/(dom[1]-dom[0])*(w-90),base=98;
        marks=text(30,30,`${p.n} 条原始观测`,teal,14)+text(30,205,'一个样本均值',gold,14)+line(48,base,w-42,base)+line(48,252,w-42,252);
        for(let i=0;i<=4;i++){const x=dom[0]+(dom[1]-dom[0])*i/4;marks+=text(sx(x),base+42,fmt(x,2),muted,12,'middle')+text(sx(x),280,fmt(x,2),muted,12,'middle');}
        sample.forEach((value,i)=>{const x=sx(value),y=base-9-(i%5)*9;marks+=dot(x,y,p.n>60?2.5:4,teal,'opacity=".34"');marks+=dot(x+(sx(mean)-x)*ease,y+(242-y)*ease,p.n>60?2.5:4,t<.9?teal:gold);});
        marks+=line(sx(mean),223,sx(mean),252,gold,'stroke-width="2"')+dot(sx(mean),242,7,gold,`opacity="${t}"`)+text(sx(mean),310,`x̄ = ${fmt(mean)}`,gold,16,'middle');
        eq.innerHTML=formula(`\\bar{x}=\\frac{1}{${p.n}}\\sum_{i=1}^{${p.n}}x_i\\approx${fmt(mean)}`);
        keyEl.innerHTML=`<span><i class="sf-dot"></i>青色：单次观测</span><span><i class="sf-dot gold"></i>金色：整份样本的均值</span>`;
        noteText=(boot?'原始样本在后面的重抽样中保持不变。':`第 ${state.example+1} / 6 份示例 · 总体：${p.distribution==='normal'?'标准正态':p.distribution==='exponential'?'指数 Exp(1)':'标准柯西'}。`)+'两排共用数值轴；点的竖向错开只为便于辨认。';
      } else if(boot&&step===1) {
        const indices=d.resampleExamples[state.example],limit=Math.min(p.n,compact?6:12),pos=i=>i<limit?40+i*(w-120)/Math.max(1,limit-1):w-32;
        marks=text(30,22,'原始样本：编号 = 观测身份',teal,13)+text(30,192,'本轮抽取：同号可以出现多次',gold,13);
        for(let j=0;j<limit;j++){
          const from=pos(indices[j]),to=pos(j),y=65+166*ease;
          marks+=`<path d="M${from},72 C${from},136 ${to},155 ${to},224" fill="none" stroke="${teal}" stroke-width="1" opacity=".2"/>`;
          marks+=dot(from+(to-from)*ease,y,14,t>.6?gold:teal)+text(from+(to-from)*ease,y+4,indices[j]+1,'#102030',11,'middle');
          marks+=text(to,266,fmt(d.sample[indices[j]],2),gold,12,'middle');
        }
        for(let j=0;j<limit;j++)marks+=dot(pos(j),65,14,teal)+text(pos(j),69,j+1,'#102030',11,'middle')+text(pos(j),103,fmt(d.sample[j],2),muted,12,'middle');
        if(p.n>limit)marks+=text(w-32,69,`+${p.n-limit}`,teal,12,'middle')+text(w-32,103,`${limit+1}–${p.n}`,muted,10,'middle');
        const unique=new Set(indices).size;
        eq.innerHTML=formula(`\\bar{x}^{*}_{${state.example+1}}\\approx${fmt(means[state.example])}`);
        keyEl.innerHTML=`<span>${p.n} 次抽取</span><span><strong>${unique}</strong> 个不同观测被选中</span><span>${p.n-unique} 个原观测未被选中</span>`;
        noteText=`第 ${state.example+1} / 6 轮真实重抽样。${unique===p.n?'本轮恰好全不同；有放回允许重复，但不强迫重复。':'编号相同的金色点，是同一原始观测的多个副本。'}${p.n>limit?`只画前 ${limit} 个位置，均值使用全部 ${p.n} 项。`:''}`;
      } else if(!boot&&step===2) {
        const baseline=globalThis.LabMath.compute('sampling',{...p,n:3});
        const curves=[baseline.data.densityCurve,d.densityCurve];
        const max=Math.max(...curves.flat().filter(q=>q[0]>=fixedDomain[0]&&q[0]<=fixedDomain[1]).map(q=>q[1]),.5)*1.2;
        const a=axes(fixedDomain,max,w,fmt,{yLabel:'概率密度'});marks=a.svg;
        curves.forEach((curve,i)=>{const pts=curve.filter(q=>q[0]>=fixedDomain[0]&&q[0]<=fixedDomain[1]);marks+=`<path d="${pts.map((q,j)=>`${j?'L':'M'}${a.sx(q[0])},${a.sy(q[1])}`).join(' ')}" fill="none" stroke="${i?gold:blue}" stroke-width="${i?3:2}" ${i?'':'stroke-dasharray="7 5"'}/>`;});
        eq.innerHTML=formula(p.distribution==='cauchy'?'\\bar{X}_n\\sim\\mathrm{Cauchy}(0,1)':'\\mathrm{SE}(\\bar{X}_n)=\\frac{1}{\\sqrt{n}}\\approx'+fmt(d.theoreticalSE,4));
        keyEl.innerHTML=`<span><i class="sf-line blue"></i>蓝色虚线 n = 3</span><span><i class="sf-line gold"></i>金色实线 n = ${p.n}</span>`;
        noteText=p.distribution==='cauchy'?'两条精确密度重合。总体均值与方差不存在；增大 n 不会使样本均值的分布收窄。':'曲线是模型的精确密度；指数总体的均值在有限 n 下仍偏斜。横轴固定，纵轴按峰值调整。';
      } else if(!boot&&step===3) {
        const path=d.runningMeans,ys=span(path.map(q=>q[1]).concat(d.populationMean===null?[]:[d.populationMean]));
        const a=axes([1,200],1,w,fmt,{xLabel:'累计观测数 k',yLabel:'前 k 项的平均值'}),sy=y=>a.bottom-(y-ys[0])/(ys[1]-ys[0])*(a.bottom-a.top);
        // Running means may be negative: draw their true signed axis, rather than reusing a density axis.
        marks=line(a.left,a.bottom,a.right,a.bottom)+text(a.left,22,'前 k 项的平均值',muted,12);
        for(let j=0;j<=4;j++){const y=ys[0]+(ys[1]-ys[0])*j/4;marks+=line(a.left,sy(y),a.right,sy(y),'#253246')+text(a.left-8,sy(y)+4,fmt(y,2),muted,11,'end');}
        for(const k of [1,50,100,150,200])marks+=text(a.sx(k),295,k,muted,12,'middle');
        marks+=text(w/2,321,'累计观测数 k',muted,13,'middle');
        if(d.populationMean!==null)marks+=line(a.left,sy(d.populationMean),a.right,sy(d.populationMean),gold,'stroke-dasharray="6 5"');
        const k=Math.max(1,Math.round(1+(p.n-1)*t)),points=path.slice(0,k);
        marks+=`<path d="${points.map((q,i)=>`${i?'L':'M'}${a.sx(q[0])},${sy(q[1])}`).join(' ')}" fill="none" stroke="${teal}" stroke-width="2.5"/>`+dot(a.sx(k),sy(points.at(-1)[1]),5,teal);
        eq.innerHTML=formula(`\\bar{x}_{${k}}\\approx${fmt(points.at(-1)[1])}`);
        keyEl.innerHTML=`<span><i class="sf-line"></i>当前路径 · 前 ${k} 项</span>${d.populationMean===null?'<span>总体均值不存在</span>':'<span><i class="sf-line gold"></i>总体均值 μ = '+d.populationMean+'</span>'}`;
        noteText='使用另一条固定的独立观测序列，横轴保留 1–200；这不是上一页 B 个均值的排列。';
      } else if(distribution) {
        const k=Math.max(1,Math.round(means.length*t)),partial=counts(means.slice(0,k),fixedDomain),a=axes(fixedDomain,maxCount,w,fmt,{xLabel:boot?'重抽样均值 x̄*':'样本均值 x̄'});marks=a.svg;
        partial.out.forEach((n,i)=>{const x=a.sx(fixedDomain[0]+i*partial.width)+1,bw=(a.right-a.left)/partial.out.length-2;marks+=`<rect x="${x}" y="${a.sy(n)}" width="${bw}" height="${a.bottom-a.sy(n)}" rx="2" fill="${teal}" opacity=".82"/>`;});
        if(boot&&step===3){for(const x of d.interval)marks+=line(a.sx(x),a.top,a.sx(x),a.bottom,blue,'stroke-width="2" stroke-dasharray="6 5"');marks+=line(a.sx(d.sampleMean),a.top,a.sx(d.sampleMean),a.bottom,gold,'stroke-width="2"');}
        else {const last=means[k-1];if(last>=fixedDomain[0]&&last<=fixedDomain[1])marks+=dot(a.sx(last),a.top-12,4,gold);}
        eq.innerHTML=formula(boot&&step===3?`[q_{0.025},q_{0.975}]\\approx[${fmt(d.interval[0])},${fmt(d.interval[1])}]`:`${k}\\;\\text{个均值}\\quad(n=${p.n})`);
        keyEl.innerHTML=boot&&step===3?'<span><i class="sf-line blue"></i>95% 百分位区间端点</span><span><i class="sf-line gold"></i>原样本均值</span>':`<span><i class="sf-dot"></i>柱高：落在这一段的均值个数</span><span>已汇入 ${k} / ${p.reps}</span>`;
        noteText=`横轴在本情境下保持固定。${partial.omitted?`窗口外还有 ${partial.omitted} 个均值，未丢弃数据；完整结果可导出。`:'当前汇入的均值全部在窗口内。'}${boot?'增加 B 不会改变原样本。':'B 是独立样本份数，不是每份样本量。'}`;
      }
      svg.innerHTML=`<title>${esc(captions[step])}</title>${marks}`;note.textContent=noteText;
    }
    function stop(){if(state.raf)cancelAnimationFrame(state.raf);state.raf=0;if(play){play.textContent='播放图解 ▷';play.setAttribute('aria-label','播放图解');}}
    if(play)play.addEventListener('click',()=>{
      if(state.raf){stop();return;}
      if(matchMedia('(prefers-reduced-motion: reduce)').matches){state.progress=1;draw();return;}
      if(state.progress>=1)state.progress=0;
      const start=performance.now()-state.progress*4200;
      play.textContent='暂停 Ⅱ';play.setAttribute('aria-label','暂停图解');
      function tick(now){state.progress=clamp((now-start)/4200,0,1);draw();if(state.progress<1)state.raf=requestAnimationFrame(tick);else stop();}
      state.raf=requestAnimationFrame(tick);
    });
    if(slider)slider.addEventListener('input',()=>{stop();state.progress=Number(slider.value)/1000;draw();});
    el.querySelector('.sf-another')?.addEventListener('click',()=>{stop();state.example=(state.example+1)%6;state.progress=0;draw();});
    if(boot&&step===1)el.querySelector('.sf-another').textContent='另一轮重抽样 →';
    draw();
  }
  stories.sampling={steps:samplingSteps,render:options=>render('sampling',options),dispose};
  stories.bootstrap={steps:bootstrapSteps,render:options=>render('bootstrap',options),dispose};
})();
