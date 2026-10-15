/* One numerical scene for the live explanation and the narrated movie. */
(function(root){
  'use strict';
  const base={x:2,w:.25,bias:0,v:2,target:1.5};
  const blue='#2374b9',orange='#ba591c',ink='#152d3b',muted='#647681',green='#21765a';
  const escape=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const n=x=>x===0?'0':(Math.abs(x)<.001?x.toExponential(2):Number(x.toPrecision(4)).toString()).replace(/-/g,'−');
  const clamp=x=>Math.max(0,Math.min(1,x)),ease=x=>{x=clamp(x);return x*x*(3-2*x);};
  const titles=['前向：先把数算出来','干预：一个小改动怎样传到损失','反向：沿原路累计对损失的影响','更新：用梯度调整参数'];
  const names=['权重 w','乘加 z','激活 h','预测 ŷ','损失 L'],symbols=['w','z','h','ŷ','L'];
  function model(p){const d=root.LabMath.compute('backprop',p).data;return {d,values:[p.w,d.forward.z,d.forward.h,d.forward.output,d.forward.loss],gains:[p.x,1-d.forward.h**2,p.v,d.dOutput],grads:[d.gradients.w,d.dZ,d.dH,d.dOutput,1]};}
  const xPositions=[95,345,595,845,1095];
  function svg(state){
    const p=state.params||base,mode=state.mode||0,progress=clamp(state.progress??1),requested=state.delta??.05;
    const delta=Math.max(-3-p.w,Math.min(3-p.w,requested));
    const old=model(p),next=root.BackpropLesson.probe(p,'w',delta),newModel=model(next.after.params);
    const update=mode===3?root.BackpropLesson.update(p,.1):null,shownParams={...p};
    if(update?.allowed)for(const key of ['w','bias','v'])shownParams[key]=p[key]+(update.next[key]-p[key])*ease(progress);
    const travel=progress*4,reverse=mode===2,index=Math.max(0,Math.min(3,Math.ceil(travel-1e-7)-1)),focus=Math.max(0,Math.min(3,state.focus??(reverse?3-index:index)));
    const fractions=Array.from({length:5},(_,i)=>mode===1||mode===0?(i===0?1:ease(travel-i+1)):1);
    const functions=[x=>p.x*x+p.bias,Math.tanh,x=>p.v*x,x=>.5*(x-p.target)**2];
    const vals=mode===3?model(shownParams).values:old.values.map((v,i)=>mode!==1?v:i===0?v+delta:functions[i-1](old.values[i-1]+(newModel.values[i-1]-old.values[i-1])*fractions[i]));
    const fn=functions[focus];
    const a=old.values[focus],b=old.values[focus+1],da=newModel.values[focus]-a,db=newModel.values[focus+1]-b,g=old.gains[focus];
    const funcs=[`× 输入 ${n(p.x)} ＋ 偏置 ${n(shownParams.bias)}`,'经过 tanh',`× 权重 ${n(shownParams.v)}`,`与目标 ${n(p.target)} 比较`];
    let out='';
    const text=(x,y,t,size=20,color=ink,weight=400,anchor='start')=>`<text x="${x}" y="${y}" fill="${color}" font-size="${size}" font-weight="${weight}" text-anchor="${anchor}">${escape(t)}</text>`;
    const box=(x,y,w,h,fill='#fff',stroke='#dce5ea',rx=16)=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx}" fill="${fill}" stroke="${stroke}"/>`;
    const arrow=(x1,y1,x2,y2,color,width=3)=>{const angle=Math.atan2(y2-y1,x2-x1),len=Math.min(8,Math.hypot(x2-x1,y2-y1)*.4),x=x2-len*Math.cos(angle),y=y2-len*Math.sin(angle);return `<path d="M${x1} ${y1}L${x2} ${y2}M${x-4*Math.sin(angle)} ${y+4*Math.cos(angle)}L${x2} ${y2}L${x+4*Math.sin(angle)} ${y-4*Math.cos(angle)}" stroke="${color}" stroke-width="${width}" fill="none" stroke-linejoin="round"/>`;};
    const lines=(x,y,rows,size=20,color=ink,gap=31)=>rows.map((r,i)=>text(x,y+i*gap,r,size,color)).join('');
    out+=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 700" role="img" aria-label="${escape(titles[mode])}：权重经过乘加、激活、输出和损失四个运算。"><defs>${[[blue,'blue'],[orange,'orange']].map(([c,id])=>`<marker id="${id}-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 1L8 5L0 9" fill="none" stroke="${c}" stroke-width="1.5"/></marker>`).join('')}<clipPath id="curve-clip"><rect x="86" y="390" width="598" height="196"/></clipPath></defs><rect width="1200" height="700" rx="20" fill="#f6f9fb"/><g font-family="Arial, Hiragino Sans GB, PingFang SC, sans-serif">`;
    out+=text(36,45,titles[mode],29,ink,700);
    out+=text(36,78,['输入 x 固定，追踪 w 这条依赖：每一步的输出，都是下一步的输入。','灰色留在原处；蓝色显示只改变 w 后的新位置。','前向值留在原处。橙色表示损失对各个量的敏感度。','反向求出梯度后，才执行参数更新，再重新前向计算。'][mode],19,muted);
    for(let i=0;i<4;i++){
      const x1=xPositions[i]+66,x2=xPositions[i+1]-66,y=166;
      out+=`<path d="M${x1} ${y}H${x2}" stroke="#c4d0d8" stroke-width="2"/>`;
      out+=text((x1+x2)/2,126,funcs[i],14,muted,400,'middle');
      const t=mode===0||mode===1?clamp(travel-i):1;
      if(t>0)out+=arrow(x1,y,x1+(x2-x1)*t,y,blue,3);
      if(mode===0||mode===1)if(t>0&&t<1)out+=`<circle cx="${x1+(x2-x1)*t}" cy="${y}" r="7" fill="${blue}"/>`;
      if(reverse){const r=clamp(travel-(3-i));out+=`<path d="M${x2} 266H${x1}" stroke="#eddbcf" stroke-width="3"/>`;if(r>0)out+=arrow(x2,266,x2-(x2-x1)*r,266,orange,5);}
    }
    for(let i=0;i<5;i++){
      const x=xPositions[i],known=mode!==0||fractions[i]>.995,selected=i===focus||i===focus+1;
      out+=`<g data-causal-node="${Math.min(i,3)}" role="button" tabindex="0" aria-label="查看${escape(names[i])}的局部运算">`;
      out+=box(x-65,141,130,88,selected?'#edf5fa':'#fff',selected?blue:'#dce5ea',13);
      out+=text(x,163,names[i],17,muted,400,'middle');out+=text(x,199,known?n(vals[i]):'···',30,known?ink:muted,600,'middle');
      if(mode===1){const change=vals[i]-old.values[i];out+=text(x,253,`Δ ${n(change)}`,18,Math.abs(change)>1e-10?blue:muted,600,'middle');}
      if(reverse&&travel>=4-i){out+=box(x-58,247,116,39,'#fff0e6','#edd1bc',9);out+=text(x,273,n(old.grads[i]),22,orange,700,'middle');}
      out+='</g>';
    }
    out+=text(36,311,mode===3?`更新进度 ${Math.round(progress*100)}% · 输入与目标保持 · w、b、v 改变时重新计算顶部数值`:`固定条件：输入 x=${n(p.x)} · 偏置 b=${n(p.bias)} · 输出权重 v=${n(p.v)} · 目标 t=${n(p.target)}`,16,muted);
    out+=box(30,336,704,296);out+=text(52,365,`${mode===3?'更新前的局部关系':'放大看这一段'}：${symbols[focus]} → ${symbols[focus+1]}`,21,ink,600);
    const left=87,right=680,top=395,bottom=585,span=focus===1?Math.max(1.2,Math.abs(a)*.45):Math.max(.45,Math.abs(a)*.5);
    let xmin=focus===1?Math.min(-3,a-.8):focus===3?Math.min(a,p.target)-.5:a-span;
    let xmax=focus===1?Math.max(5,a+.8):focus===3?Math.max(a,p.target)+.5:a+span;
    const samples=Array.from({length:101},(_,i)=>{const x=xmin+(xmax-xmin)*i/100;return[x,fn(x)];});
    let ymin=Math.min(...samples.map(x=>x[1])),ymax=Math.max(...samples.map(x=>x[1]));
    if(focus===1){ymin=-1.15;ymax=1.15;}else if(ymax-ymin<.001){ymin=b-.5;ymax=b+.5;}else{const pad=(ymax-ymin)*.12;ymin-=pad;ymax+=pad;}
    const sx=x=>left+(x-xmin)/(xmax-xmin)*(right-left),sy=y=>bottom-(y-ymin)/(ymax-ymin)*(bottom-top);
    out+=`<path d="M${left} ${top}V${bottom}H${right}" fill="none" stroke="#b8c8d2"/>`;
    out+=text(left-8,top-8,symbols[focus+1],16,muted,400,'end')+text(right,610,symbols[focus],16,muted,400,'end');
    out+=text(left,top+3,n(ymax),12,muted)+text(left,bottom-6,n(ymin),12,muted);
    out+=text(left,bottom+15,n(xmin),12,muted)+text(right,bottom+15,n(xmax),12,muted,400,'end');
    out+=`<path d="${samples.map(([x,y],i)=>(i?'L':'M')+sx(x).toFixed(2)+' '+sy(y).toFixed(2)).join('')}" fill="none" stroke="${blue}" stroke-width="4"/>`;
    if(focus===3){out+=`<path d="M${sx(p.target)} ${top}V${bottom}" stroke="#9ab5a4" stroke-dasharray="3 5"/>`+text(sx(p.target)+8,top+20,`目标 ${n(p.target)}`,14,green);}
    const tx1=a-span*.42,tx2=a+span*.42;
    if(reverse)out+=`<path d="M${sx(tx1)} ${sy(b+g*(tx1-a))}L${sx(tx2)} ${sy(b+g*(tx2-a))}" stroke="${orange}" stroke-width="3" stroke-dasharray="8 6" clip-path="url(#curve-clip)"/>`;
    out+=`<path d="M${sx(a)} ${bottom}V${sy(b)}H${left}" fill="none" stroke="#a3b3bd" stroke-dasharray="4 5"/><circle cx="${sx(a)}" cy="${sy(b)}" r="7" fill="#fff" stroke="#8194a1" stroke-width="3"/>`;
    if(mode===0){const t=clamp(travel-focus);if(t>0){const first=clamp(t*2),second=clamp(t*2-1);out+=`<path d="M${sx(a)} ${bottom}V${bottom+(sy(b)-bottom)*first}${second?'H'+(sx(a)+(left-sx(a))*second):''}" stroke="${blue}" stroke-width="4" fill="none"/><circle cx="${second?sx(a)+(left-sx(a))*second:sx(a)}" cy="${bottom+(sy(b)-bottom)*first}" r="6" fill="${blue}"/>`;}}
    if(mode===1){const f=fractions[focus+1],nx=a+da*f,ny=fn(nx);out+=`<path d="M${sx(nx)} ${bottom}V${sy(ny)}H${left}" fill="none" stroke="${blue}" stroke-dasharray="4 5"/><circle cx="${sx(nx)}" cy="${sy(ny)}" r="8" fill="${blue}"/>`;if(Math.abs(da*f)>.00001){out+=arrow(sx(a),sy(b),sx(nx),sy(b),blue,2);out+=arrow(sx(nx),sy(b),sx(nx),sy(ny),blue,2);}out+=text(sx(a)-8,sy(b)-16,'原位置',14,muted,400,'end');}
    out+=text(left+12,bottom+35,mode===1?'沿横轴移一点 → 在曲线上读出纵向变化':reverse?'○ 当前前向值    ┄ 当前局部斜率':'横轴：这一步的输入     纵轴：这一步的输出',15,muted);
    out+=box(754,336,416,296,reverse?'#fff8f1':'#fff');
    if(reverse){
      const upstream=old.grads[focus+1],product=old.grads[focus];
      out+=text(778,369,'为什么在这里相乘？',21,ink,700);
      out+=lines(778,405,[`${symbols[focus]} 每变一点，${symbols[focus+1]} 先按斜率变化；`,`${symbols[focus+1]} 的变化，再影响最终损失。`],18,muted,29);
      out+=text(778,480,`${n(upstream)} × ${n(g)} = ${n(product)}`,27,orange,700);
      out+=lines(778,514,['收到的损失梯度 × 这一步的局部斜率',`得到损失对 ${symbols[focus]} 的梯度。`],17,muted,27);
      out+=text(778,595,Math.abs(g)<.01?'这一段很平，前面的改动很难传过去。':g<0?'这一段翻转了影响的方向。':'这一段按局部斜率缩放影响。',17,ink);
    }else if(mode===1){
      out+=text(778,369,'一个改动，经过一个运算',21,ink,700);
      const shown=fractions[focus+1],inputChange=da*shown,outputChange=fn(a+inputChange)-b;
      out+=text(778,414,`${symbols[focus]} 改变 ${n(inputChange)}`,25,blue,600);
      out+=text(778,460,`${symbols[focus+1]} 改变 ${n(outputChange)}`,25,blue,600);
      out+=lines(778,508,[Math.abs(inputChange)<1e-10?'等待改动传到这一段。':`这次变化比例：${n(outputChange/inputChange)}`,`当前位置的局部斜率：${n(g)}`,'缩小改动，变化比例趋近局部斜率。'],17,muted,29);
    }else if(mode===3){
      const u=root.BackpropLesson.update(p,.1);
      out+=text(778,369,'梯度给方向，更新改变参数',21,ink,700);
      out+=lines(778,413,[`∂L/∂w = ${n(old.grads[0])}`,`w 新 = ${n(p.w)} − 0.1 × (${n(old.grads[0])})`,`w 新 = ${n(u.next.w)}`],22,ink,39);
      out+=lines(778,550,u.allowed?[`同时更新 w、b、v 后：`, `损失 ${n(old.values[4])} → ${n(u.after.data.forward.loss)}`]:['这一步超出参数范围。','可在逐项核算区减小学习率。'],20,green,32);
    }else{
      const expressions=[`${n(p.x)} × ${n(p.w)} ＋ ${n(p.bias)} = ${n(b)}`,`tanh(${n(a)}) = ${n(b)}`,`${n(p.v)} × ${n(a)} = ${n(b)}`,`½ × (${n(a)} − ${n(p.target)})²`];
      out+=text(778,369,['乘加把输入与权重组合','曲线决定这一段如何传递','输出权重缩放隐藏值','损失衡量预测偏离目标'][focus],21,ink,700);
      out+=text(778,425,expressions[focus],25,blue,600);
      out+=lines(778,487,[['先使用给定输入 x 和权重 w。','得到 z，交给下一个运算。'],['沿横轴找到 z，再抬到曲线上。','读出纵轴，就得到 h。'],['把隐藏值 h 乘输出权重 v。','得到模型的预测 ŷ。'],['预测越接近目标，损失越小。','这个结果是 L，还不是梯度。']][focus],20,muted,36);
    }
    const banner=mode===1?(progress<1?'沿蓝色路径观察：改动走到哪里，哪里才显示新的值。':`w 改变 ${n(delta)}：损失实际改变 ${n(next.actual)}；梯度的局部预测是 ${n(next.predicted)}。`):reverse?'反向传播正在计算“影响有多大”；这时没有修改任何参数。':mode===3?'一次更新结束后，用新的参数重新做前向与反向计算。':'前向传播保存计算结果，反向传播稍后会复用这些值。';
    out+=text(36,671,banner,19,reverse?orange:ink,500);out+='</g></svg>';return out;
  }
  root.BackpropCausal={base,titles,model,svg,n,clamp};
})(globalThis);
