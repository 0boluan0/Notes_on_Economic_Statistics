/* Twelve transparent neural-network experiments. No training claims from toy data. */
(function(root){
'use strict';
const M=root.LabMath||(typeof require==='function'?require('./math.js'):null);
if(!M)throw Error('LabMath must load before dl-math.js');
const {sum,linspace,metric,plot,line,scatter,segments,result}=M.helpers;
const avg=x=>sum(x)/x.length, sq=x=>x*x, dot=(a,b)=>sum(a.map((v,i)=>v*b[i]));
const zeros=(n,m)=>Array.from({length:n},()=>Array(m).fill(0));
const sigmoid=x=>x>=0?1/(1+Math.exp(-x)):Math.exp(x)/(1+Math.exp(x));
function act(kind,z){if(kind==='sigmoid')return sigmoid(z);if(kind==='tanh')return Math.tanh(z);if(kind==='relu')return Math.max(0,z);if(kind==='leaky')return z>=0?z:.1*z;return z;}
function deriv(kind,z){if(kind==='sigmoid'){const s=sigmoid(z);return s*(1-s);}if(kind==='tanh')return 1-sq(Math.tanh(z));if(kind==='relu')return z===0?null:z>0?1:0;if(kind==='leaky')return z===0?null:z>0?1:.1;return 1;}
const bar=(label,points,color=0)=>({type:'bars',label,points,color});
function heat(title,matrix,{labelsX,labelsY,zDomain,extraSeries=[],annotations=[],equalAspect=false}={}){
 const n=matrix.length,m=matrix[0].length,points=matrix.flatMap((row,i)=>row.map((z,j)=>[j,i,z]));
 const values=points.map(p=>p[2]),lo=Math.min(...values),hi=Math.max(...values);
 return plot(title,'列 / 特征索引','行 / 样本索引',[{type:'heatmap',label:title,points,cellWidth:1,cellHeight:1,zDomain:zDomain||(lo===hi?[lo-1,hi+1]:[lo,hi]),showValues:true},...extraSeries],{xDomain:[-.5,m-.5],yDomain:[-.5,n-.5],xTickLabels:labelsX||Array.from({length:m},(_,i)=>String(i)),yTickLabels:labelsY||Array.from({length:n},(_,i)=>String(i)),reverseY:true,annotations,equalAspect});
}
function frame(r,c,size){return [[c-.5,r-.5,c+size-.5,r-.5],[c+size-.5,r-.5,c+size-.5,r+size-.5],[c+size-.5,r+size-.5,c-.5,r+size-.5],[c-.5,r+size-.5,c-.5,r-.5]];}
const outline=(label,points)=>({...segments(label,points,3),width:2.5,halo:true});
const seed=[42,1,99999,true];
const schemas={
 activations:{kind:['sigmoid',['sigmoid','tanh','relu','leaky']],gain:[1,.25,4],bias:[0,-3,3],x:[1,-3,3]},
 neuron:{w1:[1,-3,3],w2:[-1,-3,3],bias:[0,-3,3],x1:[1,-2,2],x2:[.5,-2,2]},
 mlp:{width:[3,[3,5,7]],kind:['relu',['relu','tanh','linear']],gain:[1,.25,3],shift:[0,-1,1],offset:[0,-1,1]},
 backprop:{x:[2,-2,2],w:[.25,-3,3],bias:[0,-3,3],v:[2,-3,3],target:[1.5,-2,2]},
 initialization:{depth:[12,1,20,true],mode:['he',['small','xavier','he','large']],kind:['relu',['relu','tanh']],gain:[1,.5,2],seed},
 optimizers:{rate:[.05,.005,.5],condition:[12,1,20,true],steps:[60,10,100,true],momentum:[.9,0,.95],angle:[25,0,75]},
 regularization:{dropout:[.5,0,.8],scale:[1,.25,2],target:[1,-2,3],reps:[500,100,1500,true],seed},
 normalization:{mode:['batch',['batch','layer']],pattern:['mixed',['mixed','constant']],scale:[1,.25,4],shift:[0,-3,3],gamma:[1,0,2],beta:[0,-2,2]},
 convolution:{pattern:['edge',['edge','impulse','checker']],kernel:['edge',['edge','blur','laplace']],stride:[1,[1,2]],padding:[0,[0,1]],row:[1,0,2,true],col:[1,0,2,true]},
 pooling:{mode:['max',['max','mean']],pattern:['peak',['peak','ramp','tied']],size:[2,[2,3]],stride:[2,[1,2,3]],row:[1,0,1,true],col:[1,0,1,true]},
 residual:{weight:[.2,-1,1],depth:[20,1,30,true],x:[.5,-2,2],kind:['tanh',['tanh','linear']]},
 attention:{query:[2,0,3,true],strength:[1,0,4],temperature:[1,.25,3],causal:[0,[0,1]],valueScale:[1,.25,3]}
};
const models={};
models.activations=p=>{
 const xs=linspace(-4,4,321),curve=xs.map(x=>[x,act(p.kind,p.gain*x+p.bias)]),derivative=xs.map(x=>{const z=p.gain*x+p.bias,d=deriv(p.kind,z),previous=p.gain*(x-(xs[1]-xs[0]))+p.bias;return d===null||((p.kind==='relu'||p.kind==='leaky')&&previous<0&&z>0)?null:[x,p.gain*d];});
 const z=p.gain*p.x+p.bias,y=act(p.kind,z),d=deriv(p.kind,z),dy=d===null?null:p.gain*d;
 return result([metric('预激活 z',z),metric('输出 φ(z)',y),metric('输出对输入的导数',dy===null?'拐点处不可导':dy)],
 [plot('激活变换：输出随输入变化','输入 x','φ(gx+b)',[line('当前激活',curve),scatter('观察点',[[p.x,y]],1)]),plot('真实输入导数（拐点断开）','输入 x','dφ(gx+b)/dx',[line('链式法则导数',derivative,2),...(dy===null?[]:[scatter('观察点导数',[[p.x,dy]],1)])])],
 {xs,curve,derivative,z,y,dy},['ReLU 与 Leaky ReLU 在 z=0 的数学导数不存在；图中断开，不把框架约定的次梯度当作唯一导数。','输入缩放 g 同时改变位置与导数幅度；这里展示 dφ(gx+b)/dx，包含链式法则的 g。']);
};
models.neuron=p=>{
 const grid=linspace(-2,2,25),values=grid.map(y=>grid.map(x=>sigmoid(p.w1*x+p.w2*y+p.bias))),z=p.w1*p.x1+p.w2*p.x2+p.bias,probability=sigmoid(z);
 const points=grid.flatMap((y,i)=>grid.map((x,j)=>[x,y,values[i][j]])),boundary=[];
 if(Math.abs(p.w2)>1e-12){for(const x of linspace(-2,2,161)){const y=-(p.w1*x+p.bias)/p.w2;boundary.push(y>=-2&&y<=2?[x,y]:null);}}
 else if(Math.abs(p.w1)>1e-12){const x=-p.bias/p.w1;if(Math.abs(x)<=2)boundary.push([x,-2],[x,2]);}
 const derivatives=[p.w1*probability*(1-probability),p.w2*probability*(1-probability)],step=grid[1]-grid[0];
 return result([metric('仿射分数 z',z),metric('Sigmoid 输出',probability),metric('0.5 阈值分类',probability>=.5?'1（包括相等）':'0')],
 [plot('一个 Sigmoid 神经元的响应面','输入 x₁','输入 x₂',[{type:'heatmap',label:'Sigmoid 输出',points,cellWidth:step,cellHeight:step,zDomain:[0,1],showValues:false},line('p=0.5 边界',boundary,6),{...scatter('当前输入',[[p.x1,p.x2]],3),radius:7,shape:'diamond'}],{xDomain:[-2-step/2,2+step/2],yDomain:[-2-step/2,2+step/2],equalAspect:true,annotations:[{x:p.x1,y:p.x2,text:'当前输入'}]}),plot('当前输入的局部敏感度','输入坐标','∂p/∂xⱼ',[bar('输入导数',derivatives.map((v,i)=>[i,v,.65]),2)],{xTickLabels:['x₁','x₂']})],
 {grid,values,z,probability,derivatives,boundary},['权重和偏置由你指定，没有在数据上训练或校准；输出在 (0,1) 只说明 Sigmoid 的范围。',p.w1===0&&p.w2===0?'两个权重为零：响应处处相同；若偏置也为零，整张平面都是 0.5 等值集，不能画成一条唯一边界。':'在非零权重下，0.5 等值集由 w₁x₁+w₂x₂+b=0 给出；它可能落在当前观察窗外。']);
};
models.mlp=p=>{
 const knots=linspace(-1.5,1.5,p.width).map(t=>t+p.shift),coefficients=knots.map((_,i)=>i===0||i===p.width-1?1:(i%2?-2:2)),xs=linspace(-3,3,301);
 const hidden=knots.map(k=>xs.map(x=>act(p.kind,p.gain*(x-k)))),contributions=hidden.map((h,j)=>h.map(v=>v*coefficients[j])),output=xs.map((_,i)=>p.offset+sum(contributions.map(h=>h[i])));
 const labels=knots.map((_,i)=>'h'+(i+1));
 return result([metric('隐藏单元数',p.width,0),metric('输出偏置',p.offset),metric('激活方式',p.kind),metric('本实验参数来源','手工构造；未训练')],
 [plot('隐藏单元：不同偏置产生不同拐点','输入 x','隐藏激活 hⱼ',[...hidden.map((h,j)=>line(labels[j],xs.map((x,i)=>[x,h[i]]),j,j%2===1))]),plot('带符号输出权重组合隐藏表示','输入 x','输出 y',[...contributions.map((h,j)=>line('c'+(j+1)+'h'+(j+1),xs.map((x,i)=>[x,h[i]]),j,true)),{...line('最终输出（含偏置）',xs.map((x,i)=>[x,output[i]]),7),width:3.3}]),plot('输出层权重','隐藏单元','cⱼ',[bar('手工输出权重',coefficients.map((v,i)=>[i,v,.6]),1)],{xTickLabels:labels})],
 {xs,knots,coefficients,hidden,contributions,output},['所有第一层权重等于 gain，偏置为 −gain×knot；输出权重为两端 1、内部交替 −2 与 2。图形来自明确计算。','这是表达能力与组合机制实验，不进行训练，不用一条曲线证明一般的通用逼近定理。','选择 linear 时两层仿射变换合成仍为仿射；本对称构造进一步抵消为常数 offset。']);
};
models.backprop=p=>{
 const forward=w=>{const z=w*p.x+p.bias,h=Math.tanh(z),output=p.v*h,loss=sq(output-p.target)/2;return {z,h,output,loss};};
 const f=forward(p.w),dOutput=f.output-p.target,dH=dOutput*p.v,dZ=dH*(1-f.h*f.h),gradients={w:dZ*p.x,bias:dZ,v:dOutput*f.h,x:dZ*p.w};
 const eps=linspace(-7,-1,25).map(t=>10**t),numeric=eps.map(h=>(forward(p.w+h).loss-forward(p.w-h).loss)/(2*h)),error=numeric.map(v=>Math.abs(v-gradients.w));
 const nodes=[p.x,f.z,f.h,f.output,f.loss],names=['x','z=wx+b','h=tanh(z)','ŷ=vh','L=½(ŷ−t)²'];
 return result([metric('预测 ŷ',f.output),metric('损失 L',f.loss),metric('解析 ∂L/∂w',gradients.w,7),metric('h=10⁻⁴ 中心差分',(forward(p.w+1e-4).loss-forward(p.w-1e-4).loss)/2e-4,7)],
 [plot('前向计算图：节点写出实际数值','运算顺序','节点位置',[segments('计算依赖',names.slice(1).map((_,i)=>[i,0,i+1,0]),6),scatter('中间值节点',names.map((_,i)=>[i,0]),0)],{yDomain:[-.65,.65],xDomain:[-.45,4.45],xTickLabels:names,diagram:{nodes:[{label:'x',value:p.x,operation:'输入'},{label:'z',value:f.z,operation:'w×x+b'},{label:'h',value:f.h,operation:'tanh(z)'},{label:'ŷ',value:f.output,operation:'v×h'},{label:'L',value:f.loss,operation:'½(ŷ−t)²'}]},annotations:nodes.map((v,i)=>({x:i,y:.25,text:Number(v.toFixed(4)).toString()}))}),plot('反向链条：梯度逐层传播','被求导的量','损失的偏导数',[bar('反向梯度',[[0,dOutput,.6],[1,dH,.6],[2,dZ,.6],[3,gradients.w,.6]],1)],{xTickLabels:['ŷ','h','z','w']}),plot('独立中心差分核对 w 梯度','log₁₀(h)','log₁₀(max(绝对误差,10⁻¹⁶))',[line('数值梯度误差',eps.map((h,i)=>[Math.log10(h),Math.log10(Math.max(1e-16,error[i]))]),2)])],
 {params:p,forward:f,dOutput,dH,dZ,gradients,eps,numeric,error},['采用标量、一个 tanh 隐藏单元和平方损失，全部导数可手算；不是完整自动微分框架。','有限差分是核验手段，极小 h 会受舍入影响；误差图的显示下限为 10⁻¹⁶，原始误差完整导出。']);
};
models.initialization=p=>{
 const width=24,batch=48,normal=M.normalGenerator(M.rng(p.seed)),factor={small:.1,xavier:1,he:Math.SQRT2,large:2}[p.mode]*p.gain,sd=factor/Math.sqrt(width);
 const weights=[],preactivations=[],activations=[Array.from({length:batch},()=>Array.from({length:width},normal))],forwardStats=[];
 function stats(a){const v=a.flat(),mean=avg(v);return {mean,variance:avg(v.map(x=>sq(x-mean))),secondMoment:avg(v.map(sq)),zeroFraction:v.filter(x=>x===0).length/v.length};}
 forwardStats.push(stats(activations[0]));
 for(let l=0;l<p.depth;l++){
  const W=Array.from({length:width},()=>Array.from({length:width},()=>sd*normal()));weights.push(W);
  const z=activations[l].map(row=>W.map(w=>dot(w,row))),a=z.map(row=>row.map(v=>act(p.kind,v)));preactivations.push(z);activations.push(a);forwardStats.push(stats(a));
 }
 const gradients=Array(p.depth+1),gradientRMS=Array(p.depth+1);gradients[p.depth]=Array.from({length:batch},()=>Array(width).fill(1/Math.sqrt(width)));
 for(let l=p.depth;l>=0;l--){gradientRMS[l]=Math.sqrt(avg(gradients[l].flat().map(sq)));if(l){const W=weights[l-1];gradients[l-1]=gradients[l].map((row,b)=>Array.from({length:width},(_,j)=>sum(row.map((g,i)=>g*(deriv(p.kind,preactivations[l-1][b][i])||0)*W[i][j]))));}}
 const series=[line('log₁₀ 二阶矩',forwardStats.map((s,i)=>[i,Math.log10(Math.max(s.secondMoment,1e-18))]),0),line('log₁₀ 方差',forwardStats.map((s,i)=>[i,Math.log10(Math.max(s.variance,1e-18))]),1,true)];
 return result([metric('权重标准差',sd),metric('末层激活二阶矩',forwardStats.at(-1).secondMoment,6),metric('输入梯度 / 输出梯度 RMS',gradientRMS[0]/gradientRMS.at(-1),6),metric('每层宽度 / 样本数','24 / 48')],
 [plot('真实随机前向传播的幅度','层号（0 为输入）','log₁₀(max(统计量,10⁻¹⁸))',series),plot('反向传播到各层的梯度幅度','层号（从右往左反传）','log₁₀(max(梯度 RMS,10⁻¹⁸))',[line('固定末层线性读出的梯度',gradientRMS.map((v,i)=>[i,Math.log10(Math.max(v,1e-18))]),2)]),plot('各层恰为零的激活比例','层号','零激活比例',[line('样本中的零比例',forwardStats.map((s,i)=>[i,s.zeroFraction]),3)],{yDomain:[0,1]})],
 {width,batch,sd,weights,preactivations,activations,forwardStats,gradientRMS,inputGradients:gradients[0],outputGradient:1/Math.sqrt(width)},['使用有限宽度24、48个独立输入和零偏置；He/Xavier的尺度依据是简化独立性与对称性分析，不保证每一份样本逐层守恒。','He模式权重方差为 2/fan-in×gain²；Xavier模式等宽时为 1/fan-in×gain²。ReLU应区分二阶矩 E[h²] 和中心方差。','反向目标为末层各单元之和除以 √24；这里测的是这个固定读出的输入敏感度，不是训练性能。','对数图将低于10⁻¹⁸的量显示在下限，原始统计量完整保留。']);
};
models.optimizers=p=>{
 const theta=p.angle*Math.PI/180,c=Math.cos(theta),s=Math.sin(theta),H=[[c*c+p.condition*s*s,c*s*(1-p.condition)],[c*s*(1-p.condition),s*s+p.condition*c*c]];
 const loss=x=>dot(x,H.map(r=>dot(r,x)))/2,trajectories={},beta2=.999,epsilon=1e-8;
 for(const method of ['gd','momentum','adam']){let x=[3,2],m=[0,0],v=[0,0];const path=[{step:0,x:x.slice(),loss:loss(x)}];
  for(let t=1;t<=p.steps;t++){const g=H.map(r=>dot(r,x));
   for(let j=0;j<2;j++){if(method==='gd')x[j]-=p.rate*g[j];else if(method==='momentum'){m[j]=p.momentum*m[j]+g[j];x[j]-=p.rate*m[j];}else{m[j]=p.momentum*m[j]+(1-p.momentum)*g[j];v[j]=beta2*v[j]+(1-beta2)*g[j]*g[j];x[j]-=p.rate*(m[j]/(1-p.momentum**t))/(Math.sqrt(v[j]/(1-beta2**t))+epsilon);}}
   path.push({step:t,x:x.slice(),loss:loss(x)});
  }trajectories[method]=path;
 }
 const names=['GD（精确梯度）','Momentum','Adam'],ids=Object.keys(trajectories);
 return result(ids.map((id,i)=>metric(names[i]+' 最后损失',trajectories[id].at(-1).loss,6)),
 [plot('参数轨迹：固定观察窗 |坐标|≤4','参数 θ₁','参数 θ₂',ids.map((id,i)=>line(names[i],trajectories[id].map(v=>v.x.every(x=>Math.abs(x)<=4)?v.x:null),i,i===1)).concat([scatter('最小值',[[0,0]],6)]),{xDomain:[-4,4],yDomain:[-4,4],equalAspect:true}),plot('所有迭代的真实损失','更新次数','log₁₀(1+损失)',ids.map((id,i)=>line(names[i],trajectories[id].map(v=>[v.step,Math.log10(1+v.loss)]),i,i===1)))],
 {H,beta2,epsilon,trajectories},['同一学习率只用于揭示更新规则，不是优化器公平性能排行榜；每种算法有不同的适宜学习率。','所有方法使用精确全梯度，没有小批噪声。Momentum采用 v←βv+g；Adam采用一、二阶矩指数平均及偏差修正。','离开 |θⱼ|≤4 的轨迹在第一图断开；第二图和导出数据保留全部步，不能把绘图截断当成停止迭代。']);
};
models.regularization=p=>{
 const contributions=[1,.6,-.4,.8].map(x=>x*p.scale),q=1-p.dropout,random=M.rng(p.seed),masks=[],outputs=[];
 for(let i=0;i<p.reps;i++){const mask=contributions.map(()=>random()<q?1:0);masks.push(mask);outputs.push(dot(contributions,mask)/q);}
 const expected= sum(contributions),variance=p.dropout/q*sum(contributions.map(sq)),baseLoss=sq(expected-p.target)/2,penalty=variance/2,expectedLoss=baseLoss+penalty,empiricalMean=avg(outputs),empiricalVariance=M.variance(outputs,0),empiricalLoss=avg(outputs.map(v=>sq(v-p.target)/2));
 const distribution=[];for(let bits=0;bits<16;bits++){const mask=contributions.map((_,j)=>(bits>>j)&1),kept=sum(mask);distribution.push({mask,output:dot(contributions,mask)/q,probability:q**kept*p.dropout**(4-kept)});}
 const probabilityByKey=new Map(),counts=new Map(),integerWeights=[5,3,-2,4];for(const row of distribution){const key=dot(integerWeights,row.mask);probabilityByKey.set(key,(probabilityByKey.get(key)||0)+row.probability);}for(const mask of masks){const key=dot(integerWeights,mask);counts.set(key,(counts.get(key)||0)+1);}const pmf=[...probabilityByKey].sort((a,b)=>a[0]-b[0]).map(([key,probability])=>({output:key*p.scale/(5*q),probability,count:counts.get(key)||0,empiricalProbability:(counts.get(key)||0)/p.reps})).filter(r=>r.probability>0||r.count>0),barWidth=.7*p.scale/(5*q),maskPlot=heat('前 12 次抽样的保留掩码（1保留/0丢弃）',masks.slice(0,12),{labelsX:['a₁','a₂','a₃','a₄'],zDomain:[0,1]});maskPlot.yLabel='重复序号（前12次）';maskPlot.height=420;
 return result([metric('推理时无随机掩码的输出',expected),metric('Dropout 输出的精确均值',expected),metric('精确 / 模拟方差',variance.toFixed(4)+' / '+empiricalVariance.toFixed(4)),metric('精确 / 模拟期望损失',expectedLoss.toFixed(4)+' / '+empiricalLoss.toFixed(4))],
 [maskPlot,plot('离散 Dropout 输出：精确与模拟概率质量','线性读出值','该输出的概率（离散质量）',[{type:'bars',label:'模拟相对频率',points:pmf.map(r=>[r.output,r.empiricalProbability,barWidth]),color:0},{...scatter('精确概率质量',pmf.map(r=>[r.output,r.probability]),1),shape:'diamond',radius:4.5}],{yDomain:[0,Math.max(.1,...pmf.map(r=>Math.max(r.probability,r.empiricalProbability)))*1.15]}),plot('期望平方损失的精确分解','损失组成','½平方误差',[bar('解析量',[[0,baseLoss,.6],[1,penalty,.6],[2,expectedLoss,.6]],2)],{xTickLabels:['无掩码损失','方差附加项','期望总损失']})],
 {contributions,keepProbability:q,masks,outputs,distribution,pmf,expected,variance,baseLoss,penalty,expectedLoss,empiricalMean,empiricalVariance,empiricalLoss},['使用 inverted dropout：独立 Bernoulli 保留后除以保留率，p<1；推理时使用不加掩码的线性读出。','本页精确二次惩罚等价只成立于固定贡献值的线性读出与平方损失。不能据此把一般深层 Dropout 等同为普通 L2 权重衰减。','模拟方差采用分母 B，以对照条件分布总体方差；有限B的结果会有 Monte Carlo 波动。']);
};
models.normalization=p=>{
 const source=p.pattern==='mixed'?[[1,2,3],[2,4,5],[-1,0,2],[3,5,8]]:[[1,2,3],[1,4,3],[1,6,3],[1,8,3]],X=source.map(row=>row.map((v,j)=>j===2?p.scale*v+p.shift:v)),epsilon=1e-5;
 const groups=p.mode==='batch'?Array.from({length:3},(_,j)=>X.map(row=>row[j])):X,means=groups.map(avg),variances=groups.map(row=>{const m=avg(row);return avg(row.map(x=>sq(x-m)));});
 const normalized=X.map((row,i)=>row.map((x,j)=>{const g=p.mode==='batch'?j:i;return (x-means[g])/Math.sqrt(variances[g]+epsilon);})),Y=normalized.map(row=>row.map(v=>p.gamma*v+p.beta)),postGroups=p.mode==='batch'?Array.from({length:3},(_,j)=>normalized.map(row=>row[j])):normalized;
 const bound=Math.max(1,...normalized.flat().map(Math.abs),...Y.flat().map(Math.abs)),labelsX=['特征0','特征1','特征2'],labelsY=['样本0','样本1','样本2','样本3'];
 return result([metric('统计轴',p.mode==='batch'?'每列跨样本':'每行跨特征'),metric('中心化后的最大组均值绝对值',Math.max(...postGroups.map(v=>Math.abs(avg(v)))),8),metric('零方差统计组数',variances.filter(v=>v===0).length,0),metric('稳定项 ε',epsilon,5)],
 [heat('输入 X（只调整第3特征）',X,{labelsX,labelsY}),heat('标准化值 z（尚未施加 γ、β）',normalized,{labelsX,labelsY,zDomain:[-bound,bound]}),heat('输出 y=γz+β（与 z 共用色标）',Y,{labelsX,labelsY,zDomain:[-bound,bound]})],
 {X,means,variances,epsilon,normalized,Y},['BN展示训练态当前批次统计，方差用分母N；LN每个样本单独沿特征统计。BN推理态还需要训练得到的运行统计，本页没有伪造它。','为隔离统计轴，本页 γ、β 在各特征上取同一个可调数；实际模型通常为逐特征参数。','ε使非恒定组的归一化方差为 var/(var+ε)，并非严格等于1；零方差组标准化为0，仿射后为β。']);
};
const kernels={edge:[[-1,0,1],[-1,0,1],[-1,0,1]],blur:Array.from({length:3},()=>Array(3).fill(1/9)),laplace:[[0,1,0],[1,-4,1],[0,1,0]]};
function image7(pattern){return Array.from({length:7},(_,i)=>Array.from({length:7},(_,j)=>pattern==='edge'?(j>=3?1:0):pattern==='impulse'?(i===3&&j===3?1:0):(i+j)%2));}
models.convolution=p=>{
 const input=image7(p.pattern),kernel=kernels[p.kernel],size=7+2*p.padding,padded=zeros(size,size);input.forEach((row,i)=>row.forEach((v,j)=>padded[i+p.padding][j+p.padding]=v));
 const outSize=Math.floor((size-3)/p.stride)+1,output=zeros(outSize,outSize);
 for(let i=0;i<outSize;i++)for(let j=0;j<outSize;j++)for(let r=0;r<3;r++)for(let c=0;c<3;c++)output[i][j]+=padded[i*p.stride+r][j*p.stride+c]*kernel[r][c];
 const row=p.row,col=p.col,patch=Array.from({length:3},(_,r)=>Array.from({length:3},(_,c)=>padded[row*p.stride+r][col*p.stride+c])),contributions=patch.map((a,r)=>a.map((v,c)=>v*kernel[r][c])),bound=Math.max(1,...kernel.flat().map(Math.abs),...output.flat().map(Math.abs),...contributions.flat().map(Math.abs));
 return result([metric('输出形状',outSize+' × '+outSize),metric('选定输出 y['+row+','+col+']',output[row][col]),metric('逐项乘积之和',sum(contributions.flat())),metric('运算约定','互相关：不翻转核')],
 [heat('补零后的输入；框为选定感受野',padded,{equalAspect:true,zDomain:[0,1],extraSeries:[outline('3×3输入窗口',frame(row*p.stride,col*p.stride,3))]}),heat('3×3核（不翻转）',kernel,{equalAspect:true,zDomain:[-bound,bound]}),heat('输出响应；框为当前读数',output,{equalAspect:true,zDomain:[-bound,bound],extraSeries:[outline('当前输出',frame(row,col,1))]}),heat('选定窗口的九项乘积',contributions,{equalAspect:true,zDomain:[-bound,bound]})],
 {input,padded,kernel,patch,contributions,output,outSize,row,col},['采用深度学习库常见的二维 cross-correlation，不反转kernel；严格数学卷积需要反转两个空间轴。','单输入通道、单输出通道、3×3固定核；stride为1或2，padding为0或1层零。图不是训练出的边缘检测器。','形状公式 floor((7+2padding−3)/stride)+1；补零区域和真实输入区别由零值与源矩阵共同说明。']);
};
models.pooling=p=>{
 const input=Array.from({length:6},(_,i)=>Array.from({length:6},(_,j)=>p.pattern==='ramp'?i*6+j:p.pattern==='tied'?1:((i===2&&j===3)?5:((i+j)%3)*.5))),n=Math.floor((6-p.size)/p.stride)+1,output=zeros(n,n);
 for(let i=0;i<n;i++)for(let j=0;j<n;j++){const vals=[];for(let r=0;r<p.size;r++)for(let c=0;c<p.size;c++)vals.push(input[i*p.stride+r][j*p.stride+c]);output[i][j]=p.mode==='max'?Math.max(...vals):avg(vals);}
 const r0=p.row*p.stride,c0=p.col*p.stride,patch=Array.from({length:p.size},(_,r)=>Array.from({length:p.size},(_,c)=>input[r0+r][c0+c])),flat=patch.flat(),maximum=Math.max(...flat),winners=flat.map((v,i)=>v===maximum?i:-1).filter(i=>i>=0),gradient=zeros(6,6);
 if(p.mode==='mean')for(let r=0;r<p.size;r++)for(let c=0;c<p.size;c++)gradient[r0+r][c0+c]=1/(p.size*p.size);
 else gradient[r0+Math.floor(winners[0]/p.size)][c0+winners[0]%p.size]=1;
 const max=Math.max(...input.flat()),windowOutline=outline('当前池化窗口',frame(r0,c0,p.size));
 return result([metric('输出形状',n+' × '+n),metric('选定池化结果',output[p.row][p.col]),metric('当前窗口最大值个数',winners.length,0),metric('输入梯度的总和',sum(gradient.flat()))],
 [heat('输入；框为当前窗口',input,{equalAspect:true,zDomain:[0,max||1],extraSeries:[windowOutline]}),heat('池化输出（与输入共用色标）',output,{equalAspect:true,zDomain:[0,max||1],extraSeries:[outline('当前输出',frame(p.row,p.col,1))]}),heat('选定输出对各输入的导数 / 次梯度',gradient,{equalAspect:true,zDomain:[0,1]})],
 {input,patch,output,gradient,winners,row:p.row,col:p.col,size:p.size,stride:p.stride},['仅演示空间池化，无可学习权重；下采样会损失位置与非最大值信息，不能从输出唯一恢复输入。','Max出现并列最大值时普通导数不存在、次梯度不唯一，本实现按行优先选择第一个最大位置作为一个次梯度；Mean为每个窗口位置分配1/k²。','局部池化可降低某些小位移的敏感度，但不保证任意平移不变；跨窗口边界或改变步幅可改变输出。']);
};
models.residual=p=>{
 const paths={};for(const mode of ['plain','residual']){let h=p.x,jacobian=1;const path=[{layer:0,value:h,jacobian}];for(let l=1;l<=p.depth;l++){const d=p.kind==='linear'?1:1-sq(Math.tanh(h)),f=act(p.kind,h),local=(mode==='residual'?1:0)+p.weight*d;h=(mode==='residual'?h:0)+p.weight*f;jacobian*=local;path.push({layer:l,value:h,jacobian,local});}paths[mode]=path;}
 const names=['普通链','恒等残差链'];
 return result([metric('普通链末层输出',paths.plain.at(-1).value,6),metric('残差链末层输出',paths.residual.at(-1).value,6),metric('普通链 dy/dx',paths.plain.at(-1).jacobian,7),metric('残差链 dy/dx',paths.residual.at(-1).jacobian,7)],
 [plot('同一分支函数的逐层输出','层数','层输出（signed log₁₀(1+|h|)）',Object.keys(paths).map((key,i)=>line(names[i],paths[key].map(v=>[v.layer,Math.sign(v.value)*Math.log10(1+Math.abs(v.value))]),i,i===0))),plot('对初始输入的链式敏感度','层数','log₁₀(max(|dy/dx|,10⁻¹⁸))',Object.keys(paths).map((key,i)=>line(names[i],paths[key].map(v=>[v.layer,Math.log10(Math.max(1e-18,Math.abs(v.jacobian)))]),i,i===0)))],
 {paths},['普通链为 h←aφ(h)，残差链为 h←h+aφ(h)。分支参数在各层相同、标量且没有归一化/训练，方便核对雅可比乘积。','恒等捷径将局部导数由 aφ′ 改为1+aφ′；仍可发生放大、抵消或梯度消失，不承诺所有深层网络训练成功。','输出用保符号对数显示，梯度绝对值低于10⁻¹⁸压到显示下限；负梯度和精确零仍在导出数据中保留。']);
};
models.attention=p=>{
 const Q=[[1,0],[0,1],[1,1],[-1,1]],K=Q.map(r=>r.slice()),V=[[1,0],[0,1],[1,1],[-1,2]].map(r=>r.map(v=>v*p.valueScale)),n=4,labels=['A','B','C','D'];
 const scores=Q.map(q=>K.map(k=>p.strength*dot(q,k)/(Math.sqrt(2)*p.temperature))),mask=Q.map((_,i)=>K.map((_,j)=>!p.causal||j<=i)),weights=scores.map((row,i)=>{const max=Math.max(...row.filter((_,j)=>mask[i][j])),e=row.map((v,j)=>mask[i][j]?Math.exp(v-max):0),total=sum(e);return e.map(v=>v/total);}),outputs=weights.map(row=>[sum(row.map((w,j)=>w*V[j][0])),sum(row.map((w,j)=>w*V[j][1]))]),selected=weights[p.query],output=outputs[p.query],entropy=-sum(selected.filter(v=>v>0).map(v=>v*Math.log(v))),annotations=[];
 mask.forEach((row,i)=>row.forEach((v,j)=>{if(!v)annotations.push({x:j,y:i,text:'×'});}));
 const bound=Math.max(1,...scores.flat().map(Math.abs)),vectorSeries=V.map((v,i)=>segments('V_'+labels[i],[[0,0,...v]],i,true));vectorSeries.push(segments('所选query输出',[[0,0,...output]],7),{...scatter('加权输出',[output],7),shape:'diamond',radius:7});
 const scorePlot=heat('缩放分数；×位置在归一化前屏蔽',scores,{labelsX:labels,labelsY:labels,zDomain:[-bound,bound],annotations}),weightPlot=heat('按query行归一化的实际注意力权重',weights,{labelsX:labels,labelsY:labels,zDomain:[0,1],extraSeries:[outline('所选query行',[[-.5,p.query-.5,3.5,p.query-.5],[3.5,p.query-.5,3.5,p.query+.5],[3.5,p.query+.5,-.5,p.query+.5],[-.5,p.query+.5,-.5,p.query-.5]])]});scorePlot.xLabel=weightPlot.xLabel='Key / Value 位置';scorePlot.yLabel=weightPlot.yLabel='Query 位置';
 return result([metric('选定 Query',labels[p.query]),metric('该行权重和',sum(selected)),metric('该行熵（自然对数）',entropy),metric('二维读取输出',output.map(v=>v.toFixed(4)).join(', '))],
 [scorePlot,weightPlot,plot('Value 向量与所选 Query 的加权输出','Value 第1坐标','Value 第2坐标',vectorSeries,{equalAspect:true,annotations:V.map((v,i)=>({x:v[0],y:v[1],text:labels[i]})).concat([{x:output[0],y:output[1],text:'加权输出'}])})],
 {Q,K,V,scores,mask,weights,outputs,selectedQuery:p.query,entropy},['四个token是手工指定的二维向量，未使用语言模型或学习出的参数；注意力热图不代表自然语言语义。','temperature与query强度改变分数差异，valueScale只改变读取内容、不改变权重；原始缩放点积注意力相当于temperature=1。','因果掩码保留对角线，每行至少一个可读key；先限制允许集合再稳定softmax。×显示被屏蔽的原始有限分数，不表示该位置仍有权重。','输出是Value的凸组合；注意力系数本身不是token对最终预测的因果贡献。']);
};
for(const id of Object.keys(models))M.register(id,schemas[id],models[id]);
root.LabDLMath={schemas,models,act,deriv,sigmoid,heat};
if(typeof module!=='undefined'&&module.exports)module.exports=root.LabDLMath;
})(typeof globalThis!=='undefined'?globalThis:this);
