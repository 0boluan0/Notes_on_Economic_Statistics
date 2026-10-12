/* Six explicit models: recurrence, gated memory, contrastive loss, differential maps,
   Gaussian posterior prediction, and a known synthetic structural causal model. */
(function(root){
'use strict';
const M=root.LabMath||(typeof require==='function'?require('./math.js'):null);if(!M)throw Error('Load LabMath first');
const {sum,linspace,metric,plot,line,scatter,segments,result}=M.helpers;
const dot=(a,b)=>sum(a.map((x,i)=>x*b[i])),sq=x=>x*x,norm=a=>Math.sqrt(dot(a,a));
const bar=(label,points,color=0)=>({type:'bars',label,points,color});
const point=(label,xy,color=3)=>({...scatter(label,[xy],color),shape:'diamond',radius:7});
const slog=x=>Math.sign(x)*Math.log10(1+Math.abs(x));
function inputs(pattern,n){return Array.from({length:n},(_,i)=>pattern==='zero'?0:pattern==='pulse'?(i===0?1:0):pattern==='alternating'?(i%2?-1:1):Math.sin((i+1)*Math.PI/4));}
const schemas={
 rnn:{weight:[.8,-1.5,1.5],inputWeight:[.8,0,2],steps:[16,2,30,true],kind:['tanh',['tanh','linear']],pattern:['pulse',['pulse','sine','alternating','zero']],initial:[0,-1,1],target:[.5,-1,1]},
 lstm:{forget:[.95,0,1],input:[.4,0,1],output:[.8,0,1],recurrent:[0,-2,2],steps:[20,2,40,true],pattern:['zero',['zero','pulse','sine']],initial:[1,-2,2]},
 contrastive:{angle:[30,-180,180],negativeAngle:[60,-180,180],temperature:[.5,.05,2],negatives:[3,[1,2,3]]},
 jacobian:{alpha:[.8,-1.5,1.5],beta:[.4,-1,1],x:[.5,-1,1],y:[.5,-1,1],radius:[.5,.05,1.5]},
 uncertainty:{n:[12,2,30,true],sigma:[.4,.1,1.5],priorSD:[2,.25,4],span:[1,.2,2],query:[2.5,-3,3],seed:[42,1,99999,true]},
 causal:{selection:[.8,-1,1],effect:[.1,-.15,.15],prevalence:[.5,.1,.9]}
};
const models={};
models.rnn=p=>{
 const xs=inputs(p.pattern,p.steps),states=[p.initial],pre=[],local=[];
 for(let t=0;t<p.steps;t++){const z=p.weight*states[t]+p.inputWeight*xs[t],h=p.kind==='tanh'?Math.tanh(z):z;pre.push(z);states.push(h);local.push(p.kind==='tanh'?1-h*h:1);}
 const sensitivities=Array(p.steps+1).fill(0);sensitivities[p.steps]=1;
 for(let t=p.steps-1;t>=0;t--)sensitivities[t]=sensitivities[t+1]*p.weight*local[t];
 const error=states.at(-1)-p.target,loss=error*error/2,weightContributions=xs.map((_,t)=>error*sensitivities[t+1]*local[t]*states[t]),inputGradients=xs.map((_,t)=>error*sensitivities[t+1]*local[t]*p.inputWeight);
 return result([metric('末状态 h_T',states.at(-1),6),metric('末端损失 ½(h_T−y)²',loss,6),metric('共享权重梯度 ∂L/∂a',sum(weightContributions),7),metric('末状态对初态 ∂h_T/∂h₀',sensitivities[0],7)],
 [plot('真实序列输入与递归状态','时间 t（0为初态）','输入 x_t / 状态 h_t',[line('隐藏状态',states.map((h,t)=>[t,h]),0),line('输入序列',xs.map((x,t)=>[t+1,x]),1,true)]),plot('末状态对各时刻状态的敏感度','时间 t（从右向左反传）','log₁₀(max(|∂h_T/∂h_t|,10⁻¹⁸))',[line('精确链式乘积',sensitivities.map((v,t)=>[t,Math.log10(Math.max(Math.abs(v),1e-18))]),2)]),plot('共享参数的梯度须累加所有时间路径','使用权重的时间 t','该步对 ∂L/∂a 的贡献',[bar('带符号贡献',weightContributions.map((v,t)=>[t+1,v,.65]),3)])],
 {xs,states,pre,local,sensitivities,loss,weightContributions,inputGradients,weightGradient:sum(weightContributions)},['标量RNN参数由控件指定，没有训练；损失只在最后一步定义。反传完整展开，不做截断。','共享权重在每个时间步出现，梯度是各条路径贡献的和；不能只取最后一步的偏导。','第二图显示敏感度绝对值及10⁻¹⁸下限；符号和精确零保留在原始数据中。']);
};
models.lstm=p=>{
 const xs=inputs(p.pattern,p.steps),states=[{t:0,c:p.initial,h:p.output*Math.tanh(p.initial),direct:1,dc:1,dh:p.output*(1-sq(Math.tanh(p.initial)))}];
 for(let t=0;t<p.steps;t++){const prev=states.at(-1),g=Math.tanh(xs[t]+p.recurrent*prev.h),c=p.forget*prev.c+p.input*g,h=p.output*Math.tanh(c),dc=p.forget*prev.dc+p.input*(1-g*g)*p.recurrent*prev.dh,dh=p.output*(1-sq(Math.tanh(c)))*dc;states.push({t:t+1,c,h,g,direct:prev.direct*p.forget,dc,dh});}
 const last=states.at(-1);
 return result([metric('末细胞状态 c_T',last.c,6),metric('直通路径贡献 f^T',last.direct,7),metric('完整 dc_T/dc₀',last.dc,7),metric('完整 dh_T/dc₀',last.dh,7)],
 [plot('记忆状态与对外隐藏输出不同','时间 t','c_t / h_t',[line('细胞状态 c',states.map(s=>[s.t,s.c]),0),line('隐藏输出 h',states.map(s=>[s.t,s.h]),1,true)]),plot('固定门下：直通路径与完整导数','时间 t','signed log₁₀(1+|导数|)',[line('只沿细胞捷径 f^t',states.map(s=>[s.t,slog(s.direct)]),0,true),line('完整 dc_t/dc₀',states.map(s=>[s.t,slog(s.dc)]),2),line('完整 dh_t/dc₀',states.map(s=>[s.t,slog(s.dh)]),3,true)]),plot('固定门值：分别保留、写入与输出','门','门值（0关闭，1极限开启）',[bar('门值',[[0,p.forget,.6],[1,p.input,.6],[2,p.output,.6]],1)],{xTickLabels:['遗忘 f','输入 i','输出 o'],yDomain:[0,1]})],
 {xs,states,gates:{f:p.forget,i:p.input,o:p.output},last},[p.recurrent===0?'当前r=0：候选不依赖过去隐藏态，完整dc_t/dc₀与直通路径f^t精确重合。可用“候选反馈增加额外路径”预设分开观察。':'当前r≠0：完整导数包含隐藏态经候选返回细胞的路径；直通路径与总导数可能不同。','仅把三种门值固定为可调常数，候选g_t=tanh(x_t+r h_{t−1})保留可调反馈；实际LSTM通常让门也依赖输入和隐藏态。0与1是Sigmoid门的理想极限。','初态约定h₀=o tanh(c₀)，求导时随c₀一起变化；没有截断这个依赖。','f^t只表示沿细胞自连路径的贡献。r≠0时还有h→候选→c路径；即使本页固定门，完整导数也一般不等于f^t。']);
};
function unit(degrees){const a=degrees*Math.PI/180;return [Math.cos(a),Math.sin(a)];}
models.contrastive=p=>{
 const keys=[unit(0),unit(p.negativeAngle),unit(120),unit(-120)].slice(0,p.negatives+1),labels=['正样本','负样本1','负样本2','负样本3'].slice(0,keys.length);
 function evaluate(angle){const q=unit(angle),scores=keys.map(k=>dot(q,k)/p.temperature),max=Math.max(...scores),e=scores.map(s=>Math.exp(s-max)),total=sum(e),probabilities=e.map(v=>v/total),loss=Math.log(total)+max-scores[0];return {q,scores,probabilities,loss};}
 const d=evaluate(p.angle),gradient=[0,1].map(j=>(sum(keys.map((k,i)=>d.probabilities[i]*k[j]))-keys[0][j])/p.temperature),theta=p.angle*Math.PI/180,angleGradient=dot(gradient,[-Math.sin(theta),Math.cos(theta)])*Math.PI/180;
 const sweep=linspace(-180,180,361).map(a=>[a,evaluate(a).loss]),circle=linspace(0,360,161).map(unit);
 return result([metric('正样本分配概率',d.probabilities[0],6),metric('单Query InfoNCE损失',d.loss,6),metric('二维自由Query梯度范数',norm(gradient),6),metric('角度约束导数 ∂L/∂θ（每度）',angleGradient,7)],
 [plot('单位Query与单位候选向量','第1坐标','第2坐标',[line('单位圆',circle,6,true),...keys.map((k,i)=>({...segments(labels[i],[[0,0,...k]],i),arrow:true})),{...segments('Query',[[0,0,...d.q]],7),arrow:true,width:3},point('',d.q,7)],{equalAspect:true,xDomain:[-1.25,1.25],yDomain:[-1.25,1.25],annotations:keys.map((k,i)=>({x:k[0],y:k[1],text:i===0?'正样本':'负'+i})).concat([{x:d.q[0],y:d.q[1],text:'Query'}])}),plot('同一归一化分母中的竞争概率','候选','exp(相似度/温度) / 总和',[bar('候选概率',d.probabilities.map((v,i)=>[i,v,.6]),0)],{xTickLabels:labels,yDomain:[0,1]}),plot('沿单位圆旋转Query时的真实损失','Query角度 θ（度）','单Query InfoNCE损失',[line('固定候选的角度损失',sweep,2),point('当前Query',[p.angle,d.loss])])],
 {...d,keys,labels,gradient,angleGradient,sweep},['一个Query、一个正样本和1–3个固定负样本，均为二维单位向量；这里展示单项对比损失，没有训练编码器。','自由二维Query梯度与沿单位圆每度旋转的导数不同，后者还乘单位圆切向量及π/180。','固定手工负样本不满足任意互信息估计的采样条件；不把log(N)−本次损失称为已测互信息。']);
};
models.jacobian=p=>{
 const F=([x,y])=>[x+p.alpha*Math.sin(y),y+p.beta*x*x],center=[p.x,p.y],mapped=F(center),J=[[1,p.alpha*Math.cos(p.y)],[2*p.beta*p.x,1]],affine=q=>mapped.map((v,i)=>v+dot(J[i],q.map((x,j)=>x-center[j])));
 const coords=linspace(-p.radius,p.radius,5),curve=linspace(-p.radius,p.radius,41),inputLines=[],actualLines=[],linearLines=[];
 for(const c of coords)for(const vertical of [false,true]){const points=curve.map(v=>vertical?[p.x+c,p.y+v]:[p.x+v,p.y+c]);inputLines.push(points);actualLines.push(points.map(F));linearLines.push(points.map(affine));}
 const radii=linspace(-3,Math.log10(p.radius),61).map(v=>10**v),direction=[Math.SQRT1_2,Math.SQRT1_2],coefficient=.5*Math.sqrt(p.alpha*p.alpha+4*p.beta*p.beta),errors=radii.map(h=>{const q=center.map((v,i)=>v+h*direction[i]),exact=F(q),approx=affine(q);return {h,error:norm(exact.map((v,i)=>v-approx[i])),bound:coefficient*h*h};});
 const maxGridError=Math.max(...actualLines.flatMap((curve,i)=>curve.map((v,j)=>norm(v.map((a,k)=>a-linearLines[i][j][k])))));
 return result([metric('中心处 Jacobian 第一行',J[0].map(x=>x.toFixed(4)).join(', ')),metric('第二行',J[1].map(x=>x.toFixed(4)).join(', ')),metric('中心行列式 det J',J[0][0]*J[1][1]-J[0][1]*J[1][0],6),metric('当前网格最大近似误差',maxGridError,6)],
 [plot('输入局部网格：中心与半宽ρ','输入 x','输入 y',[line('输入网格',inputLines.flatMap(g=>g.concat([null])),0),point('展开中心',center)],{equalAspect:true,annotations:[{x:p.x,y:p.y,text:'中心 a'}]}),plot('真实非线性映射与局部仿射近似','输出 F₁','输出 F₂',[line('局部仿射',linearLines.flatMap(g=>g.concat([null])),1,true),line('真实映射',actualLines.flatMap(g=>g.concat([null])),0),point('F(a)',mapped)],{equalAspect:true,annotations:[{x:mapped[0],y:mapped[1],text:'F(a)'}]}),plot('沿单位对角方向的泰勒余项','log₁₀(位移长度 h)','log₁₀(max(误差,10⁻¹⁶))',[line('真实误差',errors.map(e=>[Math.log10(e.h),Math.log10(Math.max(e.error,1e-16))]),2),line('保守二阶上界',errors.map(e=>[Math.log10(e.h),Math.log10(Math.max(e.bound,1e-16))]),1,true)])],
 {center,mapped,J,determinant:J[0][0]*J[1][1]-J[0][1]*J[1][0],inputLines,actualLines,linearLines,direction,errors,coefficient,maxGridError},['光滑映射固定为F(x,y)=(x+αsin y, y+βx²)，局部近似为F(a)+J(a)(x−a)；没有把J直接乘绝对坐标当作一般展开。','第三图使用单位对角方向，Taylor余项范数≤½√(α²+4β²)h²。原始零误差保留；图上对数下限10⁻¹⁶。','det J只描述该点的一阶局部面积与取向；为零不能独自推出全局不可逆，也不能证明邻域面积完全为零。']);
};
function inverse2(A){const det=A[0][0]*A[1][1]-A[0][1]*A[1][0];return [[A[1][1]/det,-A[0][1]/det],[-A[1][0]/det,A[0][0]/det]];}
models.uncertainty=p=>{
 const random=M.rng(p.seed),noise=M.normalGenerator(M.rng(p.seed+7919)),pool=Array.from({length:30},()=>{const x=p.span*(2*random()-1);return {x,y:.4+1.2*x+p.sigma*noise()};}),sample=pool.slice(0,p.n),precision=[[1/sq(p.priorSD),0],[0,1/sq(p.priorSD)]],rhs=[0,0];
 for(const {x,y}of sample){const phi=[1,x];for(let i=0;i<2;i++){rhs[i]+=phi[i]*y/sq(p.sigma);for(let j=0;j<2;j++)precision[i][j]+=phi[i]*phi[j]/sq(p.sigma);}}
 const covariance=inverse2(precision),mean=covariance.map(row=>dot(row,rhs)),z=M.normalQuantile(.975),predict=x=>{const phi=[1,x],mu=dot(phi,mean),latent=dot(phi,covariance.map(row=>dot(row,phi))),predictive=latent+sq(p.sigma);return {x,mean:mu,latentVariance:latent,noiseVariance:sq(p.sigma),predictiveVariance:predictive,latentInterval:[mu-z*Math.sqrt(latent),mu+z*Math.sqrt(latent)],predictiveInterval:[mu-z*Math.sqrt(predictive),mu+z*Math.sqrt(predictive)]};},grid=linspace(-3,3,201).map(predict),selected=predict(p.query);
 const l00=Math.sqrt(covariance[0][0]),l10=covariance[1][0]/l00,l11=Math.sqrt(covariance[1][1]-l10*l10),radius=Math.sqrt(-2*Math.log(.05)),ellipse=linspace(0,2*Math.PI,161).map(t=>[mean[0]+radius*l00*Math.cos(t),mean[1]+radius*(l10*Math.cos(t)+l11*Math.sin(t))]);
 return result([metric('后验截距 / 斜率均值',mean.map(x=>x.toFixed(4)).join(' / ')),metric('x*处参数不确定性方差',selected.latentVariance,6),metric('x*处不可约噪声方差',sq(p.sigma),6),metric('x*处新观测预测方差',selected.predictiveVariance,6)],
 [plot('同一后验均值，两种逐点95%区间','输入 x','潜在均值 f(x) / 新观测 y',[...['predictiveInterval','latentInterval'].flatMap((key,j)=>[0,1].map(k=>line((j?'均值可信':'新观测预测')+(k?'上界':'下界'),grid.map(g=>[g.x,g[key][k]]),j+1,j===0))),{...line('后验均值',grid.map(g=>[g.x,g.mean]),0),width:3},scatter('已观察数据',sample.map(s=>[s.x,s.y]),7),point('查询位置',[p.query,selected.mean])]),plot('预测方差 = 参数方差 + 观测噪声','输入 x','方差（输出单位²）',[line('参数/函数方差',grid.map(g=>[g.x,g.latentVariance]),2),line('已知噪声方差',grid.map(g=>[g.x,g.noiseVariance]),1,true),line('新观测总方差',grid.map(g=>[g.x,g.predictiveVariance]),0),point('查询总方差',[p.query,selected.predictiveVariance])]),plot('两个参数的联合95%后验可信椭圆','截距 w₀','斜率 w₁',[line('后验联合95%边界',ellipse,0),point('后验均值',mean,2),point('生成参数',[.4,1.2],3)],{equalAspect:true,annotations:[{x:mean[0],y:mean[1],text:'后验均值'},{x:.4,y:1.2,text:'生成参数'}]})],
 {sample,pool,precision,rhs,mean,covariance,grid,selected,z,ellipse,jointRadiusSquared:radius*radius},['已指定线性Gaussian模型、已知σ、零均值各向同性Gaussian参数先验；所有概率区间条件于这些假设。数据由同一模型生成。','种子固定时n取同一30点样本池的前n个；改σ同时改变生成噪声和已知似然噪声，改span改变设计范围。','逐点95%区间不是整条函数同时95%带；联合二维参数椭圆用χ²₂的95%半径²=−2log(0.05)，不是1.96²。','“参数不确定性”可随资料减少；噪声方差被本模型固定为σ²。两者的拆分依赖模型，不自动代表所有现实误差来源。']);
};
function structural(p){
 const pz=[1-p.prevalence,p.prevalence],propensity=[.5+.4*p.selection,.5-.4*p.selection],response=[[.2,.2+p.effect],[.7,.7+p.effect]],joint=[];
 for(let z=0;z<2;z++)for(let t=0;t<2;t++)for(let y=0;y<2;y++){const pt=t?propensity[z]:1-propensity[z],py=y?response[z][t]:1-response[z][t];joint.push({z,t,y,probability:pz[z]*pt*py});}
 const treatmentProb=[0,1].map(t=>sum(joint.filter(r=>r.t===t).map(r=>r.probability))),zGivenT=[0,1].map(t=>pz.map((v,z)=>v*(t?propensity[z]:1-propensity[z])/treatmentProb[t])),observed=[0,1].map(t=>sum(pz.map((_,z)=>zGivenT[t][z]*response[z][t]))),interventional=[0,1].map(t=>sum(pz.map((v,z)=>v*response[z][t]))),adjusted=[0,1].map(t=>sum(pz.map((v,z)=>v*response[z][t])));
 return {pz,propensity,response,joint,treatmentProb,zGivenT,observed,interventional,adjusted,observationalDifference:observed[1]-observed[0],causalEffect:interventional[1]-interventional[0],adjustedDifference:adjusted[1]-adjusted[0]};
}
models.causal=p=>{
 const d=structural(p),groups=['观察T=0','观察T=1','目标总体'],weights=[d.zGivenT[0][1],d.zGivenT[1][1],p.prevalence];
 return result([metric('总体观察均值差',d.observationalDifference,6),metric('按Z标准化后均值差',d.adjustedDifference,6),metric('已知模型do干预效应',d.causalEffect,6),metric('两组最低处理概率',Math.min(...d.propensity,...d.propensity.map(x=>1-x)),3)],
 [plot('组内变化与总体观察变化可能反向','处理 T（0或1）','P(Y=1 | 条件)',[line('Z=0组内',[[0,d.response[0][0]],[1,d.response[0][1]]],0),line('Z=1组内',[[0,d.response[1][0]],[1,d.response[1][1]]],2),{...line('总体观察',d.observed.map((v,t)=>[t,v]),1,true),width:3},scatter('观测端点',d.observed.map((v,t)=>[t,v]),1)],{xDomain:[-.1,1.1],xTickLabels:['T=0','T=1'],yDomain:[0,1]}),plot('观察分组混合比例不同；do保留总体Z分布','比较对象','Z=1的比例',[bar('Z=1混合权重',weights.map((v,i)=>[i,v,.6]),2)],{xTickLabels:groups,yDomain:[0,1]}),plot('同一已知模型的三个效应量','目标量','均值差（Y=1概率之差）',[bar('精确总体量',[[0,d.observationalDifference,.6],[1,d.adjustedDifference,.6],[2,d.causalEffect,.6]],0)],{xTickLabels:['观察关联','Z标准化','do干预'],yDomain:[-.55,.55]})],
 d,['已知合成SCM：独立U_Z,U_T,U_Y∼Uniform(0,1)；Z=1[U_Z<q]，T=1[U_T<e_Z]，Y=1[U_Y<0.2+0.5Z+τT]。没有使用真实人群或干预数据。','e₀=0.5+0.4s、e₁=0.5−0.4s均在[0.1,0.9]，保证本实验的正值性；Z同时影响T与Y。','do(T=t)替换T的结构方程，保持Z与Y机制；Z标准化等于do结果依赖已指定无遗漏混杂的SCM。观察表本身不能证明这些因果假设。','三图都是对8个二元联合格精确求和的总体量，不含有限样本置信区间，也不把相关方向翻转必然归因于因果。']);
};
for(const id of Object.keys(models))M.register(id,schemas[id],models[id]);
root.LabSequentialMath={schemas,models,inputs,unit,inverse2,structural};if(typeof module!=='undefined'&&module.exports)module.exports=root.LabSequentialMath;
})(typeof globalThis!=='undefined'?globalThis:this);
