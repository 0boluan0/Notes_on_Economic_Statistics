/* Transparent finite models for representations, generation, decisions and fairness. */
(function(root){
 'use strict';
 const M=root.LabMath||(typeof require==='function'?require('./math.js'):null);
 if(!M)throw new Error('Load math.js before advanced-math.js');
 const {sum,linspace,metric,plot,line,scatter,segments,result}=M.helpers;
 const dot=(a,b)=>sum(a.map((v,i)=>v*b[i]));
 const norm=a=>Math.sqrt(dot(a,a));
 const sigmoid=x=>x>=0?1/(1+Math.exp(-x)):Math.exp(x)/(1+Math.exp(x));
 const softplus=x=>Math.max(x,0)+Math.log1p(Math.exp(-Math.abs(x)));
 const density=(x,mu,sd)=>M.normalPDF((x-mu)/sd)/sd;
 const heat=(label,points,zDomain,showValues=false)=>({type:'heatmap',label,points,zDomain,showValues,cellWidth:1,cellHeight:1});
 const bars=(label,points,color=0)=>({type:'bars',label,points,color});
 const arrow=(label,points,color=0)=>({...segments(label,points,color),arrow:true});
 const seed=[42,1,99999,true];
 const models={};
 const schemas={
  positional:{position:[8,0,31,true],shift:[3,0,32,true],pairs:[4,1,8,true],base:[10000,[10,100,10000]]},
  gnn:{graph:['chain',['chain','star']],layers:[3,0,20,true],selfLoop:[1,[0,1]],normalization:['mean',['mean','symmetric']],residual:[0,0,.9]},
  embeddings:{angle:[45,0,180],normA:[1,0,3],normB:[2,0,3]},
  autoencoder:{angle:[35,0,175],gain:[1,0,2],ratio:[3,1,6],seed},
  vae:{mu:[1,-3,3],logvar:[0,-4,2],decoder:[1,-2,2],x:[2,-3,3],beta:[1,0,4],epsilon:[.5,-2,2]},
  gan:{theta:[-2,-4,4],weight:[1,-4,4],bias:[0,-4,4],rate:[.2,.01,1],steps:[20,1,50,true]},
  flow:{scale:[1,.25,3],shift:[0,-3,3],warp:[.8,-.8,2],z:[1,-3,3]},
  diffusion:{alpha:[.5,.01,.99],mu:[2,-3,3],sigma:[1,.25,2],observed:[1,-5,5]},
  'rl-bellman':{gamma:[.9,0,.99],success:[.5,.1,1],cost:[-.2,-1,1],iterations:[12,1,50,true]},
  'rl-policy':{theta:[0,-5,5],gamma:[.9,0,.99],success:[.5,.1,1],cost:[-.2,-1,1],rate:[.5,.05,2],steps:[30,1,60,true]},
  'double-descent':{width:[20,2,80,true],n:[24,12,40,true],noise:[.3,0,1],lambda:[0,0,1],seed},
  fairness:{baseA:[.2,.05,.95],baseB:[.6,.05,.95],tpr:[.8,0,1],fpr:[.2,0,1]}
 };
 function positionalVector(position,pairs,base){return Array.from({length:2*pairs},(_,j)=>{const angle=position/Math.pow(base,2*Math.floor(j/2)/(2*pairs));return j%2?Math.cos(angle):Math.sin(angle);});}
 models.positional=p=>{
  const vectors=Array.from({length:32},(_,i)=>positionalVector(i,p.pairs,p.base));
  const selected=positionalVector(p.position,p.pairs,p.base),shifted=positionalVector(p.position+p.shift,p.pairs,p.base);
  const similarity=dot(selected,shifted)/p.pairs;
  const cells=vectors.flatMap((v,i)=>v.map((x,j)=>[j,i,x]));
  const offsets=Array.from({length:65},(_,delta)=>[delta,dot(positionalVector(0,p.pairs,p.base),positionalVector(delta,p.pairs,p.base))/p.pairs]);
  return result([metric('编码维数',2*p.pairs,0),metric('任意位置的平方范数',p.pairs,0),metric('相距 Δ 的余弦相似度',similarity),metric('最高频相位差 Δ',p.shift,0,' 弧度')],
   [plot('正弦与余弦的位置编码','编码维度','绝对位置',[heat('编码值',cells,[-1,1])],{xDomain:[-.5,2*p.pairs-.5],yDomain:[-.5,31.5],reverseY:true,xTickLabels:Array.from({length:2*p.pairs},(_,j)=>(j%2?'cos':'sin')+Math.floor(j/2))}),
    plot('相对距离决定内积，但相似度不必单调','位置差 Δ','余弦相似度',[line('任意起点的相对相似度',offsets),scatter('当前 Δ',[[p.shift,similarity]],1)],{yDomain:[-1.05,1.05]}),
    plot('第一频率对是圆上的相位旋转','sin(position)','cos(position)',[line('单位圆',linspace(0,2*Math.PI,101).map(t=>[Math.sin(t),Math.cos(t)]),6,true),arrow('当前位置',[[0,0,selected[0],selected[1]]],0),arrow('平移后位置',[[0,0,shifted[0],shifted[1]]],1)],{equalAspect:true})],
   {vectors,selected,shifted,similarity,offsets,frequencies:Array.from({length:p.pairs},(_,i)=>Math.pow(p.base,-i/p.pairs))},['每个 sin/cos 对的平方和为 1；内积等于各频率的 cos(Δω) 之和，所以不依赖起点。','这是原始 Transformer 的固定正弦位置编码；不是 RoPE，也没有训练注意力权重。相似度出现周期变化，不是距离度量。']);
 };
 models.gnn=p=>{
  const edges=p.graph==='chain'?[[0,1],[1,2],[2,3],[3,4],[4,5]]:[[0,1],[0,2],[0,3],[0,4],[0,5]];
  const A=Array.from({length:6},()=>Array(6).fill(0));for(const [a,b]of edges){A[a][b]=A[b][a]=1;}if(p.selfLoop)for(let i=0;i<6;i++)A[i][i]=1;
  const degree=A.map(sum),P=A.map((row,i)=>row.map((x,j)=>p.normalization==='mean'?x/degree[i]:x/Math.sqrt(degree[i]*degree[j])));
  const H=[Array.from({length:6},(_,i)=>i===0?1:0)];
  for(let k=0;k<p.layers;k++)H.push(P.map((row,i)=>(1-p.residual)*dot(row,H[k])+p.residual*H[k][i]));
  const final=H[H.length-1],positions=Array.from({length:6},(_,i)=>[Math.cos(2*Math.PI*i/6),Math.sin(2*Math.PI*i/6)]);
  return result([metric('传播层数',p.layers,0),metric('节点特征均值',M.mean(final)),metric('节点特征方差（分母 6）',M.variance(final,0),6),metric('特征平方范数',dot(final,final),6)],
   [plot('固定图与当前节点特征','布局 x（仅展示）','布局 y（仅展示）',[segments('无向边',edges.map(([i,j])=>[...positions[i],...positions[j]]),6),{...scatter('节点',positions),shape:'diamond',radius:6}],{equalAspect:true,xDomain:[-1.35,1.35],yDomain:[-1.35,1.35],annotations:positions.map((v,i)=>({x:v[0],y:v[1],text:i+': '+final[i].toFixed(3)}))}),
    plot('同一初始脉冲经过逐层聚合','节点编号','传播层',[heat('节点特征',H.flatMap((h,k)=>h.map((v,j)=>[j,k,v])),[0,Math.max(1,...H.flat())],p.layers<=8)],{xDomain:[-.5,5.5],yDomain:[-.5,p.layers+.5],xTickLabels:['0','1','2','3','4','5'],reverseY:true})],
   {A,degree,propagation:P,history:H,final,edges},['初始特征只有节点 0 为 1；每层进行 H←rH+(1−r)PH，权重固定为单位映射且没有激活函数或训练。','mean 使用 D⁻¹Ã，symmetric 使用 D⁻¹ᐟ²ÃD⁻¹ᐟ²。非正则图上两者不同；前者的极限通常是度数加权均值。','无自环、无残差的二部图可持续奇偶振荡；不能把“更多层总会得到同一个常数”当作无条件结论。']);
 };
 models.embeddings=p=>{
  const a=[p.normA,0],rad=p.angle*Math.PI/180,b=[p.normB*Math.cos(rad),p.normB*Math.sin(rad)],inner=dot(a,b),cosine=p.normA*p.normB===0?null:inner/(p.normA*p.normB),distance=norm(a.map((v,i)=>v-b[i]));
  return result([metric('内积 aᵀb',inner),metric('余弦相似度',cosine===null?'零向量：未定义':cosine),metric('欧氏距离',distance),metric('两个向量的夹角',cosine===null?'零向量：未定义':p.angle,0,cosine===null?'':'°')],
   [plot('两个合成嵌入向量','第一坐标','第二坐标',[arrow('向量 a',[[0,0,...a]],0),arrow('向量 b',[[0,0,...b]],1),segments('两端点之间的距离',[[...a,...b]],2,true)],{equalAspect:true}),
    plot('固定长度，改变夹角','夹角（度）','相似度或内积',[line('内积',linspace(0,180,181).map(t=>[t,p.normA*p.normB*Math.cos(t*Math.PI/180)]),0),...(cosine===null?[]:[line('余弦相似度',linspace(0,180,181).map(t=>[t,Math.cos(t*Math.PI/180)]),1,true),scatter('当前余弦',[[p.angle,cosine]],2)])])],
   {a,b,inner,cosine,distance},['向量是可控的二维合成示例，未声称来自任何训练好的词向量模型。几何相近是否对应语义相近取决于训练目标和数据。','正的长度缩放改变内积与欧氏距离，但不改变两个非零向量的余弦相似度；零向量的余弦未定义。']);
 };
 models.autoencoder=p=>{
  const normal=M.normalGenerator(M.rng(p.seed)),theta=35*Math.PI/180,raw=Array.from({length:80},()=>{const a=p.ratio*normal(),b=normal();return [a*Math.cos(theta)-b*Math.sin(theta),a*Math.sin(theta)+b*Math.cos(theta)];});
  const centers=[M.mean(raw.map(v=>v[0])),M.mean(raw.map(v=>v[1]))],X=raw.map(v=>v.map((a,j)=>a-centers[j]));
  const cov=[[M.variance(X.map(v=>v[0])),M.covariance(X.map(v=>v[0]),X.map(v=>v[1]))],[M.covariance(X.map(v=>v[0]),X.map(v=>v[1])),M.variance(X.map(v=>v[1]))]],eig=M.eigen2(cov);
  const u=[Math.cos(p.angle*Math.PI/180),Math.sin(p.angle*Math.PI/180)],codes=X.map(v=>dot(v,u)),reconstruction=codes.map(z=>u.map(v=>p.gain*z*v));
  const mse=M.mean(X.map((v,i)=>sum(v.map((x,j)=>(x-reconstruction[i][j])**2)))),optimalMSE=eig.values[1]*79/80;
  const objective=angle=>{const v=[Math.cos(angle*Math.PI/180),Math.sin(angle*Math.PI/180)];return M.mean(X.map(x=>{const z=dot(x,v);return sum(x.map((a,j)=>(a-p.gain*z*v[j])**2));}));};
  return result([metric('每点重构平方误差',mse),metric('最优秩一重构误差',optimalMSE),metric('编码维数',1,0),metric('解码增益',p.gain)],
   [plot('线性编码、瓶颈与重构','中心化第一坐标','中心化第二坐标',[scatter('输入',X,6),scatter('重构',reconstruction,0),segments('重构残差',X.map((v,i)=>[...v,...reconstruction[i]]),6,true)],{equalAspect:true}),plot('旋转编码方向，观察完整重构损失','编码方向（度）','每点重构平方误差',[line('当前增益下的损失',linspace(0,180,181).map(a=>[a,objective(a)])),line('PCA 最优误差（增益 1）',[[0,optimalMSE],[180,optimalMSE]],1,true),scatter('当前方向',[[p.angle,mse]],2)])],
   {X,centers,u,codes,reconstruction,mse,optimalMSE,eigenvalues:eig.values,eigenvectors:eig.vectors},['编码 z=uᵀx，解码 x̂=guz，u 是单位向量；这是参数可直接控制的线性瓶颈，不是在后台训练网络。','g=1 时为正交投影。对平方误差和线性秩一重构，最优子空间由样本 PCA 给出；不能把这个结论推广到任意非线性自编码器。']);
 };
 function vaeTerms(mu,logvar,w,x,beta){const variance=Math.exp(logvar),reconstruction=((x-w*mu)**2+w*w*variance)/2,kl=(mu*mu+variance-1-logvar)/2;return {variance,reconstruction,kl,total:reconstruction+beta*kl,gradMu:w*(w*mu-x)+beta*mu,gradLogvar:.5*w*w*variance+.5*beta*(variance-1)};}
 models.vae=p=>{
  const d=vaeTerms(p.mu,p.logvar,p.decoder,p.x,p.beta),sigma=Math.sqrt(d.variance),z=p.mu+sigma*p.epsilon,bound=Math.max(4,Math.abs(p.mu)+4*sigma,Math.abs(z)+1),grid=linspace(-bound,bound,321);
  return result([metric('重参数化 z',z),metric('KL(q ‖ N(0,1))',d.kl),metric('期望重构负对数似然（去常数）',d.reconstruction),metric('β 加权目标',d.total),metric('目标对 μ 的精确梯度',d.gradMu)],
   [plot('近似后验、先验与一个固定噪声样本','潜变量 z','概率密度',[line('先验 N(0,1)',grid.map(x=>[x,M.normalPDF(x)]),6,true),line('q(z|x)',grid.map(x=>[x,density(x,p.mu,sigma)])),scatter('z = μ + σε',[[z,density(z,p.mu,sigma)]],1)]),
    plot('固定其他参数，改变 μ','近似后验均值 μ','目标值（省略统一常数）',[line('期望重构项',linspace(-3,3,121).map(mu=>[mu,vaeTerms(mu,p.logvar,p.decoder,p.x,p.beta).reconstruction])),line('β × KL',linspace(-3,3,121).map(mu=>[mu,p.beta*vaeTerms(mu,p.logvar,p.decoder,p.x,p.beta).kl]),1,true),line('两项之和',linspace(-3,3,121).map(mu=>[mu,vaeTerms(mu,p.logvar,p.decoder,p.x,p.beta).total]),2),scatter('当前 μ',[[p.mu,d.total]],3)])],
   {...d,sigma,z,dzDmu:1,dzDlogvar:.5*sigma*p.epsilon},['q=N(μ,exp(logvar))，先验 N(0,1)，解码似然 x|z~N(wz,1)。重构项使用解析期望，而非一个噪声样本估计。','β=1 对应这个指定模型的负 ELBO（去掉 ½log(2π) 常数）；β≠1 是加权变体，不能直接称为原模型的负 ELBO。','改变 ε 只改变展示的样本 z，不改变解析期望目标或 KL。']);
 };
 function ganTerms(theta,weight,bias){const logit=weight*theta+bias,D=sigmoid(logit);return {D,saturating:-softplus(logit),nonSaturating:softplus(-logit),gradSaturating:-weight*D,gradNonSaturating:weight*(D-1)};}
 models.gan=p=>{
  const d=ganTerms(p.theta,p.weight,p.bias),sat=[[0,p.theta]],non=[[0,p.theta]];let a=p.theta,b=p.theta;
  for(let k=1;k<=p.steps;k++){a-=p.rate*ganTerms(a,p.weight,p.bias).gradSaturating;b-=p.rate*ganTerms(b,p.weight,p.bias).gradNonSaturating;sat.push([k,a]);non.push([k,b]);}
  const xs=linspace(-6,6,241),Dreal=sigmoid(2*p.weight+p.bias);
  return result([metric('判别器给假样本的 D',d.D),metric('饱和目标的梯度',d.gradSaturating),metric('非饱和目标的梯度',d.gradNonSaturating),metric('判别器负对数似然',softplus(-(2*p.weight+p.bias))+softplus(p.weight*p.theta+p.bias))],
   [plot('固定判别器与两个点质量样本','样本 x','D(x)',[line('D(x)=σ(kx+b)',xs.map(x=>[x,sigmoid(p.weight*x+p.bias)])),scatter('假样本 θ',[[p.theta,d.D]],1),scatter('真实示例 x=2',[[2,Dreal]],2)],{yDomain:[-.05,1.05]}),
    plot('生成器在相同判别器下收到的梯度','生成参数 θ','dL / dθ',[line('min log(1−D) 的梯度',xs.map(x=>[x,ganTerms(x,p.weight,p.bias).gradSaturating]),0),line('min −log D 的梯度',xs.map(x=>[x,ganTerms(x,p.weight,p.bias).gradNonSaturating]),1,true),scatter('当前非饱和梯度',[[p.theta,d.gradNonSaturating]],2)]),
    plot('判别器冻结后的生成器迭代','更新步数','生成点 θ',[line('饱和目标',sat,0),line('非饱和目标',non,1,true),line('真实示例位置',[[0,2],[p.steps,2]],6,true)])],
   {...d,Dreal,saturatingPath:sat,nonSaturatingPath:non,discriminatorGradient:{weight:2*(Dreal-1)+p.theta*d.D,bias:Dreal-1+d.D}},['这里生成器只输出一个标量 θ，判别器全程冻结；轨迹只比较两个生成器目标的梯度，不是完整 GAN 训练。','D 接近 0 且 k 非零时，log(1−D) 的梯度接近 0；−log D 的梯度可保持较大。k=0 时二者对 θ 的梯度都为 0。','固定判别器会奖励朝其高分区域移动，并不保证生成点停在真实示例或匹配数据分布。']);
 };
 function flowTransform(z,p){return p.shift+p.scale*(z+p.warp*Math.tanh(z));}
 function flowDerivative(z,p){return p.scale*(1+p.warp*(1-Math.tanh(z)**2));}
 function flowInverse(y,p){let lo=-1,hi=1;while(flowTransform(lo,p)>y)lo*=2;while(flowTransform(hi,p)<y)hi*=2;for(let i=0;i<65;i++){const mid=(lo+hi)/2;if(flowTransform(mid,p)<y)lo=mid;else hi=mid;}return (lo+hi)/2;}
 models.flow=p=>{
  const zs=linspace(-5,5,401),ys=zs.map(z=>flowTransform(z,p)),pdf=zs.map((z,i)=>[ys[i],M.normalPDF(z)/flowDerivative(z,p)]),y=flowTransform(p.z,p),jacobian=flowDerivative(p.z,p);
  let area=0;for(let i=1;i<pdf.length;i++)area+=(pdf[i][0]-pdf[i-1][0])*(pdf[i][1]+pdf[i-1][1])/2;
  return result([metric('选定 z 的像 y',y),metric('局部 Jacobian dy/dz',jacobian),metric('对应输出密度 p(y)',M.normalPDF(p.z)/jacobian),metric('图示区间密度积分（梯形近似）',area,6)],
   [plot('单调可逆变换及选定点','基变量 z','输出变量 y',[line('g(z)',zs.map((z,i)=>[z,ys[i]])),scatter('当前变换点',[[p.z,y]],1)]),
    plot('密度用 Jacobian 修正后搬运','输出坐标 y','概率密度 p(y)',[line('p_y(g(z))=φ(z)/g′(z)',pdf),scatter('当前密度',[[y,M.normalPDF(p.z)/jacobian]],1)]),
    plot('局部拉伸决定密度变化','基变量 z','正 Jacobian',[line('g′(z)',zs.map(z=>[z,flowDerivative(z,p)])),scatter('当前 Jacobian',[[p.z,jacobian]],1)])],
   {zs,ys,density:pdf,y,jacobian,area,inverse:flowInverse(y,p)},['g(z)=shift+scale·(z+warp·tanh z)，scale>0 且 warp>−1，故 g′>0 并且映射覆盖整条实线。','密度乘的是逆变换的 Jacobian 绝对值；在本页单调情况下为除以 g′。变换样本位置却不修改密度会破坏总质量。','图展示 z∈[−5,5] 对应的区间，省略标准正态约 5.73×10⁻⁷ 的尾部质量；梯形积分还包含离散近似误差。']);
 };
 function gaussianConditional(alpha,mu,sigma,observed){const v=alpha*sigma*sigma+1-alpha,m=Math.sqrt(alpha)*mu,gain=Math.sqrt(alpha)*sigma*sigma/v,posteriorMean=mu+gain*(observed-m),posteriorVariance=sigma*sigma*(1-alpha)/v;return {marginalMean:m,marginalVariance:v,gain,posteriorMean,posteriorVariance,noiseMean:(observed-Math.sqrt(alpha)*posteriorMean)/Math.sqrt(1-alpha),score:-(observed-m)/v};}
 models.diffusion=p=>{
  const d=gaussianConditional(p.alpha,p.mu,p.sigma,p.observed),postSD=Math.sqrt(d.posteriorVariance),margSD=Math.sqrt(d.marginalVariance),lo=Math.min(p.mu-4*p.sigma,d.posteriorMean-4*postSD),hi=Math.max(p.mu+4*p.sigma,d.posteriorMean+4*postSD);
  return result([metric('精确 E[x₀ | xₜ]',d.posteriorMean),metric('精确 Var[x₀ | xₜ]',d.posteriorVariance),metric('精确 E[ε | xₜ]',d.noiseMean),metric('边际 score ∂log p(xₜ)/∂xₜ',d.score)],
   [plot('已知高斯先验下的精确去噪后验','原始数据坐标 x₀','概率密度',[line('原始分布 N(μ,σ²)',linspace(lo,hi,321).map(x=>[x,density(x,p.mu,p.sigma)]),6,true),line('x₀ | 当前 xₜ',linspace(lo,hi,321).map(x=>[x,density(x,d.posteriorMean,postSD)]),0)]),
    plot('加噪后的边际分布与观测','噪声数据坐标 xₜ','概率密度',[line('p(xₜ)',linspace(Math.min(d.marginalMean-4*margSD,p.observed-1),Math.max(d.marginalMean+4*margSD,p.observed+1),321).map(x=>[x,density(x,d.marginalMean,margSD)])),scatter('当前观测 xₜ',[[p.observed,density(p.observed,d.marginalMean,margSD)]],1)]),
    plot('信号保留量决定不可消除的条件不确定性','累计信号比例 ᾱ','Var[x₀ | xₜ]',[line('精确条件方差',linspace(.01,.99,99).map(a=>[a,gaussianConditional(a,p.mu,p.sigma,p.observed).posteriorVariance])),scatter('当前 ᾱ',[[p.alpha,d.posteriorVariance]],1)])],
   d,['前向式 xₜ=√ᾱ·x₀+√(1−ᾱ)·ε，x₀~N(μ,σ²)、ε~N(0,1) 独立；条件分布由高斯联合分布精确计算。','此处不是训练好的去噪网络，也不是多步生成采样器。真实图像分布通常不是单个高斯，其后验不会由本页公式直接给出。','后验均值最小化条件平方误差，但不等于恢复每个样本的原始噪声或原图；条件方差说明仍有不确定性。']);
 };
 function mdpPolicy(q,p){const B=p.cost+3*p.gamma*p.success,A=p.gamma*(1-p.success),den=1-A*q,num=1+q*(B-1),value=num/den,dq=((B-1)*den+A*num)/(den*den);return {value,dq,den,Qsafe:1,Qrisky:p.cost+p.gamma*(3*p.success+(1-p.success)*value),visits:1/den};}
 models['rl-bellman']=p=>{
  const exact=[Math.max(1,(p.cost+3*p.gamma*p.success)/(1-p.gamma*(1-p.success))),3,0],history=[[0,0,0]];
  const T=v=>[Math.max(1,p.cost+p.gamma*(p.success*v[1]+(1-p.success)*v[0])),3,0];
  for(let i=0;i<p.iterations;i++)history.push(T(history[history.length-1]));
  const final=history[history.length-1],residual=Math.max(...T(final).map((v,i)=>Math.abs(v-final[i]))),error=Math.max(...final.map((v,i)=>Math.abs(v-exact[i])));
  return result([metric('当前 V(S₀)',final[0]),metric('精确最优 V*(S₀)',exact[0]),metric('最大值函数误差',error,6),metric('残差 / (1−γ) 上界',residual/(1-p.gamma),6)],
   [plot('有限 MDP 的实际转移（终止状态价值为 0）','示意 x','示意 y',[arrow('安全：即时奖励 1',[[0,0,2.6,0]],0),arrow('冒险成功：即时奖励 '+p.cost+'，p='+p.success,[[0,0,1.3,1.2]],1),arrow('S₁：即时奖励 3',[[1.3,1.2,2.6,0]],2),line('冒险失败：奖励相同，回到 S₀',[[0,0],[-.65,.3],[-.65,.8],[-.15,.8]],3),arrow('失败返回方向',[[-.15,.8,0,.08]],3),{...scatter('状态',[[0,0],[1.3,1.2],[2.6,0]],0),shape:'diamond',radius:6}],{xDomain:[-1.5,3.2],yDomain:[-.4,1.55],annotations:[{x:0,y:0,text:'S₀'},{x:1.3,y:1.2,text:'S₁'},{x:2.6,y:0,text:'终止'},{x:1.3,y:-.22,text:'安全 r=1'},{x:.48,y:.63,text:'r='+p.cost+', p='+p.success.toFixed(2)},{x:-.6,y:1.03,text:'返回 1−p='+(1-p.success).toFixed(2)},{x:2.23,y:.85,text:'r=3'}]}),
    plot('同步 Bellman 最优备份','迭代次数','状态价值',[line('V(S₀)',history.map((v,i)=>[i,v[0]])),line('V(S₁)',history.map((v,i)=>[i,v[1]]),1),line('V*(S₀)',[[0,exact[0]],[p.iterations,exact[0]]],6,true)]),
    plot('实际误差与可计算残差界','迭代次数','最大误差 / 上界',[line('对精确解的最大误差',history.map((v,k)=>[k,Math.max(...v.map((x,i)=>Math.abs(x-exact[i])))])),line('Bellman 残差 / (1−γ)',history.map((v,k)=>[k,Math.max(...T(v).map((x,i)=>Math.abs(x-v[i])))/(1-p.gamma)]),1,true)])],
   {exact,history,final,residual,error,actions:{safe:{reward:1,terminalProbability:1},risky:{reward:p.cost,toState1:p.success,toState0:1-p.success},state1:{reward:3,terminalProbability:1}}},['奖励在执行动作时收到；从 S₀ 到 S₁ 后的奖励 3 需要乘一次折扣 γ。失败会回到 S₀，并再次作决定。','展示的是已知模型下的同步 value iteration，不是通过采样训练的 Q-learning。γ<1 保证该有限折扣问题的 Bellman 算子为压缩映射。']);
 };
 models['rl-policy']=p=>{
  const initial=mdpPolicy(sigmoid(p.theta),p),path=[];let theta=p.theta;
  for(let i=0;i<=p.steps;i++){const q=sigmoid(theta),d=mdpPolicy(q,p),gradient=d.dq*q*(1-q);path.push({step:i,theta,q,value:d.value,gradient});theta+=p.rate*gradient;}
  const q=sigmoid(p.theta),gradient=initial.dq*q*(1-q),last=path[path.length-1];
  return result([metric('初始冒险概率 σ(θ)',q),metric('初始精确回报 J',initial.value),metric('初始 dJ/dθ',gradient),metric('最终精确回报 J',last.value),metric('最终冒险概率',last.q)],
   [plot('策略参数直接决定概率和回报','策略参数 θ','从 S₀ 出发的期望折扣回报',[line('精确 J(θ)',linspace(-8,8,241).map(t=>[t,mdpPolicy(sigmoid(t),p).value])),scatter('初始策略',[[p.theta,initial.value]],1)]),
    plot('精确策略梯度上升的回报','更新次数','期望回报 J',[line('当前策略回报',path.map(v=>[v.step,v.value]))]),
    plot('策略选择概率的变化','更新次数','冒险概率',[line('π(冒险 | S₀)',path.map(v=>[v.step,v.q]))],{yDomain:[0,1]})],
   {initial,initialProbability:q,gradient,path,final:last,occupancyGradient:initial.visits*q*(1-q)*(initial.Qrisky-initial.Qsafe)},['环境与 Bellman 实验相同；此处把冒险概率参数化为 sigmoid(θ)，使用解析期望与精确梯度，没有 Monte Carlo 估计噪声。','折扣访问次数 1/[1−γq(1−p)] 不能漏掉：动作会改变未来再次访问 S₀ 的概率。','策略梯度的目标是回报而非分类标签。概率接近 0 或 1 时 sigmoid 导数很小，有限步训练不保证到达确定性最优策略。']);
 };
 function randomFeatureData(n,noise,seedValue){
  const normal=M.normalGenerator(M.rng(seedValue^0xA76812)),phase=M.rng(seedValue^0x5414);
  const weights=Array.from({length:80},()=>Array.from({length:5},()=>normal())),bias=weights.map(()=>2*Math.PI*phase());
  const feature=x=>weights.map((w,j)=>Math.SQRT2*Math.cos(dot(w,x)/Math.sqrt(5)+bias[j]));
  function data(count,s){const draw=M.normalGenerator(M.rng(s)),xs=[],X=[],truth=[],y=[];for(let i=0;i<count;i++){const x=Array.from({length:5},()=>draw()),row=feature(x),target=sum(row.slice(0,10))/Math.sqrt(10);xs.push(x);X.push(row);truth.push(target);y.push(target+noise*draw());}return {xs,X,truth,y};}
  return {train:data(n,seedValue),validation:data(240,seedValue^0xF089A),weights,bias};
 }
 function fitFeatures(data,width,lambda){const X=data.train.X.map(row=>[1,...row.slice(0,width)]),fit=M.leastSquares(X,data.train.y,lambda);if(!fit.coefficients)return {width,rank:fit.rank,trainMSE:null,validationMSE:null,truthMSE:null,coefficients:null};const predict=row=>dot([1,...row.slice(0,width)],fit.coefficients),trainPrediction=data.train.X.map(predict),validationPrediction=data.validation.X.map(predict);return {width,rank:fit.rank,unique:fit.unique,method:fit.method,coefficients:fit.coefficients,trainPrediction,validationPrediction,trainMSE:M.mean(trainPrediction.map((v,i)=>(v-data.train.y[i])**2)),validationMSE:M.mean(validationPrediction.map((v,i)=>(v-data.validation.y[i])**2)),truthMSE:M.mean(validationPrediction.map((v,i)=>(v-data.validation.truth[i])**2))};}
 models['double-descent']=p=>{
  const data=randomFeatureData(p.n,p.noise,p.seed),widths=[...new Set([2,4,8,12,16,20,24,28,32,40,48,64,80,p.n-2,p.n-1,p.n,p.n+1,p.width])].filter(x=>x>=2&&x<=80).sort((a,b)=>a-b),fits=widths.map(w=>fitFeatures(data,w,p.lambda)),selected=fits.find(f=>f.width===p.width);
  const log=v=>v===null?null:Math.log10(Math.max(v,1e-16));
  const validationLogs=fits.flatMap(f=>[f.validationMSE,f.truthMSE]).filter(v=>v!==null).map(log),vmin=Math.min(...validationLogs),vmax=Math.max(...validationLogs),interp=p.n-1;
  const interpolationLine=(lo,hi)=>({...segments('参数数目 = n（m = n−1）',[[interp,lo,interp,hi]],6,true),width:1.6});
  return result([metric('当前训练 MSE',selected.trainMSE===null?'数值秩不足':selected.trainMSE,6),metric('独立验证 MSE',selected.validationMSE===null?'数值秩不足':selected.validationMSE,6),metric('参数数目 / 样本数',(p.width+1)/p.n),metric('当前求解方式',selected.method||'rank-deficient')],
   [plot('训练插值与验证风险的共同尺度','随机特征数（另含 1 个截距）','log₁₀(MSE)，下限 −16',[line('训练误差',fits.map(f=>f.trainMSE===null?null:[f.width,log(f.trainMSE)])),line('独立验证误差',fits.map(f=>f.validationMSE===null?null:[f.width,log(f.validationMSE)]),1),line('验证点相对无噪声真值的误差',fits.map(f=>f.truthMSE===null?null:[f.width,log(f.truthMSE)]),2,true),interpolationLine(-16,vmax),...(selected.validationMSE===null?[]:[scatter('当前宽度',[[p.width,log(selected.validationMSE)]],3)])],{xDomain:[2,80]}),
    plot('单独放大实际验证风险：本次曲线是否双降？','随机特征数（另含 1 个截距）','log₁₀(验证 MSE)，独立纵轴尺度',[line('独立验证误差',fits.map(f=>f.validationMSE===null?null:[f.width,log(f.validationMSE)]),1),line('验证点相对无噪声真值的误差',fits.map(f=>f.truthMSE===null?null:[f.width,log(f.truthMSE)]),2,true),interpolationLine(vmin,vmax),...(selected.validationMSE===null?[]:[scatter('当前宽度',[[p.width,log(selected.validationMSE)]],3)])],{xDomain:[2,80]}),
    plot('当前模型的独立验证预测','真实无噪声函数值','模型预测值',[scatter('验证点',selected.validationPrediction?selected.validationPrediction.map((v,i)=>[data.validation.truth[i],v]):[]),line('预测 = 真值',[[Math.min(...data.validation.truth),Math.min(...data.validation.truth)],[Math.max(...data.validation.truth),Math.max(...data.validation.truth)]],6,true)])],
   {...data,widths,fits,selected,interpolationWidth:p.n-1},['使用真实随机 Fourier 特征 φⱼ(x)=√2 cos(wⱼᵀx/√5+bⱼ)；目标函数固定为前 10 个特征之和除以 √10，再加独立高斯噪声。','所有宽度共享同一训练集、独立验证集和特征前缀。λ=0 的宽矩阵使用最小欧氏范数解；λ>0 惩罚非截距系数。','这是本次有限样本真正算出的曲线，某个种子不一定出现双降。插值位置是特征数+截距=n；图中没有预制 U 形或双降形状。','反复按验证误差选宽度会产生选择偏差；本页固定验证集用于解释，不能当作最终无偏评估。误差低于 10⁻¹⁶ 时仅在对数图上用 −16 显示，原始值完整导出。']);
 };
 function confusion(base,tpr,fpr){const tp=base*tpr,fn=base*(1-tpr),fp=(1-base)*fpr,tn=(1-base)*(1-fpr),selected=tp+fp;return {tp,fn,fp,tn,tpr,fpr,selection:selected,precision:selected===0?null:tp/selected};}
 models.fairness=p=>{
  const A=confusion(p.baseA,p.tpr,p.fpr),B=confusion(p.baseB,p.tpr,p.fpr),cells=[[0,0,A.tn],[1,0,A.fp],[0,1,A.fn],[1,1,A.tp],[0,2,B.tn],[1,2,B.fp],[0,3,B.fn],[1,3,B.tp]],metrics=['TPR','FPR','选择率','PPV'];
  const grouped=(d,shift)=>[d.tpr,d.fpr,d.selection,d.precision].flatMap((v,i)=>v===null?[]:[[i+shift,v,.32]]);
  return result([metric('A 的 PPV',A.precision===null?'无正预测：未定义':A.precision),metric('B 的 PPV',B.precision===null?'无正预测：未定义':B.precision),metric('选择率差 A−B',A.selection-B.selection),metric('相同 TPR 与 FPR','构造上满足 equalized odds')],
   [plot('各组联合混淆概率（每组四格合计 1）','预测标签','组别与真实标签',[heat('联合概率',cells,[0,1],true)],{xDomain:[-.5,1.5],yDomain:[-.5,3.5],xTickLabels:['预测 0','预测 1'],yTickLabels:['A: 真 0','A: 真 1','B: 真 0','B: 真 1'],reverseY:true}),
    plot('同一个分类器条件性能，不同基准率','评价指标','概率',[bars('组 A',grouped(A,-.18),0),bars('组 B',grouped(B,.18),1)],{xTickLabels:metrics,xDomain:[-.6,3.6],yDomain:[0,1.05]}),
    plot('固定 TPR 与 FPR 后，PPV 随基准率改变','总体正类比例 π','PPV = P(Y=1 | Ŷ=1)',[line('条件概率恒等式',linspace(.01,.99,99).map(base=>{const q=confusion(base,p.tpr,p.fpr);return q.precision===null?null:[base,q.precision];})),...(A.precision===null?[]:[scatter('组 A',[[p.baseA,A.precision]],1)]),...(B.precision===null?[]:[scatter('组 B',[[p.baseB,B.precision]],2)])],{xDomain:[0,1],yDomain:[0,1.05]})],
   {A,B,countsPer1000:{A:Object.fromEntries(['tp','fn','fp','tn'].map(k=>[k,1000*A[k]])),B:Object.fromEntries(['tp','fn','fp','tn'].map(k=>[k,1000*B[k]]))}},['A/B 是完全合成的抽象组别，不代表现实人群。热图显示总体联合概率，而不是带抽样波动的整数人数。','当两组基准率不同且 TPR>0、FPR>0 时，相同 TPR/FPR 导致不同 PPV；这是 Bayes 公式的代数结果。FPR=0、TPR=0 或相同基准率等边界需另看。','没有正预测时 PPV 未定义，不能把它记作 0。本实验不把不同公平指标的数值差异自动解释为一种政策应取代另一种政策。']);
 };
 for(const [id,model]of Object.entries(models))M.register(id,schemas[id],model);
 M.advanced={positionalVector,vaeTerms,ganTerms,flowTransform,flowDerivative,flowInverse,gaussianConditional,mdpPolicy,randomFeatureData,fitFeatures,confusion,sigmoid};
 if(typeof module!=='undefined'&&module.exports)module.exports=M.advanced;
})(typeof globalThis!=='undefined'?globalThis:this);
