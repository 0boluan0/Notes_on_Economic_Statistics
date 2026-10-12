/* Neural lab acceptance: independent closed forms, raw identities and perturbations. */
'use strict';
const assert=require('assert/strict'),path=require('path'),fs=require('fs'),crypto=require('crypto');
const root=path.resolve(__dirname,'..'),M=require('../assets/math.js'),DL=require('../assets/dl-math.js'),C=require('../assets/dl-content.js');
let assertions=0,cases=0,coordinateAssertions=0;const tested={};
const check=(b,label)=>{assert.ok(b,label);assertions++;};
const close=(a,b,tol=1e-9,label='numeric identity')=>{if(Array.isArray(a)){check(Array.isArray(b)&&a.length===b.length,label+' shape');a.forEach((v,i)=>close(v,b[i],tol,label));}else check(Math.abs(a-b)<=tol*Math.max(1,Math.abs(a),Math.abs(b)),`${label}: ${a} vs ${b}`);};
const mean=a=>a.reduce((s,v)=>s+v,0)/a.length, dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0), sum=a=>a.reduce((s,v)=>s+v,0);
const sigmoid=x=>1/(1+Math.exp(-x));
function inspect(id,params){const r=M.compute(id,params);cases++;tested[id]=(tested[id]||0)+1;check(r.plots.length>=2,id+' multiple actual mechanisms');for(const pl of r.plots){check(Boolean(pl.title&&pl.xLabel&&pl.yLabel),id+' plot labelled');for(const se of pl.series){for(const point of se.points){if(point===null)continue;check(point.every(Number.isFinite),id+' finite plot values');coordinateAssertions++;if(se.type==='heatmap'){const [a,b]=se.zDomain;check(a<b&&point[2]>=a-1e-10&&point[2]<=b+1e-10,id+' heatmap domain');coordinateAssertions++;}const xy=se.type==='segments'?[point.slice(0,2),point.slice(2,4)]:[point.slice(0,2)];for(const [x,y]of xy){if(pl.xDomain){check(x>=pl.xDomain[0]-1e-9&&x<=pl.xDomain[1]+1e-9,id+' x domain');coordinateAssertions++;}if(pl.yDomain){check(y>=pl.yDomain[0]-1e-9&&y<=pl.yDomain[1]+1e-9,id+' y domain');coordinateAssertions++;}}}}}return r;}
for(const [id,content]of Object.entries(C)){
 const schema=DL.schemas[id];check(content.challenges.length>=3,id+' worked challenges');check(content.presets.length>=3,id+' presets');
 check(content.sources.every(s=>s.url&&s.section&&s.support),id+' traceable references');
 check(content.controls.length===Object.keys(schema).length,id+' all controls match schema');
 const def=inspect(id,{});for(const ctl of content.controls){check(Object.hasOwn(schema,ctl.key),id+' control declared');check(ctl.value===schema[ctl.key][0],id+' default parity');if(ctl.type==='select'){close(ctl.options.length,schema[ctl.key][1].length);for(const opt of ctl.options)inspect(id,{[ctl.key]:opt.value});}else{check(ctl.min===schema[ctl.key][1]&&ctl.max===schema[ctl.key][2],id+' range parity');inspect(id,{[ctl.key]:ctl.min});inspect(id,{[ctl.key]:ctl.max});}}
 for(const preset of content.presets)inspect(id,preset.params);
 const repeat=M.compute(id,{});check(JSON.stringify(def.data)===JSON.stringify(repeat.data),id+' deterministic computation');
}
// Derivatives through an affine input, checked against direct scalar perturbation.
for(const kind of ['sigmoid','tanh','relu','leaky'])for(const x of [-2,-.6,.3,2]){
 const p={kind,gain:1.7,bias:.2,x},d=M.compute('activations',p).data,h=1e-6;
 const value=t=>kind==='sigmoid'?sigmoid(1.7*t+.2):kind==='tanh'?Math.tanh(1.7*t+.2):kind==='relu'?Math.max(0,1.7*t+.2):Math.max(1.7*t+.2,.1*(1.7*t+.2));
 close(d.dy,(value(x+h)-value(x-h))/(2*h),2e-7,'activation centered difference');
}
check(M.compute('activations',{kind:'relu',x:0}).data.dy===null,'ReLU kink undefined');
const shifted=M.compute('activations',{kind:'relu',bias:.123,gain:1.7}).data.derivative;check(shifted.includes(null),'off-grid derivative jump has line break');
const neuron=M.compute('neuron',{w1:2,w2:0,bias:-1,x1:.5}).data;close(neuron.probability,.5);close(neuron.derivatives,[.5,0]);check(M.compute('neuron',{w1:0,w2:0}).data.boundary.length===0,'constant neuron no fictitious unique boundary');
for(const width of [3,5,7]){const d=M.compute('mlp',{width,kind:'linear',gain:2,shift:.3,offset:.7}).data;for(const y of d.output)close(y,.7,1e-12,'affine collapse');}
const tent=M.compute('mlp',{}).data;for(let i=0;i<tent.xs.length;i++){const x=tent.xs[i],expected=Math.max(0,1.5-Math.abs(x));close(tent.output[i],expected,1e-12,'three ReLU tent closed form');}
for(const p of [{},{x:0,bias:.5},{w:3,x:2,target:0},{v:0}]){
 const d=M.compute('backprop',p),q=d.params,directLoss=z=>(z.v*Math.tanh(z.w*z.x+z.bias)-z.target)**2/2;for(const key of ['w','bias','v','x']){const h=1e-5,a=directLoss({...q,[key]:q[key]+h}),b=directLoss({...q,[key]:q[key]-h});close(d.data.gradients[key],(a-b)/(2*h),1e-7,'backprop '+key+' perturbation');}}
const init=M.compute('initialization',{depth:4,kind:'tanh'}).data;
function applyNet(x){for(const W of init.weights)x=W.map(w=>Math.tanh(dot(w,x)));return sum(x)/Math.sqrt(init.width);}
for(let j=0;j<6;j++){const a=init.activations[0][0].slice(),b=a.slice(),h=1e-6;a[j]+=h;b[j]-=h;close(init.inputGradients[0][j],(applyNet(a)-applyNet(b))/(2*h),2e-6,'initialization full-network finite difference');}
for(let l=0;l<init.activations.length;l++){const values=init.activations[l].flat(),mu=mean(values);close(init.forwardStats[l].secondMoment,mean(values.map(x=>x*x)));close(init.forwardStats[l].variance,mean(values.map(x=>(x-mu)**2)));}
const xavier=M.compute('initialization',{depth:2,mode:'xavier'}).data,he=M.compute('initialization',{depth:2,mode:'he'}).data;close(he.weights[0],xavier.weights[0].map(row=>row.map(x=>x*Math.SQRT2)),1e-12,'common base initialization draws');
const opt=M.compute('optimizers',{momentum:0}).data;close(opt.trajectories.gd.map(x=>x.x),opt.trajectories.momentum.map(x=>x.x),1e-13,'zero momentum equals GD');
const first=M.compute('optimizers',{angle:0,rate:.05,condition:12}).data;close(first.trajectories.gd[1].x,[2.85,.8]);close(first.trajectories.adam[1].x,[3-.05*3/(3+1e-8),2-.05*24/(24+1e-8)],1e-12,'Adam bias corrected first step');
for(const [method,trajectory]of Object.entries(first.trajectories))for(const p of trajectory)close(p.loss,(p.x[0]**2+12*p.x[1]**2)/2,1e-11,method+' actual objective');
for(const p of [0,.1,.5,.8]){const d=M.compute('regularization',{dropout:p}).data,prob=d.distribution.map(r=>r.probability),mu=sum(d.distribution.map(r=>r.output*r.probability)),variance=sum(d.distribution.map(r=>(r.output-mu)**2*r.probability)),expectedLoss=sum(d.distribution.map(r=>(r.output-1)**2/2*r.probability));close(sum(prob),1);close(d.expected,mu);close(d.variance,variance);close(d.expectedLoss,expectedLoss);close(sum(d.pmf.map(r=>r.probability)),1);close(sum(d.pmf.map(r=>r.empiricalProbability)),1);close(sum(d.pmf.map(r=>r.output*r.probability)),d.expected);close(sum(d.pmf.map(r=>(r.output-d.expected)**2*r.probability)),d.variance);check(sum(d.pmf.map(r=>r.count))===d.outputs.length,'all discrete outputs counted');if(p===0)close(d.empiricalVariance,0);for(let i=0;i<d.outputs.length;i++)close(d.outputs[i],dot(d.contributions,d.masks[i])/d.keepProbability,1e-12,'actual sampled dropout');}
for(const mode of ['batch','layer'])for(const pattern of ['mixed','constant']){const d=M.compute('normalization',{mode,pattern}).data,rows=mode==='batch'?d.X[0].map((_,j)=>d.X.map(row=>row[j])):d.X;rows.forEach((row,i)=>{const mu=mean(row),variance=mean(row.map(x=>(x-mu)**2));close(d.means[i],mu);close(d.variances[i],variance);const normalized=mode==='batch'?d.normalized.map(r=>r[i]):d.normalized[i];close(mean(normalized),0);close(mean(normalized.map(x=>x*x)),variance/(variance+d.epsilon));});}
const bnA=M.compute('normalization',{mode:'batch',shift:0}).data,bnB=M.compute('normalization',{mode:'batch',shift:2}).data;close(bnA.normalized,bnB.normalized,1e-12,'BN common shift invariance');
const collapsed=M.compute('normalization',{gamma:0,beta:1}).data;close(collapsed.Y,Array.from({length:4},()=>[1,1,1]));
for(const pattern of ['edge','impulse','checker'])for(const kernel of ['edge','blur','laplace'])for(const stride of [1,2])for(const padding of [0,1]){
 const d=inspect('convolution',{pattern,kernel,stride,padding}).data;
 close(d.outSize,Math.floor((7+2*padding-3)/stride)+1);
 for(let i=0;i<d.outSize;i++)for(let j=0;j<d.outSize;j++){let expected=0;for(let r=0;r<3;r++)for(let c=0;c<3;c++){const y=i*stride+r-padding,x=j*stride+c-padding;if(y>=0&&y<7&&x>=0&&x<7)expected+=d.input[y][x]*d.kernel[r][c];}close(d.output[i][j],expected,1e-12,'direct unpadded cross correlation');}
 close(sum(d.contributions.flat()),d.output[d.row][d.col]);
}
for(const mode of ['max','mean'])for(const size of [2,3])for(const stride of [1,2,3])for(const pattern of ['peak','ramp','tied']){const d=inspect('pooling',{mode,size,stride,pattern}).data;close(d.output.length,Math.floor((6-size)/stride)+1);const flat=d.patch.flat();close(d.output[d.row][d.col],mode==='max'?Math.max(...flat):mean(flat));close(sum(d.gradient.flat()),1);if(mode==='mean')check(d.gradient.flat().filter(x=>x>0).length===size*size,'mean window derivative support');else check(d.gradient.flat().filter(x=>x>0).length===1,'max selects one subgradient');}
for(const a of [-1,-.2,0,.2,1]){const d=M.compute('residual',{weight:a,kind:'linear',depth:8,x:.5}).data;close(d.paths.plain.at(-1).jacobian,a**8);close(d.paths.residual.at(-1).jacobian,(1+a)**8);close(d.paths.plain.at(-1).value,.5*a**8);close(d.paths.residual.at(-1).value,.5*(1+a)**8);}
for(const a of [-.5,.2,.9])for(const mode of ['plain','residual']){const p={weight:a,depth:12,x:.5},d=M.compute('residual',p).data,h=1e-6,y1=M.compute('residual',{...p,x:.5+h}).data.paths[mode].at(-1).value,y0=M.compute('residual',{...p,x:.5-h}).data.paths[mode].at(-1).value;close(d.paths[mode].at(-1).jacobian,(y1-y0)/(2*h),2e-6,'residual composition finite difference');}
for(const causal of [0,1])for(const strength of [0,1,4]){const d=M.compute('attention',{causal,strength,temperature:.25}).data;for(let i=0;i<4;i++){close(sum(d.weights[i]),1);for(let j=0;j<4;j++){if(!d.mask[i][j])close(d.weights[i][j],0);if(strength===0)close(d.weights[i][j],d.mask[i][j]?1/(causal?i+1:4):0);}for(let k=0;k<2;k++){close(d.outputs[i][k],sum(d.weights[i].map((w,j)=>w*d.V[j][k])));check(d.outputs[i][k]>=Math.min(...d.V.map(v=>v[k]))-1e-10&&d.outputs[i][k]<=Math.max(...d.V.map(v=>v[k]))+1e-10,'attention convex hull bounds');}}}
const att=M.compute('attention',{valueScale:1}).data,att3=M.compute('attention',{valueScale:3}).data;close(att.weights,att3.weights);close(att3.outputs,att.outputs.map(r=>r.map(x=>3*x)));close(M.compute('attention',{causal:1,query:0}).data.outputs[0],[1,0]);
const report={status:'passed',utc:new Date().toISOString(),cases,assertions,coordinateAssertions,experimentCases:tested,sourceSHA256:crypto.createHash('sha256').update(fs.readFileSync(path.join(root,'assets','dl-math.js'))).digest('hex'),methods:['Independent scalar finite differences','Exact ReLU tent and linear-collapse identities','Full initialized network perturbation','Exact enumeration of all 16 dropout masks','Independent input-index cross correlation','Direct normalization moments','Optimizer first-step and objective checks','Residual closed forms and perturbations','Attention row sums, masks, value linearity and convex hull']};
fs.writeFileSync(path.join(__dirname,'dl-report.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));
