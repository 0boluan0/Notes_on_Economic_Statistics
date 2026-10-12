/* Independent criteria: scalar perturbations, closed forms, recursive Bayesian
   updating (different algorithm from batch inversion), joint-probability SCM. */
'use strict';
const assert=require('assert/strict'),fs=require('fs'),path=require('path'),crypto=require('crypto'),cp=require('child_process');
const root=path.resolve(__dirname,'..'),M=require('../assets/math.js'),S=require('../assets/sequential-math.js'),C=require('../assets/sequential-content.js'),katex=require('../assets/vendor/katex/katex.min.js');
let assertions=0,coordinateAssertions=0,cases=0;const tested={},sum=a=>a.reduce((a,b)=>a+b,0),dot=(a,b)=>sum(a.map((v,i)=>v*b[i]));
function check(ok,label){assert.ok(ok,label);assertions++;}
function close(a,b,tol=1e-9,label='identity'){if(Array.isArray(a)){check(a.length===b.length,label+' shape');a.forEach((v,i)=>close(v,b[i],tol,label));}else check(Math.abs(a-b)<=tol*Math.max(1,Math.abs(a),Math.abs(b)),`${label}: ${a} vs ${b}`);}
function inspect(id,params){const r=M.compute(id,params);cases++;tested[id]=(tested[id]||0)+1;check(r.plots.length===3,id+' three mechanisms');for(const p of r.plots){check(Boolean(p.title&&p.xLabel&&p.yLabel),'labels');for(const s of p.series){check(typeof s.label==='string','legend field');for(const v of s.points){if(v===null)continue;check(v.every(Number.isFinite),id+' finite plotted data');coordinateAssertions++;for(const xy of s.type==='segments'?[v.slice(0,2),v.slice(2,4)]:[v.slice(0,2)]){if(p.xDomain){check(xy[0]>=p.xDomain[0]-1e-9&&xy[0]<=p.xDomain[1]+1e-9,id+' x limits');coordinateAssertions++;}if(p.yDomain){check(xy[1]>=p.yDomain[0]-1e-9&&xy[1]<=p.yDomain[1]+1e-9,id+' y limits');coordinateAssertions++;}}}}}return r;}
for(const [id,c]of Object.entries(C)){const schema=S.schemas[id];check(c.controls.length===Object.keys(schema).length,id+' complete controls');check(c.presets.length>=3&&c.challenges.length>=3,'teaching scenarios');katex.renderToString(c.formula,{throwOnError:true,strict:'error'});check(c.sources.every(s=>s.url&&s.section&&s.support),'source support');const first=inspect(id,{});for(const ctl of c.controls){check(ctl.value===schema[ctl.key][0],id+' default parity');if(ctl.type==='range'){check(ctl.min===schema[ctl.key][1]&&ctl.max===schema[ctl.key][2],id+' control limits');inspect(id,{[ctl.key]:ctl.min});inspect(id,{[ctl.key]:ctl.max});}else for(const o of ctl.options)inspect(id,{[ctl.key]:o.value});}for(const p of c.presets)inspect(id,p.params);check(JSON.stringify(first)===JSON.stringify(M.compute(id,{})),id+' deterministic');}
// RNN finite differences recompute only the explicit scalar recurrence, no model gradient calls.
for(const params of [{},{kind:'linear',weight:-.8,initial:.4,pattern:'alternating',steps:10},{weight:1.4,pattern:'sine',steps:20}]){const r=M.compute('rnn',params),p=r.params,d=r.data,h=1e-6;
 const forward=(a,h0,xs)=>{let value=h0;for(const x of xs){const z=a*value+p.inputWeight*x;value=p.kind==='linear'?z:Math.tanh(z);}return value;};
 const loss=(a,h0,xs)=>(forward(a,h0,xs)-p.target)**2/2;
 close(d.weightGradient,(loss(p.weight+h,p.initial,d.xs)-loss(p.weight-h,p.initial,d.xs))/(2*h),2e-6,'shared RNN parameter finite difference');
 close(d.sensitivities[0],(forward(p.weight,p.initial+h,d.xs)-forward(p.weight,p.initial-h,d.xs))/(2*h),2e-6,'initial-state sensitivity');
 for(let i=0;i<d.xs.length;i++){const a=d.xs.slice(),b=d.xs.slice();a[i]+=h;b[i]-=h;close(d.inputGradients[i],(loss(p.weight,p.initial,a)-loss(p.weight,p.initial,b))/(2*h),2e-6,'each sequence input gradient');}
 close(d.weightGradient,sum(d.weightContributions));}
for(const a of [-1.2,-.5,0,.5,1,1.2]){const d=M.compute('rnn',{kind:'linear',weight:a,inputWeight:0,initial:.5,steps:10,target:0}).data;close(d.states.at(-1),.5*a**10);close(d.sensitivities[0],a**10);close(d.weightGradient,2.5*a**19);}
// LSTM: independently perturb initial cell including h0=o*tanh(c0).
for(const params of [{},{forget:1,input:0,output:1},{forget:.5,input:1,output:1,recurrent:.5,initial:0,steps:10},{forget:.9,recurrent:-1.4,pattern:'sine',steps:18}]){const r=M.compute('lstm',params),p=r.params,d=r.data;
 const f=c0=>{let c=c0,h=p.output*Math.tanh(c0);for(const x of d.xs){c=p.forget*c+p.input*Math.tanh(x+p.recurrent*h);h=p.output*Math.tanh(c);}return [c,h];},eps=1e-6,a=f(p.initial+eps),b=f(p.initial-eps);close([d.last.dc,d.last.dh],a.map((x,i)=>(x-b[i])/(2*eps)),2e-6,'LSTM full derivative');close(d.last.direct,p.forget**p.steps);
 if(p.recurrent===0)close(d.last.dc,d.last.direct);}
const memory=M.compute('lstm',{forget:1,input:0,output:1,initial:1}).data;close(memory.last.c,1);close(memory.last.dc,1);close(memory.last.dh,1-Math.tanh(1)**2);
const feedback=M.compute('lstm',{forget:.5,input:1,output:1,recurrent:.5,initial:0,steps:10}).data;close(feedback.states[1].dc,1);close(feedback.states[1].direct,.5);
// InfoNCE free-vector and angle derivatives, duplicate-label impossibility, true negative count effect.
for(const params of [{},{angle:60,temperature:.1},{angle:-80,negativeAngle:-90,negatives:1}]){const r=M.compute('contrastive',params),p=r.params,d=r.data,eps=1e-6;
 const loss=q=>{const logits=d.keys.map(k=>dot(q,k)/p.temperature),m=Math.max(...logits);return m+Math.log(sum(logits.map(s=>Math.exp(s-m))))-logits[0];};
 for(let j=0;j<2;j++){const a=d.q.slice(),b=d.q.slice();a[j]+=eps;b[j]-=eps;close(d.gradient[j],(loss(a)-loss(b))/(2*eps),2e-6,'free query gradient');}
 const unit=angle=>[Math.cos(angle*Math.PI/180),Math.sin(angle*Math.PI/180)];close(d.angleGradient,(loss(unit(p.angle+eps))-loss(unit(p.angle-eps)))/(2*eps),2e-6,'degree gradient');close(sum(d.probabilities),1);close(d.loss,-Math.log(d.probabilities[0]));}
for(const temperature of [.05,.5,2]){const d=M.compute('contrastive',{angle:60,negativeAngle:0,negatives:1,temperature}).data;close(d.probabilities,[.5,.5]);close(d.loss,Math.log(2));close(d.gradient,[0,0]);}
const opposite=M.compute('contrastive',{angle:0,negativeAngle:180,temperature:1,negatives:1}).data;close(opposite.probabilities[0],1/(1+Math.exp(-2)));close(opposite.loss,Math.log1p(Math.exp(-2)));
const negativeLoss=[1,2,3].map(negatives=>M.compute('contrastive',{negatives}).data.loss);check(negativeLoss[2]>negativeLoss[1]&&negativeLoss[1]>negativeLoss[0],'fixed-score loss strictly increases with added negative');
// Actual nonlinear map Jacobian finite differences and all plotted Taylor remainder bounds.
for(const params of [{},{alpha:1,beta:.5,x:1,y:0},{alpha:-1.5,beta:1,x:-1,y:1,radius:1.5},{alpha:0,beta:0}]){const r=M.compute('jacobian',params),p=r.params,d=r.data,eps=1e-6,F=([x,y])=>[x+p.alpha*Math.sin(y),y+p.beta*x*x];
 for(let j=0;j<2;j++){const a=d.center.slice(),b=d.center.slice();a[j]+=eps;b[j]-=eps;const fa=F(a),fb=F(b);for(let i=0;i<2;i++)close(d.J[i][j],(fa[i]-fb[i])/(2*eps),2e-7,'Jacobian column perturbation');}
 for(const e of d.errors)check(e.error<=e.bound+1e-14,'actual error below analytic Taylor bound');
 for(let i=0;i<d.inputLines.length;i++)for(let j=0;j<d.inputLines[i].length;j++){const q=d.inputLines[i][j],expected=F(q);close(d.actualLines[i][j],expected);close(d.linearLines[i][j],d.mapped.map((v,k)=>v+dot(d.J[k],q.map((x,k)=>x-d.center[k]))));}}
close(M.compute('jacobian',{alpha:0,beta:0}).data.maxGridError,0);close(M.compute('jacobian',{alpha:1,beta:.5,x:1,y:0}).data.determinant,0);
// Bayesian posterior via sequential Gaussian conditioning, independent from batch matrix inversion.
const posteriorCases=[];
for(const params of [{},{n:2,span:.2,priorSD:4,sigma:.1},{n:30,sigma:1.5,priorSD:.25},{n:30,seed:18,span:2}]){const r=M.compute('uncertainty',params),p=r.params,d=r.data;let mean=[0,0],cov=[[p.priorSD**2,0],[0,p.priorSD**2]];
 for(const obs of d.sample){const phi=[1,obs.x],cphi=cov.map(row=>dot(row,phi)),den=p.sigma**2+dot(phi,cphi),innovation=obs.y-dot(phi,mean);mean=mean.map((v,i)=>v+cphi[i]*innovation/den);cov=cov.map((row,i)=>row.map((v,j)=>v-cphi[i]*cphi[j]/den));}
 close(d.mean,mean,2e-10,'posterior batch vs sequential mean');close(d.covariance,cov,2e-10,'posterior batch vs sequential covariance');
 for(const g of d.grid){check(g.latentVariance>0,'positive latent uncertainty');close(g.predictiveVariance-g.latentVariance,p.sigma**2);check(g.predictiveInterval[1]-g.predictiveInterval[0]>g.latentInterval[1]-g.latentInterval[0],'predictive interval wider');}
 for(const v of d.ellipse){const delta=v.map((x,i)=>x-d.mean[i]);close(dot(delta,d.precision.map(row=>dot(row,delta))),-2*Math.log(.05),1e-9,'joint credible ellipse Mahalanobis radius');}
 posteriorCases.push({params:p,data:d});}
const sparse=M.compute('uncertainty',{n:2}).data,dense=M.compute('uncertainty',{n:30}).data;check(JSON.stringify(sparse.sample)===JSON.stringify(dense.sample.slice(0,2)),'nested sample prefix');for(let i=0;i<sparse.grid.length;i++)check(dense.grid[i].latentVariance<=sparse.grid[i].latentVariance+1e-12,'additional nested observations reduce posterior variance');
// Causal quantities recomputed from eight joint cells, then compared with direct SCM intervention.
for(const selection of [-1,-.8,0,.8,1])for(const effect of [-.15,0,.1,.15])for(const prevalence of [.1,.5,.9]){const d=M.compute('causal',{selection,effect,prevalence}).data;close(sum(d.joint.map(r=>r.probability)),1);check(d.joint.every(r=>r.probability>0),'SCM joint positivity');
 const observed=[0,1].map(t=>sum(d.joint.filter(r=>r.t===t&&r.y===1).map(r=>r.probability))/sum(d.joint.filter(r=>r.t===t).map(r=>r.probability)));close(d.observed,observed);close(d.causalEffect,effect);close(d.adjustedDifference,effect);
 for(let t=0;t<2;t++){const conditional=[0,1].map(z=>sum(d.joint.filter(r=>r.t===t&&r.z===z&&r.y===1).map(r=>r.probability))/sum(d.joint.filter(r=>r.t===t&&r.z===z).map(r=>r.probability)));close(d.interventional[t],(1-prevalence)*conditional[0]+prevalence*conditional[1]);}
 if(selection===0)close(d.observationalDifference,effect);}
const simpson=M.compute('causal',{}).data;close(simpson.observed,[.61,.39]);close(simpson.interventional,[.45,.55]);check(simpson.observationalDifference<0&&simpson.causalEffect>0,'Simpson reversal');close(M.compute('causal',{effect:0}).data.observationalDifference,-.32);
// Optional direct NumPy solve; default command remains dependency-free.
let numpyStatus='not requested (set LAB_NUMPY_PYTHON to a NumPy Python executable)';
if(process.env.LAB_NUMPY_PYTHON){const program=`import json,sys,numpy as np\ncases=json.load(sys.stdin)\nfor c in cases:\n p,d=c['params'],c['data']; X=np.array([[1,r['x']] for r in d['sample']]); y=np.array([r['y'] for r in d['sample']]); A=np.eye(2)/p['priorSD']**2+X.T@X/p['sigma']**2; S=np.linalg.solve(A,np.eye(2)); m=np.linalg.solve(A,X.T@y/p['sigma']**2)\n np.testing.assert_allclose(S,d['covariance'],rtol=1e-11,atol=1e-12); np.testing.assert_allclose(m,d['mean'],rtol=1e-11,atol=1e-12)\nprint('passed '+str(len(cases))+' independent NumPy posterior solves')\n`;const run=cp.spawnSync(process.env.LAB_NUMPY_PYTHON,['-c',program],{input:JSON.stringify(posteriorCases),encoding:'utf8'});check(run.status===0,'NumPy oracle: '+run.stderr);numpyStatus=run.stdout.trim();}
const report={status:'passed',utc:new Date().toISOString(),cases,experimentCases:tested,assertions,coordinateAssertions,scientificAndStructuralAssertions:assertions-coordinateAssertions,numpyStatus,sourceSHA256:crypto.createHash('sha256').update(fs.readFileSync(path.join(root,'assets/sequential-math.js'))).digest('hex')};fs.writeFileSync(path.join(__dirname,'sequential-report.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));
