'use strict';
const assert=require('node:assert/strict');
const M=require('../assets/math.js'),A=require('../assets/advanced-math.js'),C=require('../assets/advanced-content.js');
let checks=0,cases=0;
const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
const close=(x,y,t=2e-10)=>{checks++;assert.ok(Number.isFinite(x)&&Math.abs(x-y)<=t*Math.max(1,Math.abs(y)),`${x} != ${y}, tolerance ${t}`);};
const finite=(value,path)=>{if(typeof value==='number'){checks++;assert.ok(Number.isFinite(value),path+' is nonfinite');}else if(value&&typeof value==='object')for(const [k,v]of Object.entries(value))finite(v,path+'.'+k);};
function inspect(id,params={}){
 const out=M.compute(id,params);cases++;finite(out,id);assert.ok(out.plots.length>=2&&out.notes.length&&out.metrics.length);checks++;
 for(const p of out.plots){assert.ok(p.title&&p.xLabel&&p.yLabel);for(const s of p.series){assert.ok(s.label);for(const v of s.points){if(v===null){assert.equal(s.type,'line');continue;}assert.ok(v.every(Number.isFinite));checks++;for(let i=0;i<(s.type==='segments'?4:2);i++){const domain=i%2?p.yDomain:p.xDomain;if(domain){assert.ok(v[i]>=domain[0]-1e-9&&v[i]<=domain[1]+1e-9,`${id} ${p.title} point ${v} outside ${domain}`);checks++;}}}}}
 return out;
}
for(const [id,c]of Object.entries(C)){
 const defaults=Object.fromEntries(c.controls.map(v=>[v.key,v.value]));assert.deepEqual(defaults,M.validate(id));checks++;
 assert.ok(c.presets.length>=3&&c.challenges.length>=3&&c.sources.every(s=>/^https:\/\//.test(s.url)&&s.section&&s.support));checks++;
 const a=inspect(id),b=M.compute(id);assert.deepEqual(a,b);checks++;
 for(const preset of c.presets)inspect(id,{...defaults,...preset.params});
 for(const [key,schema]of Object.entries(M.schemas[id])){
  for(const value of Array.isArray(schema[1])?schema[1]:[schema[1],schema[2]])inspect(id,{[key]:value});
  for(const value of [NaN,Infinity,{},null]){assert.throws(()=>M.compute(id,{[key]:value}));checks++;}
 }
}
// Independent trigonometric identities: the same Δ works for every start position.
for(const m of [1,4,8])for(const B of [10,10000])for(const p of [0,7,31])for(const delta of [0,3,32]){
 const a=A.positionalVector(p,m,B),b=A.positionalVector(p+delta,m,B);
 close(dot(a,a),m);close(dot(a,b),Array.from({length:m},(_,j)=>Math.cos(delta*B**(-j/m))).reduce((x,y)=>x+y,0));
}
// Graph invariants distinguish row-stochastic and symmetric normalization.
for(const graph of ['chain','star'])for(const selfLoop of [0,1])for(const normalization of ['mean','symmetric']){
 const d=M.compute('gnn',{graph,selfLoop,normalization,layers:20}).data;
 for(let i=0;i<6;i++){
  if(normalization==='mean')close(d.propagation[i].reduce((x,y)=>x+y,0),1);
  else {for(let j=0;j<6;j++)close(d.propagation[i][j],d.propagation[j][i]);close(dot(d.propagation[i],d.degree.map(Math.sqrt)),Math.sqrt(d.degree[i]));}
 }
 if(normalization==='mean'){const conserved=d.degree[0];for(const h of d.history)close(dot(h,d.degree),conserved);}
}
const graph1=M.compute('gnn',{graph:'star',selfLoop:0,layers:1}).data,graph2=M.compute('gnn',{graph:'star',selfLoop:0,layers:2}).data;
assert.deepEqual(graph1.final,[0,1,1,1,1,1]);assert.deepEqual(graph2.final,[1,0,0,0,0,0]);checks+=2;
for(const angle of [0,60,90,180]){const d=M.compute('embeddings',{angle,normA:2,normB:3}).data;close(d.inner,6*Math.cos(angle*Math.PI/180));close(d.distance*d.distance,13-2*d.inner);}
assert.equal(M.compute('embeddings',{normA:0}).data.cosine,null);checks++;
assert.equal(typeof M.compute('embeddings',{normA:0}).metrics[3].value,'string');checks++;
const productOne=M.compute('embeddings',{normA:2,normB:.5,angle:60}).data;close(productOne.inner,productOne.cosine);
const orthogonal=M.compute('embeddings',{normA:2,normB:3,angle:90}).data;close(orthogonal.inner,orthogonal.cosine);
// Linear bottleneck exact loss decomposition and sample covariance normalization.
for(const angle of [0,35,125])for(const gain of [0,1,2]){
 const d=M.compute('autoencoder',{angle,gain}).data;
 const second=M.mean(d.X.map(x=>dot(x,x))),projected=M.mean(d.codes.map(x=>x*x));
 close(d.mse,second+(gain*gain-2*gain)*projected);close(d.optimalMSE,d.eigenvalues[1]*79/80);assert.ok(d.mse>=d.optimalMSE-1e-10);checks++;
 const optimum=d.eigenvectors[0],error=M.mean(d.X.map(x=>dot(x,x)-dot(x,optimum)**2));close(error,d.optimalMSE);
}
// VAE gradients: centered differences of analytic expectation, not implementation derivatives.
const dh=1e-5;
for(const mu of [-3,0,2])for(const v of [-4,0,2])for(const w of [-2,0,1])for(const beta of [0,1,4]){
 const t=A.vaeTerms(mu,v,w,2,beta);assert.ok(t.kl>=-1e-14);checks++;
 close(t.gradMu,(A.vaeTerms(mu+dh,v,w,2,beta).total-A.vaeTerms(mu-dh,v,w,2,beta).total)/(2*dh),1e-8);
 close(t.gradLogvar,(A.vaeTerms(mu,v+dh,w,2,beta).total-A.vaeTerms(mu,v-dh,w,2,beta).total)/(2*dh),1e-8);
}
close(A.vaeTerms(0,0,1,2,1).kl,0);
const va=M.compute('vae',{epsilon:-2}).data,vb=M.compute('vae',{epsilon:2}).data;close(va.total,vb.total);assert.notEqual(va.z,vb.z);checks++;
// Exact GAN gradient checks include zero and reversed discriminator slopes and severe logits.
for(const theta of [-4,0,4])for(const weight of [-4,0,4])for(const bias of [-4,0,4]){
 const g=A.ganTerms(theta,weight,bias),left=A.ganTerms(theta-dh,weight,bias),right=A.ganTerms(theta+dh,weight,bias);
 close(g.gradSaturating,(right.saturating-left.saturating)/(2*dh),1e-8);close(g.gradNonSaturating,(right.nonSaturating-left.nonSaturating)/(2*dh),1e-8);
}
// Invertibility, derivative, density mass. Simpson integration is independent of plotted trapezoids.
for(const scale of [.25,1,3])for(const warp of [-.8,0,2]){
 const p={scale,warp,shift:2};for(const z of [-8,-3,0,1,8]){const y=A.flowTransform(z,p);close(A.flowInverse(y,p),z);assert.ok(A.flowDerivative(z,p)>0);checks++;close(A.flowDerivative(z,p),(A.flowTransform(z+dh,p)-A.flowTransform(z-dh,p))/(2*dh),1e-8);}
 const lo=A.flowTransform(-7,p),hi=A.flowTransform(7,p),n=6000,h=(hi-lo)/n;let total=0;
 for(let k=0;k<=n;k++){const y=lo+k*h,z=A.flowInverse(y,p),pdf=M.normalPDF(z)/A.flowDerivative(z,p);total+=(k===0||k===n?1:k%2?4:2)*pdf;}
 close(total*h/3,1,2e-8);
}
// Gaussian conditional formulas verified through covariance identities and the score identity.
for(const alpha of [.01,.5,.99])for(const sigma of [.25,1,2])for(const observed of [-5,0,5]){
 const d=A.gaussianConditional(alpha,2,sigma,observed),cov=Math.sqrt(alpha)*sigma*sigma;
 close(d.gain,cov/d.marginalVariance);close(d.posteriorVariance,sigma*sigma-cov*cov/d.marginalVariance);
 close(d.noiseMean,-Math.sqrt(1-alpha)*d.score);assert.ok(d.posteriorVariance>0&&d.posteriorVariance<=sigma*sigma);checks++;
}
// MDP Bellman fixed point, residual error certificate and direct-policy value equation.
for(const gamma of [0,.9,.99])for(const success of [.1,.5,1])for(const cost of [-1,-.2,1]){
 const p={gamma,success,cost},d=M.compute('rl-bellman',{...p,iterations:50}).data;
 close(d.exact[0],Math.max(1,cost+gamma*(3*success+(1-success)*d.exact[0])));
 assert.ok(d.error<=d.residual/(1-gamma)+1e-11);checks++;
 for(const theta of [-5,0,5]){
  const q=A.sigmoid(theta),t=A.mdpPolicy(q,p),grad=t.dq*q*(1-q),fd=(A.mdpPolicy(A.sigmoid(theta+dh),p).value-A.mdpPolicy(A.sigmoid(theta-dh),p).value)/(2*dh);
  close(t.value,(1-q)*1+q*(cost+gamma*(3*success+(1-success)*t.value)));close(grad,fd,2e-8);close(grad,t.visits*q*(1-q)*(t.Qrisky-t.Qsafe));
 }
}
// Real random features: explicit design, full/ridge normal equations, interpolation, minimum norm.
for(const n of [12,24,40])for(const width of [2,10,n-1,80])for(const lambda of [0,.2]){
 const d=A.randomFeatureData(n,.3,42),f=A.fitFeatures(d,width,lambda),X=d.train.X.map(row=>[1,...row.slice(0,width)]);
 assert.ok(f.coefficients);checks++;
 for(let j=0;j<=width;j++){
  const grad=X.reduce((acc,row,i)=>acc+row[j]*(dot(row,f.coefficients)-d.train.y[i]),0)+(j?lambda*f.coefficients[j]:0);
  close(grad,0,2e-7);
 }
 if(width+1>=n&&lambda===0){close(f.trainMSE,0,1e-14);
  // If coefficients are in the row space, they are orthogonal to every null-space direction.
  const transpose=X[0].map((_,j)=>X.map(row=>row[j])),a=M.leastSquares(transpose,f.coefficients,0).coefficients;
  assert.ok(a);checks++;for(let j=0;j<=width;j++)close(dot(transpose[j],a),f.coefficients[j],2e-7);
 }
}
const noNoise=A.fitFeatures(A.randomFeatureData(24,0,42),10,0);close(noNoise.trainMSE,0,1e-20);close(noNoise.truthMSE,0,1e-20);
const widths=M.compute('double-descent',{width:2}).data,wide=M.compute('double-descent',{width:80}).data;assert.deepEqual(widths.train,wide.train);assert.deepEqual(widths.validation,wide.validation);checks+=2;
// Bayes/confusion boundaries: no fabricated PPV at zero prediction mass.
for(const base of [.05,.2,.6,.95])for(const tpr of [0,.8,1])for(const fpr of [0,.2,1]){
 const d=A.confusion(base,tpr,fpr);close(d.tp+d.fp+d.tn+d.fn,1);close(d.tp/(d.tp+d.fn),tpr);close(d.fp/(d.fp+d.tn),fpr);
 if(d.selection===0){assert.equal(d.precision,null);checks++;}else close(d.precision,base*tpr/(base*tpr+(1-base)*fpr));
}
for(const [id,p]of [['gnn',{layers:1.2}],['double-descent',{seed:0}],['positional',{base:50}],['flow',{warp:-1}],['diffusion',{alpha:0}],['rl-policy',{gamma:1}]]){assert.throws(()=>M.compute(id,p));checks++;}
console.log(`PASS advanced: ${cases} default/preset/boundary cases; ${checks} numerical, content and plotted-data assertions.`);
