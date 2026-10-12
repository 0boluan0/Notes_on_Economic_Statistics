'use strict';
/* Independent reviewer checks; these are not copied from the implementation tests. */
const assert=require('node:assert/strict'),M=require('../assets/math.js'),A=require('../assets/advanced-math.js'),C=require('../assets/advanced-content.js');
let checks=0;
const near=(a,b,t=1e-9)=>{assert.ok(Math.abs(a-b)<t,`${a} != ${b}`);checks++;};
const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
// Joint shifts are checked at different absolute positions, not against the same formula call.
for(const pairs of [1,4,8])for(const base of [10,10000]){const a=A.positionalVector(4,pairs,base),b=A.positionalVector(11,pairs,base),c=A.positionalVector(30,pairs,base),d=A.positionalVector(37,pairs,base);near(dot(a,a),pairs);near(dot(a,b),dot(c,d));}
// A bipartite star has an exact period-two trajectory without loops or residuals.
const star=M.compute('gnn',{graph:'star',normalization:'mean',selfLoop:0,residual:0,layers:2}).data;
assert.deepEqual(star.history,[[1,0,0,0,0,0],[0,1,1,1,1,1],[1,0,0,0,0,0]]);checks++;
for(const graph of ['star','chain']){const g=M.compute('gnn',{graph,normalization:'symmetric'}).data,u=g.degree.map(Math.sqrt);g.propagation.forEach((row,i)=>near(dot(row,u),u[i]));}
// Non-unit norms with product one contradict a claimed necessary unit-length condition.
const embedding=M.compute('embeddings',{normA:2,normB:.5,angle:30}).data;near(embedding.inner,embedding.cosine);
const zero=M.compute('embeddings',{normA:0});assert.equal(zero.data.cosine,null);checks++;
// Orthogonal decomposition gives equality for gain zero and gain two pointwise.
const ac0=M.compute('autoencoder',{gain:0}).data,ac1=M.compute('autoencoder',{gain:1}).data,ac2=M.compute('autoencoder',{gain:2}).data;
near(ac0.mse,ac2.mse);near(ac0.mse-ac1.mse,M.mean(ac1.codes.map(z=>z*z)));near(ac1.optimalMSE,ac1.eigenvalues[1]*79/80);
// Central differences test both variational parameters at nine independent settings.
for(const mu of [-2,0,2])for(const logvar of [-3,0,1]){const h=1e-5,w=-1.4,x=1.7,beta=.6,r=A.vaeTerms(mu,logvar,w,x,beta);near((A.vaeTerms(mu+h,logvar,w,x,beta).total-A.vaeTerms(mu-h,logvar,w,x,beta).total)/(2*h),r.gradMu,1e-8);near((A.vaeTerms(mu,logvar+h,w,x,beta).total-A.vaeTerms(mu,logvar-h,w,x,beta).total)/(2*h),r.gradLogvar,1e-8);}
// Both GAN objectives are differentiated numerically for positive, zero and negative slopes.
for(const theta of [-3,0,3])for(const weight of [-2,0,2]){const h=1e-5,r=A.ganTerms(theta,weight,.7),a=A.ganTerms(theta+h,weight,.7),b=A.ganTerms(theta-h,weight,.7);near((a.saturating-b.saturating)/(2*h),r.gradSaturating,1e-8);near((a.nonSaturating-b.nonSaturating)/(2*h),r.gradNonSaturating,1e-8);}
// Regression checks for the two review findings: do not show a numerical angle for zero vectors,
// and do not describe unit norms as necessary for equality of cosine and inner product.
for(const params of [{normA:0},{normB:0},{normA:0,normB:0}]){const r=M.compute('embeddings',params);assert.ok(typeof r.metrics[3].value==='string'&&r.metrics[3].value.includes('未定义'));checks++;}
assert.ok(!C.embeddings.pitfalls.some(v=>v.includes('只有在单位长度')));checks++;
console.log(JSON.stringify({suite:'independent cross-review: six advanced models',reviewed:['positional','gnn','embeddings','autoencoder','vae','gan'],independentChecks:checks,status:'passed'}));
