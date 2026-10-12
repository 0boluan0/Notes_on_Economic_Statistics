/* Analytical invariants and output-contract checks. Run: node tests/core.test.js */
'use strict';
const assert=require('node:assert/strict');
const M=require('../assets/math.js');
let checks=0;
function close(actual,expected,tol=1e-10){checks++;assert.ok(Number.isFinite(actual)&&Math.abs(actual-expected)<=tol*Math.max(1,Math.abs(expected)),`${actual} != ${expected} (tol ${tol})`);}
function checkOutput(id,params={}){
  const out=M.compute(id,params);assert.ok(out.metrics.length&&out.plots.length&&out.data&&out.params);checks++;
  for(const metric of out.metrics){assert.ok(typeof metric.value==='string'||Number.isFinite(metric.value),id+' metric '+metric.label);checks++;}
  for(const plot of out.plots){assert.ok(plot.title&&plot.xLabel&&plot.yLabel);for(const series of plot.series){assert.ok(series.label);for(const point of series.points){if(point===null){assert.equal(series.type,'line');continue;}assert.ok(point.every(Number.isFinite),id+' nonfinite plotted data');checks++;
    for(let k=0;k<(series.type==='segments'?4:2);k++){const domain=k%2?plot.yDomain:plot.xDomain;if(domain)assert.ok(point[k]>=domain[0]-1e-12&&point[k]<=domain[1]+1e-12,id+' domain excludes plotted point');}
  }}}
  return out;
}
for(const id of Object.keys(M.schemas)){
  const a=checkOutput(id),b=M.compute(id);assert.deepEqual(a,b);checks++;
  for(const [key,schema] of Object.entries(M.schemas[id])){
    const values=Array.isArray(schema[1])?schema[1]:[schema[1],schema[2]];
    for(const value of values)checkOutput(id,{[key]:value});
  }
}
// Animated examples must be actual rows from the same calculation, including duplicates.
for(const distribution of ['normal','exponential'])for(const n of [5,8,120]){
  const s=M.compute('sampling',{n,distribution,reps:100}).data;
  assert.equal(s.sampleExamples.length,6);
  for(const [i,row] of s.sampleExamples.entries()){assert.equal(row.length,n);close(row.reduce((a,b)=>a+b,0)/n,s.means[i]);}
  const b=M.compute('bootstrap',{n,distribution,reps:100}).data;
  assert.equal(b.resampleExamples.length,6);
  for(const [i,indices] of b.resampleExamples.entries()){
    assert.equal(indices.length,n);assert.ok(indices.every(j=>Number.isInteger(j)&&j>=0&&j<n));
    close(indices.reduce((total,j)=>total+b.sample[j],0)/n,b.bootstrapMeans[i]);
  }
  const more=M.compute('bootstrap',{n,distribution,reps:500}).data;
  assert.deepEqual(more.sample,b.sample);assert.deepEqual(more.resampleExamples,b.resampleExamples);
  assert.deepEqual(more.bootstrapMeans.slice(0,100),b.bootstrapMeans);
}
// Comparison curves cover the full fixed window, independently of simulated extrema.
for(const distribution of ['normal','exponential','cauchy']){
  const d=M.compute('sampling',{distribution,n:1}).data;
  assert.equal(d.densityCurve[0][0],d.densityDomain[0]);assert.equal(d.densityCurve.at(-1)[0],d.densityDomain[1]);
  for(const [x,y] of d.densityCurve)close(y,distribution==='normal'?Math.exp(-x*x/2)/Math.sqrt(2*Math.PI):distribution==='exponential'?Math.exp(-x):1/(Math.PI*(1+x*x)));
  assert.deepEqual(d.densityCurve,M.compute('sampling',{distribution,n:1,reps:2000,seed:17}).data.densityCurve);
}
assert.deepEqual(M.compute('sampling',{distribution:'cauchy',n:3}).data.densityCurve,M.compute('sampling',{distribution:'cauchy',n:100}).data.densityCurve);
// Validation rejects invalid values; it does not silently clip or coerce configuration.
for(const [id,p] of [['sampling',{n:0}],['sampling',{n:1.5}],['sampling',{seed:0}],['sampling',{seed:NaN}],['testing',{alpha:.03}],['linear',{a:Infinity}],['pca',{standardize:'1'}]]){
  assert.throws(()=>M.compute(id,p));checks++;
}
assert.throws(()=>M.compute('missing'));checks++;
// Independently tabulated Gaussian probabilities and quantiles, plus symmetry/tails.
for(const [x,p] of [[0,.5],[1,.8413447460685429],[-1,.15865525393145705],[1.96,.9750021048517795],[-5,2.866515718791933e-7],[-8,6.220960574271784e-16]])close(M.normalCDF(x),p,2e-14);
close(M.normalQuantile(.975),1.959963984540054,2e-14);
for(const p of [1e-10,1e-6,.01,.1,.5,.9,.99,1-1e-10])close(M.normalCDF(M.normalQuantile(p)),p,2e-14);
close(M.betaCDF(.3,1,1),.3);close(M.betaCDF(.3,2,1),.09);close(M.betaCDF(.3,1,2),.51);
close(M.betaQuantile(.95,1,4),1-Math.pow(.05,.25));
for(const [a,b] of [[1,1],[1,65],[65,1],[65,65],[9,5]])for(const p of [.001,.025,.5,.975,.999])close(M.betaCDF(M.betaQuantile(p,a,b),a,b),p,2e-12);
// Tangent identities are exact algebra for the cubic.
for(const x of [-2,0,1.5])for(const h of [.001,.25,1]){const d=M.compute('tangent',{x,h}).data;close(d.central,3*x*x-1+h*h,2e-11);close(d.forward,3*x*x-1+3*x*h+h*h,2e-11);}
const singular=M.compute('linear',{a:1,b:2,c:1,d:2});close(singular.data.determinant,0);assert.ok(singular.metrics[2].value.includes('奇异'));checks++;
for(const angle of [0,30,90,175])for(const v of [[0,0],[2,1],[-3,3]]){const d=M.compute('projection',{angle,vx:v[0],vy:v[1]}).data;close(d.orthogonality,0);close(d.norm2,d.projectionNorm2+d.residualNorm2);}
const g=M.compute('gradient',{rate:.2,condition:4,steps:30}).data;for(const v of g.path){close(v.x,3*.8**v.step);close(v.y,2*.2**v.step);}
const boundary=M.compute('gradient',{rate:.5,condition:4,x0:0,y0:2}).data;close(boundary.path.at(-1).loss,8);assert.ok(boundary.status.includes('边界'));checks++;
const diverging=checkOutput('gradient',{rate:1.2,condition:20,steps:60,x0:4,y0:4}).data;assert.ok(diverging.path.at(-1).loss>1e100&&diverging.visibleCount<diverging.path.length);checks++;
const stationary=M.compute('gradient',{rate:1.2,condition:20,x0:0,y0:0}).data;close(stationary.path.at(-1).loss,0);
// Seeded means preserve their underlying sample and existing replicate prefixes.
for(const distribution of ['normal','exponential','cauchy']){
  const a=M.compute('sampling',{distribution,n:10,reps:100}).data,b=M.compute('sampling',{distribution,n:20,reps:200}).data;
  assert.deepEqual(a.samples,b.samples.slice(0,10));assert.deepEqual(a.runningMeans,b.runningMeans);close(a.means[0],M.mean(a.samples));checks+=2;
  if(distribution==='cauchy'){assert.equal(a.populationMean,null);assert.equal(a.theoreticalSE,null);checks+=2;}
}
const ci=M.compute('confidence').data;for(const v of ci.intervals){close(v.high-v.mean,ci.halfWidth);assert.equal(v.covers,v.low<=0&&v.high>=0);checks++;}
for(const alpha of [.01,.05,.1]){const t=M.compute('testing',{effect:0,alpha,observed:0}).data;close(t.power,alpha,2e-14);close(t.pValue,1);}
const prior=M.compute('bayes',{success:0,failure:0}).data;assert.deepEqual(prior.posterior,prior.prior);checks++;
// QR solves exact tall and wide systems; the wide result is the minimum norm solution.
const qr=M.leastSquares([[1,0],[1,1],[1,2]],[1,3,5]);close(qr.coefficients[0],1);close(qr.coefficients[1],2);
const wide=M.leastSquares([[1,0,1],[0,1,1]],[1,1]);wide.coefficients.forEach((v,i)=>close(v,[1/3,1/3,2/3][i]));assert.equal(wide.unique,false);checks++;
const deficient=M.leastSquares([[1,1],[2,2]],[2,4]);assert.equal(deficient.coefficients,null);checks++;
// Ridge normal-equation residual is an independent optimality criterion.
for(const lambda of [0,.3,10]){
  const d=M.compute('regression',{lambda,rho:.98}).data;
  for(let j=0;j<2;j++)close(M.mean(d.X.map((r,i)=>r[j]*d.residuals[i]))*d.X.length,lambda*d.beta[j],2e-10);
  close(M.mean(d.residuals),0);close(d.vif,1/(1-d.sampleCorrelation**2));
}
for(const [degree,n,lambda] of [[5,20,0],[15,10,0],[15,10,.001],[15,80,1]]){
  const d=checkOutput('overfit',{degree,n,lambda}).data;assert.ok(d.coefficients);checks++;
  for(let j=0;j<=degree;j++){const gradient=d.design.reduce((acc,row,i)=>acc+row[j]*(d.predictions[i]-d.train.y[i]),0)+(j?lambda*d.coefficients[j]:0);close(gradient,0,2e-8);}
  if(degree+1>n&&lambda===0)close(d.trainMSE,0,1e-16);
}
// PCA uses sample covariance (n-1), while reconstruction error averages over n rows.
for(const standardize of [0,1])for(const ratio of [1,10]){
  const d=M.compute('pca',{standardize,ratio,scale:4}).data;
  close(d.reconstructionMSE,d.eigenvalues[1]*(d.observations.length-1)/d.observations.length,2e-11);
  close(d.explainedRatio,d.eigenvalues[0]/(d.eigenvalues[0]+d.eigenvalues[1]));
  for(let k=0;k<2;k++){const v=d.eigenvectors[k];close(v[0]*v[0]+v[1]*v[1],1);for(let j=0;j<2;j++)close(d.covariance[j][0]*v[0]+d.covariance[j][1]*v[1],d.eigenvalues[k]*v[j],2e-11);}
}
assert.equal(M.eigen2([[1,0],[0,1]]).equal,true);checks++;
const bootA=M.compute('bootstrap',{reps:100}).data,bootB=M.compute('bootstrap',{reps:1000}).data;
assert.deepEqual(bootA.sample,bootB.sample);assert.deepEqual(bootA.bootstrapMeans,bootB.bootstrapMeans.slice(0,100));checks+=2;
close(bootB.conditionalExactSE,bootB.analyticSE*Math.sqrt((bootB.sample.length-1)/bootB.sample.length));
console.log(`PASS: ${checks} analytical, boundary, deterministic, and plot-contract checks across 12 labs.`);
