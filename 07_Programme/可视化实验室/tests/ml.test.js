'use strict';
const assert=require('node:assert/strict');
const M=require('../assets/math.js');
const A=require('../assets/ml-math.js');
const ids=['knn','logistic','svm','tree','ensemble','boosting','kmeans','gmm','kde','modelselection','metrics','likelihood'];
let checks=0;
const ok=(v,msg)=>{assert.ok(v,msg);checks++;};
const close=(a,b,tol=1e-9,msg='')=>ok(Number.isFinite(a)&&Math.abs(a-b)<=tol,`${msg}: ${a} vs ${b}`);
const approx=(a,b,r=.03)=>close(a,b,Math.abs(b)*r);
function checkPlot(plot){for(const s of plot.series){for(const point of s.points){if(point===null)continue;ok(point.every(Number.isFinite),'finite plot point');const pairs=s.type==='segments'?[[point[0],point[1]],[point[2],point[3]]]:[[point[0],point[1]]];for(const [x,y]of pairs){if(plot.xDomain)ok(x>=plot.xDomain[0]-1e-9&&x<=plot.xDomain[1]+1e-9,'x within domain');if(plot.yDomain)ok(y>=plot.yDomain[0]-1e-9&&y<=plot.yDomain[1]+1e-9,'y within domain');}if(s.type==='heatmap')ok(point[2]>=s.zDomain[0]-1e-9&&point[2]<=s.zDomain[1]+1e-9,'heat z domain');}}}
for(const id of ids){const a=M.compute(id),b=M.compute(id);assert.deepEqual(a,b);checks++;a.plots.forEach(checkPlot);for(const [key,s]of Object.entries(M.schemas[id])){const choices=Array.isArray(s[1])?s[1]:[s[1],s[2]];for(const v of choices){const r=M.compute(id,{[key]:v});r.plots.forEach(checkPlot);ok(r.metrics.every(m=>typeof m.value!=='number'||Number.isFinite(m.value)),'finite metrics '+id);}}}
// KNN: explicit distance and deterministic tie conventions.
const points=[{x:[0,0],y:0},{x:[1,0],y:1},{x:[0,2],y:1}];
close(A.knnPredict(points,[0,0],1).probability,0);
close(A.knnPredict(points,[0,0],3).probability,2/3);
assert.equal(A.knnPredict(points,[0,0],2,.1).neighbors[1].i,2);checks++;
// Stable logistic BCE: constant probability, no overflow at large logits.
close(A.sigmoid(0),.5);close(A.softplus(0),Math.log(2));close(A.softplus(1000),1000);
const logistic=M.compute('logistic',{weight:0,intercept:0});close(logistic.metrics[0].value,Math.log(2));
assert.deepEqual(M.compute('logistic',{threshold:.2}).data.probabilities,M.compute('logistic',{threshold:.8}).data.probabilities);checks++;
// SVM: direct objective and zero normal boundary.
const svm=M.compute('svm',{norm:0,offset:0,C:2});close(svm.data.objective,96);ok(svm.data.slack.every(v=>v===1));
const svmC1=M.compute('svm',{C:1}),svmC2=M.compute('svm',{C:2});close(svmC2.data.objective-svmC1.data.objective,svmC1.data.slack.reduce((a,b)=>a+b,0));
// CART: exact pure split, minimum leaf and no-positive-gain XOR stopping.
const separable=[{x:[-2,0],y:0},{x:[-1,0],y:0},{x:[1,0],y:1},{x:[2,0],y:1}];const fit=A.cart(separable,1,1);close(fit.tree.threshold,0);close(fit.tree.gain,.5);ok(fit.leaves.length===2&&fit.leaves.every(l=>l.impurity===0));
const balancedXor=[{x:[-1,-1],y:1},{x:[-1,1],y:0},{x:[1,-1],y:0},{x:[1,1],y:1}];ok(A.cart(balancedXor,4,1).leaves.length===1,'greedy XOR stall');
for(const n of [30,100,150]){const symmetricTree=M.compute('tree',{dataset:'balancedxor',n,depth:5,noise:.3});ok(symmetricTree.data.leaves.length===1,'balanced mode stalls independent of requested depth/noise');ok(symmetricTree.data.observations.length===4*Math.floor(n/4));ok(symmetricTree.data.rootCandidates.every(c=>Math.abs(c.gain)<1e-12));}
ok(M.compute('tree',{depth:0}).data.leaves.length===1);ok(M.compute('tree',{minleaf:12,depth:5}).data.leaves.every(l=>l.n>=12));
// Ensemble: equicorrelation identities and exact common-noise invariance.
const ens=M.compute('ensemble',{members:100,rho:0,reps:1000});close(ens.data.theoreticalVariance.at(-1)[1],.01);approx(ens.metrics[0].value,.01,.12);
const common=M.compute('ensemble',{members:100,rho:1,reps:1000});for(const [_,v]of common.data.empiricalVariance)close(v,common.data.empiricalVariance[0][1],1e-12);
const ens1=M.compute('ensemble',{members:1,rho:.4,reps:1000,bias:0});approx(ens1.metrics[0].value,1,.15);
// Squared-error boosting: actual stagewise minimization and held-out data independence.
const boost=M.compute('boosting',{steps:40,rate:1});for(let i=1;i<boost.data.history.length;i++)ok(boost.data.history[i].trainMSE<=boost.data.history[i-1].trainMSE+1e-12,'boosting training monotonic');
const boost0=M.compute('boosting',{steps:0});ok(boost0.data.models.length===0);close(boost0.data.base,M.mean(boost0.data.train.map(v=>v.y)));assert.deepEqual(boost.data.train,boost0.data.train);checks++;
// Lloyd: known optimum on simple clusters, monotonic objective, empty-cluster policy.
const ll=A.lloyd([[0,0],[0,2],[10,0],[10,2]],[[0,0],[10,0]],3);close(ll.centers[0][1],1);close(ll.centers[1][1],1);close(ll.history.at(-1).sse,4);
const km=M.compute('kmeans',{steps:15,k:5});for(let i=1;i<km.data.history.length;i++)ok(km.data.history[i].sse<=km.data.history[i-1].sse+1e-10,'Lloyd SSE nonincrease');
const empty=A.lloyd([[0,0],[1,0]],[[0,0],[0,0],[20,20]],1);ok(empty.emptyEvents.length>=1);
// EM: log-likelihood ascent; fixed-variance Gaussian mixture and symmetry trap.
for(const sigma of [.3,.7,1.5]){const g=M.compute('gmm',{steps:20,sigma});for(let i=1;i<g.data.history.length;i++)ok(g.data.history[i].logLikelihood>=g.data.history[i-1].logLikelihood-1e-9,'EM ascent');ok(g.data.responsibilities.every(v=>v>=0&&v<=1));}
const symmetric=M.compute('gmm',{initial:0,steps:10});close(symmetric.data.means[0],symmetric.data.means[1],1e-12);close(symmetric.data.weight,.5,1e-12);
close(A.mixtureLog(0,[0,0],.3,1),-Math.log(Math.sqrt(2*Math.PI)));
// KDE: normalization on dense grid, bandwidth change must preserve observations.
const kde=M.compute('kde',{bandwidth:.5,n:200});close(kde.data.visibleIntegral,1,2e-4);assert.deepEqual(kde.data.observations,M.compute('kde',{bandwidth:.05,n:200}).data.observations);checks++;
const kernelPanel=kde.plots[1],partial=kernelPanel.series.at(-1).points;for(let i=0;i<partial.length;i++)close(partial[i][1],kernelPanel.series.slice(0,5).reduce((a,s)=>a+s.points[i][1],0),1e-12,'kernel superposition');
// Selection: same-risk candidates, exact independent risk, prefix preservation.
const selection=M.compute('modelselection',{candidates:80,reps:100});close(selection.data.trueRisk,2);ok(selection.data.averageValidation<selection.data.trueRisk-.3,'selection optimism visible');approx(selection.data.averageTest,2,.07);
const fewer=M.compute('modelselection',{candidates:2,reps:100});assert.deepEqual(selection.data.firstValidationScores.slice(0,2),fewer.data.firstValidationScores);checks++;for(let i=0;i<100;i++)ok(selection.data.runs[i].validation<=fewer.data.runs[i].validation,'same draw minimum over superset');
// ROC/AP: analytic tiny cases, tie grouping, endpoint precision distinction.
const perfect=A.binaryCurves([0,0,1,1],[.1,.2,.8,.9]);close(perfect.auc,1);close(perfect.ap,1);
const reversed=A.binaryCurves([0,0,1,1],[.9,.8,.2,.1]);close(reversed.auc,0);close(reversed.ap,(1/3+2/4)/2);
const tied=A.binaryCurves([0,1],[.5,.5]);close(tied.auc,.5);close(tied.ap,.5);
const none=M.compute('metrics',{threshold:1});assert.equal(none.data.precision,null);close(none.data.recall,0);ok(none.data.tp+none.data.tn+none.data.fp+none.data.fn===200);
close(M.compute('metrics',{threshold:.1}).data.auc,M.compute('metrics',{threshold:.9}).data.auc);
// Likelihood: boundary MLEs, candidate contradictions, derivative identity and Gaussian MLE.
close(A.bernoulliLog(0,0,10),0);close(A.bernoulliLog(1,10,10),0);assert.equal(A.bernoulliLog(0,1,10),-Infinity);checks++;
const lik=M.compute('likelihood');close(lik.data.bernoulliMLE,.7);close(lik.data.normalMLE,M.mean(lik.data.normalObservations));
const all=M.compute('likelihood',{observedRate:1,candidateP:0});close(all.data.bernoulliMLE,1);ok(all.data.bernoulliLogLikelihood===null);close(all.data.bernoulliRelativeLikelihood.at(-1)[1],1);
const teaching=require('../assets/ml-content.js');for(const id of ids){const defaults=Object.fromEntries(teaching[id].controls.map(c=>[c.key,c.value]));assert.deepEqual(Object.keys(defaults).sort(),Object.keys(M.schemas[id]).sort());checks++;for(const preset of [{params:{}},...teaching[id].presets]){const config={...defaults,...preset.params};assert.deepEqual(M.validate(id,config),config);M.compute(id,config).plots.forEach(checkPlot);checks++;}}
console.log(JSON.stringify({suite:'classical ML independent invariants and control boundaries',models:ids.length,checks,status:'passed'}));
