'use strict';
const assert=require('node:assert/strict'),M=require('../assets/math.js');
const close=(a,b,t=1e-9)=>assert.ok(Math.abs(a-b)<=t*Math.max(1,Math.abs(b)),`${a} ≠ ${b}`);
for(const x of [-1,0,1]){const r=M.compute('tangent',{x,h:.1,function:'cubic'});for(const q of r.plots[1].series[1].points){if(!q)continue;close(q[1],2*q[0],1e-6);}}
for(const n of [10,20,80]){const r=M.compute('overfit',{n,lambda:0});const c=r.data.complexity;for(let i=1;i<c.length;i++)assert.ok(c[i].train<=c[i-1].train+1e-9,'Training error cannot grow for nested unpenalized polynomial spaces');const at=c.find(q=>q.degree===r.params.degree);close(at.train,r.data.trainMSE);close(at.validation,r.data.validationMSE);}
for(const observed of [-1.5,-.3,0,.3,1.5]){const r=M.compute('testing',{observed});const tails=r.plots[1].series[0];assert.ok(tails.fill);for(const q of tails.points){if(!q||q[1]===0)continue;assert.ok(Math.abs(q[0])>=Math.abs(observed)-1e-10,'P-value shading must lie in observed or more extreme tails');close(q[1],M.normalPDF(q[0]/r.data.se)/r.data.se);}}
const a=M.compute('projection');assert.equal(a.params.angle,60);close(a.data.orthogonality,0);assert.equal(a.plots[0].series.filter(s=>s.arrow).length,3);
console.log('PASS: revised error slopes, nested fits, p-value tails and vector geometry.');
