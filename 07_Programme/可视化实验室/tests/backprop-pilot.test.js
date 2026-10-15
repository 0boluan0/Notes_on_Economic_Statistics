'use strict';
const assert=require('node:assert/strict');
const M=require('../assets/math.js');require('../assets/dl-math.js');
const B=require('../assets/story-backprop-00-model.js');
const loss=p=>.5*(p.v*Math.tanh(p.w*p.x+p.bias)-p.target)**2;
const near=(a,b,t=1e-7)=>assert.ok(Math.abs(a-b)<=t,`${a} ≠ ${b}`);
let checks=0;
for(const p of [
 {x:1,w:.5,bias:0,v:2,target:1}, {x:1,w:0,bias:0,v:2,target:1},
 {x:1,w:1,bias:0,v:0,target:1}, {x:0,w:1,bias:.5,v:1,target:1},
 {x:-1.3,w:-.7,bias:.4,v:-1.2,target:.3}, {x:2,w:3,bias:0,v:1,target:0}
]){
 const snapshot=JSON.stringify(p),r=B.evaluate(p),h=1e-5;
 near(r.data.forward.loss,loss(p));
 for(const key of B.parameters){near(r.data.gradients[key],(loss({...p,[key]:p[key]+h})-loss({...p,[key]:p[key]-h}))/(2*h));checks++;}
 const update=B.update(p,.01);
 if(update.allowed){for(const item of update.deltas)near(item.next,p[item.key]-.01*r.data.gradients[item.key]);near(update.after.data.forward.loss,loss(update.next));}
 assert.equal(JSON.stringify(p),snapshot);checks++;
}
const zero=B.evaluate({x:1,w:1,bias:0,v:0,target:1});
near(zero.data.gradients.w,0);near(zero.data.gradients.bias,0);near(zero.data.gradients.v,-Math.tanh(1));
const start={x:1,w:0,bias:0,v:2,target:.2};
assert.ok(B.update(start,.05).after.data.forward.loss<loss(start));
assert.ok(B.update(start,1).after.data.forward.loss>loss(start));
const p={x:1,w:.5,bias:.2,v:2,target:1};
const coarse=B.probe(p,'w',.1),fine=B.probe(p,'w',.01);
assert.ok(Math.abs(fine.actual-fine.predicted)<Math.abs(coarse.actual-coarse.predicted));
const branch=B.branches(p,1,-1);near(branch.dh,0);assert.ok(branch.viaA!==0);
const q=B.branches(p,1,.6),f=h=>.5*(1.6*h-p.target)**2;
near(q.dh,(f(q.h+1e-6)-f(q.h-1e-6))/2e-6);
assert.throws(()=>B.probe(p,'target',.1));assert.throws(()=>B.probe({...p,w:3},'w',.1));assert.throws(()=>B.update(p,0));
assert.equal(B.update({x:2,w:2,bias:2,v:2,target:-2},2).allowed,false);
const transfer=B.evaluate({x:2,w:0,bias:0,v:1,target:-.5});near(transfer.data.gradients.w,1);
console.log(`PASS: ${checks} parameter/finite-difference checks; simultaneous update, zero branches, overshoot, approximation, boundary rejection and transfer example.`);
