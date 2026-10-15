'use strict';
const assert=require('node:assert/strict');
require('../assets/math.js');require('../assets/dl-math.js');require('../assets/story-backprop-00-model.js');require('../assets/story-backprop-02-film.js');require('../assets/story-backprop-03-causal.js');
const C=globalThis.BackpropCausal,B=globalThis.BackpropLesson,F=globalThis.BackpropFilm;
const loss=p=>.5*(p.v*Math.tanh(p.x*p.w+p.bias)-p.target)**2;
let checks=0;
for(const p of [C.base,{...C.base,w:2},{...C.base,v:0},{...C.base,v:-2},{...C.base,x:0},{...C.base,w:3},{...C.base,w:-3}]){
 const m=C.model(p),product=m.gains.reduce((a,b)=>a*b,1),h=1e-5,numerical=(loss({...p,w:p.w+h})-loss({...p,w:p.w-h}))/(2*h);
 assert(Math.abs(product-numerical)<1e-6);assert(Math.abs(m.grads[0]-numerical)<1e-6);checks+=2;
 for(let mode=0;mode<4;mode++)for(let step=0;step<=4;step++){
  const svg=C.svg({params:p,mode,progress:step/4});assert(!svg.includes('NaN'));assert(!svg.includes('Infinity'));assert(svg.includes('</svg>'));checks++;
 }
}
const coarse=B.probe(C.base,'w',.05),fine=B.probe(C.base,'w',.005);
assert(coarse.actual<0&&coarse.predicted<0);assert(Math.abs(fine.actual/fine.predicted-1)<Math.abs(coarse.actual/coarse.predicted-1));
assert(C.model({...C.base,w:2}).gains[1]<.002);assert(C.model({...C.base,v:0}).grads[0]===0);
assert(B.update(C.base,.1).after.data.forward.loss<loss(C.base));checks+=5;
for(const f of F.frames){assert(f.narration.length>25);assert(f.duration>0);assert(f.from>=0&&f.to<=1);assert(!C.svg({...f,progress:f.to}).includes('NaN'));checks++;}
assert.equal(Math.round(F.duration*24),F.frames.reduce((n,f)=>n+Math.round(f.duration*24),0));
console.log(`PASS: ${checks} checks: independent chain differences, saturation, zero path, approximation, boundary scenes and narrated timeline.`);
