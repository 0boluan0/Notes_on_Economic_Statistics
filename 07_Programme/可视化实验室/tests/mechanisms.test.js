const assert=require('node:assert/strict'),M=require('../assets/math.js');
for(const name of ['dl','foundations','ml'])require('../assets/'+name+'-math.js');
require('../assets/story-mechanisms.js');
const close=(a,b)=>assert(Math.abs(a-b)<1e-10*Math.max(1,Math.abs(b)),`${a} != ${b}`);
const scene=(id,p)=>{const r=M.compute(id,p);return [LabMechanisms.scene(id,r.params,r),r];};
for(const padding of [0,1])for(const stride of [1,2])for(const kernel of ['edge','blur','laplace'])for(const row of [0,2])for(const col of [0,2]){
 const [s,r]=scene('convolution',{padding,stride,kernel,row,col});close(s.result,r.data.output[row][col]);
 for(const t of s.terms){for(const [pi,i,j]of t.cells)assert(s.panels[pi].values[i][j]!==undefined);const [a,b]=t.cells;close(s.panels[a[0]].values[a[1]][a[2]]*s.panels[b[0]].values[b[1]][b[2]],t.value);}
}
for(const mode of ['max','mean']){const [s,r]=scene('pooling',{pattern:'tied',mode});close(s.result,1);const gradients=r.data.gradient.flat();close(gradients.reduce((a,b)=>a+b),1);assert.equal(gradients.filter(v=>v>0).length,mode==='max'?1:4);}
for(const mode of ['batch','layer']){const [s,r]=scene('normalization',{mode,gamma:0,beta:1});assert(s.panels[2].values.flat().every(v=>v===1));assert.equal(s.panels[0].groupAxis,mode==='batch'?'column':'row');}
for(const shift of [-100,0,100]){const [s,r]=scene('softmax',{shift,temperature:.25});close(s.result,1);const total=s.panels[1].values.flat().reduce((a,b)=>a+b);s.panels[2].values.forEach(([p],i)=>close(p,s.panels[1].values[i][0]/total));}
assert.match(scene('softmax',{z1:-4,z2:4,temperature:.25})[0].terms[0].expression,/e-\d+/,'small positive probabilities must not be printed as exact zero');
assert.equal(scene('entropy',{q1:0,p1:.5})[0].result,null);
for(const dropout of [0,.2,.8]){const [s,r]=scene('regularization',{dropout});close(s.panels[2].values.flat().reduce((a,b)=>a+b),r.data.outputs[0]);}
for(const id of ['regularization','softmax','neuron']){const r=M.compute(id);assert.equal(LabMechanisms.scene(id,r.params,r,1),null,'retain the distribution or derivative graph when it is the teaching target');}
for(const w of [-2,0,2]){const r=M.compute('backprop',{w}),s=LabMechanisms.scene('backprop',r.params,r,1);assert.equal(s.accumulate,false);for(const t of s.terms){const [a,b]=t.cells;close(s.panels[a[0]].values[a[1]][a[2]]*s.panels[b[0]].values[b[1]][b[2]],t.value);}close(s.result,r.data.gradients.w);}
for(const causal of [0,1])for(const query of [0,3]){const r=M.compute('attention',{causal,query}),s=LabMechanisms.scene('attention',r.params,r,2);for(let j=0;j<4;j++)for(let k=0;k<2;k++)close(s.panels[2].values[j][k],r.data.weights[query][j]*r.data.V[j][k]);}
console.log('PASS: linked scene arithmetic, traced cell identities, pooling ties, normalization groups, stable softmax, support mismatch, dropout draw');
