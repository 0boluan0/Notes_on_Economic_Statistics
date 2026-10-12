/* One reproducible input per object-level lesson, shared with the interactive model. */
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'..'),out=path.resolve(process.argv[2]||'');
assert(process.argv[2]&&!out.startsWith(root+path.sep),'Supply a render directory outside the vault');
const ctx=vm.createContext({});
for(const f of ['math','ml-math','dl-math','advanced-math','foundations-math','sequential-math','content','ml-content','dl-content','advanced-content','foundations-content','sequential-content'])vm.runInContext(fs.readFileSync(path.join(root,'assets',f+'.js'),'utf8'),ctx);
const choices={linear:{a:1.2,b:.8,c:.4,d:1},normalization:{gamma:1.5,beta:.5},sampling:{n:8},confidence:{reps:30,seed:7},backprop:{x:1,w:.5,bias:0,v:2,target:1},convolution:{row:1,col:1},pooling:{row:1,col:1},gradient:{steps:12},pca:{n:40},knn:{n:20,k:5},tree:{depth:2,n:40},kmeans:{n:30,steps:4},gmm:{n:40,steps:4},kde:{n:10},initialization:{depth:6},attention:{query:2},rnn:{steps:6},lstm:{steps:6},bayes:{a:2,b:2,success:3,failure:1}};
fs.mkdirSync(out,{recursive:true});
for(const id of Object.keys(ctx.LabMath.schemas).filter(x=>x!=='bootstrap')){
 const result=ctx.LabMath.compute(id,choices[id]);
 for(const [k,v] of Object.entries(choices[id]||{}))assert.equal(result.params[k],v,id+' '+k);
 const content=ctx.LabContent[id];assert(content);
 const comparisons=id==='attention'?{causal:ctx.LabMath.compute(id,{...result.params,causal:1})}:{};
 fs.writeFileSync(path.join(out,id+'.json'),JSON.stringify({id,title:content.title,en:content.en,sources:content.sources,result,comparisons},(_,v)=>typeof v==='number'&&!Number.isFinite(v)?String(v):v));
}
console.log('Exported 59 reproducible narrative inputs; existing media and manifests untouched.');
