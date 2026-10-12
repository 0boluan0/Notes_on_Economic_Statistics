/* Assemble current narrative metadata and the existing optional interactive comparisons. */
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'..'),ctx=vm.createContext({});
for(const f of ['math','ml-math','dl-math','advanced-math','foundations-math','sequential-math','content','ml-content','dl-content','advanced-content','foundations-content','sequential-content','guides-core','guides-ml','guides-dl','guides-advanced'])vm.runInContext(fs.readFileSync(path.join(root,'assets',f+'.js'),'utf8'),ctx);
const specs=require('./film-specs.cjs'),catalog={},examples={};
for(const [id,steps]of Object.entries(specs)){
 const c=ctx.LabContent[id],timeline=JSON.parse(fs.readFileSync(path.join(__dirname,id+'-timeline.json'),'utf8'));
 assert.equal(timeline.method,'object-narrative',id+' must have a current narrative before assembly');
 assert(timeline.chapters.length>=3&&timeline.checkpoints.length>=4&&!timeline.layout.length,id+' incomplete review record');
 assert(fs.existsSync(path.join(root,'media',id+'.mp4')),id);
 const canonical=ctx.LabMath.validate(id,timeline.params);assert.deepEqual(JSON.parse(JSON.stringify(canonical)),timeline.params,id+' exact film parameters');
 catalog[id]={title:c.title,en:c.en,method:timeline.method,params:timeline.params,chapters:timeline.chapters,checkpoints:timeline.checkpoints,sources:c.sources};
 examples[id]=steps.map((s,i)=>{
  const g=ctx.LabGuides[id]?.steps[i]||{};const from={...ctx.LabMath.validate(id),...g.params,...s.from},to={...from,...s.to};
  for(const params of [from,to])for(const [key,v]of Object.entries(params))assert.equal(ctx.LabMath.validate(id,params)[key],v,id+' parameter '+key);
  return {title:s.title||g.title,caption:s.caption||g.caption,plot:s.plot??g.plot??0,metrics:s.metrics??g.metrics??[0,1],focus:Object.keys(to).filter(k=>from[k]!==to[k]),from,to,action:'载入两组条件进行延伸对照；也可以只改变其中一个参数。'};
 });
 examples[id].push({title:'载入本片的模型与数据',caption:'核对影片里的对象与数值，再一次只改变一个条件。动画中的逐步运算使用这份模型与数据。',plot:0,metrics:[0,1],focus:examples[id][0].focus,from:timeline.params,to:timeline.params,single:true,action:'先核对一个具体对象，再预测改变参数后的结果。'});
}
assert.equal(Object.keys(catalog).length,59);
examples.bootstrap=[{title:'影片中的八个观测',caption:'这份样本使用指数总体、n=8、B=500、seed=42。先检查原样本，再回到“有放回地重新抽取”查看真实抽样身份。',plot:1,metrics:[0,1],focus:['n','reps'],from:{n:8,reps:500,distribution:'exponential',seed:42},to:{n:8,reps:2000,distribution:'exponential',seed:42},action:'只增加 B，核对原样本均值不变，再观察 Bootstrap 分布的数值稳定性。'}];
fs.writeFileSync(path.join(__dirname,'film-manifest.json'),JSON.stringify(catalog,null,2)+'\n');
fs.writeFileSync(path.join(root,'assets/story-film-examples.js'),'/* Current film inputs and further comparisons; video bytes stay in native MP4 files. */\nglobalThis.LabFilmExamples='+JSON.stringify(examples)+';\n');
console.log('Assembled 59 narratives, 59 exact model snapshots and 177 further comparisons; Bootstrap retained.');
