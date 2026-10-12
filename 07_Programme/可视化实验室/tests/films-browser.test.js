/* Tests execute within this owned fixture, through the same controls a learner uses. */
(()=>{
const frame=document.getElementById('subject'),report=document.getElementById('report'),run=document.getElementById('run');
const wait=ms=>new Promise(r=>setTimeout(r,ms));
run.onclick=async()=>{
run.disabled=true;report.textContent='正在检查…';const failures=[];let checks=0;
const assert=(ok,msg)=>{checks++;if(!ok)failures.push(msg);};
try{
const w=frame.contentWindow,d=w.document,ids=Object.keys(w.LabContent);assert(ids.length===60,'60 experiments');
for(const id of ids){
 w.VisualLab.navigate(id);await wait(30);const panel=d.querySelector('.film-examples');assert(panel,id+' film controls');panel.open=true;
 const examples=w.LabFilmExamples[id];
 if(id!=='bootstrap'){
  const snapshots=examples.map((s,i)=>({s,i})).filter(({s})=>s.single);
  assert(snapshots.length===1,id+' has one exact film snapshot');
  const i=snapshots[0].i;
  assert(panel.querySelectorAll('[data-film-example="'+i+'"]').length===1,id+' snapshot has one load button');
 }
 for(const b of panel.querySelectorAll('button')){
  b.click();await wait(35);const sample=examples[Number(b.dataset.filmExample)],state=w.VisualLab.getState();
  assert(d.querySelector('#application').dataset.ready==='true',id+' computes');
  assert(d.querySelector('#story-title').textContent===sample.title,id+' matching teaching context');
  assert(!d.querySelector('.story-step[aria-current="step"]'),id+' film comparison does not highlight an unrelated step');
  for(const [key,value]of Object.entries(sample[b.dataset.endpoint]))assert(state.params[key]===value,id+' exact '+key);
  assert(!!d.querySelector('#hero-chart svg, .mechanism table'),id+' matching numerical plot or linked calculation');
  for(const [key,value]of Object.entries(state.params))assert(String(d.querySelector('#control-'+key).value)===String(value),id+' slider agrees '+key);
 }
 d.querySelector('[data-story-step="0"]').click();assert(d.querySelector('#story-count').textContent.includes('第 1 步'),id+' returns to normal exploration');
}
for(const id of w.LabMechanisms.ids){
 w.VisualLab.navigate(id);await wait(30);const cells=d.querySelectorAll('.mechanism td');assert(cells.length>0,id+' intermediate values are visible');
 for(const button of d.querySelectorAll('[data-mechanism-term]')){button.click();assert(button.getAttribute('aria-pressed')==='true',id+' selected term');assert(d.querySelectorAll('.is-traced').length>=2,id+' corresponding operands highlight together');assert(d.querySelector('.mechanism-equation').textContent.includes('前 '),id+' running sum is explicit');}
}
for(const width of [320,390,768,1200]){
 frame.style.width=width+'px';await wait(120);
 for(const id of ids){w.VisualLab.navigate(id);d.querySelector('.film-examples').open=true;await wait(15);assert(d.documentElement.scrollWidth<=width+2,id+' expanded comparisons fit '+width);for(const cell of d.querySelectorAll('.mechanism td'))assert(parseFloat(w.getComputedStyle(cell).fontSize)>=12,id+' calculation values remain readable at '+width);}
}
w.VisualLab.navigate('contrastive');d.querySelector('.film-examples').open=true;d.querySelector('[data-film-example="2"][data-endpoint="to"]').click();
const p=d.querySelector('#hero-chart .chart g[clip-path] path');const ys=[...p.getAttribute('d').matchAll(/[ML][\d.-]+,([\d.-]+)/g)].map(m=>Number(m[1]));assert(ys.length>5&&Math.max(...ys)-Math.min(...ys)<.001,'constant log(2) has no visible floating-point spikes');
frame.style.width='1200px';w.VisualLab.navigate('softmax');d.querySelector('.film-examples').open=true;d.querySelector('[data-film-example="0"][data-endpoint="to"]').click();
const result={passed:!failures.length,checks,experiments:ids.length,failures,build:d.querySelector('meta[name=visual-lab-build]').content};report.textContent=JSON.stringify(result,null,2);document.body.dataset.passed=String(result.passed);
}catch(e){report.textContent=e.stack;document.body.dataset.passed='false';}finally{run.disabled=false;}
};
if(location.search.includes('autorun'))frame.addEventListener('load',()=>run.click(),{once:true});
})();
