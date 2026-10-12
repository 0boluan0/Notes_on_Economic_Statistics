/* Runs only against this project's own page; never inspects another app. */
(() => {
  const frame=document.getElementById('subject'),report=document.getElementById('report');
  const sleep=ms=>new Promise(resolve=>setTimeout(resolve,ms));
  async function run(){
    let checks=0,failures=[],errors=[];const outcomes=[];
    const assert=(condition,message)=>{checks++;if(!condition)failures.push(message);};
    report.textContent='正在运行…';document.getElementById('run').disabled=true;
    try{
      const w=frame.contentWindow,d=w.document;
      w.addEventListener('error',e=>errors.push(e.message));
      const content=w.LabContent;
      assert(!!w.VisualLab,'App booted');assert(Object.keys(content).length===Object.keys(w.LabMath.schemas).length,'All model/content IDs agree');
      for(const [id,c] of Object.entries(content)){
        w.VisualLab.navigate(id);await sleep(80);
        assert(d.querySelector('#application').dataset.ready==='true',id+' ready');
        assert(d.querySelector('h1').textContent===c.title,id+' title');
        assert(d.querySelectorAll('#plots .chart').length===w.VisualLab.getState().result.plots.length,id+' all plots rendered');
        assert(d.querySelectorAll('.formula-error').length===0,id+' formulas render');
        assert(d.querySelectorAll('.challenge').length>=3,id+' challenges');
        assert(d.querySelectorAll('.source-list a').length>0,id+' sources');
        for(const stage of d.querySelectorAll('[data-story-step]')){stage.click();await sleep(40);assert(d.querySelector('#application').dataset.ready==='true',id+' story step '+stage.dataset.storyStep);assert(d.querySelector('#story-scene').textContent.trim().length>10,id+' substantive visible story');assert(d.querySelector('#story-title').textContent.length>1,id+' story title');for(const c of content[id].controls)assert(String(d.getElementById('control-'+c.key).value)===String(w.VisualLab.getState().params[c.key]),id+'/'+c.key+' native input matches scene parameter');}
        for(const control of c.controls){
          const el=d.getElementById('control-'+control.key);
          assert(!!el,id+'/'+control.key+' control exists');
          const values=control.type==='select'?control.options.map(o=>o.value):[control.min,control.max];
          for(const value of values){
            el.value=String(value);el.dispatchEvent(new w.Event(control.type==='select'?'change':'input',{bubbles:true}));await sleep(65);
            assert(d.querySelector('#application').dataset.ready==='true',id+'/'+control.key+'='+value+' computes');
            assert(String(w.VisualLab.getState().params[control.key])===String(value),id+'/'+control.key+'='+value+' reaches computation');
            assert(!/NaN|undefined/.test(d.querySelector('#metrics').textContent),id+'/'+control.key+' labels finite/defined');
          }
          d.getElementById('reset-params').click();await sleep(55);
        }
        for(const button of d.querySelectorAll('.preset')){button.click();await sleep(55);assert(d.querySelector('#application').dataset.ready==='true',id+' preset '+button.textContent);}
        for(const tab of d.querySelectorAll('.learning-tab')){tab.click();assert(d.getElementById(tab.getAttribute('aria-controls')).hidden===false,id+' tab '+tab.dataset.panel);assert(d.querySelectorAll('.formula-error').length===0,id+' tab formulas '+tab.dataset.panel);}
        for(const challenge of d.querySelectorAll('.challenge details')){challenge.open=true;assert(!!challenge.querySelector('.challenge-answer').textContent,id+' substantive answer');}
        d.getElementById('reset-params').click();await sleep(55);
        if(c.animation){const key=c.animation.key,old=w.VisualLab.getState().params[key];d.getElementById('next-step').click();assert(w.VisualLab.getState().params[key]>=old,id+' single step');d.getElementById('play-steps').click();await sleep(520);assert(d.getElementById('play-steps').textContent.includes('暂停'),id+' playback active');d.getElementById('reset-params').click();assert(d.getElementById('play-steps').textContent.includes('播放'),id+' reset stops playback');}
        outcomes.push({id,plots:d.querySelectorAll('#plots .chart').length,controls:c.controls.length,presets:c.presets.length});
      }
      for(const width of [320,390,768,1200]){
        frame.style.width=width+'px';await sleep(180);
        assert(d.documentElement.clientWidth===width,'Responsive check actually uses '+width+'px');
        for(const id of Object.keys(content)){
          w.VisualLab.navigate(id);await sleep(40);
          assert(d.documentElement.scrollWidth<=d.documentElement.clientWidth+2,id+' fits '+width+'px');
          for(const el of d.querySelectorAll('.control input,.control select,.chart,.ts-chart,.bp-graph,.attn-story,.sf-stage,.sf-transport')){const r=el.getBoundingClientRect();assert(r.left>=-1&&r.right<=d.documentElement.clientWidth+1,id+' control/plot within '+width+'px');}
        }
      }
      for(const id of ['sampling','bootstrap']){
        w.VisualLab.navigate(id);await sleep(70);
        if(id==='bootstrap')d.querySelector('[data-story-step="1"]').click();
        const initial=d.querySelector('.sf-formula').textContent;
        d.querySelector('.sf-another').click();
        assert(d.querySelector('.sf-formula').textContent!==initial,id+' next actual example changes mean');
        const progress=d.querySelector('.sf-time input');progress.value='500';progress.dispatchEvent(new w.Event('input',{bubbles:true}));
        assert(progress.value==='500',id+' timeline scrub');
        d.querySelector('.sf-play').click();await sleep(150);
        if(!w.matchMedia('(prefers-reduced-motion: reduce)').matches){assert(Number(progress.value)>500,id+' playback advances');d.querySelector('.sf-play').click();const stopped=progress.value;await sleep(100);assert(progress.value===stopped,id+' pause stops movement');}
        const detached=progress;w.VisualLab.navigate('tangent');const stopped=detached.value;await sleep(100);assert(detached.value===stopped,id+' navigation cancels animation');
      }
      frame.style.width='1200px';w.VisualLab.navigate('library');const search=d.getElementById('library-search');search.value='PCA';search.dispatchEvent(new w.Event('input'));assert([...d.querySelectorAll('.library-card')].some(x=>!x.hidden&&x.textContent.includes('PCA')),'Library search returns PCA');search.value='no-such-experiment-xyz';search.dispatchEvent(new w.Event('input'));assert([...d.querySelectorAll('.library-card')].every(x=>x.hidden),'Library search empty state');w.VisualLab.navigate('tangent');
      assert(errors.length===0,'No window errors: '+errors.join('; '));
      const result={time:new Date().toISOString(),build:w.document.querySelector('meta[name=visual-lab-build]')?.content,checks,failures,errors,outcomes,passed:failures.length===0};globalThis.QA_RESULT=result;
      report.className=result.passed?'pass':'fail';report.textContent=JSON.stringify(result,null,2);document.body.dataset.passed=String(result.passed);document.getElementById('save-report').disabled=false;
    }catch(error){report.className='fail';report.textContent='ERROR: '+error.stack;document.body.dataset.passed='false';}
    document.getElementById('run').disabled=false;
  }
  document.getElementById('save-report').addEventListener('click',()=>{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(globalThis.QA_RESULT,null,2)],{type:'application/json'}));a.download='browser-report.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);});
  document.getElementById('run').addEventListener('click',run);
  if(location.search.includes('autorun'))frame.addEventListener('load',()=>run(),{once:true});
})();
