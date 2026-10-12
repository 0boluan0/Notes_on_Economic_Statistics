'use strict';
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),path=require('node:path');
const source=fs.readFileSync(path.join(__dirname,'../obsidian-plugin/main.js'),'utf8');
const fake={Plugin:class{},ItemView:class{},MarkdownRenderChild:class{},TFile:class{},normalizePath:s=>s.replace(/\/{2,}/g,'/')};
const context={module:{exports:{}},URL,require:name=>{assert.equal(name,'obsidian');return fake;}};vm.runInNewContext(source,context);
const T=context.module.exports.testing;
const ids=T.parseCatalog({experiments:['tangent','linear','projection','gradient','sampling','confidence','testing','bayes','regression','overfit','pca','bootstrap'].map(id=>({id}))});
assert.throws(()=>T.parseCatalog({experiments:[{id:'../bad'}]}));
assert.throws(()=>T.parseCatalog({experiments:[{id:'a'},{id:'a'}]}));
assert.throws(()=>T.parseCatalog({experiments:[{id:'library'}]}));
const entries=new Map([['tangent',{id:'tangent',note:'07_Programme/可视化实验室/实验/tangent.md',sources:['https://example.org/source','http://example.org/insecure','https://user:secret@example.org/source']}]]);
assert.equal(T.allowedSource(entries,'tangent','https://example.org/source'),'https://example.org/source');
for(const [id,url]of [['other','https://example.org/source'],['tangent','https://example.org/source?extra=1'],['tangent','http://example.org/insecure'],['tangent','https://user:secret@example.org/source']])assert.equal(T.allowedSource(entries,id,url),null);
assert.equal(T.noteLink(entries,'tangent'),'[[07_Programme/可视化实验室/实验/tangent]]');assert.equal(T.noteLink(entries,'missing'),null);
assert.equal(T.parseConfig('id: bayes\nheight: 700',ids).id,'bayes');
assert.equal(T.parseConfig('',ids).height,900);
for(const text of ['id: invalid','id: tangent\nid: pca','url: https://example.com','height: 399','height: 2001','height: 800.5','height: Infinity'])assert.throws(()=>T.parseConfig(text,ids));
for(const id of ids)assert.equal(T.parseConfig('id: '+id,ids).id,id);
for(const raw of ['/00_Knowledge/a.md','00_Knowledge/../secret.md','08_quant/private.md','99_学习情况记录/test.md','00_Knowledge/%2e%2e/a.md','https://example.com/a.md','00_Knowledge/a.md?x=1','00_Knowledge/a.html','00_Knowledge/a\\b.md'])assert.equal(T.allowedNotePath(raw),null);
assert.equal(T.allowedNotePath('00_Knowledge/正交投影.md'),'00_Knowledge/正交投影.md');
assert.equal(T.allowedNotePath('07_Programme/可视化实验室/00_实验库入口.md'),'07_Programme/可视化实验室/00_实验库入口.md');
const injected=T.buildSrcdoc('<!doctype html><html><head><script>app()</script></head><body></body></html>','pca','',ids);
assert.ok(injected.indexOf('Content-Security-Policy')<injected.indexOf('<script>'));
assert.ok(injected.indexOf('LAB_INITIAL_EXPERIMENT="pca"')<injected.indexOf('app()'));
assert.ok(T.CSP.includes("connect-src 'none'"));assert.ok(T.CSP.includes("default-src 'none'"));
assert.ok(!source.includes('allow-same-origin'));assert.ok(!source.includes('registerExtensions'));
assert.ok(source.includes('event.source!==this.frame.contentWindow'));
assert.throws(()=>T.buildSrcdoc('<html></html>','pca','',ids));
assert.throws(()=>T.buildSrcdoc('<head></head>','invalid','',ids));
const sender={},opened=[],file=new fake.TFile();
const component=new T.LabFrame({}, {ids,app:{vault:{getAbstractFileByPath:p=>p==='00_Knowledge/正交投影.md'?file:null},workspace:{getLeaf:()=>({openFile:f=>opened.push(f)})}}}, {id:'tangent',height:900});
component.frame={contentWindow:sender};component.status={textContent:''};
component.onMessage({source:{},data:{source:'visual-knowledge-lab',type:'ready',id:'pca'}});assert.equal(component.currentId,undefined);
component.onMessage({source:sender,data:{source:'different',type:'ready',id:'pca'}});assert.equal(component.currentId,undefined);
component.onMessage({source:sender,data:{source:'visual-knowledge-lab',type:'ready',id:'pca'}});assert.equal(component.currentId,'pca');assert.equal(component.ready,true);
component.onMessage({source:sender,data:{source:'visual-knowledge-lab',type:'state',id:'invalid'}});assert.equal(component.currentId,'pca');
for(const raw of ['08_quant/private.md','00_Knowledge/../secret.md','00_Knowledge/missing.md'])component.onMessage({source:sender,data:{source:'visual-knowledge-lab',type:'open-note',path:raw}});
assert.equal(opened.length,0);
component.onMessage({source:sender,data:{source:'visual-knowledge-lab',type:'open-note',path:'00_Knowledge/正交投影.md'}});assert.equal(opened.length,1);assert.equal(opened[0],file);
console.log('PASS: plugin catalog/config, CSP ordering, isolated sandbox, message-source validation, exact source allowlist, safe note links and navigation scope.');
// A newly built experiment must work in a note before any existing frame reloads.
(async()=>{
  const Plugin=context.module.exports,plugin=new Plugin();let processor;
  let liveCatalog={experiments:[{id:'first'}]};
  plugin.app={vault:{adapter:{read:async()=>JSON.stringify(liveCatalog)}}};
  plugin.registerMarkdownCodeBlockProcessor=(id,fn)=>{processor=fn};plugin.registerView=()=>{};plugin.addCommand=()=>{};
  await plugin.onload();assert.ok(plugin.ids.has('first'));assert.ok(!plugin.ids.has('new-lab'));
  liveCatalog={experiments:[{id:'first'},{id:'new-lab'}]};let child;
  const el={classList:{add:()=>{throw Error('Fresh catalog was not accepted')}},textContent:''};
  await processor('id: new-lab',el,{sourcePath:'07_Programme/可视化实验室/实验/new-lab.md',addChild:c=>{child=c}});
  assert.equal(child.config.id,'new-lab');console.log('PASS: note embeds refresh the catalog before validating newly added IDs.');
})().catch(e=>{console.error(e);process.exitCode=1});

// A second device may load the plugin before the fixed catalog finishes syncing.
// Keep all three entry points available, and put read failures inside the frame's
// existing error/reload UI rather than rejecting before a view can be opened.
(async()=>{
  function startupHost(available){
    const plugin=new context.module.exports(),registered=new Set(),commands=new Map(),opened=[];
    let catalogAvailable=available;
    const leaf={setViewState:async state=>{opened.push(state);}};
    plugin.app={
      vault:{adapter:{read:async()=>{
        if(!catalogAvailable)throw new Error('catalog has not synced yet');
        return JSON.stringify({experiments:[{id:'tangent'}]});
      }}},
      workspace:{getLeaf:()=>leaf,revealLeaf:()=>{}}
    };
    plugin.registerMarkdownCodeBlockProcessor=()=>registered.add('processor');
    plugin.registerView=()=>registered.add('view');
    plugin.addCommand=command=>{registered.add('command');commands.set(command.id,command);};
    return {plugin,registered,commands,opened,setCatalogAvailable:value=>{catalogAvailable=value;}};
  }
  let failed=0;
  async function check(label,run){
    try{await run();console.log('PASS: '+label);}
    catch(error){failed++;console.error('FAIL: '+label+' — '+error.message);}
  }
  await check('unavailable startup catalog does not block processor/view/command registration',async()=>{
    const h=startupHost(false);
    await assert.doesNotReject(()=>h.plugin.onload());
    assert.deepEqual([...h.registered].sort(),['command','processor','view']);
  });
  await check('command opens after delayed catalog arrival without reloading the plugin',async()=>{
    const h=startupHost(false);
    try{await h.plugin.onload();}catch(_){/* The assertion below identifies the missing entry point. */}
    h.setCatalogAvailable(true);
    const command=h.commands.get('open-laboratory');
    assert.equal(typeof command?.callback,'function','open command must already be registered');
    await command.callback();
    assert.equal(h.opened.length,1);
    assert.equal(h.opened[0].type,'visual-knowledge-lab-view');
    assert.equal(h.opened[0].state.id,'library');
  });
  await check('command opens a frame even while the catalog remains unavailable',async()=>{
    const h=startupHost(true);
    await h.plugin.onload();
    h.setCatalogAvailable(false);
    const command=h.commands.get('open-laboratory');
    await assert.doesNotReject(()=>Promise.resolve(command.callback()));
    assert.equal(h.opened.length,1,'the view must open so its frame can show the read failure and reload action');
    assert.equal(h.opened[0].type,'visual-knowledge-lab-view');
  });
  if(failed){console.error(`Startup recovery regressions: ${failed} failed.`);process.exitCode=1;}
})().catch(e=>{console.error(e);process.exitCode=1});
