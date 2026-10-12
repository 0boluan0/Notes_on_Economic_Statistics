'use strict';
const {Plugin,ItemView,MarkdownRenderChild,TFile,normalizePath,Platform}=require('obsidian');
const PROJECT='07_Programme/可视化实验室';
const BUNDLE=PROJECT+'/实验室.html';
const VIEW='visual-knowledge-lab-view';
const CATALOG=PROJECT+'/assets/catalog.json';
function parseCatalog(value){
  if(!value||!Array.isArray(value.experiments)||value.experiments.length>5000)throw new Error('实验目录需要 experiments 数组。');
  const ids=new Set(['library']);
  for(const entry of value.experiments){if(!entry||typeof entry.id!=='string'||! /^[a-z][a-z0-9_-]{0,63}$/.test(entry.id)||ids.has(entry.id))throw new Error('实验目录中有无效或重复的 id。');ids.add(entry.id);}
  return ids;
}
const CSP="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; font-src data:; img-src data: blob:; connect-src 'none'; object-src 'none'; frame-src 'none'; base-uri 'none'; form-action 'none';";

function parseConfig(source,ids){
  const config={id:'library',height:900},seen=new Set();
  for(const line of source.split(/\r?\n/)){
    if(!line.trim()||line.trim().startsWith('#'))continue;
    const match=/^\s*(id|height)\s*:\s*(.*?)\s*$/.exec(line);
    if(!match||seen.has(match[1]))throw new Error('只接受唯一的 id: 实验编号 和 height: 高度。');
    const [,key,value]=match;seen.add(key);
    if(key==='id'){if(!ids.has(value))throw new Error('未知实验编号：'+value);config.id=value;}
    else{if(!/^\d+$/.test(value))throw new Error('height 必须是整数。');const n=Number(value);if(n<400||n>2000)throw new Error('height 应在 400–2000 像素之间。');config.height=n;}
  }
  return config;
}
function allowedNotePath(raw){
  if(typeof raw!=='string'||raw.length>1000||/[\0\\?#:%]/.test(raw)||raw.split('/').some(part=>part==='..'||part==='.')||raw.startsWith('/'))return null;
  const path=normalizePath(raw.trim());
  if(!path.endsWith('.md')||!(path.startsWith('00_Knowledge/')||path.startsWith(PROJECT+'/')))return null;
  return path;
}
function allowedSource(entries,id,raw){
  const entry=entries.get(id);if(!entry||!Array.isArray(entry.sources)||!entry.sources.includes(raw))return null;
  try{const url=new URL(raw);return url.protocol==='https:'&&!url.username&&!url.password?raw:null;}catch(_){return null;}
}
function noteLink(entries,id){const path=allowedNotePath(entries.get(id)?.note);return path?'[['+path.slice(0,-3)+']]':null;}
function buildSrcdoc(html,id,notePath='',ids){
  if(!ids.has(id))throw new Error('未知实验编号。');
  if(typeof html!=='string'||!/<head(?:\s[^>]*)?>/i.test(html))throw new Error('实验室离线包缺少有效的 HTML head。');
  // This policy is inserted before every script/style in the fixed, locally owned bundle.
  const head='<meta http-equiv="Content-Security-Policy" content="'+CSP+'">'+
    '<script>window.LAB_INITIAL_EXPERIMENT='+JSON.stringify(id)+';window.LAB_EMBEDDED=true;window.LAB_OBSIDIAN_NOTE='+JSON.stringify(allowedNotePath(notePath)||'').replace(/</g,'\\u003c')+';</script>';
  return html.replace(/<head(?:\s[^>]*)?>/i,match=>match+head);
}
class LabFrame extends MarkdownRenderChild{
  constructor(containerEl,plugin,config){super(containerEl);this.plugin=plugin;this.config=config;this.generation=0;}
  onload(){
    this.containerEl.classList.add('visual-lab-renderer');
    const toolbar=document.createElement('div');toolbar.className='visual-lab-toolbar';
    this.status=document.createElement('span');this.status.className='visual-lab-status';this.status.setAttribute('role','status');this.status.setAttribute('aria-live','polite');
    const reload=document.createElement('button');reload.type='button';reload.textContent='重新加载';reload.addEventListener('click',()=>this.render());
    const open=document.createElement('button');open.type='button';open.textContent='在独立标签页打开';open.addEventListener('click',()=>this.plugin.openLab(this.currentId||this.config.id));
    toolbar.append(this.status,reload,open);this.host=document.createElement('div');this.host.className='visual-lab-frame-host';this.containerEl.replaceChildren(toolbar,this.host);
    this.ownerWindow=this.containerEl.ownerDocument.defaultView;
    this.listener=event=>this.onMessage(event);this.ownerWindow.addEventListener('message',this.listener);
    this.register(()=>this.ownerWindow.removeEventListener('message',this.listener));
    this.render();
  }
  async render(){
    const generation=++this.generation;this.ready=false;this.status.textContent='正在读取离线实验室…';this.host.replaceChildren();
    try{
      await this.plugin.loadCatalog();
      // The fixed HTML/JSON assets need not be registered as Obsidian file views.
      const html=await this.plugin.app.vault.adapter.read(BUNDLE);
      if(generation!==this.generation)return;
      const frame=this.containerEl.ownerDocument.createElement('iframe');this.frame=frame;
      frame.className='visual-lab-frame';frame.title='可视化知识实验室 · '+this.config.id;
      frame.setAttribute('sandbox','allow-scripts allow-downloads');
      frame.setAttribute('referrerpolicy','no-referrer');
      frame.setAttribute('allow',"camera 'none'; microphone 'none'; geolocation 'none'; clipboard-read 'none'; clipboard-write 'none'");
      frame.style.height=this.config.height+'px';frame.srcdoc=buildSrcdoc(html,this.config.id,this.config.notePath,this.plugin.ids);
      frame.addEventListener('load',()=>{if(!this.ready&&generation===this.generation)this.status.textContent='页面已载入，等待实验初始化…';});
      this.host.replaceChildren(frame);
    }catch(error){if(generation===this.generation){this.status.textContent='实验室加载失败';const message=document.createElement('p');message.className='visual-lab-error';message.textContent=error.message;this.host.replaceChildren(message);}}
  }
  onMessage(event){
    if(!this.frame||event.source!==this.frame.contentWindow||!event.data||typeof event.data!=='object')return;
    const data=event.data;if(data.source!=='visual-knowledge-lab')return;
    if(data.type==='ready'||data.type==='state'){
      if(!this.plugin.ids.has(data.id))return;this.ready=true;this.currentId=data.id;
      this.status.textContent='实验已就绪 · '+data.id+(data.type==='state'?' · 参数已更新':'');
    }else if(data.type==='open-note'){
      const path=allowedNotePath(data.path);if(!path)return;
      const file=this.plugin.app.vault.getAbstractFileByPath(path);
      if(file instanceof TFile)this.plugin.app.workspace.getLeaf('tab').openFile(file);
    }else if(data.type==='open-source'){
      const url=allowedSource(this.plugin.entries,data.id,data.url);if(!url)return;
      if(Platform?.isDesktopApp)require('electron').shell.openExternal(url);
      else this.ownerWindow.open(url,'_blank','noopener,noreferrer');
    }else if(data.type==='copy-link'){
      const link=noteLink(this.plugin.entries,data.id);if(!link)return;
      const copied=()=>{this.status.textContent='已复制当前实验的 Obsidian 双链';this.frame.contentWindow.postMessage({type:'visual-lab:copied',id:data.id},'*');};
      if(Platform?.isDesktopApp){require('electron').clipboard.writeText(link);copied();}
      else this.ownerWindow.navigator.clipboard.writeText(link).then(copied).catch(()=>{this.status.textContent='剪贴板不可用；请从实验笔记复制双链。';});
    }
  }
  onunload(){this.generation++;if(this.frame)this.frame.srcdoc='';this.containerEl.replaceChildren();}
}
class LabView extends ItemView{
  constructor(leaf,plugin){super(leaf);this.plugin=plugin;this.initialId='library';}
  getViewType(){return VIEW;}
  getDisplayText(){return '可视化实验室';}
  getIcon(){return 'chart-scatter';}
  getState(){return {id:this.child?.currentId||this.initialId};}
  async setState(state,result){if(state&&this.plugin.ids.has(state.id))this.initialId=state.id;await super.setState(state,result);if(this.child){this.child.config.id=this.initialId;await this.child.render();}}
  async onOpen(){this.contentEl.classList.add('visual-lab-item-view');this.child=new LabFrame(this.contentEl,this.plugin,{id:this.initialId,height:900});this.addChild(this.child);}
  async onClose(){if(this.child){this.removeChild(this.child);this.child=null;}this.contentEl.replaceChildren();}
}
class VisualKnowledgeLab extends Plugin{
  async onload(){
    // Vault assets can finish syncing after the plugin itself arrives.
    try{await this.loadCatalog();}catch(_){this.ids=new Set(['library']);this.entries=new Map();}
    this.registerMarkdownCodeBlockProcessor('visual-lab',async(source,el,ctx)=>{
      try{await this.loadCatalog();ctx.addChild(new LabFrame(el,this,{...parseConfig(source,this.ids),notePath:ctx.sourcePath}));}
      catch(error){el.classList.add('visual-lab-error');el.textContent=error.message;}
    });
    this.registerView(VIEW,leaf=>new LabView(leaf,this));
    this.addCommand({id:'open-laboratory',name:'打开可视化实验室',callback:()=>this.openLab('library')});
  }
  async loadCatalog(){const catalog=JSON.parse(await this.app.vault.adapter.read(CATALOG));this.ids=parseCatalog(catalog);this.entries=new Map(catalog.experiments.map(entry=>[entry.id,entry]));}
  async openLab(id){
    try{await this.loadCatalog();}catch(_){/* LabFrame.render shows the read failure beside its reload button. */}
    const leaf=this.app.workspace.getLeaf('tab');await leaf.setViewState({type:VIEW,active:true,state:{id:this.ids.has(id)?id:'library'}});this.app.workspace.revealLeaf(leaf);
  }
}
module.exports=VisualKnowledgeLab;
module.exports.testing={parseConfig,allowedNotePath,allowedSource,noteLink,buildSrcdoc,parseCatalog,CSP,BUNDLE,CATALOG,LabFrame};
