"use strict";
const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const escapeHTML = value => String(value ?? "").replace(/[&<>"']/g, c => ({ "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;" }[c]));
const safeHref = value => {
  try { const url = new URL(value); return ["http:","https:","obsidian:"].includes(url.protocol) ? url.href : "#"; }
  catch { return "#"; }
};
const link = (url, title, cls = "") => '<a class="'+cls+'" href="'+escapeHTML(safeHref(url))+'" target="_blank" rel="noopener noreferrer">'+escapeHTML(title)+'</a>';
const roleNames = { teach:"讲解", use:"使用", require:"要求基础" };
const relationNames = { prerequisite:"知识前置", application:"应用", method:"计算方式" };
const state = { data:null, view:"map", topic:null, course:null, relation:null, note:null, zoom:1, focus:false, onlyPrereq:false, editor:null, layout:null };
let searchTimer, notesTimer, toastTimer, searchVersion = 0, editorVersion = 0;
async function api(path, options = {}) {
  const response = await fetch(path, { ...options, headers:{ "Content-Type":"application/json", ...options.headers } });
  const result = await response.json();
  if (!response.ok) { const error = new Error(result.error || "读取失败，请重试。"); error.status = response.status; throw error; }
  return result;
}
function showError(error) { $("#alert").hidden = false; $("#alert").textContent = error.message || String(error); }
function toast(message) {
  clearTimeout(toastTimer); $("#toast").textContent=message; $("#toast").hidden=false;
  toastTimer=setTimeout(()=>$("#toast").hidden=true,3500);
}
const topics = () => state.data.topics;
const courseById = id => state.data.courses.find(c=>c.id===id);
const topicById = id => topics().find(t=>t.id===id);
const mappingsFor = id => state.data.mappings.filter(m=>m.topic_id===id);
function atomsFor(topic) { return [...new Set([...topic.atom_ids,...state.data.note_links.filter(l=>l.topic_id===topic.id).map(l=>l.atom_id)])]; }
function noteURI(path) { return "obsidian://open?vault=Academic&file="+encodeURIComponent(path); }
function updateURL() {
  const params = new URLSearchParams();
  if(state.view!=="map")params.set("view",state.view);
  if(state.topic)params.set("topic",state.topic);
  if(state.course)params.set("course",state.course);
  history.replaceState(null,"",location.pathname+(params.size?"?"+params:""));
}
async function loadData(initial=false) {
  try {
    $("#alert").hidden=true;
    const data=await api("/api/map");state.data=data;
    $("#topic-count").textContent=data.topics.length;
    $("#course-count").textContent=data.courses.length;
    $("#note-count").textContent=data.stats.notes.toLocaleString();
    $("#connection-summary").textContent=data.stats.notes.toLocaleString()+" 篇笔记 · "+data.stats.linked_notes+" 篇已关联";
    if(initial) {
      const params=new URLSearchParams(location.search);
      state.view=["map","courses","notes"].includes(params.get("view"))?params.get("view"):"map";
      state.topic=topicById(params.get("topic"))?params.get("topic"):null;
      state.course=courseById(params.get("course"))?params.get("course"):null;
    }
    renderCourseFilters();renderProgress();renderMap();renderCourses();setView(state.view,false);
    if(initial)fitMap();
    if(state.topic)renderTopic(state.topic,false);
    if(data.missing_note_ids.length)showError(new Error("有 "+data.missing_note_ids.length+" 个已关联原子暂未找到，请核对原笔记是否移动。"));
  } catch(error) { showError(error);$("#connection-summary").textContent="暂未连接，请启动本地知识地图。"; }
}
function renderCourseFilters() {
  $("#course-filters").innerHTML='<button class="course-filter '+(!state.course?"active":"")+'" data-filter=""><i class="dot"></i><span>全部课程<small>看见共同知识与不同讲法</small></span></button>'+
  state.data.courses.map(c=>'<button class="course-filter '+(state.course===c.id?"active":"")+'" data-filter="'+escapeHTML(c.id)+'" style="--course-color:'+escapeHTML(c.color||"#658574")+'"><i class="dot"></i><span>'+escapeHTML(c.title)+'<small>'+escapeHTML(c.version)+'</small></span></button>').join("");
}
function setCourse(id) { state.course=id||null;renderCourseFilters();renderMap();renderCourses();updateURL();if(state.topic)renderTopic(state.topic,false); }
function renderProgress() {
  const p=state.data.course_progress;
  $("#progress-summary").innerHTML=p?.available?'<div>18.06SC 学习任务 <strong>'+p.completed+'</strong> / '+p.total+'</div><small>'+link(p.uri,"打开学习记录 ↗")+' · 独立掌握另记</small>':'<small>个人掌握状态：尚未评估</small>';
}
function setView(view, update=true) {
  state.view=view;
  $$(".view").forEach(el=>el.hidden=el.id!==view+"-view");
  $$(".nav-item").forEach(el=>el.classList.toggle("active",el.dataset.view===view));
  $("#view-crumb").textContent={map:"知识地图",courses:"课程对照",notes:"Obsidian 笔记"}[view];
  $("#page-title").textContent={map:"让课程，落在你的知识上。",courses:"同一个知识，换一种讲法。",notes:"已有的思考，都从这里连接。"}[view];
  $("#page-subtitle").textContent={map:"从知识点找到讲解，也从一门课看见它能补足什么。",courses:"比较具体章节的讲解范围，保留各自的版本与前置条件。",notes:"检索原有知识原子，打开笔记，或把它关联到地图主题。"}[view];
  if(view==="notes")searchNotes($("#notes-search").value);
  if(update)updateURL();
}
function ancestors(id) {
  const found=new Set([id]); let changed=true;
  while(changed){changed=false;state.data.relations.filter(r=>r.kind==="prerequisite").forEach(r=>{if(found.has(r.to)&&!found.has(r.from)){found.add(r.from);changed=true;}});}
  return found;
}
function buildLayout() {
  const rank=Object.fromEntries(topics().map(t=>[t.id,0]));
  // Only reviewed knowledge/application links establish order; an alternative method is not a prerequisite.
  for(let i=0;i<topics().length;i++) {
    let changed=false;
    state.data.relations.filter(r=>r.kind!=="method").forEach(r=>{const next=rank[r.from]+1;if(rank[r.to]<next){rank[r.to]=next;changed=true;}});
    if(!changed)break;
  }
  const pos={},lanes=[];let y=48;
  const maxRank=Math.max(...Object.values(rank));const width=108+(maxRank+1)*196;
  state.data.domains.forEach(d=>{
    const nodes=topics().filter(t=>t.domain===d.id),counts={};
    nodes.forEach(t=>counts[rank[t.id]]=(counts[rank[t.id]]||0)+1);
    const height=Math.max(141,Math.max(...Object.values(counts),1)*108+33);
    const used={};
    nodes.forEach(t=>{const row=used[rank[t.id]]||0;used[rank[t.id]]=row+1;pos[t.id]={x:98+rank[t.id]*196,y:y+23+row*108,w:164,h:91};});
    lanes.push({...d,y,height});y+=height;
  });
  return {pos,lanes,width,height:y+12,maxRank};
}
function renderMap() {
  if(!state.data)return;
  const l=buildLayout();state.layout=l;
  const focused=state.topic&&state.focus?ancestors(state.topic):null;
  const covered=new Set(state.data.mappings.filter(m=>!state.course||m.course_id===state.course).map(m=>m.topic_id));
  let content='<div class="axis-title">知识关系 →</div>';
  for(let r=0;r<=l.maxRank;r++)content+='<span class="stage-label" style="left:'+(98+r*196)+'px">'+(r===0?"基础与入口":"展开 "+r)+'</span>';
  l.lanes.forEach(d=>content+='<div class="lane" style="top:'+d.y+'px;height:'+d.height+'px;--lane-color:'+d.color+'"><div class="lane-label">'+escapeHTML(d.title)+'<small>'+escapeHTML(d.english)+'</small></div></div>');
  content+='<svg class="graph-edges" width="'+l.width+'" height="'+l.height+'" aria-label="知识关系连线"><defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#9ab3a4"/></marker></defs>';
  state.data.relations.forEach((r,index)=>{
    if(state.onlyPrereq&&r.kind!=="prerequisite")return;
    const a=l.pos[r.from],b=l.pos[r.to];if(!a||!b)return;
    const x1=a.x+a.w,y1=a.y+a.h/2,x2=b.x,y2=b.y+b.h/2;
    const bend=Math.max(35,Math.abs(x2-x1)*.43);
    const path='M '+x1+' '+y1+' C '+(x1+bend)+' '+y1+', '+(x2-bend)+' '+y2+', '+x2+' '+y2;
    const highlighted=state.topic&&(r.to===state.topic||r.from===state.topic);
    const dimmed=focused&&!(focused.has(r.to)&&focused.has(r.from));
    content+='<g><path class="edge '+(r.kind!=="prerequisite"?"secondary ":"")+(highlighted?"highlight ":"")+(dimmed?"dimmed":"")+'" d="'+path+'" marker-end="url(#arrow)"/><path class="edge-hit" d="'+path+'" data-relation="'+index+'" tabindex="0" role="button" aria-label="'+escapeHTML(topicById(r.from).title+" → "+topicById(r.to).title+"："+r.label)+'"/><text class="edge-label" x="'+((x1+x2)/2)+'" y="'+((y1+y2)/2-8)+'" text-anchor="middle">'+escapeHTML(r.label)+'</text></g>';
  });
  content+='</svg>';
  topics().forEach(t=>{
    const p=l.pos[t.id],d=state.data.domains.find(d=>d.id===t.domain);
    const count=new Set(mappingsFor(t.id).map(m=>m.course_id)).size;
    const classes=["node",state.topic===t.id?"selected":"",focused&&!focused.has(t.id)?"dimmed":"",state.course&&!covered.has(t.id)?"not-covered":"",state.course&&covered.has(t.id)?"filtered-hit":""].join(" ");
    content+='<button class="'+classes+'" data-topic="'+t.id+'" style="left:'+p.x+'px;top:'+p.y+'px;--node-color:'+d.color+'" aria-label="'+escapeHTML(t.title+'，'+count+'门关联课程')+'"><strong>'+escapeHTML(t.title)+'</strong><em>'+escapeHTML(t.english)+'</em><span class="node-meta"><span>'+atomsFor(t).length+' 篇笔记</span><span class="coverage">'+count+' 门课程 ↗</span></span></button>';
  });
  $("#map-world").innerHTML=content;$("#map-world").style.width=l.width+"px";$("#map-world").style.height=l.height+"px";
  applyZoom();
}
function applyZoom() {
  const l=state.layout;if(!l)return;
  $("#map-world").style.transform="scale("+state.zoom+")";
  $("#map-space").style.width=l.width*state.zoom+"px";$("#map-space").style.height=l.height*state.zoom+"px";
  $("#zoom-label").textContent=Math.round(state.zoom*100)+"%";
}
function fitMap() { if(!state.layout)return;state.zoom=Math.min(1,Math.max(.55,($("#map-scroller").clientWidth-8)/state.layout.width));applyZoom(); }
function focusNode(id) {
  const p=state.layout?.pos[id];if(!p)return;
  const scroll=$("#map-scroller");
  scroll.scrollTo({left:Math.max(0,(p.x+p.w/2)*state.zoom-scroll.clientWidth*.48),top:Math.max(0,p.y*state.zoom-scroll.clientHeight*.34),behavior:"smooth"});
}
function openDetail(kicker) { $("#detail").hidden=false;$("#detail-kicker").textContent=kicker; }
function closeDetail() { $("#detail").hidden=true;state.note=null;state.relation=null; }
function renderTopic(id, scroll=true) {
  const t=topicById(id);if(!t)return;
  state.topic=id;state.note=null;state.relation=null;updateURL();renderMap();
  openDetail(state.data.domains.find(d=>d.id===t.domain).title+" / 知识主题");
  const maps=mappingsFor(id).filter(m=>!state.course||m.course_id===state.course);
  const notes=atomsFor(t).map(id=>state.data.notes_by_id[id]).filter(Boolean);
  const prerequisites=state.data.relations.filter(r=>r.to===id);
  const next=state.data.relations.filter(r=>r.from===id);
  let html='<h2>'+escapeHTML(t.title)+'</h2><div class="en-title">'+escapeHTML(t.english)+'</div><p class="summary">'+escapeHTML(t.summary)+'</p><div class="detail-actions"><button data-action="focus-topic">查看前置路径</button><button data-action="mapping">＋ 关联课程章节</button></div>';
  html+='<div class="detail-section"><h3>课程里的这个知识 <span>'+maps.length+' 处</span></h3>';
  if(state.course)html+='<p class="muted">当前筛选：'+escapeHTML(courseById(state.course).title)+' <button class="text-button" data-action="clear-course">显示全部</button></p>';
  html+=maps.length?maps.map(mappingCard).join(""):'<p class="notice">这里还没有已整理的课程映射，可以添加具体章节与说明。</p>';
  html+='</div><div class="detail-section"><h3>你的 Obsidian 笔记 <button data-action="attach-note">＋ 关联笔记</button></h3>';
  html+=notes.length?notes.map(n=>'<div class="note-link-card"><button data-note="'+escapeHTML(n.id)+'">'+escapeHTML(n.title)+'<small>'+escapeHTML(n.id)+' · '+(n.status==="source-checked"?"原笔记已核源":"原笔记待核源")+'</small></button>'+link(n.uri,"打开 ↗")+'</div>').join(""):'<p class="notice">尚未关联原笔记。主题存在不代表已经学习。</p>';
  html+='</div><div class="detail-section"><h3>这部分知识从哪里来</h3>'+relationList(prerequisites,"from")+'</div>';
  html+='<div class="detail-section"><h3>接下来用于什么</h3>'+relationList(next,"to")+'</div><p class="notice">这是所选专题的关系切片。个人掌握保持未知；更多准备知识请在原笔记和课程说明中核对。</p>';
  $("#detail-content").innerHTML=html;$("#detail").scrollTop=0;if(scroll)focusNode(id);
}
function relationList(relations,endpoint) {
  return relations.length?relations.map(r=>'<button class="relation-button" data-evidence="'+escapeHTML(r.id)+'">'+escapeHTML(topicById(r[endpoint]).title)+' <small>'+escapeHTML(relationNames[r.kind]+" · "+r.label)+'</small></button>').join(""):'<p class="muted">这张图中尚未整理，不表示没有其他关联。</p>';
}
function mappingCard(m) {
  const c=courseById(m.course_id);
  return '<article class="mapping-card"><div class="mapping-course">'+escapeHTML(c.title+" · "+c.version)+'</div><span class="role-badge '+m.role+'">'+roleNames[m.role]+'</span>'+link(m.url||c.url,m.title+' ↗')+'<p>'+escapeHTML(m.statement)+'</p><footer><span>'+(m.verification==="official"?"官方单元已核对":"你添加的关联 · 待核验")+'</span>'+(m.verification==="user"?'<button data-remove="'+m.id+'">撤销此关联</button>':escapeHTML(m.reviewed||""))+'</footer></article>';
}
function renderRelation(id) {
  const r=state.data.relations.find(r=>r.id===id);if(!r)return;
  state.relation=id;openDetail("关系依据 / "+relationNames[r.kind]);
  $("#detail-content").innerHTML='<h2>'+escapeHTML(r.label)+'</h2><p class="summary">'+escapeHTML(topicById(r.from).title+' → '+topicById(r.to).title)+'</p><div class="relation-explanation">'+escapeHTML(r.evidence.statement)+'</div>'+
  (r.evidence.url?link(r.evidence.url,"查看官方来源 ↗","evidence-link"):"")+
  (r.evidence.note?link(noteURI(r.evidence.note),"打开关系所依据的笔记 ↗","evidence-link"):"")+
  '<p class="notice">'+(r.kind==="prerequisite"?"表达所说明推导中的知识准备，不代表必须修完整门课程。":"应用或计算联系不构成必须先学整门课程的要求。")+'</p><div class="detail-actions"><button data-topic="'+r.from+'">查看起点</button><button data-topic="'+r.to+'">查看终点</button></div>';
  $("#detail").scrollTop=0;
}
async function renderNote(id) {
  try {
    const n=await api("/api/note?id="+encodeURIComponent(id));
    state.note=n;openDetail("OBSIDIAN / 原笔记");
    const requirements=(n.requires||[]).map(x=>String(x).replace(/\[\[|\]\]/g,"").split("|")[0]);
    $("#detail-content").innerHTML='<h2>'+escapeHTML(n.title)+'</h2><p class="muted">'+escapeHTML(n.id)+'</p><div class="detail-actions">'+link(n.uri,"在 Obsidian 打开 ↗")+'<button data-action="note-to-topic">＋ 关联到主题</button></div><p class="notice">来源状态：'+(n.status==="source-checked"?"原笔记标为已核源":"原笔记待核源")+'。此状态不代表个人掌握。</p>'+
    (requirements.length?'<div class="detail-section"><h3>原笔记列出的准备知识</h3><p class="muted">'+escapeHTML(requirements.join(" · "))+'</p></div>':"")+
    '<div class="detail-section"><h3>原文预览</h3><div class="excerpt">'+escapeHTML(n.excerpt.replace(/<!--[\s\S]*?-->/g,"").trim())+'</div></div>';
    $("#detail").scrollTop=0;
  }catch(error){showError(error);}
}
function renderCourses() {
  $("#course-cards").innerHTML=state.data.courses.map(c=>{
    const ms=state.data.mappings.filter(m=>m.course_id===c.id),count=new Set(ms.map(m=>m.topic_id)).size;
    return '<article class="course-card '+(state.course===c.id?"active":"")+'" style="--course-color:'+escapeHTML(c.color||"#658574")+'"><span class="institution">'+escapeHTML(c.institution||"Open course")+'</span><h3>'+escapeHTML(c.title)+'</h3><div class="version">'+escapeHTML(c.version)+'</div><p>'+escapeHTML(c.description||"你添加的公开课程。可以继续关联具体章节。")+'</p><button data-show-course="'+escapeHTML(c.id)+'">在图上看 '+count+' 个知识主题</button>'+link(c.url,"课程主页 ↗")+'</article>';
  }).join("");
  const cs=state.data.courses;
  let html='<table class="comparison-table"><thead><tr><th>知识主题</th>'+cs.map(c=>'<th>'+escapeHTML(c.title)+'</th>').join("")+'</tr></thead><tbody>';
  topics().forEach(t=>{
    html+='<tr><td><button data-topic="'+t.id+'">'+escapeHTML(t.title)+'</button></td>';
    cs.forEach(c=>{
      const maps=mappingsFor(t.id).filter(m=>m.course_id===c.id);
      html+='<td>'+(maps.length?maps.map(m=>'<span class="role-badge '+m.role+'">'+roleNames[m.role]+'</span>'+link(m.url||c.url,m.title)+'<br>').join(""):'<span class="no-mapping">尚未整理</span>')+'</td>';
    });html+='</tr>';
  });
  $("#course-comparison").innerHTML=html+'</tbody></table><p class="scope-note">空白表示尚未整理对应章节，不代表课程完全不涉及这个知识。</p>';
}
async function searchNotes(query) {
  try {
    const result=await api("/api/notes?q="+encodeURIComponent(query)+"&limit=50");
    if($("#notes-search").value!==query)return;
    $("#notes-caption").textContent="找到 "+result.total.toLocaleString()+" 篇 · 显示前 "+result.notes.length+" 篇";
    $("#notes-list").innerHTML=result.notes.length?result.notes.map(n=>'<button class="note-row" data-note="'+escapeHTML(n.id)+'"><div><strong>'+escapeHTML(n.title)+'</strong><small>'+escapeHTML(n.id)+' · '+escapeHTML(n.heading||"")+'</small></div><span>查看笔记 ↗</span></button>').join(""):'<div class="empty">没有匹配的笔记，试试英文别名或更短的词。</div>';
  }catch(error){showError(error);}
}
async function globalSearch(query) {
  const version=++searchVersion,box=$("#search-results");query=query.trim();
  if(!query){box.hidden=true;return;}
  const q=query.toLowerCase();
  const ts=topics().filter(t=>(t.title+" "+t.english).toLowerCase().includes(q));
  const cs=state.data.courses.filter(c=>(c.title+" "+c.institution).toLowerCase().includes(q));
  try {
    const ns=await api("/api/notes?q="+encodeURIComponent(query)+"&limit=8");
    if(version!==searchVersion)return;
    let html=ts.length?'<h4>地图主题</h4>'+ts.map(t=>'<button class="search-result" data-topic="'+t.id+'">'+escapeHTML(t.title)+'<small>'+escapeHTML(t.english)+'</small></button>').join(""):"";
    if(cs.length)html+='<h4>公开课程</h4>'+cs.map(c=>'<button class="search-result" data-show-course="'+c.id+'">'+escapeHTML(c.title)+'<small>'+escapeHTML(c.version)+'</small></button>').join("");
    if(ns.notes.length)html+='<h4>Obsidian 笔记</h4>'+ns.notes.map(n=>'<button class="search-result" data-note="'+escapeHTML(n.id)+'">'+escapeHTML(n.title)+'<small>'+escapeHTML(n.id)+'</small></button>').join("");
    box.innerHTML=html||'<div class="empty">没有匹配项</div>';box.hidden=false;
  }catch(error){showError(error);}
}
const field=(label,name,type="text",value="",required=true)=>'<label class="field">'+label+'<input type="'+type+'" name="'+name+'" value="'+escapeHTML(value)+'" '+(required?"required":"")+' maxlength="'+({title:240,institution:120,version:120,url:2048}[name]||1000)+'"></label>';
const options=(records,selected)=>records.map(x=>'<option value="'+escapeHTML(x.id)+'" '+(x.id===selected?"selected":"")+'>'+escapeHTML(x.title)+'</option>').join("");
function openEditor(kind) {
  const session=++editorVersion;
  state.editor=kind;$("#editor-error").textContent="";$("#save-editor").disabled=false;
  let html="",title="";
  if(kind==="course"){
    title="添加一门公开课";
    html=field("课程名称","title")+field("学校 / 提供者","institution","text","",false)+field("版本 / 学期（避免混用年份）","version")+field("官方课程主页","url","url");
  } else if(kind==="mapping"){
    title="把课程章节连接到知识";
    html='<label class="field">知识主题<select name="topic_id">'+options(topics(),state.topic)+'</select></label><label class="field">公开课程<select name="course_id">'+options(state.data.courses,state.course)+'</select></label>'+field("具体章节 / 讲义名称","title")+field("章节或证据链接","url","url")+
    '<label class="field">这段材料与知识的关系<select name="role"><option value="teach">讲解这个知识</option><option value="use">使用这个知识</option><option value="require">明确要求此基础</option></select></label><label class="field">覆盖范围或要求的依据<textarea name="statement" required maxlength="2000" placeholder="写清楚教了什么，或哪里明确列出要求"></textarea></label><p class="muted">你的新增关联会标记为待核验，不自动成为课程硬先修。</p>';
  } else if(kind==="attach-note") {
    title="关联已有笔记";
    html='<label class="field">地图主题<select name="topic_id">'+options(topics(),state.topic)+'</select></label><label class="field">按原子编号或标题搜索<input id="attach-search" placeholder="例如：正交投影" autocomplete="off"></label><label class="field">选择原笔记<select name="atom_id" id="attach-options" required><option value="">搜索后选择</option></select></label><p class="muted">只增加地图关联，原笔记内容保持原样。</p>';
  } else {
    title="把原笔记关联到主题";
    html='<p class="summary">'+escapeHTML(state.note.title)+' · '+escapeHTML(state.note.id)+'</p><input type="hidden" name="atom_id" value="'+escapeHTML(state.note.id)+'"><label class="field">地图主题<select name="topic_id">'+options(topics(),state.topic)+'</select></label>';
  }
  $("#editor-title").textContent=title;$("#editor-fields").innerHTML=html;
  $("#save-editor").textContent=kind==="course"?"添加课程":"保存关联";
  $("#editor").showModal();
  if(kind==="attach-note"){
    let timer,version=0;
    const select=$("#attach-options"),save=$("#save-editor");
    select.disabled=true;save.disabled=true;
    select.addEventListener("change",()=>{save.disabled=!select.value;});
    $("#attach-search").addEventListener("input",e=>{
      clearTimeout(timer);const query=e.target.value.trim(),request=++version;
      select.innerHTML='<option value="">'+(query?"正在搜索…":"输入关键词后选择")+'</option>';
      select.disabled=true;save.disabled=true;$("#editor-error").textContent="";
      if(!query)return;
      timer=setTimeout(async()=>{
        try{
          const r=await api("/api/notes?q="+encodeURIComponent(query)+"&limit=20");
          if(request!==version||session!==editorVersion||!$("#editor").open)return;
          select.innerHTML=r.notes.length?'<option value="" selected>请选择一篇笔记</option>'+options(r.notes,null):'<option value="">没有匹配的笔记</option>';
          select.disabled=!r.notes.length;
        }catch(error){if(request===version&&session===editorVersion&&$("#editor").open)$("#editor-error").textContent=error.message;}
      },180);
    });
  }
}
async function saveEditor(event) {
  event.preventDefault();$("#save-editor").disabled=true;$("#editor-error").textContent="";
  const data=Object.fromEntries(new FormData($("#editor-form")).entries());data.revision=state.data.revision;
  const endpoint=state.editor==="course"?"courses":state.editor==="mapping"?"mappings":"note-links";
  try {
    const result=await api("/api/"+endpoint,{method:"POST",body:JSON.stringify(data)});
    $("#editor").close();
    if(endpoint==="courses")state.course=result.saved.id;
    await loadData();
    if(endpoint==="courses")setView("courses");
    if(data.topic_id)renderTopic(data.topic_id,false);
    toast(endpoint==="courses"?"课程已保存，可以继续关联具体章节。":"关联已保存到你的本地地图。");
  } catch(error) {
    if(error.status===409){
      try{
        const latest=await api("/api/map");state.data.revision=latest.revision;
        $("#editor-error").textContent="地图已在另一窗口更新。表单已保留，请再次保存。";
      }catch(refreshError){$("#editor-error").textContent="表单已保留，但暂时无法刷新地图版本："+refreshError.message;}
    }else $("#editor-error").textContent=error.message;
    $("#save-editor").disabled=state.editor==="attach-note"&&!$("#attach-options")?.value;
  }
}
async function removeMapping(id) {
  try{await api("/api/mappings/"+encodeURIComponent(id),{method:"DELETE"});await loadData();toast("已撤销这条自建关联；上一版保留在本地备份。");}catch(error){showError(error);}
}
document.addEventListener("click",event=>{
  const el=event.target.closest("button,a,path");if(!el)return;
  if(el.dataset.view){closeDetail();setView(el.dataset.view);}
  if(el.hasAttribute("data-filter"))setCourse(el.dataset.filter);
  if(el.dataset.topic){$("#search-results").hidden=true;setView("map");renderTopic(el.dataset.topic);}
  if(el.dataset.note){$("#search-results").hidden=true;renderNote(el.dataset.note);}
  if(el.dataset.showCourse){$("#search-results").hidden=true;closeDetail();setCourse(el.dataset.showCourse);setView("map");toast("已高亮这门课程关联的知识主题。");}
  if(el.dataset.relation!==undefined)renderRelation(state.data.relations[Number(el.dataset.relation)].id);
  if(el.dataset.evidence)renderRelation(el.dataset.evidence);
  if(el.dataset.remove)removeMapping(el.dataset.remove);
  const action=el.dataset.action;
  if(action==="focus-topic"){state.focus=true;$("#focus-path").checked=true;renderMap();focusNode(state.topic);}
  if(action==="mapping")openEditor("mapping");
  if(action==="attach-note")openEditor("attach-note");
  if(action==="note-to-topic")openEditor("note-to-topic");
  if(action==="clear-course")setCourse(null);
});
document.addEventListener("keydown",event=>{
  if(event.key==="/"&&!["INPUT","TEXTAREA","SELECT"].includes(document.activeElement.tagName)){event.preventDefault();$("#global-search").focus();}
  if(event.key==="Escape"){$("#search-results").hidden=true;if(!$("#editor").open)closeDetail();}
  if((event.key==="Enter"||event.key===" ")&&event.target.hasAttribute("data-relation")){event.preventDefault();renderRelation(state.data.relations[Number(event.target.dataset.relation)].id);}
});
$("#global-search").addEventListener("input",e=>{clearTimeout(searchTimer);searchTimer=setTimeout(()=>globalSearch(e.target.value),160);});
$("#notes-search").addEventListener("input",e=>{clearTimeout(notesTimer);const value=e.target.value;notesTimer=setTimeout(()=>searchNotes(value),180);});
document.addEventListener("click",e=>{if(!e.target.closest(".search-wrap"))$("#search-results").hidden=true;});
$("#close-detail").addEventListener("click",closeDetail);
$("#focus-path").addEventListener("change",e=>{state.focus=e.target.checked;renderMap();if(!state.topic&&state.focus)toast("先选择一个知识主题，即可高亮它的前置路径。");});
$("#all-relations").addEventListener("click",()=>{state.onlyPrereq=false;$("#all-relations").classList.add("chosen");$("#prereq-only").classList.remove("chosen");renderMap();});
$("#prereq-only").addEventListener("click",()=>{state.onlyPrereq=true;$("#prereq-only").classList.add("chosen");$("#all-relations").classList.remove("chosen");renderMap();});
$("#zoom-out").addEventListener("click",()=>{state.zoom=Math.max(.55,state.zoom-.1);applyZoom();});
$("#zoom-in").addEventListener("click",()=>{state.zoom=Math.min(1.6,state.zoom+.1);applyZoom();});
$("#fit-map").addEventListener("click",fitMap);
$("#add-course").addEventListener("click",()=>openEditor("course"));
$("#cancel-editor").addEventListener("click",()=>$("#editor").close());
$("#editor-form").addEventListener("submit",saveEditor);
$("#refresh").addEventListener("click",async()=>{await loadData();toast("已重新读取现有笔记和学习记录。");});
loadData(true);

