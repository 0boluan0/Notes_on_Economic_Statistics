/* Read-only structural audit. Writes only its own JSON report.
 * node tests/structure.test.js [--expected-count=60]
 * Reads the laboratory, public knowledge atoms, and Git HEAD for those atoms.
 */
'use strict';
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),crypto=require('node:crypto');
const {execFileSync}=require('node:child_process');
const root=path.resolve(__dirname,'..'),vault=path.resolve(root,'../..');
const relative=p=>path.relative(vault,p).split(path.sep).join('/');
const groups=['ml','dl','advanced','foundations','sequential'];
const expectedCount=Number((process.argv.find(v=>v.startsWith('--expected-count='))||'--expected-count=60').split('=')[1]);
const failures=[],checks=[],preservation=[],wikiLinks=[],hashes={};
const check=(name,ok,detail='')=>{checks.push(name);if(!ok)failures.push({name,detail});};
const hash=data=>crypto.createHash('sha256').update(data).digest('hex');
const read=p=>fs.readFileSync(p,'utf8');
const snapshot=p=>{const data=fs.readFileSync(p);hashes[relative(p)]=hash(data);return data.toString('utf8');};
const same=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
const sort=a=>a.slice().sort();
function walk(dir){return fs.readdirSync(dir,{withFileTypes:true}).flatMap(e=>e.isDirectory()?walk(path.join(dir,e.name)):e.isFile()?[path.join(dir,e.name)]:[]);}
function protectedRead(p){const full=path.resolve(p);if(!(full===root||full.startsWith(root+path.sep)||full.startsWith(path.join(vault,'00_Knowledge')+path.sep)))throw Error('Out of audit read scope '+relative(p));return snapshot(full);}
const context=vm.createContext({});
for(const file of ['math.js',...groups.map(g=>g+'-math.js'),'content.js',...groups.map(g=>g+'-content.js')]){
 const full=path.join(root,'assets',file);check('asset exists '+file,fs.existsSync(full));if(fs.existsSync(full))try{vm.runInContext(protectedRead(full),context,{filename:file,timeout:15000});}catch(e){failures.push({name:'classic script loads '+file,detail:String(e)});}
}
const M=context.LabMath,C=context.LabContent||{},ids=Object.keys(C),modelIds=Object.keys(M.schemas);
check('exact expected content count',ids.length===expectedCount,String(ids.length));check('content/model id equality',same(sort(ids),sort(modelIds)));
for(const [id,c]of Object.entries(C)){
 const schema=M.schemas[id];if(!schema)continue;
 const keys=c.controls.map(v=>v.key);check(id+' unique controls',new Set(keys).size===keys.length);check(id+' controls equal schema keys',same(sort(keys),sort(Object.keys(schema))));
 for(const control of c.controls){const spec=schema[control.key];if(!spec)continue;
  check(id+'.'+control.key+' default',control.value===spec[0]);
  if(Array.isArray(spec[1])){check(id+'.'+control.key+' select type',control.type==='select');check(id+'.'+control.key+' options',same(sort((control.options||[]).map(v=>v.value)),sort(spec[1])));}
  else {check(id+'.'+control.key+' range type',control.type==='range');check(id+'.'+control.key+' bounds',control.min===spec[1]&&control.max===spec[2]);check(id+'.'+control.key+' positive step',Number.isFinite(control.step)&&control.step>0);if(spec[3])check(id+'.'+control.key+' integral UI step',Number.isInteger(control.step)&&Number.isInteger(control.value));}
 }
 const defaults=Object.fromEntries(c.controls.map(v=>[v.key,v.value]));try{check(id+' validated defaults',same(M.validate(id,defaults),M.validate(id)));}catch(e){failures.push({name:id+' validation',detail:String(e)});}
 for(const rel of c.related||[])check(id+' related '+rel,Boolean(C[rel]));
 for(const preset of c.presets||[])try{for(const key of Object.keys(preset.params))check(id+' preset key '+key,Boolean(schema[key]));M.validate(id,{...defaults,...preset.params});}catch(e){failures.push({name:id+' preset '+preset.label,detail:String(e)});}
 if(c.animation)check(id+' animation control exists',keys.includes(c.animation.key));
}
const catalog=JSON.parse(protectedRead(path.join(root,'assets/catalog.json'))).experiments;
check('catalog no duplicate ids',new Set(catalog.map(v=>v.id)).size===catalog.length);check('catalog/content id equality',same(sort(catalog.map(v=>v.id)),sort(ids)));
for(const row of catalog){const c=C[row.id];if(!c)continue;check(row.id+' catalog title/group',row.title===c.title&&row.group===c.group);check(row.id+' catalog sources',same(row.sources,c.sources.map(v=>v.url)));const expected=relative(path.join(root,'实验',row.id+'.md'));check(row.id+' fixed note target',row.note===expected);check(row.id+' note exists',fs.existsSync(path.join(vault,row.note)));if(fs.existsSync(path.join(vault,row.note))){const body=protectedRead(path.join(vault,row.note));check(row.id+' note embed id',new RegExp('```visual-lab\\s*\\nid: '+row.id+'\\s*\\n').test(body));}}
const maps=fs.readdirSync(path.join(root,'docs')).filter(v=>v==='atom-map.json'||v.endsWith('-atom-map.json')),mapIds=new Set(),atoms=new Map();
for(const file of maps){const map=JSON.parse(protectedRead(path.join(root,'docs',file)));for(const [id,links]of Object.entries(map)){check(file+' mapped id '+id,Boolean(C[id]));check(id+' only one map definition',!mapIds.has(id));mapIds.add(id);for(const link of links){check(id+' safe atom path '+link.path,/^00_Knowledge\/[^/]+\.md$/.test(link.path));check(id+' atom target exists '+link.path,fs.existsSync(path.join(vault,link.path)));check(id+' semantic map metadata',Boolean(link.title&&link.support));if(!atoms.has(link.path))atoms.set(link.path,[]);atoms.get(link.path).push(id);}}}
check('every content id has explicit map entry',same(sort([...mapIds]),sort(ids)));
vm.runInContext(protectedRead(path.join(root,'assets/atom-links.js')),context,{filename:'atom-links.js'});
for(const id of ids)check(id+' generated atom links match map',same(sort((C[id].atoms||[]).map(v=>v.path)),sort([...atoms].filter(([p,ids])=>ids.includes(id)).map(([p])=>p))));
const git=args=>execFileSync('git',args,{cwd:vault,encoding:'utf8',maxBuffer:20*1024*1024});
const head=git(['rev-parse','HEAD']).trim();
const changed=git(['diff','--name-only','-z','HEAD','--','00_Knowledge']).split('\0').filter(Boolean);
const untracked=git(['ls-files','--others','--exclude-standard','-z','--','00_Knowledge']).split('\0').filter(Boolean);
const begin='<!-- visual-lab:links:start -->',end='<!-- visual-lab:links:end -->';
function withoutManaged(text,file,requireBlock){
 const starts=text.split(begin).length-1,ends=text.split(end).length-1;
 check(file+' balanced unique managed block',starts===ends&&starts<=1&&(!requireBlock||starts===1));
 if(!starts)return text;
 const at=text.indexOf(begin),finish=text.indexOf(end,at)+end.length;
 check(file+' exact block trailing delimiter',text.slice(finish,finish+2)==='\n\n');
 return text.slice(0,at)+text.slice(finish+2);
}
const publicKnowledge=walk(path.join(vault,'00_Knowledge')).filter(p=>p.endsWith('.md'));
const managedAtoms=publicKnowledge.filter(p=>read(p).includes(begin)).map(relative);
check('managed atoms exactly match reviewed mappings',same(sort(managedAtoms),sort([...atoms.keys()])));
for(const file of new Set([...changed,...atoms.keys()])){
 const full=path.join(vault,file);check(file+' exists after edits',fs.existsSync(full));if(!fs.existsSync(full))continue;
 if(!file.endsWith('.md')){failures.push({name:'unexpected modified knowledge asset',detail:file});continue;}
 const current=protectedRead(full),pure=withoutManaged(current,file,atoms.has(file));let before;
 check(file+' current UTF-8 roundtrip preserves bytes',Buffer.from(current,'utf8').equals(fs.readFileSync(full)));
 try{const bytes=execFileSync('git',['show',head+':'+file],{cwd:vault,maxBuffer:20*1024*1024});before=bytes.toString('utf8');check(file+' HEAD UTF-8 roundtrip preserves bytes',Buffer.from(before,'utf8').equals(bytes));}catch(e){failures.push({name:'knowledge has no HEAD baseline',detail:file});continue;}
 const headPure=withoutManaged(before,file+' at HEAD',false),equal=Buffer.from(pure,'utf8').equals(Buffer.from(headPure,'utf8'));
 check(file+' original bytes match HEAD',equal);preservation.push({file,headContentSha256:hash(headPure),currentWithoutManagedSha256:hash(pure),preserved:equal});
 if(atoms.has(file)){const block=current.slice(current.indexOf(begin),current.indexOf(end)+end.length);for(const id of atoms.get(file))check(file+' backlink to '+id,block.includes('07_Programme/可视化实验室/实验/'+id+'|'));}
}
for(const file of untracked)check('no untracked knowledge rewrite '+file,false,'No Git HEAD baseline; inspect provenance rather than deleting.');
const libraryFiles=walk(root),markdown=libraryFiles.filter(p=>p.endsWith('.md'));
const resolvable=[...markdown,...publicKnowledge],byName=new Map();for(const full of resolvable){const name=path.basename(full,'.md');if(!byName.has(name))byName.set(name,[]);byName.get(name).push(full);}
function resolveWiki(raw,file){let target=raw.split('|')[0].split('#')[0].trim();if(!target)return file;try{target=decodeURIComponent(target);}catch{};const extension=path.extname(target);const suffixes=extension?['']:['','.md','.canvas'];const bases=[path.dirname(file),root,vault];for(const base of bases)for(const suffix of suffixes){const candidate=path.resolve(base,target+suffix);if(candidate.startsWith(vault+path.sep)&&fs.existsSync(candidate)&&fs.statSync(candidate).isFile())return candidate;}if(!target.includes('/')){const choices=byName.get(target.replace(/\.md$/,''))||[];if(choices.length===1)return choices[0];}return null;}
for(const file of markdown){const content=protectedRead(file),body=content.replace(/^(`{3,}|~{3,})[^\n]*\n[\s\S]*?^\1[^\n]*$/gm,'').replace(/`[^`\n]+`/g,'');for(const match of body.matchAll(/!?\[\[([^\]\n]+)\]\]/g)){const target=resolveWiki(match[1],file);check(relative(file)+' wiki target '+match[1],Boolean(target));wikiLinks.push({from:relative(file),target:match[1].split('|')[0],resolved:target?relative(target):null});}}
// Construct retired directory names without leaving false-positive literal paths in this test.
const retired=[['07','Project'].join('_'),['Math','Statistics','Lab'].join('_')];
const textExtensions=new Set(['.js','.mjs','.cjs','.json','.html','.css','.md','.py','.txt']);
for(const file of [...libraryFiles.filter(p=>textExtensions.has(path.extname(p))&&!p.includes(path.sep+'vendor'+path.sep)),...publicKnowledge]){
 if(file===path.join(root,'tests','structure-report.json'))continue;
 const text=protectedRead(file);for(const old of retired)check(relative(file)+' has no retired directory '+old.split('_')[0],!text.includes(old));
}
// Generated catalog and bundle must represent the exact current inputs, not an earlier build.
const manifest=JSON.parse(protectedRead(path.join(root,'build-manifest.json')));
for(const [file,expected]of Object.entries(manifest.inputs)){const full=path.join(root,file);check('build input current '+file,fs.existsSync(full)&&hash(fs.readFileSync(full))===expected);}
check('bundle hash current',hash(fs.readFileSync(path.join(root,manifest.entry)))===manifest.sha256);
const report={schema:1,expectedCount,contentCount:ids.length,modelCount:modelIds.length,catalogCount:catalog.length,atomMapFiles:maps.length,mappedAtoms:atoms.size,modifiedKnowledgeFiles:changed.length,preservedAtoms:preservation.length,markdownFiles:markdown.length,wikiLinks:wikiLinks.length,gitHead:head,checks:checks.length,passed:failures.length===0,failures,preservation,hashes};
fs.writeFileSync(path.join(root,'tests','structure-report.json'),JSON.stringify(report,null,2)+'\n');
console.log(`${report.passed?'PASS':'FAIL'} structure: ${ids.length} content/models/catalog entries; ${atoms.size} mapped atoms; ${wikiLinks.length} wiki links; ${checks.length} checks.`);
if(failures.length){console.error(JSON.stringify(failures,null,2));process.exitCode=1;}
