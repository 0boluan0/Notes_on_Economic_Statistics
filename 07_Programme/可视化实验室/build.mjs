import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import crypto from 'node:crypto';
import vm from 'node:vm';
const root=path.dirname(fileURLToPath(import.meta.url));
const read=p=>fs.readFileSync(path.join(root,p),'utf8');
const extraGroups=['ml','dl','advanced','foundations','sequential'].filter(group=>fs.existsSync(path.join(root,`assets/${group}-math.js`))&&fs.existsSync(path.join(root,`assets/${group}-content.js`)));
const contentFiles=['assets/content.js',...extraGroups.map(group=>`assets/${group}-content.js`)];
const context=vm.createContext({});
for(const file of contentFiles) vm.runInContext(read(file),context,{filename:file});
const atomMaps=['atom-map.json','ml-atom-map.json','dl-atom-map.json','advanced-atom-map.json','foundations-atom-map.json','sequential-atom-map.json'].filter(file=>fs.existsSync(path.join(root,'docs',file))).map(file=>JSON.parse(read('docs/'+file)));
const atoms=Object.assign({},...atomMaps);
fs.writeFileSync(path.join(root,'assets/atom-links.js'),`Object.entries(${JSON.stringify(atoms)}).forEach(([id,atoms])=>{if(LabContent[id])LabContent[id].atoms=atoms;});\n`);
const catalog={experiments:Object.entries(context.LabContent).map(([id,c])=>({id,title:c.title,group:c.group,note:`07_Programme/可视化实验室/实验/${id}.md`,sources:c.sources.map(s=>s.url)}))};
fs.writeFileSync(path.join(root,'assets/catalog.json'),JSON.stringify(catalog,null,2)+'\n');
const storyFiles=fs.readdirSync(path.join(root,'assets')).filter(f=>/^(story-|guides-).+\.(js|css)$/.test(f)).sort().map(f=>'assets/'+f);
const sourceFiles=[...storyFiles,'index.html','assets/style.css','assets/math.js',...extraGroups.map(group=>`assets/${group}-math.js`),...contentFiles,'assets/atom-links.js','assets/app.js','assets/vendor/katex/katex.min.js','assets/vendor/katex/katex.min.css'];
const hashes=Object.fromEntries(sourceFiles.map(f=>[f,crypto.createHash('sha256').update(read(f)).digest('hex')]));
let html=read('index.html');
html=html.replace('<!-- teaching-stories -->',storyFiles.map(file=>file.endsWith('.css')?`<link rel="stylesheet" href="${file}">`:`<script defer src="${file}"></script>`).join('\n'));
html=html.replace('<!-- extension-models -->',extraGroups.map(group=>`<script defer src="assets/${group}-math.js"></script>`).join('\n'));
html=html.replace('<!-- extension-content -->',extraGroups.map(group=>`<script defer src="assets/${group}-content.js"></script>`).join('\n'));
html=html.replace(/<link rel="stylesheet" href="([^"]+)">/g,(_,file)=>{
  let css=read(file);
  if(file.includes('katex')){
    css=css.replace(/src:([^;}]+);/g,(match,list)=>{const woff2=list.match(/url\([^)]*\.woff2\)[^,;]*/);return woff2?'src:'+woff2[0]+';':match;});
    css=css.replace(/url\((?:['"])?(fonts\/[^)'"\s]+)(?:['"])?\)/g,(_,font)=>{
      const bytes=fs.readFileSync(path.join(root,path.dirname(file),font));return `url(data:font/woff2;base64,${bytes.toString('base64')})`;
    });
  }
  return `<style data-bundled-source="${file}">${css.replaceAll('</style','<\\/style')}</style>`;
});
html=html.replace(/<script defer src="([^"]+)"><\/script>/g,(_,file)=>{
  let js=read(file).replace(/<\/script/gi,'<\\/script');
  if(file==='assets/app.js')js=`document.addEventListener('DOMContentLoaded',()=>{\n${js}\n});`;
  return `<script data-bundled-source="${file}">${js}</script>`;
});
html=html.replace('</head>',`<meta name="visual-lab-build" content="${crypto.createHash('sha256').update(JSON.stringify(hashes)).digest('hex')}">\n</head>`);
fs.writeFileSync(path.join(root,'实验室.html'),html);
fs.writeFileSync(path.join(root,'build-manifest.json'),JSON.stringify({schema:1,entry:'实验室.html',sha256:crypto.createHash('sha256').update(html).digest('hex'),bytes:Buffer.byteLength(html),inputs:hashes},null,2)+'\n');
console.log(`Built self-contained 实验室.html (${(Buffer.byteLength(html)/1024).toFixed(0)} KiB); scripts, styles and WOFF2 fonts inline.`);
