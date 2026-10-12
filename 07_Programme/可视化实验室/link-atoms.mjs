/* Applies only the semantically reviewed explicit maps; never infers relationships. */
import fs from 'node:fs';import path from 'node:path';import crypto from 'node:crypto';import {fileURLToPath} from 'node:url';
const root=path.dirname(fileURLToPath(import.meta.url)),vault=path.resolve(root,'../..');
const catalog=JSON.parse(fs.readFileSync(path.join(root,'assets/catalog.json'),'utf8')).experiments;
const entries=new Map(catalog.map(c=>[c.id,c])),byPath=new Map();
for(const file of ['atom-map.json','ml-atom-map.json','dl-atom-map.json','advanced-atom-map.json','foundations-atom-map.json','sequential-atom-map.json']){const full=path.join(root,'docs',file);if(!fs.existsSync(full))continue;const map=JSON.parse(fs.readFileSync(full,'utf8'));for(const [id,atoms]of Object.entries(map)){if(!entries.has(id))continue;for(const atom of atoms){if(!/^00_Knowledge\/[^/]+\.md$/.test(atom.path))throw Error('Invalid target '+atom.path);if(!byPath.has(atom.path))byPath.set(atom.path,[]);byPath.get(atom.path).push({id,...atom});}}}
const begin='<!-- visual-lab:links:start -->',end='<!-- visual-lab:links:end -->',pattern=new RegExp(begin+'[\\s\\S]*?'+end+'\\n\\n','g');
const hash=s=>crypto.createHash('sha256').update(s).digest('hex'),checks=[],pending=[];
for(const [file,links]of byPath){const full=path.join(vault,file),old=fs.readFileSync(full,'utf8'),original=old.replace(pattern,'');const marker=original.match(/^## 来源(?:与核验|与原文定位)?$/m)?.[0],at=marker?original.lastIndexOf(marker):-1;if(at<0)throw Error('Missing final sources section '+file);
 const block=begin+'\n## 交互理解\n\n'+links.map(({id})=>`- [[${entries.get(id).note.replace(/\.md$/,'')}|${entries.get(id).title}]]`).join('\n')+'\n'+end+'\n\n';
 const updated=original.slice(0,at)+block+original.slice(at);if(updated.replace(pattern,'')!==original)throw Error('Content preservation failed '+file);pending.push({full,updated});checks.push({file,experiments:links.map(x=>x.id),preservedOriginalSha256:hash(original),updatedSha256:hash(updated)});
}
for(const {full,updated} of pending) fs.writeFileSync(full,updated);
fs.writeFileSync(path.join(root,'docs','backlink-report.json'),JSON.stringify({criterion:'Removing only the managed navigation block leaves all preexisting bytes unchanged.',atoms:checks},null,2)+'\n');console.log('Linked '+checks.length+' atoms with original-content preservation verified.');
