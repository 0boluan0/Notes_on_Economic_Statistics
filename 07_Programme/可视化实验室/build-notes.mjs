import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import {fileURLToPath} from 'node:url';
const root=path.dirname(fileURLToPath(import.meta.url)),ctx=vm.createContext({});
const files=['content','ml-content','dl-content','advanced-content','foundations-content','sequential-content','atom-links'];
for(const f of files){const file=path.join(root,'assets',f+'.js');if(fs.existsSync(file))vm.runInContext(fs.readFileSync(file,'utf8'),ctx);}
const start='<!-- visual-lab:generated:start -->',end='<!-- visual-lab:generated:end -->';
function managed(file,body){const full=path.join(root,file),block=start+'\n'+body.trim()+'\n'+end;fs.mkdirSync(path.dirname(full),{recursive:true});if(fs.existsSync(full)){const old=fs.readFileSync(full,'utf8');if(!old.includes(start)||!old.includes(end))throw new Error('Refusing to overwrite unowned note: '+file);fs.writeFileSync(full,old.replace(new RegExp(start+'[\\s\\S]*?'+end),()=>block));}else fs.writeFileSync(full,block+'\n');}
const selected=process.argv.slice(2);for(const id of selected)if(!ctx.LabContent[id])throw Error('Unknown experiment '+id);
const entries=Object.entries(ctx.LabContent),groups=[...new Set(entries.map(([id,c])=>c.group))];
function filmBlock(id){
  if(id==='bootstrap')return bootstrapFilm;
  if(!fs.existsSync(path.join(root,'media',id+'.mp4')))return '';
  const timeline=JSON.parse(fs.readFileSync(path.join(root,'animations',id+'-timeline.json'),'utf8'));
  const time=t=>`${Math.floor(t/60)}:${String(Math.floor(t%60)).padStart(2,'0')}`;
  const chapters=timeline.chapters;
  const startHelp=id==='backprop'?'先播放“前向计算”，再进入“追踪一个改动”：灰点留在原处，蓝点沿曲线移动。接着切到反向，观察前向值保持、敏感度逐段相乘。可暂停、拖动进度或选择一个局部运算。影片与交互共用计算与画面；“从影片继续”可载入片中情境。':'在下方展开“影片参数与延伸对照”，选择“载入本片的模型与数据”，先核对影片中的一个对象或运算，再改变一个条件。其他对照用于继续探索。';
  const startHelpEn=id==='backprop'?'Play the forward pass, then trace a small weight change: the gray point stays at its original position while the blue point moves along the curve. Switch to the backward pass to see fixed forward values and accumulated sensitivities. Pause, scrub or inspect a local operation. The film and interaction share the same calculation and scene.':'Expand the film inputs below and load the model and data used in this film. Check one object or calculation, then change one condition. The other comparisons provide further exploration.';
  const audio=timeline.narration?'中文合成讲解与中文字幕':'中英字幕',audioEn=timeline.narration?'Chinese synthetic narration and Chinese captions':'Chinese and English captions';
  return `<!-- visual-lab:film:start -->\n## 动画：分步观察\n\n<!-- bilingual-en:start -->\n*Animation: follow the mechanism*\n<!-- bilingual-en:end -->\n\n![[07_Programme/可视化实验室/media/${id}.mp4]]\n\n约 ${Math.round(timeline.duration)} 秒，${audio}。可暂停、点击进度条定位或全屏观看。\n\n<!-- bilingual-en:start -->\nAbout ${Math.round(timeline.duration)} seconds, with ${audioEn}. Pause, seek on the timeline or watch full screen.\n<!-- bilingual-en:end -->\n\n${timeline.chapters.map((c,i)=>`- **${time(c.start)}** ${chapters[i].title}`).join('\n')}\n\n<!-- bilingual-en:start -->\n${timeline.chapters.map((c,i)=>`- **${time(c.start)}** ${chapters[i].en}`).join('\n')}\n<!-- bilingual-en:end -->\n\n## 亲手实验\n\n<!-- bilingual-en:start -->\n*Try it yourself*\n<!-- bilingual-en:end -->\n\n${startHelp}\n\n<!-- bilingual-en:start -->\n${startHelpEn}\n<!-- bilingual-en:end -->\n\n<!-- visual-lab:film:end -->\n`;
}
const bootstrapFilm=`## 动画：一份样本怎样变成一幅分布

<!-- bilingual-en:start -->
*Animation: from one sample to a distribution*
<!-- bilingual-en:end -->

![[07_Programme/可视化实验室/media/bootstrap.mp4]]

81 秒，中英字幕。可暂停、点击进度条定位或全屏观看。合成示例使用指数总体，n=8、B=500、seed=42；数值显示三位小数。这个小样本用于说明机制。

<!-- bilingual-en:start -->
81 seconds, with Chinese and English captions. Pause, click the timeline to seek or watch full screen. The synthetic example uses an exponential population with n=8, B=500 and seed=42; values are displayed to three decimal places. This small sample illustrates the mechanism.
<!-- bilingual-en:end -->

## 亲手实验

<!-- bilingual-en:start -->
*Try it yourself*
<!-- bilingual-en:end -->

先打开“有放回地重新抽取”，确认 n=8、B=500、指数总体、种子42，可复现短片；再只改变 B 或 n，比较各自改变了什么。

<!-- bilingual-en:start -->
Open the with-replacement step and set n=8, B=500, an exponential population and seed 42 to reproduce the film. Then change only B or n and compare what changes.
<!-- bilingual-en:end -->

`;
for(const [id,c] of entries.filter(([id])=>!selected.length||selected.includes(id))){
  managed(`实验/${id}.md`,`# ${c.title}\n\n<!-- bilingual-en:start -->\n*${c.en}*\n<!-- bilingual-en:end -->\n\n${c.summary}\n\n> [!question] 先预测，再操作\n> ${c.question}\n\n${filmBlock(id)}\`\`\`visual-lab\nid: ${id}\nheight: 980\n\`\`\`\n\n## 观察路线\n\n${c.steps.map((x,i)=>`${i+1}. ${x}`).join('\n')}\n\n## 公式与适用条件\n\n$$\n${c.formula}\n$$\n\n${c.assumptions.map(x=>'- '+x).join('\n')}\n\n## 回到知识点\n\n${(c.atoms||[]).map(a=>`- [[${a.path.replace(/\.md$/,'')}|${a.title}]]：${a.support}`).join('\n')||'这个主题尚未连接已有知识原子；定义、假设与来源可在本页和实验内阅读。'}\n\n[[07_Programme/可视化实验室/00_实验库入口|回到实验库]]${c.related.filter(x=>ctx.LabContent[x]).map(x=>` · [[07_Programme/可视化实验室/实验/${x}|${ctx.LabContent[x].title}]]`).join('')}\n\n## 来源与核验\n\n${c.sources.map(s=>`- [${s.label}](${s.url}) — ${s.section}。支持：${s.support}`).join('\n')}\n\n实验使用明确给定的公式、合成数据或固定小型示例。代码、可复现检查和逐轮修订记录见 [[07_Programme/可视化实验室/01_使用与维护|使用与维护]]。`);
}
if(!selected.length)managed('00_实验库入口.md',`# 可视化实验室\n\n把需要图形、动态过程和可操作例子的知识放在一起。先提出预测，改变一个条件，再用公式和数值检查观察结果。\n\n当前收录 **${entries.length} 个实验**，各自有独立入口，可从多个知识原子双链引用。知识原子负责定义与命题，实验负责看见机制、比较条件和检验理解。\n\n全部 ${entries.length} 个实验均附有原生播放的动画。打开下方任一实验笔记，先看动画，再用“影片参数与延伸对照”载入影片所用的模型与数据亲手操作。\n\n<!-- bilingual-en:start -->\nAll ${entries.length} experiments include a natively embedded film. Open an experiment note below, watch the film, then load its comparison parameters in the interactive experiment.\n<!-- bilingual-en:end -->\n\n\`\`\`visual-lab\nid: library\nheight: 980\n\`\`\`\n\n## 按问题进入\n\n${groups.map(g=>`### ${g}\n\n${entries.filter(([id,c])=>c.group===g).map(([id,c])=>`- [[07_Programme/可视化实验室/实验/${id}|${c.title}]] — ${c.question}`).join('\n')}`).join('\n\n')}\n\n## 怎么使用\n\n1. 先读每个实验的问题，写下预测。\n2. 一次只改变一个参数，比较图、数值和假设。\n3. 使用预设观察边界，展开“检验理解”中的推导。\n4. 从“来源与双链”回到知识点；需要保留结果时导出 SVG、CSV 或 JSON。\n\n[[07_Programme/可视化实验室/01_使用与维护|使用、离线兼容与新增实验]] · [[07_Programme/可视化实验室/docs/建设与验收范围|覆盖范围与验收方法]] · [[07_Programme/可视化实验室/docs/全库动画验收|动画质量与播放核验]]`);
console.log('Updated '+(selected.length||entries.length)+' managed experiment notes'+(selected.length?'': ' and index')+'.');
