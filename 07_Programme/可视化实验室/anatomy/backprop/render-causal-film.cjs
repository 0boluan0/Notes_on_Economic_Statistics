/* Offline renderer: shared SVG scene + measured Chinese narration, no browser capture. */
const fs=require('node:fs'),path=require('node:path'),{spawn,spawnSync}=require('node:child_process'),{once}=require('node:events'),crypto=require('node:crypto'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'../..'),work=process.argv[2];
assert(work&&path.isAbsolute(work),'Pass an absolute temporary output directory');
fs.mkdirSync(work,{recursive:true});
for(const f of ['math','dl-math','story-backprop-00-model','story-backprop-02-film','story-backprop-03-causal'])require(path.join(root,'assets',f+'.js'));
const F=globalThis.BackpropFilm,C=globalThis.BackpropCausal,fps=24;
function command(program,args){const r=spawnSync(program,args,{encoding:'utf8',maxBuffer:1024*1024*8});if(r.status!==0)throw Error(`${program}: ${r.stderr}`);return r.stdout;}
const duration=file=>Number(command('ffprobe',['-v','error','-show_entries','format=duration','-of','default=nw=1:nk=1',file]));
async function main(){
  const timing=[],audioFiles=[];
  for(let i=0;i<F.frames.length;i++){
    const f=F.frames[i],key=crypto.createHash('sha256').update(f.narration+JSON.stringify(F.narration)).digest('hex').slice(0,16),stem=path.join(work,`${String(i).padStart(2,'0')}-${key}`),voice=stem+'.aiff',text=stem+'.txt',wav=stem+'.wav';
    fs.writeFileSync(text,f.narration+'\n');if(!fs.existsSync(voice))command('/usr/bin/say',['-v',F.narration.voice,'-r',String(F.narration.rate),'-f',text,'-o',voice]);
    const seconds=Math.ceil((duration(voice)+1.1)*fps)/fps;timing.push(seconds);
    command('ffmpeg',['-v','error','-y','-i',voice,'-af','adelay=350,apad','-t',String(seconds),'-ar','48000','-ac','1',wav]);audioFiles.push(wav);
    console.log(`Narration ${i+1}/${F.frames.length}: ${seconds.toFixed(2)}s`);
  }
  const filmFile=path.join(root,'assets/story-backprop-02-film.js');
  fs.writeFileSync(filmFile,fs.readFileSync(filmFile,'utf8').replace(/const timing=\[[^\]]*\]; \/\/ generated audio durations/,`const timing=${JSON.stringify(timing)}; // generated audio durations`));
  let start=0;F.frames.forEach((f,i)=>{f.duration=timing[i];f.start=start;start+=f.duration;});F.duration=start;F.chapters.forEach(c=>c.start=F.frames[c.index].start);
  const concat=path.join(work,'audio-concat.txt');fs.writeFileSync(concat,audioFiles.map(f=>`file '${f.replace(/'/g,"'\\''")}'`).join('\n'));
  const audio=path.join(work,'narration.wav');command('ffmpeg',['-v','error','-y','-f','concat','-safe','0','-i',concat,'-af','loudnorm=I=-18:TP=-2:LRA=7','-ar','48000','-ac','1',audio]);
  if(process.argv.includes('--prepare'))return console.log(`Prepared ${start.toFixed(2)} seconds of narration.`);
  const {createCanvas,loadImage,GlobalFonts}=require('@napi-rs/canvas');
  assert(GlobalFonts.registerFromPath('/System/Library/Fonts/Supplemental/Arial Unicode.ttf','Arial Unicode MS'),'Chinese and math font missing');
  const width=1920,height=1080,canvas=createCanvas(width,height),ctx=canvas.getContext('2d'),output=path.join(work,'backprop.mp4');
  const encoder=spawn('ffmpeg',['-hide_banner','-loglevel','error','-y','-f','rawvideo','-pixel_format','rgba','-video_size',`${width}x${height}`,'-framerate',String(fps),'-i','pipe:0','-i',audio,'-map','0:v:0','-map','1:a:0','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-c:a','aac','-b:a','128k','-movflags','+faststart',output],{stdio:['pipe','ignore','inherit']});
  const completion=once(encoder,'exit');let totalFrames=0;
  for(let i=0;i<F.frames.length;i++){
    const f=F.frames[i],count=Math.round(f.duration*fps);
    for(let j=0;j<count;j++){
      const u=C.clamp((j/fps-.9)/Math.max(1,f.duration*.55)),e=u*u*(3-2*u),state={...f,progress:f.from+(f.to-f.from)*e};
      const svg=C.svg(state).replace('Arial, Hiragino Sans GB, PingFang SC, sans-serif','Arial Unicode MS');
      ctx.fillStyle='#f6f9fb';ctx.fillRect(0,0,width,height);ctx.drawImage(await loadImage(Buffer.from(svg)),120,8,1680,980);
      ctx.fillStyle='#173a50';ctx.fillRect(0,994,width,86);ctx.font='30px "Arial Unicode MS"';ctx.fillStyle='#ffffff';ctx.textAlign='center';ctx.fillText(f.caption,960,1049);
      const bytes=canvas.data();if(!encoder.stdin.write(bytes))await once(encoder.stdin,'drain');totalFrames++;
      if(j===Math.floor(count*.72))fs.writeFileSync(path.join(work,`review-${String(i).padStart(2,'0')}.png`),canvas.toBuffer('image/png'));
    }
    console.log(`Rendered beat ${i+1}/${F.frames.length} (${totalFrames} frames)`);
  }
  encoder.stdin.end();const [code]=await completion;assert.equal(code,0,'Video encode failed');
  command('ffmpeg',['-v','error','-i',output,'-f','null','-']);
  const probe=JSON.parse(command('ffprobe',['-v','error','-show_streams','-show_format','-of','json',output])),video=probe.streams.find(s=>s.codec_type==='video'),sound=probe.streams.find(s=>s.codec_type==='audio');
  assert.equal(video.width,1920);assert.equal(video.height,1080);assert.equal(video.pix_fmt,'yuv420p');assert.equal(sound.codec_name,'aac');assert(Math.abs(Number(probe.format.duration)-F.duration)<.12);
  const timeline={id:'backprop',method:'shared-interactive-scene',animation:'continuous-shared-svg',narration:F.narration,duration:F.duration,params:F.frames[0].params,chapters:F.chapters,checkpoints:F.frames.map((f,i)=>({name:`causal-${i}`,time:f.start+f.duration*.72})),frames:F.frames,layout:[],sha256:crypto.createHash('sha256').update(fs.readFileSync(output)).digest('hex')};
  fs.writeFileSync(path.join(work,'backprop-timeline.json'),JSON.stringify(timeline,null,2)+'\n');fs.writeFileSync(path.join(work,'video-probe.json'),JSON.stringify(probe,null,2)+'\n');
  console.log(`PASS: ${totalFrames} continuous frames, ${F.duration.toFixed(2)}s, AAC narration, full decode.`);
}
main().catch(e=>{console.error(e);process.exitCode=1;});
