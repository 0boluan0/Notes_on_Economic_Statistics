"""Render object-level lessons outside the vault; publish only decoded complete movies."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--work',required=True);p.add_argument('--review');p.add_argument('--ids',nargs='*');p.add_argument('--workers',type=int,default=3);args=p.parse_args()
    shared={f.name.removesuffix('-timeline.json') for f in (ROOT/'animations').glob('*-timeline.json') if json.loads(f.read_text()).get('method')=='shared-interactive-scene'}
    if args.ids and shared.intersection(args.ids):
        p.error('Shared interactive films use anatomy/backprop/README.md: '+', '.join(sorted(shared.intersection(args.ids))))
    work=Path(args.work).resolve();work.mkdir(parents=True,exist_ok=True)
    assert not work.is_relative_to(ROOT),'Render cache must be outside the vault'
    data=work/'data';subprocess.run(['node',str(ROOT/'animations/export-narratives.cjs'),str(data)],check=True)
    ids=args.ids or sorted(f.stem for f in data.glob('*.json'));assert len(ids)==len(set(ids)) and 'bootstrap' not in ids
    if not args.ids and shared:
        print('Retaining shared interactive films: '+', '.join(sorted(shared)),flush=True)
        ids=[id for id in ids if id not in shared]
    review=work/args.review if args.review else None
    logdir=work/'logs';logdir.mkdir(exist_ok=True)
    receipt=work/((args.review or 'final')+'-report.json');results=[]

    def run(id):
        started=time.time();cache=work/'cache'/id
        assert (data/(id+'.json')).exists(),id
        cmd=[sys.executable,str(ROOT/'animations/narrative.py'),str(data/(id+'.json')),'--media',str(cache)]
        cmd+=['--review',str(review),'--stills'] if review else ['--quality','final']
        with (logdir/(id+'.log')).open('w') as log:
            result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
        if result.returncode:return {'id':id,'passed':False,'log':str(logdir/(id+'.log'))}
        timeline=json.loads((cache/(id+'-timeline.json')).read_text())
        assert len(timeline['chapters'])>=3 and timeline['duration']>25,(id,'incomplete lesson')
        assert not timeline.get('layout'),(id,timeline['layout'])
        r={'id':id,'passed':True,'seconds':round(time.time()-started,1),'chapters':len(timeline['chapters']),'checkpoints':len(timeline['checkpoints'])}
        if not review:
            movie=next((cache/'videos').rglob(id+'.mp4'));destination=ROOT/'media'/(id+'.mp4');temp=destination.with_suffix('.partial.mp4')
            try:
                subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-i',str(movie),'-c','copy','-movflags','+faststart',str(temp)],check=True)
                subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(temp),'-f','null','-'],check=True,stdout=subprocess.DEVNULL)
                info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(temp)]));video=info['streams'][0]
                assert (video['width'],video['height'],video['codec_name'],video['pix_fmt'])==(1920,1080,'h264','yuv420p')
                r.update(bytes=temp.stat().st_size,duration=float(info['format']['duration']),sha256=hashlib.sha256(temp.read_bytes()).hexdigest())
                assert abs(r['duration']-timeline['duration'])<.1,(id,'timeline mismatch')
                os.replace(temp,destination);shutil.copyfile(cache/(id+'-timeline.json'),ROOT/'animations'/(id+'-timeline.json'))
            finally:
                if temp.exists():temp.unlink()
            proof=work/'encoded';proof.mkdir(exist_ok=True)
            for j,c in enumerate([{'time':0}]+timeline['checkpoints']):
                subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-ss',str(c['time']),'-i',str(destination),'-frames:v','1','-update','1',str(proof/f'{id}-{j:02d}.png')],check=True)
            shutil.rmtree(cache)
        return r

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(run,id):id for id in ids}
        for f in concurrent.futures.as_completed(futures):
            try:r=f.result()
            except Exception as e:r={'id':futures[f],'passed':False,'error':str(e)}
            results.append(r);receipt.write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n');print(json.dumps(r,ensure_ascii=False),flush=True)
    if not all(r['passed'] for r in results):sys.exit(1)


if __name__=='__main__':main()
