"""Execute every tutorial and create self-contained HTML previews + plot PNGs.

Usage from any directory: python path/to/python/execute_notebooks.py
No global kernelspec changes: Python's ipykernel spec is supplied temporarily.
"""
from pathlib import Path
import base64, html, json, os, sys, tempfile
import nbformat
from nbclient import NotebookClient
import mistune
ROOT=Path(__file__).resolve().parent
FIGURE_DIR=Path(os.environ.get("LAB_FIGURE_DIR",str(Path(tempfile.gettempdir())/"math-lab-figures")))
FIGURE_DIR.mkdir(parents=True,exist_ok=True)
STYLE='''body{font-family:system-ui,-apple-system,sans-serif;color:#20323b;background:#f4f6f7;margin:0;line-height:1.7}main{max-width:960px;margin:30px auto;background:white;padding:42px}h1{font-size:30px;line-height:1.3}h2{border-top:1px solid #d5e0e5;padding-top:26px;margin-top:40px}h3{margin-top:28px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f0f4f6;padding:15px;font-size:13px;line-height:1.5;border-radius:5px}code{font-family:ui-monospace,monospace}img{width:100%;height:auto}a{color:#24658c}.output{border-left:3px solid #7194a8;padding-left:14px}.cell{margin:20px 0}details{background:#f1f5f6;padding:10px}table{border-collapse:collapse}th,td{padding:8px;border-bottom:1px solid #ddd}@media(max-width:650px){main{margin:0;padding:20px}h1{font-size:24px}}
'''
markdown=mistune.create_markdown(escape=False,plugins=['table','math'])
report=[]
with tempfile.TemporaryDirectory(prefix='lab-kernel-') as temp:
    kernel=Path(temp)/'kernels'/'lab-python'; kernel.mkdir(parents=True)
    (kernel/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Lab Python','language':'python'}))
    previous=os.environ.get('JUPYTER_PATH','')
    os.environ['JUPYTER_PATH']=temp+(os.pathsep+previous if previous else '')
    for path in sorted((ROOT/'notebooks').glob('*.ipynb')):
        nb=nbformat.read(path,as_version=4); nbformat.validate(nb)
        client=NotebookClient(nb,timeout=180,kernel_name='lab-python',resources={'metadata':{'path':str(path.parent)}})
        client.execute(); nbformat.validate(nb); nbformat.write(nb,path)
        fragments=[]; figures=0; errors=[]; code_count=0
        for index,cell in enumerate(nb.cells):
            if cell.cell_type=='markdown': fragments.append('<section class="cell">'+markdown(cell.source)+'</section>')
            elif cell.cell_type=='code':
                code_count+=1
                fragments.append('<details class="cell"><summary>Python · cell '+str(index+1)+'</summary><pre>'+html.escape(cell.source)+'</pre></details>')
                for output in cell.get('outputs',[]):
                    if output.output_type=='error': errors.append(output.ename+': '+output.evalue)
                    if 'text' in output: fragments.append('<pre class="output">'+html.escape(output.text)+'</pre>')
                    data=output.get('data',{})
                    if 'image/png' in data:
                        figures+=1; png=data['image/png']
                        (FIGURE_DIR/f'{path.stem}-{figures}.png').write_bytes(base64.b64decode(png))
                        fragments.append('<figure><img alt="Executed scientific plot '+str(figures)+'" src="data:image/png;base64,'+png+'"></figure>')
                    elif 'text/plain' in data: fragments.append('<pre class="output">'+html.escape(data['text/plain'])+'</pre>')
        katex_script = r'<script>document.querySelectorAll(".math").forEach(function(el){let t=el.textContent;let block=el.tagName==="DIV";if(t.startsWith("\\(") && t.endsWith("\\)")){t=t.slice(2,-2)}if(t.startsWith("$$") && t.endsWith("$$")){t=t.slice(2,-2)}try{katex.render(t,el,{throwOnError:false,displayMode:block})}catch(e){}})</script>'
        document='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+path.stem+'</title><link rel="stylesheet" href="../../assets/vendor/katex/katex.min.css"><style>'+STYLE+'</style><main>'+''.join(fragments)+'</main><script src="../../assets/vendor/katex/katex.min.js"></script>'+katex_script+'</html>'
        (ROOT/'previews'/f'{path.stem}.html').write_text(document,encoding='utf-8')
        row=dict(notebook=path.name,codeCells=code_count,figures=figures,errors=errors,executed=True)
        report.append(row); print(json.dumps(row,ensure_ascii=False),flush=True)
    (ROOT/'previews'/'execution-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
