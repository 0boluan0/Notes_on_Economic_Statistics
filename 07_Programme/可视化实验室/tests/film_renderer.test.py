"""Geometry and typesetting regression cases found during actual frame review."""
import importlib.util
from pathlib import Path
import numpy as np
import tempfile

spec=importlib.util.spec_from_file_location('films',Path(__file__).resolve().parents[1]/'animations/films.py')
films=importlib.util.module_from_spec(spec);spec.loader.exec_module(films)
assert np.allclose(films.clipped([-2,0],[2,0],(-1,1,-1,1)),[[-1,0],[1,0]])
assert np.allclose(films.clipped([0,-2],[0,2],(-1,1,-1,1)),[[0,-1],[0,1]])
assert films.clipped([-2,2],[2,2],(-1,1,-1,1)) is None
assert np.allclose(films.clipped([-2,-2],[2,2],(-1,1,-1,1)),[[-1,-1],[1,1]])
p={'series':[{'type':'line','points':[[-180,np.log(2)],[0,np.log(2)+2e-16],[180,np.log(2)]]}]}
_,yd=films.limits([p]);assert yd[1]-yd[0]>1
assert '0.\n1' not in films.wrap('固定 x=1；把 h 从 1 缩到 0.1。',14)
assert 'Sigmoid' in films.wrap('当前 Sigmoid 输出为 0.123。',8)
with tempfile.TemporaryDirectory(prefix='lab-geometry-') as cache, films.tempconfig({'media_dir':cache}):
    p={'equalAspect':True,'xLabel':'x','yLabel':'y','series':[{'type':'line','points':[[-1,0],[0,1],[1,0],[0,-1],[-1,0]]}]}
    shape=films.chart(p,([-2,2],[-2,2]),{})[-1]
    assert abs(shape.width-shape.height)<1e-9,'Equal units must produce equal physical lengths'
    assert shape.height>1.5,'Do not shrink geometry into an unnecessarily wide data domain'
print('PASS: clipping, equal-unit viewport, near-constant scale, and unbroken numeric/Latin labels')
