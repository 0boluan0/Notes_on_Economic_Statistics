"""Regressions from frame review: false zero, empty curves and physical scale."""
from pathlib import Path
import sys,tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'animations'))
import narrative as n
import numpy as np
for x in [0.0037,-0.0037,1e-12,-1e-12]:
    displayed=n.num(x)
    assert displayed not in ('0','−0'),(x,displayed)
    assert abs(float(displayed.replace('−','-'))/x-1)<.02
assert n.num(0)=='0'
with tempfile.TemporaryDirectory(prefix='narrative-test-') as cache,n.tempconfig({'media_dir':cache}):
    _,p=n.axes([-2,2],[-2,2],size=(8,3.4),equal=True)
    assert np.isclose(np.linalg.norm(p(1,0)-p(0,0)),np.linalg.norm(p(0,1)-p(0,0)))
    n.curve([[0,0],[1,1]],p)
    for invalid in [[],[[0,0]],[[0,[0,0]],[1,[1,1]]]]:
        try:n.curve(invalid,p)
        except AssertionError:pass
        else:raise AssertionError('An empty or malformed curve must not silently render')
print('PASS: small signed values, nonempty curves, and equal physical axis units')
