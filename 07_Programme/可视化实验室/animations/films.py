"""Shared chart geometry and typography helpers."""
import math
import re

import numpy as np
from manim import *

BG, INK, MUTED, RULE = '#10171F', '#F3F0E9', '#A6B2BF', '#334353'
COLORS = ['#85D7BE', '#F0B084', '#88BCF5', '#DFA5CC', '#D8D792', '#BAACF3', '#687887', '#83CDD8']
LEFT_X, RIGHT_X, BOTTOM_Y, TOP_Y = -5.90, 6.12, -1.11, 2.15


def fmt(v):
    if v is None:
        return '未定义'
    if isinstance(v, str):
        return {'Infinity': '+∞', '-Infinity': '−∞'}.get(v, v)
    if abs(v) < 1e-13:
        return '0'
    return (f'{v:.2e}' if abs(v) >= 1e5 or abs(v) < .001 else f'{v:.3f}'.rstrip('0').rstrip('.')).replace('-', '−')


def text(s, size=24, color=INK, width=None):
    t = Text(str(s).replace('ₜ','_t'), font='PingFang SC', font_size=size, color=color, line_spacing=.85)
    if width and t.width > width:
        t.scale_to_fit_width(width)
    return t


def wrap(s, units):
    # Count CJK glyphs as full width and Latin as half width; preserve explicit lines.
    lines, line, used = [], '', 0
    for ch in re.findall(r'[A-Za-z0-9_.=+−-]+|\n|.',s):
        n = sum(1 if ord(v) > 255 else .53 for v in ch)
        if ch == '\n' or (used+n > units and line and ch not in '，。；：、！？）'):
            lines.append(line.strip()); line, used = '', 0
        if ch != '\n':
            line += ch; used += n
    if line:
        lines.append(line.strip())
    return '\n'.join(lines)


def finite(p):
    return p is not None and all(isinstance(v, (int, float)) and math.isfinite(v) for v in p)


def limits(plots, tight=False):
    ranges = []
    for axis in [0, 1]:
        vals = []
        for p in plots:
            supplied = None if tight else p.get('xDomain' if axis == 0 else 'yDomain')
            if supplied:
                vals.extend(supplied)
                continue
            for s in p['series']:
                for q in s['points']:
                    if not finite(q):
                        continue
                    vals.append(q[axis])
                    if s['type'] == 'segments':
                        vals.append(q[axis+2])
                    if s['type'] == 'bars':
                        vals.extend([q[0]-q[2]/2, q[0]+q[2]/2] if axis == 0 and len(q)>2 else [0] if axis == 1 else [])
                    if s['type'] == 'heatmap':
                        d = s.get('cellWidth' if axis == 0 else 'cellHeight', 1)/2
                        vals.extend([q[axis]-d, q[axis]+d])
        lo, hi = min(vals or [-1]), max(vals or [1])
        if hi-lo <= 64*np.finfo(float).eps*max(1,abs(lo),abs(hi)):
            lo, hi = lo-1, hi+1
        if tight or not all(p.get('xDomain' if axis == 0 else 'yDomain') for p in plots):
            pad = (hi-lo)*.07; lo -= pad; hi += pad
        ranges.append([lo, hi])
    xd, yd = ranges
    return xd, yd


def tick_values(lo, hi, n=5):
    raw = (hi-lo)/n; power = 10**math.floor(math.log10(raw)); fraction=raw/power
    step = next((s for s in [1, 2, 5, 10] if fraction <= s))*power
    return np.arange(math.ceil(lo/step)*step, hi+step*.0001, step).tolist()


def clipped(a, b, box):
    """Liang-Barsky: preserve geometry instead of clamping endpoints to the border."""
    xmin, xmax, ymin, ymax = box
    dx, dy = b[0]-a[0], b[1]-a[1]; low, high = 0., 1.
    for p, q in [(-dx,a[0]-xmin),(dx,xmax-a[0]),(-dy,a[1]-ymin),(dy,ymax-a[1])]:
        if p == 0:
            if q < 0: return None
        elif p < 0: low=max(low,q/p)
        else: high=min(high,q/p)
        if low > high: return None
    return [a[0]+low*dx,a[1]+low*dy], [a[0]+high*dx,a[1]+high*dy]


def chart(p, domains, zlimits):
    group = VGroup()
    left,right,bottom,top=LEFT_X,RIGHT_X,BOTTOM_Y,TOP_Y
    if p.get('equalAspect'):
        xd,yd=domains
        unit=min((right-left)/(xd[1]-xd[0]),(top-bottom)/(yd[1]-yd[0]))
        cx,cy=(left+right)/2,(bottom+top)/2
        left,right=cx-unit*(xd[1]-xd[0])/2,cx+unit*(xd[1]-xd[0])/2
        bottom,top=cy-unit*(yd[1]-yd[0])/2,cy+unit*(yd[1]-yd[0])/2
    if p.get('diagram'):
        nodes = p['diagram']['nodes']; step=11.25/max(1,len(nodes)-1)
        for i,n in enumerate(nodes):
            x=-5.65+i*step
            if i:
                group.add(Arrow([x-step+.4,.35,0],[x-.42,.35,0],color=MUTED,buff=0,stroke_width=2,max_tip_length_to_length_ratio=.14))
            group.add(Circle(radius=.38,color=COLORS[0],stroke_width=2).move_to([x,.35,0]))
            group.add(text(n['label'],27,COLORS[0]).move_to([x,.35,0]))
            group.add(text(fmt(n['value']),23).move_to([x,-.45,0]))
            group.add(text(n['operation'],18,MUTED,width=1.48).move_to([x,-.95,0]))
        return group
    xd, yd = domains
    sx=lambda x:left+(x-xd[0])/(xd[1]-xd[0])*(right-left)
    sy=lambda y:bottom+((yd[1]-y) if p.get('reverseY') else (y-yd[0]))/(yd[1]-yd[0])*(top-bottom)
    pos=lambda x,y:np.array([sx(x),sy(y),0])
    inside=lambda q:xd[0]<=q[0]<=xd[1] and yd[0]<=q[1]<=yd[1]
    diagram = '示意' in p['xLabel'] or '仅展示' in p['xLabel']
    for axis in [0,1]:
        d=xd if axis==0 else yd; labels=p.get('xTickLabels' if axis==0 else 'yTickLabels')
        values=p.get('xTicks' if axis==0 else 'yTicks') or ([i for i in range(len(labels)) if d[0]<=i<=d[1]] if labels else tick_values(*d))
        if diagram: values=[]
        for v in values:
            a,b = ([sx(v),bottom,0],[sx(v),top,0]) if axis==0 else ([left,sy(v),0],[right,sy(v),0])
            group.add(Line(a,b,stroke_width=1.1 if abs(v)<1e-12 else .5,color=RULE,stroke_opacity=.8))
            label=labels[int(v)] if labels and float(v).is_integer() and 0<=v<len(labels) else fmt(v)
            t=text(label,18,MUTED,width=1.3 if axis==0 else .8)
            t.next_to([sx(v),bottom,0] if axis==0 else [left,sy(v),0],DOWN if axis==0 else LEFT,buff=.12)
            group.add(t)
    if not diagram:
        group.add(text(p['xLabel'],20,MUTED,width=11.7).move_to([(left+right )/2,-1.62,0]))
        group.add(text(p['yLabel'],19,MUTED,width=3.4).rotate(PI/2).move_to([left-.75,(bottom+top)/2,0]))

    for j,s in enumerate(p['series']):
        color=COLORS[s.get('color',j)%len(COLORS)]; typ=s['type']; points=s['points']
        stroke=s.get('width',1.25 if s.get('color')==6 else 2.6)
        if typ in ('line','segments'):
            pairs = [(a,b) for a,b in zip(points,points[1:]) if finite(a) and finite(b)] if typ=='line' else [(q[:2],q[2:]) for q in points if finite(q)]
            path=VMobject(stroke_color=color,stroke_width=stroke)
            prev=None
            for a,b in pairs:
                v=clipped(a,b,(*xd,*yd))
                if v is None: prev=None; continue
                v1,v2=pos(*v[0]),pos(*v[1])
                if prev is None or np.linalg.norm(v1-prev)>1e-6:path.start_new_path(v1)
                path.add_line_to(v2);prev=v2
            if path.has_points():
                if s.get('fill'):
                    path.set_fill(color,opacity=.22)
                if s.get('dashed'):path=DashedVMobject(path,num_dashes=max(12,int(path.get_arc_length()*7)))
                group.add(path)
            if s.get('arrow'):
                for a,b in pairs:
                    if inside(b) and np.linalg.norm(pos(*a)-pos(*b))>.05:
                        delta=pos(*b)-pos(*a); angle=np.arctan2(delta[1],delta[0])
                        tip=Triangle(color=color,fill_color=color,fill_opacity=1,stroke_width=0).scale(.055).rotate(angle-PI/2).move_to(pos(*b));group.add(tip)
        elif typ=='scatter':
            r=min(.07,s.get('radius',3)/65)
            for q in points:
                if not finite(q) or not inside(q):continue
                if s.get('shape')=='diamond':m=Square(side_length=r*1.8,color=color,fill_opacity=1,stroke_width=.6).rotate(PI/4)
                elif j%2:m=VGroup(Line([-r,-r,0],[r,r,0],color=color,stroke_width=1.6),Line([-r,r,0],[r,-r,0],color=color,stroke_width=1.6))
                else:m=Dot(radius=r,color=color)
                group.add(m.move_to(pos(*q[:2])))
        elif typ=='bars':
            for k,q in enumerate(points):
                if not finite(q):continue
                x,y=q[:2]; w=q[2] if len(q)>2 else abs(points[min(k+1,len(points)-1)][0]-points[max(k-1,0)][0]) or 1
                x1,x2=max(x-w/2,xd[0]),min(x+w/2,xd[1]);y1,y2=max(min(0,y),yd[0]),min(max(0,y),yd[1])
                if x2<=x1 or y2<=y1:continue
                group.add(Rectangle(width=max(.002,sx(x2)-sx(x1)-.013),height=abs(sy(y2)-sy(y1)),stroke_width=.5,stroke_color=color,fill_color=color,fill_opacity=.63).move_to(pos((x1+x2)/2,(y1+y2)/2)))
        elif typ=='heatmap':
            zd=zlimits.get(j) or s.get('zDomain') or [min(q[2] for q in points if finite(q)),max(q[2] for q in points if finite(q))]
            sequential=s.get('scale')=='sequential' or (s.get('scale')!='diverging' and zd[0]>=0)
            for q in points:
                if not finite(q):continue
                x,y,z=q;cw=s.get('cellWidth',1);ch=s.get('cellHeight',1)
                t=.5 if zd[1]==zd[0] else min(1,max(0,(z-zd[0])/(zd[1]-zd[0])))
                c=interpolate_color(ManimColor('#1C3037'),ManimColor(COLORS[0]),t) if sequential else interpolate_color(ManimColor(COLORS[2] if t<.5 else '#27343E'),ManimColor('#27343E' if t<.5 else COLORS[1]),t*2 if t<.5 else t*2-1)
                w=abs(sx(x+cw/2)-sx(x-cw/2));h=abs(sy(y+ch/2)-sy(y-ch/2))
                group.add(Rectangle(width=w,height=h,stroke_width=.5 if len(points)<300 else 0,stroke_color=BG,fill_color=c,fill_opacity=1).move_to(pos(x,y)))
                if s.get('showValues') and w>.32 and h>.22:
                    group.add(text(fmt(z),min(21,w*31),BG if sequential and t>.55 else INK,width=w*.86).move_to(pos(x,y)))
    for a in p.get('annotations',[]):
        if not inside([a['x'],a['y']]):continue
        t=text(a['text'],19,INK,width=2.6).move_to(pos(a['x'],a['y'])+UP*.2)
        if t.get_top()[1]>top:t.shift(DOWN*(t.get_top()[1]-top+.04))
        if t.get_right()[0]>right:t.shift(LEFT*(t.get_right()[0]-right+.04))
        if t.get_left()[0]<left:t.shift(RIGHT*(left-t.get_left()[0]+.04))
        back=BackgroundRectangle(t,color=BG,fill_opacity=.9,buff=.045,stroke_width=0)
        group.add(back,t)
    return group
