"""Object-level lessons. Numerical states are exported from the interactive models."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from manim import *

BG, INK, MUTED, RULE = '#111820', '#F4F2ED', '#9AAAB9', '#344454'
BLUE, GREEN, GOLD, RED, PURPLE = '#88BCF5', '#85D7BE', '#F0C47C', '#EE9696', '#C3B0F0'
PALETTE = [BLUE, GOLD, GREEN, PURPLE, RED]
ROOT = Path(__file__).resolve().parents[1]


def num(v, digits=2):
    if v is None: return '未定义'
    if isinstance(v, str): return {'Infinity':'∞','-Infinity':'−∞'}.get(v,v)
    if v == 0: return '0'
    if abs(v)<.5*10**(-digits):return f'{v:.2g}'.replace('-','−')
    return (f'{v:.2e}' if abs(v)<.001 or abs(v)>=1e4 else f'{v:.{digits}f}'.rstrip('0').rstrip('.')).replace('-','−')


def txt(s, size=27, color=INK, width=12):
    m=Text(str(s).replace('ₜ','_t'),font='PingFang SC',font_size=size,color=color,line_spacing=1.1)
    if m.width>width: m.scale_to_fit_width(width)
    return m


def en(s,size=18,color=MUTED,width=12):
    m=Text(s,font='Avenir Next',font_size=size,color=color)
    if m.width>width:m.scale_to_fit_width(width)
    return m


def eq(s,size=38,color=GREEN):
    m=MathTypst(s,font_size=size,color=color)
    if m.width>11.8:m.scale_to_fit_width(11.8)
    return m


def pos(x,y): return np.array([x,y,0.])


def dotlabel(value,name='',color=BLUE):
    c=Circle(.32,color=color,stroke_width=2).set_fill(BG,1)
    return VGroup(c,txt(num(value),22,color,width=.58).move_to(c),en(name,18).next_to(c,DOWN,buff=.14))


def card(value,name,color=BLUE,width=1.7):
    box=RoundedRectangle(corner_radius=.08,width=width,height=.95,color=RULE,stroke_width=1.5,fill_color=BG,fill_opacity=1)
    return VGroup(box,txt(value,29,color,width=width-.15).move_to(box),txt(name,20,MUTED,width=width+.2).next_to(box,DOWN,buff=.18))


def row(values,names=None,color=BLUE,spacing=1.18,y=.7):
    names=names or [str(i+1) for i in range(len(values))]
    return VGroup(*(dotlabel(v,names[i],color).move_to(pos((i-(len(values)-1)/2)*spacing,y)) for i,v in enumerate(values)))


def grid(values,cell=.65,center=(0,0),color=BLUE,labels=True):
    a=np.array(values,dtype=object);assert a.ndim==2
    n,m=a.shape;g=VGroup();g.cells=[];g.labels=[]
    finite=[abs(float(v)) for v in a.flat if isinstance(v,(int,float)) and np.isfinite(v)]
    scale=max(finite or [1]) or 1
    for i in range(n):
        cells=[];labelsrow=[]
        for j in range(m):
            v=a[i,j];xy=pos(center[0]+(j-(m-1)/2)*cell,center[1]+((n-1)/2-i)*cell)
            opacity=.10+.24*abs(float(v))/scale if isinstance(v,(int,float)) else .08
            box=Square(cell-.03,color=RULE,stroke_width=1,fill_color=color,fill_opacity=opacity).move_to(xy)
            label=txt(num(v),min(27,cell*43),INK,width=cell*.88).move_to(xy)
            item=VGroup(box,label) if labels else VGroup(box)
            g.add(item);cells.append(item);labelsrow.append(label)
        g.cells.append(cells);g.labels.append(labelsrow)
    return g


def axes(xrange,yrange,center=(0,-.15),size=(9.6,3.45),labels=('x','y'),equal=False):
    size=(size[0],min(size[1],3.6))
    if equal:
        unit=min(size[0]/(xrange[1]-xrange[0]),size[1]/(yrange[1]-yrange[0]));size=(unit*(xrange[1]-xrange[0]),unit*(yrange[1]-yrange[0]))
    left=center[0]-size[0]/2;bottom=center[1]-size[1]/2
    def p(x,y):return pos(left+(x-xrange[0])/(xrange[1]-xrange[0])*size[0],bottom+(y-yrange[0])/(yrange[1]-yrange[0])*size[1])
    g=VGroup()
    from films import tick_values
    numberline=not labels[1] and size[1]<=1.3
    if numberline:g.add(Line(p(xrange[0],sum(yrange)/2),p(xrange[1],sum(yrange)/2),color=MUTED,stroke_width=1.5))
    for axis,r in enumerate([xrange,yrange]):
        if numberline and axis==1:continue
        for v in tick_values(r[0],r[1],5):
            a,b=(p(v,yrange[0]),p(v,yrange[1])) if axis==0 else (p(xrange[0],v),p(xrange[1],v))
            if numberline:
                midpoint=p(v,sum(yrange)/2);g.add(Line(midpoint+.06*DOWN,midpoint+.06*UP,color=MUTED,stroke_width=1.5))
            else:g.add(Line(a,b,color=RULE,stroke_width=.7))
            q=txt(num(v),17,MUTED).next_to(a,DOWN if axis==0 else LEFT,buff=.1);g.add(q)
    g.add(txt(labels[0],20,MUTED,width=size[0]+1).next_to(p(sum(xrange)/2,yrange[0]),DOWN,buff=.4))
    if labels[1]:g.add(txt(labels[1],20,MUTED,width=5).next_to(p(xrange[0],yrange[1]),UP,buff=.12).align_to(g,LEFT))
    return g,p


def curve(points,p,color=BLUE,width=3):
    m=VMobject(color=color,stroke_width=width)
    pairs=[]
    for q in points:
        if q is None or not all(isinstance(x,(int,float)) and np.isfinite(x) for x in q[:2]):
            if len(pairs)>1:m.start_new_path(p(*pairs[0]));[m.add_line_to(p(*v)) for v in pairs[1:]]
            pairs=[]
        else:pairs.append(q[:2])
    if len(pairs)>1:m.start_new_path(p(*pairs[0]));[m.add_line_to(p(*v)) for v in pairs[1:]]
    assert m.has_points(), 'A curve must contain at least two finite coordinate pairs'
    return m


def arrow(a,b,color=GOLD):
    if np.linalg.norm(np.array(a)-np.array(b))<.04:return Dot(b,radius=.035,color=color)
    return Arrow(a,b,buff=0,stroke_width=3,color=color,max_tip_length_to_length_ratio=.18)


class Lesson(Scene):
    def construct(self):
        self.payload=json.loads(Path(os.environ['LAB_NARRATIVE_DATA']).read_text())
        self.id=self.payload['id'];self.r=self.payload['result'];self.d=self.r['data'];self.p=self.r['params']
        self.camera.background_color=BG;self.heading=VGroup();self.caption=VGroup();self.chapters=[];self.marks=[];self.layout=[]
        from narrative_neural import STORIES as neural
        from narrative_geometry import STORIES as geometry
        from narrative_statistics import STORIES as statistics
        from narrative_learning import STORIES as learning
        stories={**neural,**geometry,**statistics,**learning}
        assert self.id in stories,self.id
        stories[self.id](self)
        self.hold('closing',3)
        out=Path(config.media_dir);out.mkdir(parents=True,exist_ok=True)
        (out/(self.id+'-timeline.json')).write_text(json.dumps({'id':self.id,'duration':self.time,'chapters':self.chapters,'checkpoints':self.marks,'params':self.p,'method':'object-narrative','layout':self.layout},ensure_ascii=False,indent=2))

    def chapter(self,title,subtitle,caption='',translation=''):
        self.chapters.append({'title':title,'en':subtitle,'start':round(self.time,3)})
        h=VGroup(en(self.id.upper()+'   /   '+f'{len(self.chapters):02d}',17,GREEN).move_to(pos(-5.8,3.47),aligned_edge=LEFT),txt(title,40,width=11.6).move_to(pos(-5.8,2.94),aligned_edge=LEFT),en(subtitle,18).move_to(pos(-5.8,2.45),aligned_edge=LEFT))
        if len(self.heading):self.play(FadeOut(self.heading),FadeIn(h),run_time=.6)
        else:self.add(h)
        self.heading=h
        if caption:self.say(caption,translation)

    def say(self,zh,english=''):
        c=VGroup(txt(zh,27,width=11.8),en(english,18,width=11.8)).arrange(DOWN,buff=.16).move_to(pos(0,-3.17))
        assert c.height<.84,(self.id,'caption needs rewriting',zh)
        if len(self.caption):self.play(FadeOut(self.caption),FadeIn(c),run_time=.35)
        else:self.add(c)
        self.caption=c

    def show(self,*items,run_time=.7):self.play(*(FadeIn(m,shift=.06*UP) for m in items),run_time=run_time)

    def wipe(self):
        objects=[m for m in self.mobjects if m is not self.heading and m is not self.caption]
        if objects:self.play(*(FadeOut(m) for m in objects),run_time=.55)

    def hold(self,name,seconds=2.5):
        for item in self.mobjects:
            if item is self.heading or item is self.caption or (item.width==0 and item.height==0):continue
            box=[float(item.get_left()[0]),float(item.get_right()[0]),float(item.get_bottom()[1]),float(item.get_top()[1])]
            if box[0]<-6.7 or box[1]>6.7 or box[2]<-2.7 or box[3]>2.28:self.layout.append({'checkpoint':name,'type':type(item).__name__,'bounds':box})
        self.wait(seconds);self.marks.append({'name':name,'time':round(self.time-.15,3)})
        if os.environ.get('LAB_NARRATIVE_REVIEW'):
            dest=Path(os.environ['LAB_NARRATIVE_REVIEW']);dest.mkdir(parents=True,exist_ok=True)
            self.renderer.update_frame(self)
            Image.fromarray(self.renderer.get_frame()).save(dest/f'{self.id}-{len(self.marks):02d}.png')

    def equation(self,s,y=0,size=38):
        m=eq(s,size).move_to(pos(0,y));self.play(Write(m),run_time=1);return m

    def ending(self,title,subtitle,formula,zh,english):
        self.wipe();self.chapter(title,subtitle,zh,english);self.equation(formula,y=.6);self.hold('meaning-and-boundary',4)



if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('data');parser.add_argument('--media',required=True);parser.add_argument('--review');parser.add_argument('--stills',action='store_true');parser.add_argument('--quality',choices=['draft','final'],default='draft');args=parser.parse_args()
    os.environ['LAB_NARRATIVE_DATA']=args.data
    if args.review:os.environ['LAB_NARRATIVE_REVIEW']=args.review
    with tempconfig({'media_dir':args.media,'pixel_width':1280 if args.quality=='draft' else 1920,'pixel_height':720 if args.quality=='draft' else 1080,'frame_rate':30,'write_to_movie':not args.stills,'skip_animations':args.stills,'save_last_frame':False,'disable_caching':True,'verbosity':'ERROR','progress_bar':'none','output_file':Path(args.data).stem}):Lesson().render()
