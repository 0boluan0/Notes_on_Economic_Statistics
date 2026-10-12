"""Neural-network lessons: preserve the identity of each activation and contribution."""
from narrative import *


def convolution(s):
    d=s.d;assert np.allclose(np.array(d['patch'])*d['kernel'],d['contributions'])
    value=float(np.sum(d['contributions']));assert value==d['output'][d['row']][d['col']]
    s.chapter('一个输出格，是怎样算出来的？','Follow one receptive field into one output.','输入有一条边缘。先只看跨过边缘的这九个数。','The selected 3 × 3 patch straddles the edge.')
    source=grid(d['input'],.50,(-3.6,-.05));kernel=grid(d['kernel'],.65,(1,-.05),GOLD)
    labels=VGroup(txt('输入  7 × 7',23,BLUE).move_to(pos(-3.6,2)),txt('共享的核',23,GOLD).move_to(pos(1,1.45)))
    selection=SurroundingRectangle(VGroup(*(source.cells[i][j] for i in range(1,4) for j in range(1,4))),color=GOLD,buff=.03)
    s.add(source,kernel,labels,selection);s.hold('source-and-window',4)
    patch=grid(d['patch'],.65,(-2.1,-.05))
    s.play(FadeOut(labels),source.animate.scale(.64).move_to(pos(-4.8,.9)),FadeOut(selection),TransformFromCopy(VGroup(*(source.cells[i][j] for i in range(1,4) for j in range(1,4))),patch),run_time=1.7)
    s.show(txt('选出的 3 × 3',22,BLUE).move_to(pos(-2.1,1.45)),txt('逐项相乘',22,GOLD).move_to(pos(1,1.45)))
    s.show(txt('原输入',19,MUTED).move_to(pos(-4.8,-.42)));s.hold('patch-extracted',3)
    s.chapter('相同位置相乘，九项再相加','Multiply corresponding entries, then add.','这个例子采用互相关：核保持方向，不翻转。','This convention is cross-correlation: the kernel is not flipped.')
    products=grid(d['contributions'],.65,(4.2,-.05),GREEN);s.show(txt('九个乘积',22,GREEN).move_to(pos(4.2,1.45)))
    cumulative=txt('累计  0',30,GREEN).move_to(pos(1,-2.05));s.show(cumulative)
    total=0
    for k in range(9):
        i,j=divmod(k,3);a,b=patch.cells[i][j],kernel.cells[i][j]
        rings=VGroup(SurroundingRectangle(a,color=GOLD,buff=.02),SurroundingRectangle(b,color=GOLD,buff=.02))
        term=txt(f'{num(d["patch"][i][j])} × {num(d["kernel"][i][j])} = {num(d["contributions"][i][j])}',27).move_to(pos(-2.1,-1.6))
        s.show(rings,term,run_time=.2)
        s.play(TransformFromCopy(VGroup(a,b),products.cells[i][j]),run_time=.6)
        total+=d['contributions'][i][j]
        new=txt('累计  '+num(total),30,GREEN).move_to(cumulative)
        s.play(Transform(cumulative,new),run_time=.3);s.wait(.35)
        
        if k in [2,5]:s.hold('partial-sum-'+str(k+1),1.6)
        s.play(FadeOut(rings),FadeOut(term),run_time=.2)
    s.hold('nine-products-sum-to-three',3)
    s.chapter('把这个和，放回它的输出位置','The sum belongs to one output position.','九个乘积收成一个响应；它不是九个输出。','Nine products become one response, not nine outputs.')
    s.wipe();source=grid(d['input'],.48,(-3.9,0));out=grid(d['output'],.55,(3.4,0),GREEN,labels=False)
    s.show(source,out,txt('输入',24,BLUE).move_to(pos(-3.9,1.95)),txt('输出  5 × 5',24,GREEN).move_to(pos(3.4,1.95)))
    window=SurroundingRectangle(VGroup(*(source.cells[i][j] for i in range(1,4) for j in range(1,4))),color=GOLD,buff=.02)
    sumtoken=card(num(value),'一个局部响应',GREEN).move_to(pos(0,0));s.show(window,sumtoken)
    label=txt(num(value),25,GREEN).move_to(out.cells[1][1]);s.play(TransformFromCopy(sumtoken,label),run_time=1.2);s.hold('response-placed',3)
    s.say('移动窗口，重复同一运算；核的九个权重不变。','Slide the window and reuse exactly the same nine weights.')
    for r,c in [(1,2),(1,3),(2,3),(3,3)]:
        box=SurroundingRectangle(VGroup(*(source.cells[i][j] for i in range(r,r+3) for j in range(c,c+3))),color=GOLD,buff=.02)
        v=d['output'][r][c];s.play(Transform(window,box),run_time=.65)
        s.play(Transform(sumtoken,card(num(v),'同一组核权重',GREEN).move_to(sumtoken)),TransformFromCopy(window,txt(num(v),25,GREEN).move_to(out.cells[r][c])),run_time=.7);s.wait(.5)
    s.hold('shared-kernel',3)
    s.ending('现在再读卷积的求和式','The indices now refer to visible objects.','y_(r,c)=sum_(i,j) K_(i,j) X_(r+i,c+j)','亲手只改列位置，先预测响应，再移动窗口核对。','Change only the column. Predict the response before checking it.')


def backprop(s):
    d=s.d;f=d['forward'];g=d['gradients'];print('backprop',list(d),flush=True)
    vals=[s.p['x'],f['z'],f['h'],f['output'],f['loss']]
    names=['输入 x','z = wx + b','h = tanh(z)','ŷ = vh','L = ½(ŷ−y)²']
    s.chapter('先让一个输入走到损失','One input, four operations, one loss.','先算前向数值。反向时，仍沿着同一条路径。','The backward pass will reuse this exact forward path.')
    nodes=VGroup(*(card(num(v,3),names[i],BLUE if i<4 else GOLD,1.8).move_to(pos(-4.9+i*2.45,.55)) for i,v in enumerate(vals)))
    s.add(nodes[0]);s.hold('input',2)
    arrows=VGroup()
    for i in range(1,5):
        a=arrow(nodes[i-1][0].get_right()+RIGHT*.08,nodes[i][0].get_left()-RIGHT*.08,BLUE);arrows.add(a)
        s.play(GrowArrow(a),TransformFromCopy(nodes[i-1][1],nodes[i]),run_time=.9);s.wait(1)
    s.show(txt('w = 0.5     b = 0     v = 2     y = 1',24,MUTED).move_to(pos(0,-1.2)));s.hold('forward-values',4)
    s.chapter('局部导数，把上游变化传回来','Upstream sensitivity × local derivative.','从损失出发；每跨过一个运算，就乘一次局部导数。','Start at the loss and multiply by one local derivative at each edge.')
    upstream=1.
    locals_=[f['output']-s.p['target'],s.p['v'],1-f['h']**2,s.p['w']]
    formulas=['hat(y)-y','v','1-h^2','w']
    for k,i in enumerate([3,2,1,0]):
        value=upstream*locals_[k]
        a=arrow(pos(nodes[i+1].get_x(),1.55),pos(nodes[i].get_x(),1.55),GOLD)
        expression=eq(f'{num(upstream,4).replace("−","-")} times ({num(locals_[k],4).replace("−","-")}) = {num(value,4).replace("−","-")}',32,GOLD).move_to(pos(0,-2.05))
        small=eq(formulas[k],25,GOLD).next_to(a,UP,buff=.1)
        s.play(GrowArrow(a),FadeIn(small),Write(expression),run_time=1);s.hold('backward-edge-'+str(k),2.7);s.play(FadeOut(expression),run_time=.2);upstream=value
    assert abs(upstream-g['x'])<1e-10
    s.chapter('同一个节点，还要把梯度交给参数','Branch from z into its trainable parameters.','z = wx + b：对 w 的局部导数是 x，对 b 是 1。','For z = wx + b, the local derivatives are x and 1.')
    s.wipe();up=card(num(d['dZ'],4),'已传回的 ∂L/∂z',GOLD,3).move_to(pos(0,1.2));up[2].next_to(up[0],UP,buff=.15);up.shift(.1*DOWN);s.show(up)
    for x,key,local in [(-3,'w',s.p['x']),(3,'bias',1)]:
        target=card(num(g[key],4),'∂L/∂'+('b' if key=='bias' else key),GREEN,2.6).move_to(pos(x,-1.05))
        s.play(GrowArrow(arrow(up[0].get_bottom(),target[0].get_top())),run_time=.7);s.show(target,txt('× '+num(local),25,GOLD).move_to(pos(x*.5,.1)));s.wait(2)
    s.hold('parameter-branches',3)
    s.ending('链式法则是在组合敏感度','The chain rule composes local sensitivities.','(partial L)/(partial w)=(partial L)/(partial hat(y)) (partial hat(y))/(partial h) (partial h)/(partial z) x','公式中的每一项，刚才都对应一条反向边。','Each factor corresponds to an edge you just followed.')


def attention(s):
    d=s.d;q=d['selectedQuery'];scores=d['scores'][q];weights=d['weights'][q];V=d['V'];Q=d['Q'];K=d['K']
    assert np.allclose(np.array(weights)@V,d['outputs'][q])
    s.chapter('一个 Query，先逐个比较 Key','A query scores each key before reading any value.','固定第 3 个 Query；每次内积再除以 √2，得到一个分数。','Fix query 3. Divide each query–key dot product by √2.')
    query=card(str(Q[q]),'Query 3',GOLD,2).move_to(pos(-4.6,.4));keys=VGroup(*(card(str(v),'Key '+str(i+1),BLUE,1.65).move_to(pos(-1.5+i*2.05,.4)) for i,v in enumerate(K)))
    s.add(query,keys);s.hold('query-and-keys',3)
    scoreitems=VGroup()
    for i in range(4):
        ring=SurroundingRectangle(keys[i][0],color=GOLD,buff=.04);v=card(num(scores[i],3),'分数 '+str(i+1),GOLD,1.65).move_to(pos(-1.5+i*2.05,-1.5))
        s.play(Create(ring),TransformFromCopy(VGroup(query[1],keys[i][1]),v),run_time=1);scoreitems.add(v);s.wait(.8);s.remove(ring)
    s.hold('four-dot-products',3)
    s.chapter('把分数变成共享总量的权重','Normalize across the same query row.','指数为正；除以同一个总和，权重之和才是 1。','Positive exponentials share one denominator, so the row sums to one.')
    s.wipe();logits=row(scores,['s₁','s₂','s₃','s₄'],GOLD,2.1,1.1);s.show(logits)
    exps=np.exp(np.array(scores)-max(scores));middle=row(exps,['exp(s−max s)']*4,BLUE,2.1,-.25)
    for i in range(4):s.play(TransformFromCopy(logits[i],middle[i]),run_time=.55)
    denom=txt('共用分母  '+num(sum(exps),3),27,GOLD).move_to(pos(0,-1.3));s.show(denom);s.hold('common-denominator',3)
    s.play(FadeOut(denom),run_time=.4)
    probs=row(weights,['权重 1','权重 2','权重 3','权重 4'],GREEN,2.1,-1.7)
    s.play(*(TransformFromCopy(middle[i],probs[i]) for i in range(4)),run_time=1.3);s.hold('weights',3)
    s.chapter('每个权重，都作用在一个 Value 上','The same weights mix the value vectors.','分数决定读多少；Value 决定读到什么。','Scores determine how much to read; values determine what is read.')
    s.wipe();values=VGroup(*(card(str(v),'Value '+str(i+1),BLUE,1.9).move_to(pos(-3.6+i*2.4,.85)) for i,v in enumerate(V)))
    s.show(values);products=VGroup()
    for i in range(4):
        vec=np.array(V[i])*weights[i];m=card('['+', '.join(num(x,3) for x in vec)+']','× '+num(weights[i],3),GREEN,2.15).move_to(pos(-3.6+i*2.4,-.9))
        s.play(TransformFromCopy(values[i],m),run_time=.75);products.add(m);s.wait(.5)
    s.hold('weighted-values',3)
    output=card('['+', '.join(num(x,3) for x in d['outputs'][q])+']','Query 3 的输出',GREEN,3.3).move_to(pos(0,.15))
    s.play(FadeOut(values),ReplacementTransform(products,output),run_time=1.5);s.hold('one-output-vector',4)
    s.wipe();s.chapter('第 3 个 Query，不能读第 4 个位置','Mask future scores before normalization.','先把第 4 个分数设为 −∞，再对前三个位置重新归一化。','Set the future score to −∞, then normalize over the three visible positions.')
    masked=s.payload['comparisons']['causal']['data'];masked_weights=masked['weights'][q]
    visible=np.exp(np.array(scores[:q+1])-max(scores[:q+1]));visible/=sum(visible)
    assert np.allclose(masked_weights,[*visible,0])
    logits=row(scores,['位置 1','位置 2','位置 3','未来位置 4'],GOLD,2.4,1.1);s.show(logits)
    hidden=dotlabel('-Infinity','未来位置 4',RED).move_to(logits[3]);s.play(Transform(logits[3],hidden),run_time=1.2);s.hold('future-score-masked',3)
    masked_probs=row(masked_weights,['可读','可读','可读','不可读'],GREEN,2.4,-.7)
    for i in range(4):s.play(TransformFromCopy(logits[i],masked_probs[i]),run_time=.7)
    s.show(txt('权重和 = 1；第 4 项 = 0',26,GREEN).move_to(pos(0,-1.9)));s.hold('renormalized-visible-keys',4)
    s.ending('可见范围改变，加权读取也随之改变','Visibility changes the weighted read.','O_i=sum_j A_(i,j)V_j','掩码在 Softmax 前施加；注意力权重本身不是因果效应。','Mask before softmax. Attention weights themselves are not causal effects.')


def pooling(s):
    d=s.d;v=float(d['output'][d['row']][d['col']]);patch=np.array(d['patch']);flat=patch.flatten();k=patch.shape[0]
    s.chapter('从一个窗口，只留下一个摘要','A pooling window produces one number.','先保留窗口内每个数的身份，再看谁决定输出。','Keep track of each input before summarizing the window.')
    source=grid(d['input'],.55,(-3.5,0));window=SurroundingRectangle(VGroup(*(source.cells[i][j] for i in range(d['row']*s.p['stride'],d['row']*s.p['stride']+k) for j in range(d['col']*s.p['stride'],d['col']*s.p['stride']+k))),color=GOLD,buff=.02)
    block=grid(d['patch'],.95,(1.15,.25));s.add(source,window);s.hold('window',3);s.play(TransformFromCopy(window,block),run_time=1.2);s.hold('patch',3)
    s.chapter('最大池化，让最大值留下来','Max pooling selects an input value.','最大值成为输出；它不是把窗口内的数相加。','The maximum becomes the output, without adding the other values.')
    winner=int(np.argmax(flat));i,j=divmod(winner,k);ring=SurroundingRectangle(block.cells[i][j],color=GOLD,buff=.03);out=card(num(v),'最大池化输出',GREEN,2).move_to(pos(4.4,.25))
    s.play(Create(ring));s.play(TransformFromCopy(block.cells[i][j],out),run_time=1.1);s.hold('maximum-selected',4)
    s.chapter('从输出往回问：哪个输入有影响？','Trace sensitivity back to the selected entry.','上游梯度为 1 时，这次选择把它交给最大值的位置。','With unit upstream gradient, this selection routes it to the maximum.')
    s.wipe();left=grid(patch,.95,(-2.5,0));grad=np.zeros_like(patch);grad[i,j]=1;right=grid(grad,.95,(2.5,0),GOLD);s.show(left)
    upstream=txt('上游梯度 = 1',27,GOLD).move_to(pos(2.5,1.8));s.show(upstream);s.play(TransformFromCopy(upstream,right),run_time=1.3);s.show(arrow(pos(1,0),pos(-1,0),GOLD));s.hold('gradient-location',4)
    s.chapter('平均池化，让每个输入分担','Mean pooling shares both value and sensitivity.','四个数各贡献四分之一；对应梯度也各为四分之一。','For this 2 × 2 window, each input receives weight one quarter.')
    mean=card(num(float(np.mean(patch))),'平均值',GREEN,2.5).move_to(pos(0,1.4));s.show(mean);s.play(Transform(right,grid(np.ones_like(patch)/patch.size,.95,(2.5,0),GREEN)),run_time=1.2);s.hold('mean-gradient',4)
    s.ending('选择规则，决定梯度流向','The reduction rule determines the backward route.','y_("max")=max_i x_i quad y_("mean")=1/k sum_i x_i','最大值并列时有多个有效次梯度；本实现选择第一个最大值。','At a tie, this implementation selects the first maximizer, one valid subgradient.')


def neuron(s):
    p=s.p;d=s.d;contributions=[p['w1']*p['x1'],p['w2']*p['x2'],p['bias']]
    assert abs(sum(contributions)-d['z'])<1e-9
    s.chapter('神经元先把贡献加起来','Inputs make weighted contributions.','一个输入先乘自己的权重，偏置再单独加入。','Each input is weighted; the bias contributes once.')
    ins=row([p['x1'],p['x2'],1],['x₁','x₂','偏置输入'],BLUE,3,1);s.add(ins);s.hold('inputs',3)
    out=row(contributions,['× '+num(p['w1']),'× '+num(p['w2']),'× '+num(p['bias'])],GOLD,3,-.65)
    for i in range(3):s.play(TransformFromCopy(ins[i],out[i]),run_time=.9);s.wait(.8)
    s.hold('three-contributions',3)
    z=card(num(d['z'],3),'z：加权和',GOLD,2.6).move_to(pos(0,.15));s.play(FadeOut(ins),ReplacementTransform(out,z),run_time=1.3);s.hold('sum',3)
    s.chapter('再把加权和，送进非线性函数','The sigmoid turns a score into a probability.','先有 z，再有 σ(z)；概率不是输入贡献的直接相加。','The probability is a nonlinear function of the total score.')
    s.wipe();a,f=axes([-6,6],[0,1],size=(9,3.2),labels=('加权和 z','输出 σ(z)'));c=curve([[x,1/(1+np.exp(-x))] for x in np.linspace(-6,6,181)],f,BLUE)
    s.show(a);s.play(Create(c),run_time=1.5);dot=Dot(f(d['z'],d['probability']),radius=.09,color=GOLD);s.show(dot,DashedLine(f(d['z'],0),dot.get_center(),color=GOLD));s.hold('sigmoid-mapping',4)
    s.ending('一个权重控制一条局部影响路径','A local derivative follows the weighted path.','(partial sigma(z))/(partial x_i)=sigma(z)(1-sigma(z)) w_i','改变一个输入，先看 z 怎样变，再看概率为何变化变慢。','Change one input; follow the score before interpreting saturation.')


def softmax(s):
    d=s.d;print('softmax',list(d),flush=True)
    scores=d.get('logits',d.get('scores'));probs=d.get('probabilities',d.get('p'))
    if scores is None:scores=[s.p['z1']+s.p['shift'],s.p['z2']+s.p['shift'],s.p['z3']+s.p['shift']]
    if probs is None:probs=d['probability']
    z=np.array(scores)/s.p['temperature'];e=np.exp(z-max(z));assert np.allclose(e/e.sum(),probs)
    s.chapter('三个分数，要共同分一份概率','Three scores compete for one unit of probability.','分数可以为负，概率必须非负，而且总和是 1。','Scores may be negative; probabilities must be nonnegative and sum to one.')
    a=row(scores,['分数 1','分数 2','分数 3'],BLUE,3,1);s.add(a);s.hold('scores',4)
    s.chapter('先指数化，再使用同一个分母','Exponentiate, then divide by a shared denominator.','减去最大分数只为数值稳定，不改变最终概率。','Subtracting the maximum stabilizes the calculation without changing probabilities.')
    b=row(e,['exp(z₁/T−max)','exp(z₂/T−max)','exp(z₃/T−max)'],GOLD,3,-.4)
    for i in range(3):s.play(TransformFromCopy(a[i],b[i]),run_time=.85);s.wait(.7)
    total=card(num(e.sum(),3),'共同分母',GOLD,2.5).move_to(pos(0,-1.9));s.play(TransformFromCopy(b,total),run_time=1.2);s.hold('denominator',4)
    s.chapter('一个概率变化，其他概率也要响应','Normalization couples the three outputs.','这三个数来自同一个分母，不是三个独立的 Sigmoid。','These values share a denominator; they are not independent sigmoids.')
    s.wipe();ps=row(probs,['p₁','p₂','p₃'],GREEN,3,.9);s.show(ps);bar=VGroup();x=-4.5
    for i,v in enumerate(probs):
        rect=Rectangle(width=max(.002,9*v),height=.65,color=PALETTE[i],fill_opacity=.85).move_to(pos(x+9*v/2,-.8));x+=9*v;s.play(TransformFromCopy(ps[i],rect),run_time=.9);bar.add(rect)
    s.show(txt('总长度 = 1',26,MUTED).move_to(pos(0,-1.65)));s.hold('one-unit-probability',4)
    s.ending('标签影响损失，分数决定预测','Changing the label changes loss, not this prediction.','p_i=exp(z_i/T)/(sum_j exp(z_j/T))','固定三个分数，只改真实类别；预测不变，损失与梯度会变。','Hold scores fixed and change the target: loss and gradients change, probabilities do not.')


def normalization(s):
    d=s.d;X=np.array(d['X']);Y=np.array(d['normalized']);mode=s.p['mode']
    s.chapter('先回答：哪些数一起算均值？','Choose the statistical group before normalizing.','这里是 Batch 模式：同一特征列里的四个样本组成一组。','In batch mode, observations within one feature form the group.')
    a=grid(X,.7,(-2.8,0));s.add(a,txt('行：样本   列：特征',23,MUTED).move_to(pos(-2.8,1.9)))
    s.hold('input-matrix',3)
    for j in range(X.shape[1]):
        col=VGroup(*(a.cells[i][j] for i in range(X.shape[0])));outline=SurroundingRectangle(col,color=PALETTE[j],buff=.03)
        m=card(num(float(X[:,j].mean()),3),'这一列的均值',PALETTE[j],2.8).move_to(pos(2.5,.5))
        v=txt('方差  '+num(float(X[:,j].var()),3),27,PALETTE[j]).move_to(pos(2.5,-.7))
        s.play(Create(outline),TransformFromCopy(col,m),FadeIn(v),run_time=1);s.hold('group-'+str(j),2.5);s.play(FadeOut(m),FadeOut(v),FadeOut(outline),run_time=.35)
    s.chapter('同一组，减同一个均值、除同一个尺度','Center and scale using that group only.','每个元素的位置保持不变，变化的是它相对本组的数值。','Positions stay fixed; values become deviations relative to their own group.')
    b=grid(Y,.7,(2.8,0),GREEN);s.play(*(TransformFromCopy(a.cells[i][j],b.cells[i][j]) for i in range(4) for j in range(3)),run_time=2)
    s.show(arrow(pos(-1.35,0),pos(1.35,0)),eq('(x-mu)/sqrt(v+epsilon)',29).move_to(pos(0,-2.15)));s.hold('standardized-matrix',4)
    s.chapter('仿射参数在标准化之后作用','Apply the learned scale and shift afterward.','γ 放大或缩小，β 平移；它们不重新选择统计分组。','Gamma rescales and beta shifts without changing the grouping rule.')
    out=grid(d['Y'],.7,(2.8,0),GOLD);s.play(Transform(b,out),run_time=1.2);s.show(txt('γ = '+num(s.p['gamma'])+'     β = '+num(s.p['beta']),25,GOLD).move_to(pos(2.8,1.95)));s.hold('affine-output',4)
    s.ending('Layer Norm 改的是分组方向','Layer normalization uses features within each observation.','y=gamma (x-mu)/sqrt(v+epsilon)+beta','亲手切到 Layer 模式：统计组从列变成行，再检查同一格。','Switch to layer mode: the group changes from a column to a row.')


STORIES={'convolution':convolution,'backprop':backprop,'attention':attention,'pooling':pooling,'neuron':neuron,'softmax':softmax,'normalization':normalization}


def activations(s):
    d=s.d;p=s.p
    s.chapter('一个输入，怎样穿过激活函数？','Follow a single input through the nonlinearity.','先在横轴找到输入，再向上读输出。','Locate the input on the horizontal axis, then read the output vertically.')
    a,f=axes([-6,6],[0,1],labels=('输入 z','Sigmoid 输出'));c=curve(d['curve'],f,BLUE)
    s.add(a,c);t=ValueTracker(-3.5);point=always_redraw(lambda:Dot(f(t.get_value(),1/(1+np.exp(-t.get_value()))),radius=.085,color=GOLD))
    guide=always_redraw(lambda:DashedLine(f(t.get_value(),0),point.get_center(),color=GOLD));s.add(point,guide);s.hold('far-left',3)
    for x in [0,1,3.5]:s.play(t.animate.set_value(x),run_time=2);s.hold('input-'+str(x),3)
    s.chapter('相同的输入变化，不一定有相同的输出变化','The slope changes the transmitted perturbation.','中心附近变化较快；两端饱和后，输出变化很小。','The center is sensitive; saturated tails transmit little change.')
    for x in [0,4]:
        s.play(t.animate.set_value(x),run_time=1.2);y=1/(1+np.exp(-x));dy=y*(1-y)
        tangent=Line(f(x-.75,y-.75*dy),f(x+.75,y+.75*dy),color=GREEN,stroke_width=4)
        badge=card(num(dy,3),'局部斜率',GREEN,2.4).move_to(pos(-3,1));s.show(tangent,badge);s.hold('slope-'+str(x),4);s.play(FadeOut(tangent),FadeOut(badge),run_time=.4)
    point.clear_updaters();guide.clear_updaters()
    s.ending('激活值和导数，是两个不同的问题','The output and its sensitivity answer different questions.','sigma prime(z)=sigma(z)(1-sigma(z))','亲手切换激活函数，同时比较输出曲线与导数曲线。','Compare both the output and derivative when switching activation functions.')


def mlp(s):
    d=s.d;xs=d['xs'];h=d['contributions'];assert np.allclose(np.sum(h,axis=0)+s.p['offset'],d['output'])
    s.chapter('先做出几个有转折点的部件','Hidden units create reusable pieces.','每个 ReLU 在自己的转折点之后开始响应。','Each ReLU begins responding after its own knot.')
    yr=[min(-1,float(np.min(h))),max(2,float(np.max(h)))];a,f=axes([-3,3],yr,labels=('输入 x',''));s.add(a)
    lines=[]
    for i,ys in enumerate(h):
        c=curve(list(zip(xs,ys)),f,PALETTE[i]);name=txt('单元 '+str(i+1)+'：系数 '+num(d['coefficients'][i]),23,PALETTE[i]).move_to(pos(-3+i*3,1.65))
        s.play(Create(c),FadeIn(name),run_time=1.5);lines.append(c);s.hold('unit-'+str(i),3)
    s.chapter('同一个 x，先读取每个部件的贡献','Read every contribution at the same input.','把输入位置固定，才是在相加同一个点的输出。','Fix the input before adding outputs from different units.')
    index=min(range(len(xs)),key=lambda i:abs(xs[i]-.8));x=xs[index]
    vertical=DashedLine(f(x,yr[0]),f(x,yr[1]),color=GOLD);s.show(vertical)
    tokens=row([v[index] for v in h],['贡献 '+str(i+1) for i in range(len(h))],GOLD,2,-1.5);s.show(tokens);s.hold('one-input-contributions',4)
    total=card(num(d['output'][index],3),'这些贡献的和',GREEN,3).move_to(pos(0,-1.5));s.play(ReplacementTransform(tokens,total),run_time=1.3);s.hold('sum-at-input',3)
    s.chapter('对每个 x 重复，得到整条输出曲线','Pointwise addition creates the full function.','这里展示手工设定的组合，尚未发生训练。','These coefficients are hand-set; no training has occurred in this example.')
    s.wipe();a,f=axes([-3,3],[min(d['output'])-.2,max(d['output'])+.2],labels=('输入 x','网络输出'));s.show(a);s.play(Create(curve(list(zip(xs,d['output'])),f,GREEN)),run_time=3);s.hold('network-function',5)
    s.ending('宽度增加的是可组合的部件','More hidden units provide more pieces to combine.','f(x)=b+sum_i a_i "ReLU"(x-t_i)','亲手改变隐藏层宽度，观察新增部件如何改变输出。','Change the hidden width and inspect how the added pieces change the output.')


def regularization(s):
    d=s.d;q=d['keepProbability'];values=d['contributions'];mask=d['masks'][0];scaled=np.array(values)*mask/q
    assert abs(sum(scaled)-d['outputs'][0])<1e-9
    s.chapter('一次 Dropout，究竟丢掉了什么？','Follow one actual dropout mask.','先固定四项贡献，再抽取一次保留或丢弃的掩码。','Keep the contributions fixed and sample a binary mask.')
    a=row(values,['贡献 1','贡献 2','贡献 3','贡献 4'],BLUE,2.3,1);s.add(a);s.hold('contributions',3)
    b=row(mask,['保留' if v else '丢弃' for v in mask],GOLD,2.3,-.4)
    for i in range(4):s.play(TransformFromCopy(a[i],b[i]),run_time=.6);s.wait(.4)
    s.hold('actual-mask',3)
    s.chapter('保留下来的项，还要除以保留概率','Inverted dropout rescales the survivors.','本例保留概率为 0.5，留下来的贡献乘以 2。','Here q = 0.5, so surviving contributions are doubled.')
    c=row(scaled,['m₁c₁/q','m₂c₂/q','m₃c₃/q','m₄c₄/q'],GREEN,2.3,-1.8);s.play(*(TransformFromCopy(b[i],c[i]) for i in range(4)),run_time=1.5);s.hold('scaled-survivors',4)
    s.wipe();out=card(num(float(sum(scaled))),'这一次的输出',GREEN,3).move_to(pos(0,.5));s.show(out);s.hold('one-output',3)
    s.chapter('一次会变，平均值保持什么？','Random outputs can have a stable expectation.','重复抽掩码会产生不同输出；期望等于原始贡献之和。','Masks change individual outputs; their expectation equals the original sum.')
    examples=row(d['outputs'][:8],[str(i+1) for i in range(8)],BLUE,1.28,-1.2);s.show(examples,txt('前 8 次真实掩码的输出',23,MUTED).move_to(pos(0,-2.1)));s.hold('random-outputs',4)
    s.ending('保持期望，不等于每次保持输出','Preserving the mean does not preserve each realization.','E[m c/q]=c','亲手只改丢弃率，同时观察输出均值与方差。','Change dropout probability and compare both mean and variance.')


def residual(s):
    d=s.d;paths=d['paths'];plain=paths['plain'];res=paths['residual']
    s.chapter('一条捷径，怎样改变每一层？','An identity path changes the layer, not just the diagram.','普通层只输出变换结果；残差层把输入再加回来。','A plain layer outputs the transform; a residual layer adds its input.')
    left=card('h','输入',BLUE,1.5).move_to(pos(-4,0));middle=card('f(h)','变换',GOLD,2).move_to(pos(0,0));right=card('h + f(h)','输出',GREEN,2.6).move_to(pos(4,0));s.add(left);s.hold('input',2)
    s.play(GrowArrow(arrow(pos(-3.15,0),pos(-1.1,0))),FadeIn(middle),run_time=1);s.play(GrowArrow(arrow(pos(1.1,0),pos(2.5,0))),FadeIn(right),run_time=1)
    shortcut=VMobject(color=BLUE,stroke_width=3).set_points_as_corners([pos(-4,.6),pos(-4,1.45),pos(4,1.45),pos(4,.6)]);s.play(Create(shortcut),run_time=1.5);s.show(txt('输入直接相加',23,BLUE).move_to(pos(0,1.7)));s.hold('identity-path',4)
    s.chapter('梯度也沿两条路相加','The two derivative paths meet at the addition.','恒等路径贡献 1，变换路径贡献 f′(h)。','The identity contributes 1; the transformed path contributes f′(h).')
    s.equation('J_("res")=1+f prime(h)',y=-1.8);s.hold('two-gradient-paths',4)
    s.wipe();s.chapter('多层以后，仍要看实际的连乘','Depth still multiplies the local factors.','把前六层一层层接起来；比较同一输入的两种路径。','Apply the first six layers to the same input and compare both chains.')
    top=row([plain[i]['value'] for i in range(7)],[str(i) for i in range(7)],GOLD,1.55,.85);bottom=row([res[i]['value'] for i in range(7)],[str(i) for i in range(7)],GREEN,1.55,-.9)
    s.show(txt('普通层',21,GOLD).move_to(pos(-5.5,1.7)),txt('残差层',21,GREEN).move_to(pos(-5.5,-.1)))
    for i in range(7):s.play(FadeIn(top[i]),FadeIn(bottom[i]),run_time=.55);s.wait(.45)
    s.hold('six-layers',4)
    s.ending('加上恒等路径，也不保证稳定','Identity helps a path; it does not bound every product.','J_("total")=product_l (1+f_l prime(h_l))','试线性变换 w = −1：每层局部因子为 0，梯度仍会消失。','Try a linear transform with w = −1: each residual factor is zero.')


def rnn(s):
    d=s.d;T=s.p['steps'];states=d['states'];assert len(states)==T+1
    factors=np.array(d['local'])*s.p['weight']
    assert np.allclose(np.r_[np.cumprod(factors[::-1])[::-1],1],d['sensitivities'])
    s.chapter('把同一个单元，沿时间展开','One recurrent cell is reused over time.','只有第一步有脉冲输入；后面靠隐藏状态传递影响。','A pulse enters at the first step; later states carry its influence.')
    nodes=row(states,[f'h{i}' for i in range(T+1)],BLUE,1.65,0);s.add(nodes[0]);s.hold('initial-state',2)
    for i in range(T):
        a=arrow(nodes[i][0].get_right()+.05*RIGHT,nodes[i+1][0].get_left()-.05*RIGHT,GOLD)
        inp=txt('x = '+num(d['xs'][i]),21,MUTED).move_to(pos(nodes[i+1].get_x(),1.05))
        s.play(GrowArrow(a),TransformFromCopy(nodes[i],nodes[i+1]),FadeIn(inp),run_time=.85);s.wait(.9)
    weight_label=txt('每一步都使用 a = '+num(s.p['weight']),26,GOLD).move_to(pos(0,-1.3));s.show(weight_label);s.hold('unrolled-forward',4)
    s.chapter('旧输入的影响，要穿过后续每一步','Sensitivity travels through every intervening step.','局部导数接连相乘：一个很小的因子就会削弱后续影响。','Local derivatives multiply along the temporal path.')
    s.play(FadeOut(weight_label),run_time=.3);factors=[s.p['weight']*v for v in d['local']];local=row(factors,['a × φ′']*T,GOLD,1.8,-1.4);s.show(local);s.hold('local-factors',4)
    s.wipe();s.chapter('共享参数，收到的是各次使用的总贡献','A shared parameter receives a sum across its uses.','不要把每个时刻当作独立的权重；这些贡献都交给同一个 a。','The weights are tied: every time-step contribution belongs to the same parameter.')
    terms=row(d['weightContributions'],[f't={i+1}' for i in range(T)],GOLD,1.8,.8);s.show(terms);s.hold('weight-contributions',4)
    total=card(num(d['weightGradient'],4),'∂L/∂a',GREEN,3);s.play(ReplacementTransform(terms,total),run_time=1.5);s.hold('shared-gradient',4)
    s.ending('前向递推，反向连乘，参数贡献相加','Recurrence, products along paths, sums into shared parameters.','h_t=phi(a h_(t-1)+b x_t)','固定序列，只改递归权重，观察状态保留与梯度变化。','Hold the sequence fixed and compare state retention with gradient transmission.')


def lstm(s):
    d=s.d;states=d['states'];g=d['gates'];print('lstm first',states[:2],flush=True)
    s.chapter('记忆状态，和可见输出，是两条线','Cell memory and hidden output are distinct states.','本例没有新输入：先看记忆怎样被遗忘门逐步保留。','With no new input, inspect how the forget gate retains the cell state.')
    vals=[v['c'] for v in states];out=[v['h'] for v in states];n=min(7,len(vals));top=row(vals[:n],[f'c{i}' for i in range(n)],BLUE,1.6,.85);bottom=row(out[:n],[f'h{i}' for i in range(n)],GREEN,1.6,-1.1)
    for i in range(n):
        s.play(FadeIn(top[i]),run_time=.6)
        if i:s.show(txt('× '+num(g['f']),18,GOLD).move_to(pos((top[i].get_x()+top[i-1].get_x())/2,1.65)),run_time=.25)
        s.play(TransformFromCopy(top[i],bottom[i]),run_time=.65);s.wait(.5)
    s.hold('cell-and-hidden',4)
    s.chapter('输出门控制读出，不直接改写记忆','The output gate controls exposure of the cell.','h = o·tanh(c)：输出小，不一定意味着内部记忆已经消失。','A small hidden output does not by itself imply that cell memory is gone.')
    s.equation('h_t=o tanh(c_t)',y=-2.1,size=32);s.hold('readout',4)
    s.wipe();s.chapter('只沿记忆直通路径，梯度怎么走？','Follow the direct cell-state shortcut.','每跨一步乘一次 f；本例固定门值，且没有候选状态反馈。','Each direct step contributes f. Gates are fixed and candidate feedback is disabled here.')
    products=row([g['f']**i for i in range(7)],[f'{i} 步' for i in range(7)],GOLD,1.6,.55)
    for item in products:s.show(item,run_time=.4);s.wait(.55)
    s.hold('direct-gradient-products',5)
    s.ending('直通路径的导数，不总是完整导数','The direct-path derivative need not be the full derivative.','J_("direct")=f^T','加入候选状态反馈后会多出路径；需比较完整递推导数。','Candidate-state feedback adds paths. Compare the full recurrent derivative when it is enabled.')


def gnn(s):
    d=s.d;positions=[pos(-4.8+i*1.92,.35) for i in range(6)]
    s.chapter('一层消息传递，只跨一条边','One layer carries information across one edge.','先只在节点 1 放一个信号；其余节点从 0 开始。','Only node 1 starts with a signal.')
    nodes=row(d['history'][0],[f'节点 {i+1}' for i in range(6)],BLUE,1.92,.35);positions=[n[0].get_center() for n in nodes];links=VGroup(*(Line(positions[a],positions[b],color=MUTED,stroke_width=2.5) for a,b in d['edges']));s.add(links,nodes);s.hold('initial-signal',4)
    for layer,values in enumerate(d['history'][1:],1):
        s.chapter('第 '+str(layer)+' 层：邻居先发送，再一起汇总','Layer '+str(layer)+': send, then aggregate.','同时读取上一层的邻居和自身值，不会读取本轮刚更新的值。','All nodes read the previous layer, not already-updated neighbors.')
        signals=VGroup();animations=[]
        for a,b in [(u,v) for u,v in d['edges']]+[(v,u) for u,v in d['edges']]:
            if d['history'][layer-1][a]>0:
                m=Dot(positions[a],radius=.06,color=GOLD);signals.add(m);animations.append(m.animate.move_to(positions[b]))
        s.add(signals)
        if animations:s.play(*animations,run_time=1.4)
        new=row(values,[f'节点 {i+1}' for i in range(6)],GREEN,1.92,.35);s.play(Transform(nodes,new),FadeOut(signals),run_time=1)
        s.hold('layer-'+str(layer),4)
    s.ending('传播范围，随层数逐步扩大','The receptive field grows one hop per layer.','H^(l+1)=P H^l','本例使用带自环的邻居平均；归一化方式改变传播算子。','This example averages neighbors with self-loops. Other normalizations change the operator.')


def initialization(s):
    d=s.d;stats=d['forwardStats'];n=len(stats)
    incoming=np.array(d['activations'][0][0]);weights=np.array(d['weights'][0][0]);contributions=incoming*weights;pre=float(contributions.sum());assert abs(pre-d['preactivations'][0][0][0])<1e-10
    s.chapter('先追踪第一层的一个神经元','Track one actual neuron before summarizing the network.','24 项加权贡献先相加，再经过 ReLU；这里展开前四项。','Sum 24 weighted contributions, then apply ReLU. Four terms are shown individually.')
    visible=list(contributions[:4])+[sum(contributions[4:])];parts=row(visible,['贡献 1','贡献 2','贡献 3','贡献 4','其余 20 项和'],BLUE,2.1,.8);s.add(parts);s.hold('weighted-input-terms',4)
    total=card(num(pre,3),'24 项之和；ReLU 后 '+num(max(0,pre),3),GOLD,4).move_to(pos(0,-1));s.play(TransformFromCopy(parts,total),run_time=1.6);s.hold('one-preactivation',4);s.wipe()
    s.chapter('初始化，先决定信号进入下一层的尺度','Initialization sets the scale of transmitted signals.','同一批输入穿过六层随机网络；没有训练更新。','The same input batch passes through six random layers without training.')
    a,f=axes([0,6],[0,max(v['secondMoment'] for v in stats)*1.25],labels=('层数','激活二阶矩 E[h²]'));s.add(a)
    previous=None
    for i,v in enumerate(stats):
        point=Dot(f(i,v['secondMoment']),radius=.085,color=BLUE);label=txt(num(v['secondMoment'],3),22,BLUE).next_to(point,UP,buff=.15)
        if previous is not None:s.play(Create(Line(previous,point.get_center(),color=BLUE)),FadeIn(point),FadeIn(label),run_time=.9)
        else:s.show(point,label)
        previous=point.get_center();s.wait(1)
    s.hold('forward-scale',4)
    s.chapter('梯度从另一端回来，也有自己的尺度','Backward signal scale deserves a separate check.','前向数值尚可，不代表每一层的梯度都同样稳定。','A reasonable forward scale does not establish stable gradients at every layer.')
    s.wipe();v=d['gradientRMS'];a,f=axes([0,len(v)-1],[0,max(v)*1.25],labels=('层数','梯度 RMS'));s.show(a)
    points=list(enumerate(v));s.play(Create(curve(list(reversed(points)),f,GOLD)),run_time=3);s.hold('backward-scale',5)
    s.chapter('为什么要按输入宽度缩放权重？','Weight variance compensates for accumulated inputs.','独立贡献相加会积累方差；He 缩放针对 ReLU 的信号损失作补偿。','Independent contributions accumulate variance; He scaling compensates for ReLU gating.')
    s.wipe();s.equation('"Var"(W_(i,j))=2/n_("in")',y=.75);s.show(txt('本例输入宽度 = '+str(d['width'])+'     权重标准差 = '+num(d['sd'],3),26,MUTED).move_to(pos(0,-.65)));s.hold('he-scale',5)
    s.ending('这是尺度分析，仍须检查具体网络','A scaling argument is not a guarantee for every finite network.','E[h^2] != "Var"(h) quad "when" E[h] != 0','ReLU 后均值通常非零；不要把二阶矩和方差混为一谈。','After ReLU, a nonzero mean makes the second moment differ from the variance.')


def optimizers(s):
    d=s.d;paths=d['trajectories'];names=['gd','momentum','adam'];colors=[BLUE,GOLD,GREEN]
    allxy=[v['x'] for name in names for v in paths[name]];xs=[v[0] for v in allxy];ys=[v[1] for v in allxy]
    s.chapter('同一个起点，三种更新记忆','Three update rules start at the same point.','先看第一步，再看各方法保存的历史如何影响路径。','Start with the first step, then follow how each rule uses its history.')
    a,f=axes([min(xs)-.25,max(xs)+.25],[min(ys)-.3,max(ys)+.3],labels=('x₁','x₂'),equal=True,center=(-1,-.15));s.add(a)
    labels=VGroup(*(txt(n,23,c).move_to(pos(4.5,1.3-i*.6)) for i,(n,c) in enumerate(zip(['GD','Momentum','Adam'],colors))));s.add(labels)
    dots=[Dot(f(*paths[name][0]['x']),radius=.085,color=c) for name,c in zip(names,colors)];s.add(*dots);s.hold('common-start',3)
    for t in range(1,min(13,min(map(len,paths.values())))):
        animations=[]
        for i,name in enumerate(names):
            a0=paths[name][t-1]['x'];b=paths[name][t]['x'];segment=Line(f(*a0),f(*b),color=colors[i],stroke_width=2.5);animations.extend([Create(segment),dots[i].animate.move_to(f(*b))])
        s.play(*animations,run_time=.65);s.wait(.2)
        if t in [1,5,12]:s.hold('step-'+str(t),2)
    s.chapter('路径不同，因为状态不同','Different internal states produce different updates.','Momentum 累积方向；Adam 还估计各坐标的平方梯度尺度。','Momentum accumulates direction; Adam also tracks coordinatewise squared gradients.')
    s.wipe();items=[('GD','当前梯度'),('Momentum','方向的移动平均'),('Adam','一阶与二阶矩估计')]
    for i,(name,meaning) in enumerate(items):s.show(card(name,meaning,colors[i],3).move_to(pos(-4+i*4,.4)));s.wait(2)
    s.hold('optimizer-state',4)
    s.ending('这张轨迹图不构成方法排名','A single parameter setting is not a benchmark.','x_(t+1)=x_t+Delta_t','比较时还需考虑学习率与调参预算；本例只解释更新机制。','A fair comparison also needs learning-rate tuning and a defined tuning budget.')


def positional(s):
    d=s.d;omega=d['frequencies'][0];p=s.p['position'];shift=s.p['shift']
    s.chapter('一对正弦余弦，把位置变成方向','A sine-cosine pair represents position as an angle.','先只看一个频率；位置增加，圆上的方向随之旋转。','At a single frequency, advancing position rotates a unit vector.')
    a,f=axes([-1.2,1.2],[-1.2,1.2],size=(4,4),labels=('sin(pω)','cos(pω)'),equal=True);circle=curve([[np.cos(t),np.sin(t)] for t in np.linspace(0,TAU,181)],f,BLUE);s.add(a,circle)
    t=ValueTracker(0);v=always_redraw(lambda:arrow(f(0,0),f(np.sin(t.get_value()),np.cos(t.get_value())),GOLD));s.add(v);s.hold('position-zero',2);s.play(t.animate.set_value(p*omega),run_time=3);s.hold('position-p',3)
    before=arrow(f(0,0),f(np.sin(p*omega),np.cos(p*omega)),BLUE);s.add(before)
    s.chapter('位置增加，方向按固定角速度旋转','A position shift adds a fixed angular increment.','蓝色保留原方向；位置增加 δ，金色方向再转 δω。','Blue retains the original direction. A position shift δ adds angle δω to the gold vector.')
    s.play(t.animate.set_value((p+shift)*omega),run_time=2);s.hold('position-shift',4);v.clear_updaters()
    s.wipe();s.chapter('多对频率，提供多个位置尺度','Several frequency pairs provide several scales.','高频变化快，低频变化慢；相似度会振荡，并非距离刻度。','Fast and slow frequencies coexist. Similarity can oscillate with distance.')
    for i,w in enumerate(d['frequencies']):
        g,pf=axes([0,31],[-1,1],center=(-3 if i%2==0 else 3,.9 if i<2 else -1.1),size=(4.4,1.05),labels=('' if i<2 else '位置',''))
        s.show(g);s.play(Create(curve([[j,np.sin(j*w)] for j in range(32)],pf,PALETTE[i])),run_time=1);s.show(txt('ω = '+num(w,4),21,PALETTE[i]).move_to(pos(-3 if i%2==0 else 3,1.8 if i<2 else -.2)),run_time=.3)
    s.hold('multiple-frequencies',5)
    s.ending('位置编码提供结构，不替模型理解顺序','The encoding supplies structure for learned use.','"PE"(p)=(sin(p omega_i),cos(p omega_i))_i','亲手固定频率数，只改基数，观察哪些频率真正变化。','Keep the number of pairs fixed and inspect which frequencies change with the base.')


STORIES.update({'activations':activations,'mlp':mlp,'regularization':regularization,'residual':residual,'rnn':rnn,'lstm':lstm,'gnn':gnn,'initialization':initialization,'optimizers':optimizers,'positional':positional})


def vae(s):
    d=s.d;p0=s.p;mu=p0['mu'];sigma=d['sigma'];eps=p0['epsilon'];assert abs(mu+sigma*eps-d['z'])<1e-10
    s.chapter('把随机性放在一个独立的噪声变量里','Put randomness in a separate noise variable.','ε 来自标准正态；μ 与 σ 决定怎样平移和缩放这次抽样。','Epsilon is standard normal; mu and sigma shift and scale this draw.')
    noise=card(num(eps),'固定这一次 ε',BLUE,3).move_to(pos(-4,.5));scale=card(num(sigma*eps),'先乘 σ = '+num(sigma),GOLD,3).move_to(pos(0,.5));sample=card(num(d['z']),'再加 μ = '+num(mu),GREEN,3).move_to(pos(4,.5));s.add(noise);s.hold('noise-draw',3)
    s.play(TransformFromCopy(noise,scale),run_time=1.3);s.hold('scaled-noise',3);s.play(TransformFromCopy(scale,sample),run_time=1.3);s.hold('latent-sample',4)
    s.chapter('抽样点的变化，可以沿确定性路径求导','The sampled value has a differentiable parameter path.','固定 ε 后，z 对 μ 的导数为 1，对 σ 的导数为 ε。','For fixed epsilon, the derivatives are 1 with respect to mu and epsilon with respect to sigma.')
    s.wipe();s.equation('z=mu+sigma epsilon',y=1);s.show(card('1','∂z / ∂μ',BLUE,3).move_to(pos(-2.5,-.8)),card(num(eps),'∂z / ∂σ',GOLD,3).move_to(pos(2.5,-.8)));s.hold('reparameterized-derivatives',5)
    s.chapter('重建与先验匹配，分别计算再相加','Reconstruction and prior matching are distinct terms.','这里用一维高斯和线性解码器，期望重建项可精确计算。','This scalar Gaussian model has an analytically computed expected reconstruction term.')
    s.wipe();r=card(num(d['reconstruction'],3),'期望重建项',BLUE,3.6).move_to(pos(-3,.65));kl=card(num(p0['beta']*d['kl'],3),'β × KL',GOLD,3.6).move_to(pos(3,.65));s.show(r,kl);s.hold('two-objective-terms',4)
    total=card(num(d['total'],3),'总目标',GREEN,3.6).move_to(pos(0,-1.3));s.play(TransformFromCopy(VGroup(r,kl),total),run_time=1.5);s.hold('vae-objective',4)
    s.ending('改噪声样本，和改分布参数，是两种操作','Changing a draw differs from changing its distribution.','"KL"=1/2(mu^2+sigma^2-1-ln sigma^2)','只改 ε 会改变 z；本例精确期望目标与 KL 不因此改变。','Changing epsilon changes the draw, but not this exact expected objective or KL term.')


def gan(s):
    d=s.d;p0=s.p;k=p0['weight'];b=p0['bias'];theta=p0['theta'];D=lambda x:1/(1+np.exp(-(k*x+b)))
    s.chapter('先冻结判别器，只移动一个生成点','Freeze the discriminator and move one generated point.','这条曲线保持不动；生成器参数 θ 就是点的位置。','The discriminator curve is fixed; theta specifies the generated point.')
    a,p=axes([-4,4],[0,1],labels=('生成点位置 θ','固定判别器 D(θ)'));s.add(a,curve([[x,D(x)] for x in np.linspace(-4,4,201)],p,BLUE));dot=Dot(p(theta,D(theta)),radius=.09,color=GOLD);s.show(dot);s.hold('fixed-discriminator',4)
    s.chapter('同一个低分，会产生不同强度的更新','The same low score gives different generator gradients.','饱和目标在 D 很小时梯度很小；非饱和目标保留较强的推动。','At low D, the saturating objective has a small gradient; the non-saturating objective remains stronger.')
    s.wipe();s.show(card(num(d['gradSaturating'],3),'饱和目标梯度',BLUE,4).move_to(pos(-3,.7)),card(num(d['gradNonSaturating'],3),'非饱和目标梯度',GOLD,4).move_to(pos(3,.7)));s.hold('two-gradients',5)
    s.chapter('沿各自梯度走，比较实际位置更新','Follow the two actual update sequences.','学习率相同，起点相同；这里只有生成器在更新。','Both runs share a learning rate and initial point; the discriminator never updates.')
    s.wipe();a,p=axes([0,p0['steps']],[-2.3,1],labels=('更新次数','生成点 θ'));s.show(a)
    for points,color,name in [(d['saturatingPath'],BLUE,'饱和'),(d['nonSaturatingPath'],GOLD,'非饱和')]:
        s.play(Create(curve(points,p,color)),run_time=3);s.show(txt(name,24,color).move_to(pos(-2 if color==BLUE else 2,1.85)));s.hold(name+'-updates',4)
    s.ending('这段只隔离生成器损失的影响','This isolates generator objectives under a frozen discriminator.','g_("sat")=-k D quad g_("non")=-k(1-D)','完整 GAN 会交替训练两方；不能把这里的两条路径当成完整对抗训练。','A full GAN alternates both players. These paths are not a full adversarial training run.')


def diffusion(s):
    d=s.d;p0=s.p;alpha=p0['alpha'];mu=p0['mu'];sigma=p0['sigma'];obs=p0['observed']
    s.chapter('信号与噪声，先分别缩放再相加','Scale signal and noise before adding them.','α 决定信号比例；每次加噪产生的观测不等于原信号。','Alpha controls signal retention; the noisy observation is not the original signal.')
    left=card('√α · x₀','缩放后的信号',BLUE,3.8).move_to(pos(-3,.5));right=card('√(1−α) · ε','缩放后的噪声',GOLD,3.8).move_to(pos(3,.5));s.add(left,right);s.hold('signal-and-noise',4)
    noisy=card('xₜ = '+num(obs),'现在只观察到这个值',GREEN,4).move_to(pos(0,-1.2));s.play(TransformFromCopy(VGroup(left,right),noisy),run_time=1.5);s.hold('one-noisy-observation',4)
    s.chapter('看到 xₜ 后，反过来推断可能的 x₀','Condition on the noisy observation to infer the clean variable.','这里使用已知的一维高斯模型，能精确算出条件分布。','This known one-dimensional Gaussian model permits exact conditioning.')
    s.wipe();a,p=axes([-1.5,5.5],[0,.8],labels=('原始变量 x₀','条件密度'));s.show(a)
    normal=lambda x,m,sd:np.exp(-.5*((x-m)/sd)**2)/(np.sqrt(2*np.pi)*sd)
    prior=curve([[x,normal(x,mu,sigma)] for x in np.linspace(-1.5,5.5,241)],p,BLUE);posterior=curve([[x,normal(x,d['posteriorMean'],np.sqrt(d['posteriorVariance']))] for x in np.linspace(-1.5,5.5,241)],p,GREEN)
    s.play(Create(prior),run_time=2);s.show(txt('蓝：看见 xₜ 前',24,BLUE).move_to(pos(-2,1.75)));s.hold('prior-clean-distribution',3);s.play(Create(posterior),run_time=2);s.show(txt('绿：看见 xₜ 后',24,GREEN).move_to(pos(2,1.75)));s.hold('posterior-clean-distribution',5)
    s.chapter('去噪估计，不是恢复唯一真值','A denoised estimate does not determine a unique original.','条件均值只是一个估计，条件分布仍有非零宽度。','The conditional mean is an estimate; residual conditional variance remains.')
    marker=DashedLine(p(d['posteriorMean'],0),p(d['posteriorMean'],.7),color=GOLD);s.show(marker);s.hold('remaining-uncertainty',4)
    s.ending('这个精确小模型解释去噪的条件推断','This exact toy model isolates conditional denoising.','x_t=sqrt(alpha)x_0+sqrt(1-alpha)epsilon','它没有训练神经网络，也没有执行多步反向采样。','No neural network is trained and no multistep reverse sampler is run in this example.')


def contrastive(s):
    d=s.d;q=np.array(d['q']);keys=np.array(d['keys']);scores=d['scores'];prob=d['probabilities'];T=s.p['temperature'];assert np.allclose(keys@q/T,scores)
    s.chapter('先把查询与各个候选方向摆出来','Compare a query with positive and negative keys.','所有向量长度为 1；夹角决定内积相似度。','Unit vectors make dot products equal cosine similarities.')
    a,p=axes([-1.2,1.2],[-1.2,1.2],size=(4.1,4.1),center=(-1,-.15),equal=True,labels=('第 1 维','第 2 维'));query=arrow(p(0,0),p(*q),GREEN);s.add(a,query);s.wait(.7)
    arrows=[]
    for i,key in enumerate(keys):
        v=arrow(p(0,0),p(*key),GOLD if i==0 else BLUE);s.play(GrowArrow(v),run_time=.8);arrows.append(v)
    s.show(VGroup(txt('查询',24,GREEN),txt('正样本',24,GOLD),txt('负样本',24,BLUE)).arrange(DOWN,buff=.3).move_to(pos(3.3,.5)));s.hold('query-key-geometry',5)
    s.chapter('相似度除以温度，再在候选间归一化','Scale similarities, then normalize across candidates.','正样本也要与负样本竞争同一份概率质量。','The positive key competes with the negatives for one unit of probability.')
    s.wipe();a=row(scores,d['labels'],BLUE,2.5,1);b=row(prob,['归一化权重']*len(prob),GOLD,2.5,-1);s.show(a)
    for i in range(len(prob)):s.play(TransformFromCopy(a[i],b[i]),run_time=.65);s.wait(.45)
    s.hold('contrastive-softmax',5)
    s.chapter('目标只奖励分给正样本的概率','The loss rewards probability assigned to the positive key.','本例查询与一个负样本同样对齐；温度变低也不能打破同分。','Here one negative ties the positive. Lower temperature cannot break that tie.')
    s.wipe();s.show(card(num(prob[0],3),'分给正样本的概率',BLUE,3.7).move_to(pos(-3,.5)));loss=card(num(d['loss'],3),'−log p₊',GOLD,3.7).move_to(pos(3,.5));s.show(loss);s.hold('positive-log-loss',4)
    s.ending('温度放大已有排序，不保证排序正确','Temperature sharpens the existing ranking, including mistakes.','L=-ln(exp(s_+/T)/(sum_j exp(s_j/T)))','若正负 Key 完全相同，查询再怎么转也不能把它们分开。','Identical positive and negative keys cannot be separated by changing the query.')


STORIES.update({'vae':vae,'gan':gan,'diffusion':diffusion,'contrastive':contrastive})
