"""Learning algorithms: show the actual fitting or decision operation, not a decorative sweep."""
from narrative import *


def regression(s):
    d=s.d;X=np.array(d['X']);beta=np.array(d['beta']);pred=np.array(d['predictions']);assert np.allclose(d['intercept']+X@beta,pred)
    s.chapter('线性预测，是几项贡献相加','A linear prediction adds feature contributions.','追踪第一条观测；两个特征已经按样本标准差标准化。','Track the first observation, using the standardized features of this model.')
    vals=[d['intercept'],X[0,0]*beta[0],X[0,1]*beta[1]];items=row(vals,['截距','β₁ × x₁','β₂ × x₂'],BLUE,3,.9);s.add(items);s.hold('weighted-contributions',4)
    prediction=card(num(pred[0],3),'该观测的预测值',GREEN,3.6).move_to(pos(0,-1));s.play(TransformFromCopy(items,prediction),run_time=1.5);s.hold('one-prediction',4)
    s.chapter('残差连接预测与真实观测','A residual connects a prediction to the observed response.','每个残差都来自同一条观测的 y − ŷ。','Each residual compares the observed and fitted values for the same case.')
    s.wipe();bound=max(4,max(abs(v) for v in list(pred[:8])+d['y'][:8])+.4);a,p=axes([-bound,bound],[-bound,bound],size=(4.2,4.2),equal=True,labels=('预测值 ŷ','观测值 y'));s.show(a,Line(p(-bound,-bound),p(bound,bound),color=RULE))
    n=8
    for i in range(n):
        dot=Dot(p(pred[i],d['y'][i]),radius=.065,color=BLUE);res=Line(p(pred[i],pred[i]),p(pred[i],d['y'][i]),color=GOLD,stroke_width=2.5);s.show(dot,res,run_time=.45);s.wait(.4)
    s.hold('residuals',4)
    s.chapter('拟合目标，把所有残差平方汇总','The fitting objective aggregates squared residuals.','大残差代价增长更快；回归系数共同决定整批残差。','Squaring penalizes larger errors more strongly; coefficients jointly affect all residuals.')
    s.wipe();s.show(card(num(sum(v*v for v in d['residuals']),3),'全部 '+str(len(X))+' 条观测的残差平方和',GOLD,5.5));s.hold('least-squares-objective',5)
    s.ending('预测稳定，不代表每个系数都稳定','Correlated predictors can destabilize individual coefficients.','hat(y)=b+x^T beta','两个特征高度相关时，许多系数组合给出相近预测；岭惩罚改变这种取舍。','Highly correlated features can trade coefficients while preserving similar predictions.')


def logistic(s):
    d=s.d;p0=s.p;x=d['observations'][0]['x'];prob=d['probabilities'][0];y=d['observations'][0]['y']
    s.chapter('先产生概率，再决定类别','A probability model and a decision rule are separate.','先跟踪一个输入 x 的分数和概率。','Follow one input through its score into a probability.')
    nodes=[card(num(x,3),'输入 x',BLUE,2.5).move_to(pos(-4,.4)),card(num(p0['weight']*x+p0['intercept'],3),'分数 wx + b',GOLD,2.5).move_to(pos(0,.4)),card(num(prob,3),'正类概率',GREEN,2.5).move_to(pos(4,.4))];s.add(nodes[0]);s.hold('input',2)
    for i in [1,2]:s.play(TransformFromCopy(nodes[i-1],nodes[i]),GrowArrow(arrow(pos(-2.6+(i-1)*4,.4),pos(-1.4+(i-1)*4,.4))),run_time=1.3);s.wait(2)
    s.hold('probability-before-label',3)
    s.chapter('阈值只是把概率切成两边','The threshold splits probabilities into decisions.','同一条概率曲线可以使用不同阈值；预测概率不会因此改变。','Changing the threshold changes labels, not the underlying probabilities.')
    s.wipe();a,p=axes([-3,3],[0,1],labels=('输入 x','正类概率'));s.show(a);s.play(Create(curve([[q,1/(1+np.exp(-(p0['weight']*q+p0['intercept'])))] for q in np.linspace(-3,3,201)],p,GREEN)),run_time=2)
    line=Line(p(-3,.5),p(3,.5),color=GOLD);s.show(line);s.hold('threshold-half',3);s.play(Transform(line,Line(p(-3,.8),p(3,.8),color=GOLD)),run_time=1.8);s.hold('threshold-eight-tenths',4)
    s.chapter('训练看概率是否支持真实标签','Log loss evaluates probability assigned to the observed label.','阈值没有出现在交叉熵里；本页权重由手动控件设定。','The threshold is absent from log loss. These weights are set manually, not fitted here.')
    s.wipe();s.show(card('y = '+str(y),'第一条观测的真实标签',BLUE,3.6).move_to(pos(-2.7,.5)),card(num(d['losses'][0],3),'该观测的对数损失',GOLD,3.6).move_to(pos(2.7,.5)));s.hold('label-likelihood',4)
    s.ending('分数、概率、决策，是三层不同对象','Score, probability and decision are distinct objects.','p=sigma(w x+b) quad L=-y ln(p)-(1-y)ln(1-p)','亲手只改阈值，核对交叉熵和概率都不变。','Change only the threshold and verify that probabilities and log loss stay fixed.')


def overfit(s):
    d=s.d;train=d['train'];valid=d['validation'];fit=d['fitCurve'];truth=d['trueCurve'];pred=d['predictions']
    s.chapter('模型首先看见的是这批训练点','The fit is chosen using these training observations.','真实曲线只作为模拟中的参照；拟合使用带噪声的观测。','The latent curve is a simulation reference; the fit uses noisy observations.')
    ys=train['y']+valid['y']+[q[1] for q in fit];a,p=axes([-1,1],[max(-4,min(ys)-.2),min(4,max(ys)+.2)],labels=('x','响应 y'));dots=VGroup(*(Dot(p(x,y),radius=.055,color=BLUE) for x,y in zip(train['x'],train['y'])));s.add(a,dots);s.hold('training-data',4)
    fitted=curve(fit,p,BLUE,3);s.play(Create(fitted),run_time=2.5);s.hold('fitted-to-training',4)
    s.chapter('把没参与拟合的数据放进来','Now reveal observations the fit did not use.','训练误差变小，不能直接说明新数据预测也变好。','Low training error alone does not establish good prediction on new observations.')
    vdots=VGroup(*(Dot(p(x,y),radius=.038,color=GOLD) for x,y in zip(valid['x'],valid['y'])));s.play(FadeIn(vdots),run_time=1.5);s.hold('held-out-data',4)
    s.chapter('复杂度变化，用两条误差曲线一起判断','Track training and validation error across model complexity.','这些验证点只用于评估；每个次数都重新拟合训练数据。','Each degree is fitted anew on training data and evaluated on the same holdout set.')
    s.wipe();data=d['complexity'];hi=max(max(v['train'],v['validation']) for v in data);a,p=axes([1,15],[0,hi*1.1],labels=('多项式次数','均方误差'));s.show(a)
    s.play(Create(curve([[v['degree'],v['train']] for v in data],p,BLUE)),run_time=2);s.play(Create(curve([[v['degree'],v['validation']] for v in data],p,GOLD)),run_time=2);s.show(txt('蓝：训练    金：验证',25,MUTED).move_to(pos(0,1.85)));s.hold('complexity-errors',5)
    s.ending('拟合能力，是可以使用也可以误用的自由度','Flexibility must be assessed using independent evidence.','"training error" != "generalization error"','保留同一数据种子，比较次数和惩罚；不要把单次曲线当成普遍定律。','Keep the data seed fixed and compare degree and regularization; a single curve is not a universal law.')


def knn(s):
    d=s.d;data=d['observations'];q=d['query'];near=d['neighbors'];k=s.p['k'];assert len(near)==k
    s.chapter('一个新点，先找谁离它最近','A new query first needs a neighborhood.','蓝色为类别 0，金色为类别 1；方块是需要预测的新位置。','Colors are observed classes; the square marks the new query.')
    a,p=axes([-2.3,2.3],[-2.3,2.3],size=(4.5,4.2),equal=True,labels=('特征 1','特征 2'));dots=VGroup(*(Dot(p(*v['x']),radius=.065,color=BLUE if v['y']==0 else GOLD) for v in data));query=Square(.18,color=GREEN).move_to(p(*q));s.add(a,dots,query);s.hold('query-and-data',4)
    s.chapter('按距离，选出同一批 K 个邻居','Select the K nearest observations by distance.','每选中一个邻居，就保留它的类别来投票。','Each selected neighbor contributes its observed class.')
    votes=[]
    for rank,item in enumerate(near):
        i=item['i'];line=Line(query.get_center(),dots[i].get_center(),color=GREEN,stroke_width=1.8);s.play(Create(line),Indicate(dots[i],scale_factor=1.7),run_time=.85)
        vote=dotlabel(data[i]['y'],f'第 {rank+1} 近',BLUE if data[i]['y']==0 else GOLD).scale(.85);vote[2].next_to(vote[0],RIGHT,buff=.15);vote.move_to(pos(4,1.45-rank*.7));s.play(TransformFromCopy(dots[i],vote),run_time=.7);votes.append(vote)
    s.hold('nearest-neighbor-votes',4)
    s.chapter('把邻居标签平均，得到局部正类比例','Average the selected labels into a local class proportion.','这里每个邻居权重相同；距离用于选择邻居。','This version gives selected neighbors equal voting weights.')
    result=card(num(d['probability'],3),'K 个邻居中的正类比例',GREEN,4).move_to(pos(0,0));s.wipe();s.show(result);s.hold('local-proportion',4)
    s.ending('距离的定义，决定谁有投票权','The distance metric determines who votes.','hat(p)(y=1|x)=1/K sum_(i in N_(K)(x)) y_i','改变特征尺度会改变邻居；先检查度量，再解释分类边界。','Feature scaling can change the neighborhood, so inspect the metric before interpreting the boundary.')


def svm(s):
    d=s.d;obs=d['observations'];w=np.array(d['w']);b=s.p['offset'];margins=d['margins'];slack=d['slack'];norm=np.linalg.norm(w)
    s.chapter('分类边界之外，还要看离边界多远','A separator also defines a margin.','先固定一条手动分隔线；本页没有声称它已经优化。','Start with a manually specified separator, not a claimed optimum.')
    a,p=axes([-3,3],[-2.5,2.5],size=(6,4.2),equal=True,labels=('x₁','x₂'));dots=VGroup(*(Dot(p(*v['x']),radius=.055,color=BLUE if v['y']==-1 else GOLD) for v in obs));s.add(a,dots)
    direction=np.array([-w[1],w[0]])/norm
    for target,c in [(0,GREEN),(-1,RULE),(1,RULE)]:
        center=(target-b)*w/(norm*norm)
        from films import clipped
        ends=clipped(center-5*direction,center+5*direction,[-3,3,-2.5,2.5]);assert ends
        line=Line(p(*ends[0]),p(*ends[1]),color=c,stroke_width=2.5 if target==0 else 1.5);s.play(Create(line),run_time=.65)
    s.hold('separator-and-margins',4)
    s.chapter('把真实类别乘进去，正确一侧才是正','A signed margin includes the true class label.','y(wx+b) ≥ 1 才越过间隔边界；仅分类正确还不够。','A correctly classified point can still lie inside the unit margin.')
    candidates=[int(np.argmin(margins)),int(np.argmin(np.abs(np.array(margins)-.5))),int(np.argmax(margins))]
    for i in candidates:
        ring=SurroundingRectangle(dots[i],color=GREEN,buff=.09);badge=card(num(margins[i],3),'有符号间隔',GREEN,3).move_to(pos(4.4,.7));loss=txt('Hinge 损失 = '+num(slack[i],3),24,GOLD).move_to(pos(4.4,-.8));s.show(ring,badge,loss);s.hold('margin-case-'+str(i),3);s.play(FadeOut(ring),FadeOut(badge),FadeOut(loss),run_time=.3)
    s.ending('大间隔与违例代价，需要一起权衡','The objective trades margin width against violations.','min 1/2 norm(w)^2+C sum_i max(0,1-y_(i) (w^T x_i+b))','C 改变优化目标中的权衡；手动改变 C 不会自动移动这条分隔线。','C changes the objective; this manual example does not refit the separator when C changes.')


def tree(s):
    d=s.d;obs=d['observations'];root=d['tree']
    s.chapter('一棵树，先问一个能分开样本的问题','A tree begins with one axis-aligned question.','先看根节点中的全部观测，再画第一次切分。','Inspect the observations at the root before drawing its split.')
    a,p=axes([-2,2],[-2,2],size=(4.3,4.3),equal=True,labels=('特征 1','特征 2'));dots=VGroup(*(Dot(p(*v['x']),radius=.06,color=BLUE if v['y']==0 else GOLD) for v in obs));s.add(a,dots);s.hold('root-data',4)
    for i,split in enumerate(d['splits']):
        bounds=split['bounds'];t=split['threshold'];axis=split['axis'];start=p(t,bounds[1][0]) if axis==0 else p(bounds[0][0],t);end=p(t,bounds[1][1]) if axis==0 else p(bounds[0][1],t)
        s.chapter('根节点先切一刀' if i==0 else '只在自己的子区域内继续切分','Split the root.' if i==0 else 'A child splits only its own region.','规则是 x'+str(axis+1)+' ≤ '+num(t,3)+'；每条规则只作用于到达该节点的样本。','Each rule applies only to observations that reached that node.')
        s.play(Create(Line(start,end,color=GREEN,stroke_width=3)),run_time=1.5);s.hold('split-'+str(i),4)
    s.chapter('到达叶子，就使用叶子内的频率','A leaf predicts using the observations that reached it.','每个叶子给出其中正类的比例；更深不一定更可靠。','Each leaf reports its class proportion; deeper does not necessarily mean more reliable.')
    for leaf in d['leaves']:
        bounds=leaf['bounds'];center=p(np.mean(bounds[0]),np.mean(bounds[1]));badge=txt(num(leaf['probability'],2),27,GREEN).move_to(center);s.show(BackgroundRectangle(badge,color=BG,fill_opacity=.85,buff=.07),badge,run_time=.4)
    s.hold('leaf-probabilities',5)
    s.ending('路径上的问题，共同定义一个区域','A decision path defines an intersection of regions.','"prediction"="leaf class proportion"','亲手改变最大深度，检查区域细分与每个叶子的样本数。','Change maximum depth and inspect both partition detail and leaf sample counts.')


def boosting(s):
    d=s.d;train=d['train'];xs=np.array([v['x'] for v in train]);ys=np.array([v['y'] for v in train]);pred=np.full(len(xs),d['base']);rate=s.p['rate']
    s.chapter('第一轮先给每个输入同一个预测','Begin with a constant prediction.','平方损失下，初始常数是训练响应的平均值。','For squared loss, initialize with the mean training response.')
    a,p=axes([-1,1],[-1.7,1.7],labels=('输入 x','响应与预测'));dots=VGroup(*(Dot(p(x,y),radius=.04,color=BLUE) for x,y in zip(xs,ys)));fitted=Line(p(-1,d['base']),p(1,d['base']),color=GREEN,stroke_width=3);s.add(a,dots,fitted);s.hold('constant-start',4)
    s.chapter('下一棵小树，学习的是当前没解释的部分','The next weak learner fits the current residuals.','竖线是真实响应减当前预测；第二个模型不是从零重新拟合原响应。','Residuals are response minus current prediction, not a fresh fit to the original target.')
    residuals=VGroup(*(Line(p(x,pred[i]),p(x,ys[i]),color=GOLD,stroke_width=1.3) for i,x in enumerate(xs[:12])));s.play(Create(residuals),run_time=1.5);s.hold('current-residuals',4)
    gridx=np.linspace(-1,1,301);gridpred=np.full(len(gridx),d['base'])
    for t,model in enumerate(d['models'][:4]):
        s.chapter('第 '+str(t+1)+' 次修正：只加一小步','Correction '+str(t+1)+': add a shrunken residual fit.','学习率 η = '+num(rate)+'；新预测 = 旧预测 + η × 小树。','The learning rate scales the new tree before adding it to the current prediction.')
        delta=np.where(gridx<=model['threshold'],model['left'],model['right']);gridpred+=rate*delta;new=curve(list(zip(gridx,gridpred)),p,GREEN,3)
        s.play(FadeOut(residuals),Transform(fitted,new),run_time=1.7);s.hold('additive-update-'+str(t),3.5);residuals=VGroup()
    s.ending('Boosting 保留旧模型，再累加新的修正','Boosting preserves previous learners and adds corrections.','F_(m+1)(x)=F_(m)(x)+eta h_(m)(x)','继续加树需要验证集判断；训练残差变小不保证泛化误差同步下降。','Use validation evidence to choose when to stop; lower training residuals do not guarantee better generalization.')


def kmeans(s):
    d=s.d;X=np.array(d['observations']);centers=np.array(d['initialCenters']);k=len(centers)
    s.chapter('先放中心，还没有决定谁属于谁','Start with centers before assigning observations.','十字是当前中心；圆点是保持不动的数据。','Crosses mark centers. The observed points will remain fixed.')
    a,p=axes([-3.3,3.3],[-2.2,2.2],size=(7,4.2),equal=True,labels=('特征 1','特征 2'));dots=VGroup(*(Dot(p(*x),radius=.055,color=MUTED) for x in X));markers=VGroup(*(Cross(stroke_color=PALETTE[i],stroke_width=4).scale(.1).move_to(p(*c)) for i,c in enumerate(centers)));s.add(a,dots,markers);s.hold('initial-centers',4)
    for step in [1,2]:
        assignment=np.argmin(np.sum((X[:,None,:]-centers[None,:,:])**2,axis=2),axis=1)
        s.chapter('先按最近中心分组','Assign each observation to its nearest center.','中心固定时，每个点选择距离最小的那一组。','Hold centers fixed and minimize distance for each observation.')
        lines=VGroup(*(Line(p(*x),p(*centers[assignment[i]]),color=PALETTE[assignment[i]],stroke_width=.8) for i,x in enumerate(X)))
        s.play(Create(lines),*(dot.animate.set_color(PALETTE[assignment[i]]) for i,dot in enumerate(dots)),run_time=1.8);s.hold('assignment-'+str(step),4)
        s.chapter('再把中心移到组内平均位置','Move each center to its assigned cluster mean.','分组固定时，平方距离和的最小点就是组内均值。','With assignments fixed, the mean minimizes squared distances.')
        target=np.array(d['history'][step]['centers']);expected=np.array([X[assignment==i].mean(axis=0) for i in range(k)]);assert np.allclose(target,expected)
        s.play(FadeOut(lines),*(markers[i].animate.move_to(p(*target[i])) for i in range(k)),run_time=2);centers=target;s.hold('center-update-'+str(step),4)
    s.ending('两步交替，可能停在不同的局部解','Alternating improvements need not find a global optimum.','min sum_i norm(x_i-mu_(c_i))^2','本例第二次中心已不再移动；换初始化后，收敛结果可能不同。','The second update is stationary here. Different initializations can lead to different solutions.')


def gmm(s):
    d=s.d;X=np.array(d['observations']);history=d['history'];sigma=d['sigma']
    s.chapter('一个点，不必硬归到某一组','A point can share responsibility across components.','先放两个高斯分量；一个观测对两组都可能有支持。','Two Gaussian components can both assign density to the same observation.')
    lo=min(X)-.4;hi=max(X)+.4;a,p=axes([lo,hi],[0,.45],size=(9,2.8),labels=('观测 x','加权分量密度'));dots=VGroup(*(Dot(p(x,.01),radius=.045,color=MUTED) for x in X));s.add(a,dots)
    for j,c in enumerate([BLUE,GREEN]):
        density=[[x,.5*np.exp(-.5*((x-history[0]['means'][j])/sigma)**2)/(sigma*np.sqrt(2*np.pi))] for x in np.linspace(lo,hi,241)];s.play(Create(curve(density,p,c)),run_time=1.1)
    s.show(txt('蓝：分量 1    绿：分量 2',24,MUTED).move_to(pos(0,1.85)));s.hold('mixture-start',4)
    i=int(np.argmin(abs(X)));support=np.array([history[0]['weight']*np.exp(-.5*((X-history[0]['means'][0])/sigma)**2),(1-history[0]['weight'])*np.exp(-.5*((X-history[0]['means'][1])/sigma)**2)]);responsibility=support[0]/support.sum(axis=0);r1=float(responsibility[i]);assert np.allclose([(responsibility*X).sum()/responsibility.sum(),((1-responsibility)*X).sum()/(1-responsibility).sum()],history[1]['means'])
    s.chapter('E 步：相对密度，变成分担比例','The E-step normalizes component support.','使用初始两个分量，展开最靠近 0 的一个观测的责任概率。','Using the initial components, inspect the observation closest to zero.')
    s.wipe();ob=card(num(X[i],3),'一个观测 x',GOLD,3).move_to(pos(0,1.2));left=card(num(r1,3),'对分量 1 的责任',BLUE,3.6).move_to(pos(-3,-.8));right=card(num(1-r1,3),'对分量 2 的责任',GREEN,3.6).move_to(pos(3,-.8));s.show(ob);s.play(TransformFromCopy(ob,left),TransformFromCopy(ob,right),run_time=1.6);s.hold('soft-responsibilities',5)
    s.chapter('M 步：用责任概率作权重，重算均值','The M-step uses responsibilities as averaging weights.','接下来展示实际迭代记录：两个均值逐轮移动到更支持数据的位置。','Follow the actual iteration history of the two component means.')
    s.wipe();a,p=axes([lo,hi],[0,1],size=(9,1.3),labels=('分量均值',''));s.show(a);markers=[Dot(p(history[0]['means'][j],.5),radius=.12,color=[BLUE,GREEN][j]) for j in range(2)];s.show(*markers)
    for state in history[1:]:s.play(*(markers[j].animate.move_to(p(state['means'][j],.5)) for j in range(2)),run_time=1.1);s.wait(1.3)
    s.hold('weighted-mean-updates',4)
    s.ending('软分组，依然依赖模型与初始值','Soft assignments still depend on assumptions and initialization.','mu_k=(sum_i r_(i,k)x_i)/(sum_i r_(i,k))','本例固定两组相同的已知标准差；EM 不保证找到全局最优。','This example fixes a shared known standard deviation. EM need not find the global optimum.')


def kernel(s):
    d=s.d;obs=d['observations'];alpha=np.array(d['alpha']);h=s.p['bandwidth'];x=np.array([v['x'] for v in obs]);y=np.array([v['y'] for v in obs]);K=np.array(d['K']);assert np.allclose((K+s.p['lambda']*np.eye(len(x)))@alpha,y)
    s.chapter('把一个观测，当作相似度的中心','An observation anchors a similarity function.','RBF 核随距离衰减；带宽规定“附近”有多大。','The RBF kernel decays with distance; bandwidth sets the neighborhood scale.')
    chosen=np.argsort(x)[np.linspace(2,len(x)-3,4).astype(int)];gx=np.linspace(-1.2,1.2,201)
    a,p=axes([-1.2,1.2],[0,1.1],labels=('查询 x','核相似度'));s.add(a)
    curves=VGroup()
    for rank,i in enumerate(chosen):
        color=PALETTE[rank];center=Dot(p(x[i],0),radius=.06,color=color)
        c=curve(list(zip(gx,np.exp(-.5*((gx-x[i])/h)**2))),p,color,3)
        s.show(center);s.play(Create(c),run_time=1.4);curves.add(c);s.hold('kernel-center-'+str(rank),2.5)
    s.chapter('每座小山，乘上一个拟合系数','Each similarity function is scaled by a fitted coefficient.','中心保留在原观测位置；负系数会把贡献翻到横轴下方。','Centers stay at their observations; negative coefficients reverse the contribution.')
    s.wipe();bound=max(abs(alpha[chosen]))*1.12;a,p=axes([-1.2,1.2],[-bound,bound],labels=('查询 x','对预测的贡献'));s.show(a)
    for rank,i in enumerate(chosen):
        contribution=alpha[i]*np.exp(-.5*((gx-x[i])/h)**2)
        label=txt('xᵢ = '+num(x[i])+'    αᵢ = '+num(alpha[i]),25,PALETTE[rank]).move_to(pos(3,1.95))
        s.show(label);s.play(Create(curve(list(zip(gx,contribution)),p,PALETTE[rank],3)),run_time=1.4);s.hold('weighted-kernel-'+str(rank),2.5);s.remove(label)
    s.chapter('把所有贡献，在同一个 x 上相加','Sum all weighted kernels at the same query location.','刚才四条线是四项贡献；完整预测使用全部 '+str(len(x))+' 个中心。','Those four curves are four terms; the predictor uses all observation centers.')
    s.wipe();bounds=[v['y'] for v in obs]+[v[1] for v in d['curve']];a,p=axes([-1.2,1.2],[min(bounds)-.2,max(bounds)+.2],labels=('查询 x','完整预测'));s.show(a,VGroup(*(Dot(p(v['x'],v['y']),radius=.05,color=BLUE) for v in obs)));s.play(Create(curve(d['curve'],p,GREEN)),run_time=2.5);s.hold('full-kernel-prediction',5)
    s.chapter('系数不是随手定的，来自同一组线性方程','The coefficients solve a regularized linear system.','核矩阵记录样本两两相似度；λ 调节拟合与函数大小的取舍。','The Gram matrix records pairwise similarities; lambda regularizes the function.')
    s.wipe();s.equation('(K+lambda I)alpha=y',y=.75);s.show(txt('K：'+str(len(x))+' × '+str(len(x))+'     α：'+str(len(x))+' 个系数',26,MUTED).move_to(pos(0,-.75)));s.hold('kernel-linear-system',5)
    s.ending('局部相似度，经组合产生全局预测','Local similarities combine into a global predictor.','hat(f)(x)=sum_i alpha_i K(x,x_i)','保持数据不变，分别改变带宽与 λ，区分它们改变的对象。','Keep data fixed and change bandwidth and lambda separately to distinguish their effects.')


def double_descent(s):
    d=s.d;n=s.p['n'];fits=d['fits'];selected=d['selected'];print('double descent selected',selected.keys(),flush=True)
    s.chapter('先数清参数：宽度之外，还有截距','Count parameters before locating interpolation.','本例有 n = '+str(n)+' 条训练样本，m 个随机特征加一个截距。','There are n training observations, m random features and one intercept.')
    count=card(str(n),'训练样本数',BLUE,3).move_to(pos(-3,.5));params=card('m + 1','拟合参数数目',GOLD,3).move_to(pos(3,.5));s.add(count,params);s.hold('parameter-count',4)
    s.chapter('等参数数目附近，矩阵可能变得敏感','The interpolation region can be ill-conditioned.','m = n−1 是计数位置；实际是否插值，还要检查秩与训练误差。','The count m = n−1 alone does not establish interpolation; inspect rank and training error.')
    s.wipe();s.show(card(str(d['interpolationWidth']),'插值计数对应的宽度',GOLD,4));s.hold('interpolation-count',4)
    s.chapter('沿真实计算结果，逐点看验证风险','Inspect the actual validation curve, not a promised shape.','曲线来自同一训练和验证数据；不预设一定出现双下降。','These are computed errors on fixed datasets, not a guaranteed double-descent shape.')
    s.wipe();hi=max(v['validationMSE'] for v in fits);a,p=axes([min(d['widths']),max(d['widths'])],[0,hi*1.08],labels=('随机特征宽度 m','验证均方误差'));s.show(a);line=curve([[v['width'],v['validationMSE']] for v in fits],p,GOLD);s.play(Create(line),run_time=4);s.show(DashedLine(p(n-1,0),p(n-1,hi),color=BLUE));s.hold('actual-risk-curve',6)
    s.ending('容量曲线，也受正则化与数据影响','Capacity curves depend on data and regularization.','"parameter count" != "guaranteed interpolation"','增加岭惩罚后再比较；插值位置和风险形状需要数值证据。','Compare after adding ridge regularization; interpolation and risk shape require numerical evidence.')


def rl_bellman(s):
    d=s.d;h=d['history'];gamma=s.p['gamma'];actions=d['actions']
    s.chapter('先把奖励与转移概率分开读','Rewards and transition probabilities are different quantities.','在状态 0，可安全退出得 1，也可付出成本选择冒险。','State 0 offers a safe terminal reward or a risky transition with a cost.')
    nodes=VGroup(card('s₀','当前状态',BLUE,2).move_to(pos(-4,0)),card('s₁','下一状态',GREEN,2).move_to(pos(1,1)),card('终止','未来价值 = 0',MUTED,2).move_to(pos(4,-1)))
    s.add(nodes);s.play(GrowArrow(arrow(pos(-2.9,.2),pos(-.1,.8))),GrowArrow(arrow(pos(-2.9,-.3),pos(2.9,-.9))),run_time=1.3)
    s.show(txt('冒险：即时 '+num(actions['risky']['reward'])+'，概率 '+num(actions['risky']['toState1'])+' 到 s₁',21,GOLD).move_to(pos(-.8,1.8)),txt('安全：奖励 1 后终止',22,MUTED).move_to(pos(0,-1.7)));s.show(arrow(pos(2.1,.7),pos(3.7,-.2),GREEN),txt('奖励 3 后终止',21,GREEN).move_to(pos(4.3,1.2)));s.play(Create(CurvedArrow(pos(-4.7,.6),pos(-4.7,-.45),angle=PI,color=BLUE)),run_time=1);s.show(txt('失败后回到 s₀',20,BLUE).move_to(pos(-4.2,-1.45)));s.hold('mdp-options',5)
    s.chapter('把后续价值折现，再回传到当前状态','Discount future value before backing it up.','冒险失败会回到 s₀；这条回路也必须计入期望。','A failed risky transition returns to s₀ and must be included in the expectation.')
    s.wipe();s.equation('Q_("risk")=c+gamma[p V(s_1)+(1-p)V(s_0)]',y=.75);s.hold('risky-backup',5)
    s.chapter('每次都用上一轮的价值做下一轮更新','Synchronous backups use the previous value vector.','从全 0 开始，奖励先到达近处，再影响更早的状态。','Starting from zero, reward information propagates backward through transitions.')
    s.wipe();state=row(h[0],['V(s₀)','V(s₁)','V(终止)'],BLUE,3,.7);s.show(state)
    for i in range(1,6):
        new=row(h[i],['V(s₀)','V(s₁)','V(终止)'],GREEN,3,.7);s.play(Transform(state,new),run_time=.9)
        prev=h[i-1];risk=actions['risky']['reward']+gamma*(s.p['success']*prev[1]+(1-s.p['success'])*prev[0]);assert abs(max(1,risk)-h[i][0])<1e-10
        calculation=txt('V(s₀) = max(1, '+num(actions['risky']['reward'])+' + '+num(gamma)+' × ['+num(s.p['success'])+' × '+num(prev[1],3)+' + '+num(1-s.p['success'])+' × '+num(prev[0],3)+'])',25,GOLD).move_to(pos(0,-1.3));s.show(calculation);s.hold('backup-'+str(i),2.5);s.play(FadeOut(calculation),run_time=.25)
    s.ending('取最大值，才从评估走向最优控制','The maximum chooses the best available action.','V_(k+1)(s)=max_a E[r+gamma V_(k)(s prime)]','本例有限状态且 γ < 1；残差给出误差上界，不等于实际误差。','For this discounted finite MDP, the Bellman residual bounds error but is not the error itself.')


def rl_policy(s):
    d=s.d;path=d['path'];q=d['initialProbability'];initial=d['initial']
    s.chapter('参数先改变选动作的概率','The parameter changes action probabilities.','θ 通过 Sigmoid 决定冒险概率 q；它不直接就是回报。','Theta controls risk probability q through a sigmoid; it is not the return.')
    nodes=VGroup(card(num(s.p['theta']),'θ',BLUE,2.2).move_to(pos(-4,.5)),card(num(q,3),'冒险概率 q',GOLD,2.2).move_to(pos(0,.5)),card(num(initial['value'],3),'期望折现回报 J',GREEN,2.8).move_to(pos(4,.5)));s.add(nodes[0])
    for i in [1,2]:s.play(TransformFromCopy(nodes[i-1],nodes[i]),run_time=1.3);s.wait(2)
    s.hold('policy-probability-return',4)
    s.chapter('梯度问：稍微增加 θ，回报会怎样？','The gradient asks how return changes with theta.','本例使用精确环境模型计算梯度，包含重复访问状态的影响。','The exact model-based gradient includes repeated visits to the state.')
    s.wipe();s.show(card(num(d['gradient'],4),'初始策略梯度',GOLD,3.8));s.hold('initial-policy-gradient',4)
    s.chapter('沿梯度更新，再重新评估策略','Update the parameter, then reevaluate the policy.','每步都重新计算概率与回报；环境奖励和转移保持不变。','Each update recomputes the policy and its return in the same environment.')
    s.wipe();table=VGroup()
    for i,t in enumerate([0,1,5,15,30]):
        v=path[t];line=txt(f'第 {t:>2} 步     θ = {num(v["theta"],3)}     q = {num(v["q"],3)}     J = {num(v["value"],3)}',26,PALETTE[min(i,4)]).move_to(pos(0,1.45-i*.75));s.show(line,run_time=.7);s.wait(1.5)
    s.hold('policy-updates',4)
    s.ending('更常选择哪个动作，取决于当前环境','The beneficial update direction depends on the environment.','theta_(t+1)=theta_t+eta nabla J(theta_t)','改变成功率或成本，最有利的方向可能反转；这里没有采样估计噪声。','Changing success probability or cost can reverse the preferred direction; this example uses exact gradients.')


STORIES={'regression':regression,'logistic':logistic,'overfit':overfit,'knn':knn,'svm':svm,'tree':tree,'boosting':boosting,'kmeans':kmeans,'gmm':gmm,'kernel':kernel,'double-descent':double_descent,'rl-bellman':rl_bellman,'rl-policy':rl_policy}
