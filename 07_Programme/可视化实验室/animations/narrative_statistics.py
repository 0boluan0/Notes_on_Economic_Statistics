"""Statistical lessons: make the sampling unit and each conditioning event explicit."""
from narrative import *


def sampling(s):
    d=s.d;n=s.p['n'];examples=d['sampleExamples'];means=d['means'];assert all(abs(np.mean(a)-means[i])<1e-9 for i,a in enumerate(examples))
    s.chapter('八个观测，只产生一个样本均值','Eight observations produce one sample mean.','每个圆是一次观测；先不要把观测分布和均值分布混在一起。','A circle is an observation, not a sample mean.')
    tokens=row(examples[0],[f'X{i+1}' for i in range(n)],BLUE,1.27,.8);s.add(tokens);s.hold('one-sample',4)
    mean=card(num(means[0],3),'这一份样本的均值',GOLD,3.4).move_to(pos(0,-1.1));s.play(TransformFromCopy(tokens,mean),run_time=1.5);s.hold('one-mean',4)
    s.chapter('换一整份样本，才得到下一个均值','A new sample gives the next sample mean.','固定 n = 8；每次重新抽八个，再把它们收成一个点。','Keep n = 8. Each repetition draws eight new observations.')
    s.wipe();lo=min(means);hi=max(means);a,p=axes([min(0,lo),hi*1.05],[0,1],center=(0,-1.2),size=(9.8,.55),labels=('样本均值',''));s.show(a)
    for i,values in enumerate(examples[:5]):
        group=row(values,[str(j+1) for j in range(n)],BLUE,1.27,.9);s.show(group,run_time=.4);s.wait(1)
        point=Dot(p(means[i],.4+i*.06),radius=.065,color=GOLD);s.play(ReplacementTransform(group,point),run_time=1.2);s.wait(.6)
    s.hold('five-independent-means',4)
    s.chapter('很多个均值，才形成抽样分布','The distribution is built from repeated sample means.','每个点仍代表八个观测的平均；增加重复次数不会改变 n。','Each point still averages eight observations. More repetitions do not increase n.')
    s.wipe();p0=s.r['plots'][1] if len(s.r['plots'])>1 else s.r['plots'][0]
    values=np.array(means);counts,edges=np.histogram(values,bins=22);a,p=axes([float(edges[0]),float(edges[-1])],[0,max(counts)*1.12],labels=('样本均值','频数'));s.show(a)
    bars=VGroup(*(Rectangle(width=(edges[i+1]-edges[i])/(edges[-1]-edges[0])*9.6*.9,height=max(.015,counts[i]/(max(counts)*1.12)*3.45),color=GREEN,fill_opacity=.7).move_to(p((edges[i]+edges[i+1])/2,counts[i]/2)) for i in range(len(counts))))
    s.play(LaggedStart(*(GrowFromEdge(b,DOWN) for b in bars),lag_ratio=.08),run_time=3);s.hold('mean-distribution',5)
    s.ending('n 控制每个均值，B 控制模拟精细度','Sample size and repetition count play different roles.','"SE"(overline(X))=sigma/sqrt(n)','有限方差条件很关键；柯西总体不满足这里的均值与 CLT 前提。','Finite variance matters. Cauchy observations do not satisfy these mean and CLT assumptions.')


def confidence(s):
    d=s.d;intervals=d['intervals'];width=d['halfWidth'];extent=max(max(abs(r['low']),abs(r['high'])) for r in intervals)+.1
    s.chapter('先抽样，才有一个会变的区间','A random sample produces a random interval.','总体均值固定在 0；随机的是样本和由它算出的端点。','The population mean stays at zero; samples and interval endpoints vary.')
    a,p=axes([-extent,extent],[0,1],size=(9.7,.8),center=(0,.1),labels=('均值所在的数轴',''));truth=DashedLine(p(0,-1.8),p(0,3),color=GOLD);s.add(a,truth);s.show(txt('固定真值 μ = 0',24,GOLD).move_to(pos(0,1.8)));s.hold('fixed-parameter',3)
    first=intervals[0];mean=Dot(p(first['mean'],.5),radius=.09,color=BLUE);s.show(mean);segment=Line(p(first['low'],.5),p(first['high'],.5),color=GREEN,stroke_width=6);s.play(GrowFromCenter(segment),run_time=1.3);s.show(txt('样本均值 ± '+num(width,3),27,GREEN).move_to(pos(0,-1.25)));s.hold('one-interval',4)
    s.chapter('重复整个抽样过程，再看覆盖频率','Repeat sampling to assess the interval procedure.','每一条线都来自新样本；金色竖线仍是同一个总体真值。','Every segment uses a new sample; the gold line is the same fixed truth.')
    s.wipe();a,p=axes([-extent,extent],[0,len(intervals)+1],size=(9.5,4),labels=('区间端点','样本编号'));s.show(a,DashedLine(p(0,0),p(0,len(intervals)+1),color=GOLD))
    for i,item in enumerate(intervals):
        col=GREEN if item['covers'] else RED;line=Line(p(item['low'],i+1),p(item['high'],i+1),color=col,stroke_width=3)
        s.play(GrowFromCenter(line),run_time=.18)
        if i in [0,4,14]:s.hold('intervals-'+str(i+1),2)
    s.hold('all-intervals',4)
    s.wipe();s.chapter('95% 描述的是这种造区间的方法','Coverage is a property of the procedure.','有限次模拟的覆盖比例会波动，并不必须恰好等于 95%。','Finite simulation coverage fluctuates around the nominal level.')
    s.show(card(f'{d["covered"]} / {len(intervals)}','本次模拟覆盖次数',GREEN,4));s.hold('coverage-count',4)
    s.ending('固定参数，随机区间','A fixed parameter and a random interval.','P_(mu) [mu in C(X)]=0.95','这里假设 IID 正态观测且 σ 已知；不能把它直接当成参数的后验概率。','This example uses IID normal observations with known sigma, not a posterior probability for the parameter.')


def testing(s):
    d=s.d;z=d['observedZ'];critical=d['criticalZ']
    s.chapter('先假设零假设成立，再放入观测','Locate the observation under the null model.','本例零假设是 μ = 0；把样本均值标准化为 z。','The null is μ = 0. Standardize the observed sample mean.')
    a,p=axes([-4,4],[0,.45],labels=('标准化统计量 z','零假设下的密度'));density=lambda x:np.exp(-x*x/2)/np.sqrt(2*np.pi);c=curve([[x,density(x)] for x in np.linspace(-4,4,301)],p,BLUE);s.add(a,c);s.hold('null-distribution',4)
    marker=DashedLine(p(z,0),p(z,density(z)),color=GOLD);s.play(Create(marker),run_time=1);s.show(txt('观测 z = '+num(z,3),25,GOLD).move_to(pos(3,1.35)));s.hold('observed-statistic',3)
    s.chapter('两侧更极端的区域，加起来是 p 值','A two-sided p-value counts both extreme tails.','以 |z| 为界；这里问的是零假设下这些数据有多极端。','Measure outcomes at least as extreme as |z| under the null.')
    tails=VGroup()
    for sign in [-1,1]:
        xs=np.linspace(abs(z),4,100)*sign;points=[p(xs[0],0)]+[p(x,density(x)) for x in xs]+[p(xs[-1],0)];tails.add(Polygon(*points,color=GOLD,fill_opacity=.45,stroke_width=0))
    s.play(FadeIn(tails),run_time=1.4);s.show(card(num(d['pValue'],4),'两侧尾概率之和',GOLD,3.4).move_to(pos(0,.5)));s.hold('two-tail-probability',5)
    s.chapter('显著性阈值，先于这次数据规定','The significance level specifies a rejection rule.','α 决定拒绝区；p 值说明这次观测在零假设下的位置。','Alpha defines the rejection region; the p-value describes this observation.')
    s.wipe();s.show(card('α = '+num(s.p['alpha']),'预设阈值',BLUE,3).move_to(pos(-2.6,.55)),card('p = '+num(d['pValue'],4),'这次结果',GOLD,3).move_to(pos(2.6,.55)));s.hold('p-versus-alpha',4)
    s.ending('p 值不是零假设为真的概率','A p-value is not the probability that the null is true.','p=P_(H_0)(abs(Z)>=abs(z_("obs")))','也不能把不显著解释为证明没有效应；功效取决于样本量和真实效应。','Nonsignificance does not prove no effect; power depends on sample size and the true effect.')


def bayes(s):
    d=s.d;p=s.p;a0=p['a'];b0=p['b'];success=p['success'];failure=p['failure']
    s.chapter('先验是更新之前的分布','A prior describes uncertainty before these observations.','本例从 Beta(2, 2) 出发，参数 θ 是成功概率。','Begin with Beta(2, 2) for the Bernoulli success probability θ.')
    prior=s.r['plots'][0]['series'][0]['points'];posterior=s.r['plots'][0]['series'][1]['points'];height=max(y for x,y in prior+posterior)*1.12;a,p=axes([0,1],[0,height],labels=('成功概率 θ','概率密度'));s.add(a,curve(prior,p,BLUE));s.show(txt(f'先验 Beta({a0}, {b0})',27,BLUE).move_to(pos(3.1,1.6)));s.hold('prior',4)
    s.chapter('每个新观测，贡献一个似然因子','Each observation contributes a likelihood factor.','成功乘 θ，失败乘 1−θ；它们不是直接相加的概率。','A success contributes θ, and a failure contributes 1−θ.')
    s.wipe();outcomes=[1]*success+[0]*failure;tokens=row(outcomes,['成功']*success+['失败']*failure,BLUE,2,1);s.show(tokens)
    factors=VGroup(*(eq('theta' if v else '(1-theta)',33,GOLD).move_to(pos((i-(len(outcomes)-1)/2)*2,-.45)) for i,v in enumerate(outcomes)))
    for i,m in enumerate(factors):s.play(TransformFromCopy(tokens[i],m),run_time=.85);s.wait(.7)
    s.hold('likelihood-factors',4)
    combined=eq(f'theta^{success} (1-theta)^{failure}',38,GOLD).move_to(pos(0,-1.9));s.play(TransformFromCopy(factors,combined),run_time=1.3);s.hold('combined-likelihood',3)
    s.chapter('相乘后，再归一化成新的分布','Multiply prior and likelihood, then normalize.','Beta 分布的两个指数分别增加成功次数和失败次数。','The exponents increase by the success and failure counts.')
    s.wipe();left=card(f'Beta({a0}, {b0})','先验',BLUE,3.2).move_to(pos(-3,.5));right=card(f'Beta({a0+success}, {b0+failure})','后验',GREEN,3.2).move_to(pos(3,.5));s.show(left);s.play(GrowArrow(arrow(pos(-1.25,.5),pos(1.25,.5))),TransformFromCopy(left,right),run_time=1.5);s.show(txt(f'+ {success} 次成功     + {failure} 次失败',25,GOLD).move_to(pos(0,-1.25)));s.hold('posterior-update',4);s.wipe();a,p=axes([0,1],[0,height],labels=('成功概率 θ','概率密度'));s.show(a,curve(prior,p,BLUE));s.play(Create(curve(posterior,p,GREEN)),run_time=2.4);s.show(txt('蓝：先验    绿：后验',25,MUTED).move_to(pos(0,1.85)));s.hold('updated-distribution',5)
    s.ending('后验，仍然表达不确定性','Updating changes uncertainty; it does not eliminate it.','p(theta|D)=p(D|theta)p(theta)/p(D)','这里采用独立伯努利似然；后验均值不等于保证正确的真实概率。','The model uses independent Bernoulli observations. The posterior mean is still an estimate.')


def likelihood(s):
    d=s.d;n=d['n'];k=d['success'];mle=d['bernoulliMLE']
    s.chapter('数据已经固定，移动的是参数假设','The data stay fixed while the candidate parameter varies.','本次观察到 '+str(n)+' 次中的 '+str(k)+' 次成功；这些数不再改变。','Keep the observed success and trial counts fixed.')
    successes=VGroup(*(Dot(pos(-4.3+(i%10)*.95,1.2-(i//10)*.7),radius=.12,color=GREEN if i<k else MUTED) for i in range(n)));s.add(successes);s.hold('fixed-outcomes',4)
    s.chapter('一个候选概率，给整份数据一个相对支持度','A candidate probability assigns support to the whole dataset.','相同的观测序列，对不同的 θ 产生不同似然。','Different candidate probabilities assign different likelihoods to the same sequence.')
    s.wipe();a,p=axes([0,1],[0,1.1],labels=('候选成功概率 θ','相对似然（最大值缩放为 1）'));s.show(a)
    points=d['bernoulliRelativeLikelihood'];c=curve(points,p,BLUE);s.play(Create(c),run_time=2.2);s.hold('likelihood-curve',4)
    for candidate in [.3,.5,mle]:
        val=np.exp(k*np.log(candidate/mle)+(n-k)*np.log((1-candidate)/(1-mle)));line=DashedLine(p(candidate,0),p(candidate,val),color=GOLD);dot=Dot(p(candidate,val),radius=.08,color=GOLD)
        s.show(line,dot);s.hold('candidate-'+str(candidate),3);s.play(FadeOut(line),FadeOut(dot),run_time=.3)
    s.chapter('最大似然选择最支持这份数据的参数','Maximum likelihood selects the best-supported candidate.','最大值在观测成功比例 '+num(mle)+'；纵轴不是参数的概率密度。','The maximum is at the observed success fraction. This curve is not a posterior density.')
    s.show(arrow(p(mle,.2),p(mle,.98),GREEN));s.hold('maximum-likelihood',4)
    s.ending('把概率看作数据的函数，和看作参数的函数，要分清','Likelihood is a function of the parameter for fixed observations.','L(theta)=theta^k (1-theta)^(n-k)','若要讨论参数的后验概率，还需要先验与归一化。','A posterior probability for the parameter additionally requires a prior and normalization.')


def kde(s):
    d=s.d;values=d['observations'];h=d['bandwidth'];curve_=d['curve'];xs=[v[0] for v in curve_];hi=max(v[1] for v in curve_)*1.7
    s.chapter('一个观测，先放上一座小山','Place one kernel at each observation.','每个点都贡献一个面积为 1/n 的小密度。','Each observation contributes a kernel with total mass 1/n.')
    a,p=axes([min(xs),max(xs)],[0,hi],labels=('x','估计密度'));dots=VGroup(*(Dot(p(x,0),radius=.055,color=GOLD) for x in values));s.add(a,dots);s.hold('observations',3)
    kernels=[]
    for i,x in enumerate(values[:5]):
        ys=[[q,np.exp(-.5*((q-x)/h)**2)/(np.sqrt(2*np.pi)*h*len(values))] for q in xs];c=curve(ys,p,PALETTE[i%5],2);s.play(TransformFromCopy(dots[i],c),run_time=1.1);kernels.append(c);s.wait(.6)
    s.hold('individual-kernels',4)
    s.chapter('在同一个 x，把所有小山的高度相加','Sum the kernel heights at the same evaluation point.','最后的曲线来自全部 '+str(len(values))+' 个观测；刚才只单独展开了前五项。','The final curve uses every observation; only the first five terms were shown separately.')
    total=curve(curve_,p,GREEN,4);s.play(Create(total),run_time=2.5);s.hold('sum-of-kernels',5)
    s.chapter('带宽改变的是每座小山的宽窄','Bandwidth changes the scale of smoothing.','小带宽更局部，大带宽更平滑；总概率质量仍为 1。','Small bandwidth is local; large bandwidth smooths more while preserving mass.')
    s.wipe();s.show(card('h = '+num(h),'本例共同带宽',GOLD,3));s.hold('shared-bandwidth',3)
    s.ending('平滑曲线，不等于已经发现真实结构','A smooth estimate is still a smoothing choice.','hat(f)(x)=1/(n h) sum_i K((x-x_i)/h)','保持同一批观测，只改带宽，比较哪些峰会消失或合并。','Keep the observations fixed and change bandwidth; inspect which peaks merge or disappear.')


def entropy(s):
    d=s.d;P=d['P'];Q=d['Q'];terms=d['crossTerms'];assert abs(sum(terms)-d['crossEntropy'])<1e-9
    s.chapter('预测概率，决定遇到事件时付出多少损失','A reported probability sets the surprise of an event.','事件发生后，报告概率越小，−log Q 就越大。','When an event occurs, a smaller reported probability gives larger log loss.')
    q=row(Q,['报告 Q₁','报告 Q₂','报告 Q₃'],BLUE,3,.95);s.add(q);s.hold('reported-probabilities',3)
    loss=row([-np.log(v) for v in Q],['事件 1 损失','事件 2 损失','事件 3 损失'],GOLD,3,-.9)
    for i in range(3):s.play(TransformFromCopy(q[i],loss[i]),run_time=.85);s.wait(.7)
    s.hold('event-losses',4)
    s.chapter('真正经常发生的事件，占更大的平均权重','True event frequencies weight the average loss.','现在用真实分布 P 加权；Q 是报告，P 决定发生频率。','P supplies event frequencies; Q supplies the reported probabilities.')
    s.wipe();p=row(P,['真实频率 P₁','真实频率 P₂','真实频率 P₃'],BLUE,3,1);t=row(terms,['−P₁ log Q₁','−P₂ log Q₂','−P₃ log Q₃'],GOLD,3,-.75);s.show(p)
    for i in range(3):s.play(TransformFromCopy(p[i],t[i]),run_time=.8);s.wait(.5)
    s.hold('weighted-losses',4)
    total=card(num(d['crossEntropy'],3),'交叉熵（nats）',GREEN,3.4).move_to(pos(0,-.7));s.play(ReplacementTransform(t,total),run_time=1.4);s.hold('cross-entropy',4)
    s.ending('真实不确定性，加上报告错误的代价','Cross-entropy includes irreducible uncertainty and mismatch.','H(P,Q)=H(P)+"KL"(P parallel Q)','若 Pᵢ > 0 却报告 Qᵢ = 0，平均对数损失为无穷大。','If a possible event receives zero reported probability, expected log loss is infinite.')


def metrics(s):
    d=s.d;labels=np.array(d['labels']);scores=np.array(d['scores']);threshold=s.p['threshold'];pred=scores>=threshold
    assert int(sum(pred&(labels==1)))==d['tp']
    s.chapter('先划阈值，才产生四种结果','A threshold turns scores into four types of outcomes.','真实类别已经给定；阈值只改变预测类别。','True labels are fixed; the threshold changes predictions.')
    a,p=axes([0,1],[-.5,1.5],size=(9,2.5),labels=('预测分数','真实类别'));dots=VGroup(*(Dot(p(float(v),float(labels[i])+(i%5-2)*.04),radius=.035,color=GREEN if labels[i] else GOLD) for i,v in enumerate(scores)));s.add(a,dots);line=DashedLine(p(threshold,-.5),p(threshold,1.5),color=INK);s.show(line);s.hold('threshold-splits-scores',4)
    s.chapter('按真实与预测两个维度，整理同一批样本','Cross-tabulate true and predicted classes.','四格分别是 TP、FN、FP、TN；每个样本只进入一格。','Every observation enters exactly one cell of the confusion table.')
    s.wipe();values=[[d['tp'],d['fn']],[d['fp'],d['tn']]];matrix=grid(values,1.35,(0,.2),GREEN);s.show(matrix)
    s.show(txt('预测正',24,MUTED).move_to(pos(-.675,1.8)),txt('预测负',24,MUTED).move_to(pos(.675,1.8)),txt('真实正',24,MUTED).move_to(pos(-2.5,.875)),txt('真实负',24,MUTED).move_to(pos(-2.5,-.475)));s.hold('confusion-table',5)
    s.chapter('分母不同，问题就不同','Changing the denominator changes the question.','Precision 问预测正中有多少真；Recall 问真实正中找回多少。','Precision conditions on predicted positives; recall conditions on actual positives.')
    col=SurroundingRectangle(VGroup(matrix.cells[0][0],matrix.cells[1][0]),color=GOLD,buff=.05);s.play(Create(col));badge=card(num(d['precision'],3),'Precision = TP / (TP + FP)',GOLD,4.7).move_to(pos(3.65,.8));s.show(badge);s.hold('precision-denominator',4);s.play(FadeOut(col),FadeOut(badge))
    outline=SurroundingRectangle(VGroup(*matrix.cells[0]),color=BLUE,buff=.05);s.play(Create(outline));s.show(card(num(d['recall'],3),'Recall = TP / (TP + FN)',BLUE,4.7).move_to(pos(3.65,.8)));s.hold('recall-denominator',4)
    s.ending('阈值是取舍，指标要对应使用问题','Threshold choice expresses a tradeoff.','"precision" != "recall"','亲手移动阈值，跟踪哪些样本从一格进入另一格。','Move the threshold and track observations migrating between cells.')


def biasvariance(s):
    d=s.d;grid_=d['grid'];fits=d['fits'];point=d['atPoint'];print('biasvariance point',point,flush=True)
    s.chapter('数据换一次，拟合曲线也换一次','A fitted function varies across datasets.','训练位置相同，只重新生成训练噪声；先看五次拟合。','Training locations stay fixed; the training noise changes across repetitions.')
    ys=[v for fit in fits[:5] for v in fit];a,p=axes([min(grid_),max(grid_)],[min(ys)-.2,max(ys)+.2],labels=('输入 x','拟合预测'));s.add(a)
    for i,fit in enumerate(fits[:5]):s.play(Create(curve(list(zip(grid_,fit)),p,PALETTE[i],2)),run_time=1.4);s.wait(.7)
    s.hold('different-fits',4)
    s.chapter('固定一个 x，看预测值如何散开','Fix one input and inspect the distribution of predictions.','不同数据集对同一点给出不同预测；散开程度是方差。','Different datasets predict different values at the same input.')
    s.wipe();pred=d['pointPredictions'];lo=min(pred)-.1;hi=max(pred)+.1;a,p=axes([lo,hi],[0,1],size=(9,.7),labels=('固定 x₀ 处的预测',''));s.show(a)
    dots=VGroup(*(Dot(p(v,.25+(i%7)*.07),radius=.04,color=BLUE) for i,v in enumerate(pred)));s.play(LaggedStart(*(FadeIn(v) for v in dots),lag_ratio=.02),run_time=2.5);s.hold('prediction-distribution',4)
    mean=point['average'];truth=mean-point['bias'];s.show(DashedLine(p(mean,-.1),p(mean,1.2),color=GREEN),DashedLine(p(truth,-.1),p(truth,1.2),color=GOLD));s.show(txt('绿色：平均预测   金色：真实函数值',25,MUTED).move_to(pos(0,1.5)));s.hold('bias-is-displacement',4)
    s.ending('平均值的偏移，和围绕平均值的波动，要分开','Bias is displacement; variance is spread.','E[(hat(f)-Y)^2]="bias"^2+"variance"+sigma^2','预测新的响应 Y 还要加上独立观测噪声；预测真实函数值时没有这一项。','Fresh-response risk includes independent observation noise; function-estimation risk does not.')


def uncertainty(s):
    d=s.d;q=d['selected'];print('uncertainty',q,flush=True)
    s.chapter('同一个位置，可以问两种不确定性','At one location, distinguish mean uncertainty from response noise.','先估计潜在均值，再预测一次新的带噪声观测。','First infer the latent mean, then predict a new noisy response.')
    values=d['grid'];lo=min(v['predictiveInterval'][0] for v in values);hi=max(v['predictiveInterval'][1] for v in values);a,p=axes([values[0]['x'],values[-1]['x']],[lo-.2,hi+.2],labels=('输入 x','潜在均值与预测区间'));s.add(a)
    for key,color in [('predictiveInterval',GOLD),('latentInterval',BLUE)]:
        band=Polygon(*[p(v['x'],v[key][0]) for v in values],*[p(v['x'],v[key][1]) for v in reversed(values)],stroke_width=1.3,color=color,fill_opacity=.1);s.show(band)
    s.show(curve([[v['x'],v['mean']] for v in values],p,GREEN),VGroup(*(Dot(p(v['x'],v['y']),radius=.055,color=BLUE) for v in d['sample'])),txt('蓝：均值区间    金：新响应区间',24,MUTED).move_to(pos(0,1.85)));s.hold('regression-bands',5)
    s.chapter('先只保留参数不确定性','Parameter uncertainty creates a latent-mean interval.','蓝色区间描述潜在均值；它不包含新观测本身的随机噪声。','The latent interval excludes the noise in a future observation.')
    s.wipe();mean=q['mean'];low,high=q['predictiveInterval'];a,p=axes([low-.2,high+.2],[0,1],size=(9,1.3),labels=('在查询位置处的响应值',''));s.show(a)
    latent=Line(p(q['latentInterval'][0],.7),p(q['latentInterval'][1],.7),color=BLUE,stroke_width=7);point=Dot(p(mean,.7),radius=.08,color=INK);s.play(GrowFromCenter(latent),FadeIn(point),run_time=1.2);s.hold('latent-interval',4)
    s.chapter('预测新观测，还要加入观测噪声','A new response adds another source of variation.','两种区间中心相同，但预测区间更宽。','The centers agree, but the predictive interval is wider.')
    response=Line(p(low,.25),p(high,.25),color=GOLD,stroke_width=7);s.play(TransformFromCopy(latent,response),run_time=1.6);s.hold('predictive-interval',4)
    s.ending('增加数据，不能消除新观测的固有噪声','More data do not remove irreducible response noise.','"Var"(Y_*|D)=x_*^T Sigma x_*+sigma^2','这个分解来自当前线性高斯模型；它不是任意神经网络的通用置信保证。','This decomposition belongs to the specified linear-Gaussian model.')


def causal(s):
    d=s.d;print('causal',d['response'],d['zGivenT'],d['pz'],flush=True)
    s.chapter('比较两组之前，先看组里都有谁','Group composition can change an aggregate comparison.','Z 同时影响接受处理的机会和结果；处理组的组成会变。','Z affects both treatment selection and outcome.')
    nodes=[card('Z','组别',GOLD,1.7).move_to(pos(0,1)),card('T','处理',BLUE,1.7).move_to(pos(-3,-.9)),card('Y','结果',GREEN,1.7).move_to(pos(3,-.9))];s.add(*nodes)
    s.play(*(GrowArrow(arrow(a,b)) for a,b in [(pos(-.7,.6),pos(-2.5,-.3)),(pos(.7,.6),pos(2.5,-.3)),(pos(-2.1,-.9),pos(2.1,-.9))]),run_time=1.5);s.hold('confounding-path',5)
    s.chapter('观察差异，使用两套不同的人群权重','Observed groups can have different mixtures of Z.','比较 P(Z | T=0) 和 P(Z | T=1)，先确认混合比例。','Inspect the Z mixture separately within each treatment group.')
    s.wipe();weights=d['zGivenT'];matrix=grid(weights,.95,(0,.1),BLUE);s.show(matrix,txt('行：T=0、T=1   列：Z=0、Z=1',25,MUTED).move_to(pos(0,1.7)));s.hold('different-mixtures',5)
    s.chapter('干预比较，保留同一套总体组成','An intervention compares treatments in the same population.','对每个处理方案，都按相同的 P(Z) 加权结果。','Both treatment interventions use the same population distribution of Z.')
    common=grid([d['pz'],d['pz']],.95,(0,.1),GREEN);s.play(Transform(matrix,common),run_time=1.7);s.hold('same-population-mixture',4)
    s.wipe();s.show(card(num(d['observationalDifference'],3),'观察差异',BLUE,3.6).move_to(pos(-2.7,.5)),card(num(d['causalEffect'],3),'模型中的干预效应',GREEN,3.6).move_to(pos(2.7,.5)));s.hold('different-questions',5)
    s.ending('调整有效，需要因果假设支持','Adjustment requires a justified causal model.','E[Y|"do"(T=t)]=sum_z E[Y|T=t,Z=z]P(Z=z)','这里因果结构已知且满足调整条件；不能只凭观察相关性推断结构。','The example specifies a valid adjustment structure; observational association alone does not establish it.')


def fairness(s):
    d=s.d;counts=d['countsPer1000'];print('fairness counts',counts,flush=True)
    s.chapter('同样的错误率，会给出同样的预测可靠性吗？','Equal error rates need not imply equal predictive value.','两组的 TPR 和 FPR 相同，但基础阳性比例不同。','The groups share TPR and FPR, but have different base rates.')
    a=d['A'];b=d['B'];left=grid([[counts['A']['tp'],counts['A']['fn']],[counts['A']['fp'],counts['A']['tn']]],.95,(-3,0),BLUE);right=grid([[counts['B']['tp'],counts['B']['fn']],[counts['B']['fp'],counts['B']['tn']]],.95,(3,0),GREEN)
    s.add(left,right);s.show(txt('组 A：每 1000 人',25,BLUE).move_to(pos(-3,1.7)),txt('组 B：每 1000 人',25,GREEN).move_to(pos(3,1.7)),txt('行：真实正 / 负    列：预测正 / 负',23,MUTED).move_to(pos(0,-1.8)));s.hold('two-populations',5)
    s.chapter('现在只看“预测为正”的那一列','Condition on positive predictions in each group.','可靠性问的是：所有预测正中，有多少是真的正？','Positive predictive value asks what fraction of positive predictions are true.')
    for matrix in [left,right]:s.play(Create(SurroundingRectangle(VGroup(matrix.cells[0][0],matrix.cells[1][0]),color=GOLD,buff=.05)),run_time=.8)
    s.hold('predicted-positive-columns',4)
    s.wipe();s.show(card(num(a['precision'],3),'组 A 的 PPV',BLUE,3.4).move_to(pos(-2.7,.5)),card(num(b['precision'],3),'组 B 的 PPV',GREEN,3.4).move_to(pos(2.7,.5)));s.hold('unequal-predictive-values',5)
    s.ending('先说清分母，再讨论公平约束','Specify the conditioning event before comparing fairness metrics.','"PPV"=("TPR" p)/("TPR" p+"FPR"(1-p))','指标之间的取舍取决于基础率与使用情境；一个数值不能替代规范判断。','Metric tradeoffs depend on base rates and context; a number does not settle the normative choice.')


def calibration(s):
    d=s.d;groups=[g for g in d['groups'] if g['count']];g=min(groups,key=lambda x:abs(x['meanPrediction']-.7));print('calibration group',g,flush=True)
    s.chapter('说“'+str(round(g['meanPrediction']*100))+'%”，长期意味着什么？','What should the reported probability mean over repeated cases?','先把相近预测概率的案例放在一起，再看其中的阳性频率。','Group similar predicted probabilities and measure their observed positive frequency.')
    report=card(num(g['meanPrediction'],3),'这一组的平均预测',BLUE,3.6).move_to(pos(-3,.5));count=card(str(g['count']),'落入这一组的案例数',GOLD,3.6).move_to(pos(3,.5));s.add(report,count);s.hold('probability-bin',4)
    s.chapter('在这一组内，数真实发生了多少次','Count actual outcomes within that probability group.','案例数是分母，真实阳性数是分子。','The bin count is the denominator; positive outcomes form the numerator.')
    s.wipe();success=int(g['outcomeSum']);total=g['count'];s.show(card(f'{success} / {total}','真实阳性频率',GREEN,4));s.hold('bin-frequency',4)
    s.chapter('每个组变成一个可靠性图上的点','Each bin becomes one reliability-diagram point.','横轴报告多少，纵轴实际发生多少；理想位置在对角线上。','The x-axis is predicted probability; the y-axis is observed frequency.')
    s.wipe();a,p=axes([0,1],[0,1],size=(4.2,4.2),equal=True,labels=('平均预测概率','组内阳性频率'));s.show(a,Line(p(0,0),p(1,1),color=RULE,stroke_width=2))
    for item in groups:
        dot=Dot(p(item['meanPrediction'],item['observedRate']),radius=.045+.055*np.sqrt(item['count']/max(x['count'] for x in groups)),color=BLUE);s.show(dot,run_time=.25);s.wait(.25)
    s.hold('reliability-points',5)
    s.ending('频率估计也有抽样波动','Reliability estimates themselves are noisy.','E[Y|q]=q','分箱越细，每箱证据可能越少；ECE 也会随分箱方式改变。','Finer bins may contain less evidence, and ECE depends on the binning scheme.')


def modelselection(s):
    d=s.d;scores=d['firstValidationScores'];selected=int(np.argmin(scores));truth=d['trueRisk']
    s.chapter('验证分数也会带着噪声','Validation scores contain sampling noise.','本例候选模型的真实风险相同；看到的验证分数仍不相同。','Candidates have equal true risk in this model, but noisy validation scores differ.')
    m=min(20,len(scores));lo=min(scores)-.1;hi=max(max(scores),truth)+.2;a,p=axes([0,m+1],[lo,hi],labels=('候选模型编号','验证分数'));s.add(a)
    dots=VGroup(*(Dot(p(i+1,scores[i]),radius=.07,color=BLUE) for i in range(m)));s.play(LaggedStart(*(FadeIn(dot) for dot in dots),lag_ratio=.09),run_time=2.8);s.hold('noisy-scores',4)
    s.chapter('挑最小值，也挑中了有利的噪声','Selecting the minimum selects favorable noise.','只根据这次验证集挑赢家，分数会系统性偏乐观。','Choosing the validation winner creates optimistic selection bias.')
    ring=SurroundingRectangle(dots[selected],color=GOLD,buff=.12);s.play(Create(ring),run_time=1);s.hold('selected-winner',4)
    s.show(DashedLine(p(0,truth),p(m+1,truth),color=GREEN),txt('共同真实风险',23,GREEN).move_to(pos(3.3,1.6)));s.hold('truth-versus-winner',4)
    s.chapter('用独立测试，重新衡量这个赢家','Evaluate the selected model on independent data.','独立测试没有参与选择，所以不会重复奖励同一份验证噪声。','Independent test data did not participate in selection.')
    s.wipe();s.show(card(num(d['averageValidation'],3),'重复实验：赢家验证分数均值',GOLD,4.8).move_to(pos(-3,.6)),card(num(d['averageTest'],3),'独立测试分数均值',GREEN,4.8).move_to(pos(3,.6)));s.hold('independent-evaluation',5)
    s.ending('反复看测试集，也会把它变成选择集','Repeated test-driven selection destroys that independence.','"select on validation; evaluate independently"','这个模型隔离了选择噪声；它不表示真实候选模型的风险一定相同。','Equal true risks are a controlled assumption here, not a claim about real candidate models.')


def ensemble(s):
    d=s.d;rho=s.p['rho'];sigma=s.p['sigma'];m=s.p['members'];pred=d['predictions']
    s.chapter('平均多个预测，会抵消多少误差？','Averaging can cancel errors that disagree.','先区分每个模型独有的波动与大家共同的波动。','Distinguish independent variation from variation shared across models.')
    a=row([d['prefixes'][i][0] for i in range(min(6,m))],[f'{i+1} 个模型均值' for i in range(min(6,m))],BLUE,1.8,.7);s.add(a[0]);s.hold('one-model',3)
    for i in range(1,len(a)):s.play(TransformFromCopy(a[i-1],a[i]),run_time=.9);s.wait(.7)
    s.hold('growing-average',4)
    s.chapter('独有误差被平均，共同误差留在原地','Independent noise shrinks; common noise remains.','相关系数 ρ 决定随模型数增加也无法消掉的部分。','Correlation determines the variance component that averaging cannot remove.')
    s.wipe();a,p=axes([1,m],[0,sigma*sigma*1.1],labels=('模型数 M','平均预测的理论方差'));s.show(a)
    c=curve([[i,sigma*sigma*(rho+(1-rho)/i)] for i in np.linspace(1,m,150)],p,BLUE);floor=DashedLine(p(1,sigma*sigma*rho),p(m,sigma*sigma*rho),color=GOLD);s.play(Create(c),run_time=2.5);s.show(floor);s.hold('correlation-floor',5)
    s.show(txt('共享部分  σ²ρ = '+num(sigma*sigma*rho,3),25,GOLD).move_to(pos(2.2,1.3)));s.hold('shared-component',4)
    s.ending('模型数量，不能替代误差多样性','More members do not guarantee independent errors.','"Var"(bar(f))=sigma^2(rho+(1-rho)/M)','这个等方差、等相关模型里，增加 M 不会消除共同偏差。','In this equal-variance, equal-correlation model, averaging also leaves common bias unchanged.')


STORIES={'sampling':sampling,'confidence':confidence,'testing':testing,'bayes':bayes,'likelihood':likelihood,'kde':kde,'entropy':entropy,'metrics':metrics,'biasvariance':biasvariance,'uncertainty':uncertainty,'causal':causal,'fairness':fairness,'calibration':calibration,'modelselection':modelselection,'ensemble':ensemble}
