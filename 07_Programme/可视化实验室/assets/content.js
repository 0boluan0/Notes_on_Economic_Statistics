/* Original instructional text and exercises. Source review: docs/来源核验.md. */
(function (root) {
  'use strict';
  const r = (key,label,min,max,step,value) => ({key,label,type:'range',min,max,step,value});
  const s = (key,label,value,options) => ({key,label,type:'select',value,options:options.map(([value,label])=>({value,label}))});
  const seed = () => r('seed','随机种子',1,99999,1,42);
  const src = (label,url,section,support) => ({label,url,section,support});
  const challenge = (question,hint,answer) => ({question,hint,answer});
  const source = {
    derivative: src('OpenStax · Calculus Volume 1','https://openstax.org/books/calculus-volume-1/pages/3-1-defining-the-derivative','§3.1 Tangent Lines; The Derivative of a Function at a Point','差商、导数极限和切线；本实验的三次函数误差式为直接代数推导。'),
    determinant: src('Georgia Tech · Interactive Linear Algebra','https://textbooks.math.gatech.edu/ila/determinants-volumes.html','§4.3 Determinants and Volumes','行列式绝对值给出面积缩放；二维公式的符号记录有向面积。'),
    projection: src('Georgia Tech · Interactive Linear Algebra','https://textbooks.math.gatech.edu/ila/projections.html','§6.3.1 Orthogonal Decomposition; Recipe: Orthogonal projection onto a line','正交分解、最近点及投影公式；勾股恒等式由正交性展开。'),
    gradient: src('MIT · Foundations of Computer Vision','https://visionbook.mit.edu/gradient_descent.html','§10.3 Basic Gradient Descent; §10.4 Learning Rate Schedules','梯度迭代和学习率作用；本实验严格稳定界由二次函数递推自行推导，不外推到任意损失。'),
    clt: src('OpenStax · Introductory Statistics 2e','https://openstax.org/books/introductory-statistics-2e/pages/7-1-the-central-limit-theorem-for-sample-means-averages','§7.1 The Central Limit Theorem for Sample Means','样本均值的抽样分布与标准误；实验补足 IID 与有限方差条件，正态总体的结果为精确分布。'),
    limitTheorems: src('Rick Durrett · Probability: Theory and Examples, 5th ed.','https://math.duke.edu/~rtd/PTE/PTE5_011119.pdf','Theorem 2.2.14, printed p.64 (PDF p.72); Theorem 3.4.1, printed p.144 (PDF p.152)','IID 且一阶绝对矩有限的弱大数定律；IID 且正有限方差的标准化 CLT。'),
    cauchy: src('NIST/SEMATECH · Cauchy Distribution','https://www.itl.nist.gov/div898/handbook/eda/section3/eda3663.htm','§1.3.6.6.3; Common Statistics','标准柯西的均值、标准差未定义；对称中心不是期望。'),
    ci: src('OpenStax · Introductory Statistics 2e','https://openstax.org/books/introductory-statistics-2e/pages/8-1-a-single-population-mean-using-the-normal-distribution','§8.1; known population standard deviation and error bound','已知总体标准差的 z 区间与宽度；本实验额外固定正态总体以保证有限样本精确覆盖。'),
    coverage: src('NIST/SEMATECH · Confidence Limits for the Mean','https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm','§1.3.5.2; Purpose and confidence interpretation','置信水平属于重复构造区间的方法；该页主要 t 区间公式不作为本实验 z 公式来源。'),
    test: src('NIST/SEMATECH · Test of a Process Mean','https://www.itl.nist.gov/div898/handbook/prc/section2/prc22.htm','§7.2.2; test statistic and rejection region','已知标准差时的 z 统计量与双侧拒绝域。'),
    pvalue: src('NIST/SEMATECH · Critical Values and p Values','https://www.itl.nist.gov/div898/handbook/prc/section1/prc131.htm','§7.1.3.1 Critical values and p values','p 值是在零假设与参考分布下的尾概率，和预先指定显著性水平比较。'),
    power: src('NIST/SEMATECH · Sample Sizes Required','https://www.itl.nist.gov/div898/handbook/prc/section2/prc222.htm','§7.2.2.2; Type II error and power','功效取决于指定备择效应与样本量；本实验双侧精确功效由正态尾概率自行计算。'),
    bayes: src('Holmes & Huber · Modern Statistics for Modern Biology','https://web.stanford.edu/class/bios221/book/02-chap.html','§2.9.3–2.9.6; especially §2.9.5 The posterior distribution is also a Beta','Beta 先验与二项数据更新；先验参数加成功/失败次数，允许顺序更新。'),
    beta: src('NIST/SEMATECH · Beta Distribution','https://www.itl.nist.gov/div898/handbook/eda/section3/eda366h.htm','§1.3.6.6.17; density, CDF, percent point function, mean','Beta 密度和均值；分位数需数值求逆，实验区间为中央等尾区间。'),
    leastSquares: src('Georgia Tech · Interactive Linear Algebra','https://textbooks.math.gatech.edu/ila/least-squares.html','§6.5.1 Least-Squares Solutions','最小二乘与正交投影、正规方程、系数解唯一性的满列秩条件。'),
    ridge: src('Ryan Tibshirani · CMU Modern Regression 1','https://stat.cmu.edu/~ryantibs/datamining/lectures/16-modr1.pdf','slides 3–8; Important details (slide 7)','岭回归的平方和目标、闭式解、标准化和不惩罚截距；方差讨论属于重复样本层面。'),
    overfit: src('Cosma Shalizi · CMU Optimism and Over-Fitting','https://www.stat.cmu.edu/~cshalizi/sml/21/lectures/04/lecture-04.html','Lecture 4; Estimating the true risk; The moral: approximation-estimation trade-off','训练误差的乐观性、嵌套模型类训练误差不增，以及逼近误差与估计误差的交换。'),
    selection: src('Cawley & Talbot · JMLR 11 (2010)','https://jmlr.csail.mit.edu/papers/volume11/cawley10a/cawley10a.pdf','Abstract; §5.1 An Unbiased Performance Evaluation Methodology; §5.3 Another Example of Biased Evaluation Methodology','对有限样本的模型选择准则也会过拟合；评估需隔离选择过程。'),
    pca: src('Cosma Shalizi · CMU Principal Components Analysis','https://stat.cmu.edu/~cshalizi/dm/19/lectures/05/lecture-05.html','Lecture 5; Finding the principal component; Low-rank approximation; Backup: Gory details','最大投影方差、最小重构误差和特征向量；本实验用样本协方差 n−1 约定。'),
    pcaScale: src('NIST/SEMATECH · Properties of Principal Components','https://www.itl.nist.gov/div898/handbook/pmc/section5/pmc551.htm','§6.5.5.1; Roadmap to solve the V matrix','标准化数据的协方差对应相关矩阵，特征分解给出方向和解释方差。'),
    bootstrap: src('UC Berkeley · Stat 20 Bootstrapping','https://stat20.berkeley.edu/fall-2025/3-generalization/09-bootstrapping/notes.html','Bootstrap; Bootstrap Confidence Interval (percentile method); Summary cautions','从经验分布有放回重采样，百分位区间及代表性限制；区间覆盖并非有限样本精确保证。')
  };
  const labs = {
    tangent: {
      title:'割线如何变成切线',en:'Derivatives & finite differences',group:'微积分与几何',
      summary:'移动观察点与差分步长，把局部变化率、解析导数和数值近似放到同一张图上。',
      question:'把步长缩小十倍，差商误差一定也缩小十倍吗？',
      formula:String.raw`f'(x)=\lim_{h\to0}\frac{f(x+h)-f(x)}h,\quad D_h^+f=\frac{f(x+h)-f(x)}h,\quad D_h^0f=\frac{f(x+h)-f(x-h)}{2h}`, 
      controls:[s('function','函数','cubic',[['cubic','三次函数 x³ − x'],['sine','正弦 sin x（弧度）'],['exponential','指数 exp x']]),r('x','观察位置 x',-2.5,2.5,.05,1),r('h','差分步长 h',.001,1,.001,.5)],
      presets:[{label:'弯曲处：比较两种误差',params:{function:'cubic',x:1,h:.5}},{label:'拐点：两种差商重合',params:{function:'cubic',x:0,h:.5}},{label:'正弦峰值附近',params:{function:'sine',x:1.55,h:.1}}],
      steps:['先只看原函数：预测当前点的斜率正负，再读解析导数。','固定 x=1、三次函数，分别设置 h=0.5、0.1、0.01，比较前向和中心差分的绝对误差。','把 x 改为 0，解释为什么两种误差相同；用下方挑战的展开式核对。','换成 sin x 或 exp x，区分函数值的变化与导数值的变化。'],
      explanation:'切线给出当前点的局部一阶近似；前向差商使用右侧一点，中心差商把两侧的信息对称组合。对本实验三次函数，直接展开得到前向误差 3xh+h²，中心误差 h²。中心差分通常消掉一阶误差项，但在 x=0 两种公式恰好相同。图中的相交不等于在整个范围内贴合。',
      assumptions:['当前三个函数处处光滑；中心差分需要能在 x−h 和 x+h 处评价函数。','在通常的光滑性条件下，前向截断误差为 O(h)，中心为 O(h²)；个别位置主项可能消失。','数值计算使用有限精度；本页 h≥0.001，不能用此范围证明无限缩小 h 永远改善精度。'],
      pitfalls:['导数是极限，某一个有限 h 的割线斜率一般不是导数。','中心差分更高阶不代表任意点、任意 h 下绝对误差必然更小。','sin x 的导数为 cos x 使用弧度；把度数直接代入会改变数值。'],
      challenges:[challenge('设 f(x)=x³−x、x=1、h=0.1。先算出两种差商与真实导数，再操作核对。','展开 (x+h)³ 和 (x−h)³；不要先看误差指标。','f′(1)=2。前向差商=2+3×1×0.1+0.1²=2.31；中心差商=2+0.1²=2.01。绝对误差分别 0.31 与 0.01。'),challenge('为什么三次函数在 x=0 时，前向和中心差分完全相同？这能证明前向方法也是二阶方法吗？','看前向误差中的 x。','在 x=0 时 3xh=0，两者都等于 −1+h²。该点的一阶误差恰好抵消，只说明这个函数在这个位置的特殊行为，不能推广到一般 x。'),challenge('在三次函数 x=−1 处，把 h 从 0.5 改成 0.1。前向差商会从真实导数哪一侧接近？','前向误差是 −3h+h²，当前 h 在 (0,1]。','真实导数为 2，误差为负。h=0.5 时差商 0.75；h=0.1 时 1.71，从下方接近 2。比较不同位置时不能只背“差商总是偏大”。')],
      sources:[source.derivative],related:['gradient','overfit']
    },
    linear:{
      title:'矩阵怎样改变空间',en:'Linear maps & signed area',group:'微积分与几何',summary:'观察基向量、网格和单位正方形如何被同一个矩阵变换，读懂行列式的面积与方向。',question:'面积不变，图形就一定只是旋转吗？',
      formula:String.raw`A=\begin{pmatrix}a&b\\c&d\end{pmatrix},\quad Ae_1=(a,c)^T,\quad Ae_2=(b,d)^T,\quad \det A=ad-bc,\quad \mathrm{Area}=|\det A|`,
      controls:[r('a','矩阵 a（左上）',-2,2,.1,1),r('b','矩阵 b（右上）',-2,2,.1,.7),r('c','矩阵 c（左下）',-2,2,.1,0),r('d','矩阵 d（右下）',-2,2,.1,1)],
      presets:[{label:'剪切：面积保持',params:{a:1,b:1,c:0,d:1}},{label:'反射：方向翻转',params:{a:-1,b:0,c:0,d:1}},{label:'秩一：压成一条线',params:{a:1,b:2,c:.5,d:1}},{label:'旋转 90°',params:{a:0,b:-1,c:1,d:0}}],
      steps:['先看矩阵的两列，预测 e₁ 与 e₂ 各自的终点，再对照图。','比较“剪切”和“旋转 90°”：行列式相同，但长度与角度的变化不同。','使用反射预设，沿单位正方形顶点的固定次序观察方向翻转。','使用秩一预设；尝试找两个不同输入，使它们产生同一输出。'],
      explanation:'每个向量 v=(x,y) 都是 x e₁+y e₂；线性变换的输出就是 x 倍第一列加 y 倍第二列。因此两列已经决定整个网格的去向。行列式描述有向面积的缩放，绝对值才是普通面积倍率。两列相关时面积为零，变换会失去一个或两个维度的信息。',
      assumptions:['这是过原点的二维线性变换，不含平移。','面积结论用于二维面积；行列式为零不表示所有向量都被映到零。','几何图使用相同的坐标尺度；面积由行列式计算，不靠屏幕像素估计。'],
      pitfalls:['行列式为 1 只保证面积保持，不能保证长度、角度或圆形保持。','矩阵列向量是基向量的像；把行当成列会得到不同变换。','负行列式是方向翻转，不是“负的普通面积”。'],
      challenges:[challenge('A=[[1,1],[0,1]] 的面积倍率为多少？向量 (0,1) 的长度是否保持？','先看第二列，再计算它的欧氏长度。','det A=1，单位正方形面积仍为 1；(0,1)→(1,1)，长度由 1 变为 √2。剪切保持面积，却不保持长度。'),challenge('秩一预设 A=[[1,2],[0.5,1]] 中，哪个非零向量会消失？','找第一列和第二列的线性关系。','第二列是第一列的两倍，故 A(−2,1)ᵀ=0。任意 v 与 v+t(−2,1) 都有同一输出，因而不能唯一逆推输入。'),challenge('A=diag(−1,1) 和 A=diag(−1,−1) 都出现负号，为什么只有前者翻转有向面积？','分别算两个行列式。','前者 det=−1，是沿一个轴反射；后者 det=+1，相当于旋转 180°。翻转两个独立方向会恢复原来的整体取向。')],
      sources:[source.determinant],related:['projection','pca','regression']
    },
    projection:{
      title:'投影、残差与最短距离',en:'Orthogonal projection & residuals',group:'微积分与几何',summary:'把一个向量分成沿指定方向的分量和正交残差，建立最小二乘与 PCA 的几何基础。',question:'沿直线选一个点，为什么垂直落点离原向量最近？',
      formula:String.raw`u=(\cos\theta,\sin\theta)^T,\quad p=(u^Tv)u,\quad r=v-p,\quad u^Tr=0,\quad \|v\|^2=\|p\|^2+\|r\|^2`,
      controls:[r('angle','投影方向 θ（度）',0,175,5,60),r('vx','向量 v 的 x 分量',-3,3,.1,2),r('vy','向量 v 的 y 分量',-3,3,.1,1)],
      presets:[{label:'投影到水平轴',params:{angle:0,vx:2,vy:1}},{label:'原向量垂直于直线',params:{angle:45,vx:1,vy:-1}},{label:'原向量就在直线上',params:{angle:45,vx:2,vy:2}}],
      steps:['固定 v=(2,1)，旋转方向；预测哪个角度让残差最短。','使用垂直预设，解释为什么投影长度为零但原向量不为零。','把 v 改为零，检查所有范数与残差；零向量没有唯一方向。','把投影看成一个一元最小二乘问题，完成第三道挑战。'],
      explanation:'投影 p 是直线 span(u) 上离 v 最近的向量。任意直线上的点 q 与 p 的差沿 u，而 v−p 与 u 垂直；所以 ||v−q||²=||v−p||²+||p−q||²，额外偏离 p 只会增加距离。这个恒等式同时解释残差正交性和最小二乘。',
      assumptions:['u 由三角函数构造，长度为 1；对一般非单位向量 a，应使用 a(aᵀv)/(aᵀa)。','最短距离采用欧氏范数；改变距离定义可能改变最优投影。','方向角相差 180° 对应同一条直线，投影不随基向量符号改变。'],
      pitfalls:['投影的标量坐标 uᵀv 可以为负；投影长度 ||p|| 总是非负。','r 与投影直线正交，不要求 r 与原向量 v 正交。','这是投到整条直线，不是投到射线或有限线段。'],
      challenges:[challenge('v=(2,1)，投影到水平轴后 p、r 和三条平方长度分别是多少？','u=(1,0)，先算 uᵀv。','p=(2,0)，r=(0,1)；||v||²=5，||p||²=4，||r||²=1，满足 5=4+1。'),challenge('v=(1,−1)，θ=45°。为什么投影为零？若将方向改为 135° 呢？','分别计算两个单位方向与 v 的内积。','45° 时 u=(1,1)/√2，与 v 内积为 0，故 p=0；135° 时 u=(−1,1)/√2，内积为 −√2，p=v。负坐标也能给出原向量。'),challenge('用微积分求 minₜ ||v−tu||²，并与图中投影连接。','先展开平方，使用 uᵀu=1。','目标为 ||v||²−2t uᵀv+t²。导数 2t−2uᵀv=0 给出 t*=uᵀv；二阶导数 2>0，所以唯一最小点为 p=t*u。')],
      sources:[source.projection],related:['linear','regression','pca']
    },
    gradient:{animation:{key:'steps'},
      title:'梯度下降何时收敛',en:'Gradient descent & conditioning',group:'微积分与几何',summary:'在可精确分析的椭圆二次函数上，观察步长、曲率和起点怎样决定收敛、振荡与发散。',question:'每步都沿负梯度走，为什么函数值还可能越来越大？',
      formula:String.raw`f(x,y)=\frac{x^2+\kappa y^2}{2},\quad x_{t+1}=(1-\eta)x_t,\quad y_{t+1}=(1-\eta\kappa)y_t,\quad 0<\eta<\frac2\kappa`,
      controls:[r('rate','学习率 η',.01,1.2,.01,.2),r('condition','曲率比 κ',1,20,1,4),r('steps','迭代步数',1,60,1,20),r('x0','起点 x₀',-4,4,.25,3),r('y0','起点 y₀',-4,4,.25,2)],
      presets:[{label:'稳定收敛',params:{rate:.2,condition:4,steps:20,x0:3,y0:2}},{label:'边界：持续振荡',params:{rate:.5,condition:4,steps:20,x0:3,y0:2}},{label:'越界：快速发散',params:{rate:.6,condition:4,steps:20,x0:3,y0:2}},{label:'病态但稳定',params:{rate:.09,condition:20,steps:60,x0:3,y0:2}}],
      steps:['保留 κ=4，预测 η=0.2、0.5、0.6 的轨迹，再逐个使用预设。','对照参数轨迹与损失曲线：振荡可以同时伴随损失下降。','把 κ 提高到 20，保留原学习率，先算 2/κ 再判断稳定性。','在越界步长下设 y₀=0，解释为什么这条特殊轨迹仍可能收敛。'],
      explanation:'负梯度只是局部下降方向；有限一步可能跨过谷底太远。这个二次函数的两个坐标分别乘以 1−η 和 1−ηκ。为了对所有起点收敛，两个倍率的绝对值都必须小于 1。曲率比增大会缩小稳定步长范围；被最陡方向限制的小步长又让平缓方向走得慢。',
      assumptions:['κ≥1，Hessian=diag(1,κ) 正定，唯一最小点为原点。','严格界 0<η<2/κ 保证所有起点收敛；若起点在不稳定特征方向的分量恰为零，单条轨迹可例外。','这是固定步长、精确梯度、二维二次损失；不能把其精确界直接当作神经网络训练保证。'],
      pitfalls:['在 η=2/κ 处，非零 y 分量符号反复改变且幅值不变，不能说已收敛。','损失曲线很低或参数变动很小不等于数学上已到达最优点。','发散数据可能远超初始尺度；查看图的坐标和截断说明，不能把绘图边界误认为轨迹停止。'],
      challenges:[challenge('κ=4、η=0.2、起点 (3,2)。手算一步和两步的位置。','每一步分别乘 0.8 与 0.2。','第一步 (2.4,0.4)，第二步 (1.92,0.08)。初始损失 12.5；第一步损失 (2.4²+4×0.4²)/2=3.2。'),challenge('κ=4、η=0.5、y₀=2 时，为什么增加到 60 步仍不能收敛到原点？','y 的倍率是 1−ηκ。','倍率为 −1，所以 yₜ=(−1)ᵗ×2；xₜ=(0.5)ᵗx₀ 趋于 0，但损失趋于 4×2²/2=8，而不是 0。'),challenge('κ=20、η=0.2 已超过 2/κ。若起点为 (3,0)，图却收敛，是理论错了吗？','检查不稳定方向最初有没有分量。','y₀=0 导致所有 yₜ=0；剩下 xₜ=0.8ᵗ×3 会收敛。这个特殊初值不推翻“对所有初值稳定”的界。现实舍入或扰动可能重新引入不稳定方向。')],
      sources:[source.gradient],related:['tangent','linear','regression']
    },
    sampling:{
      title:'抽样均值、大数与中心极限定理',en:'Sampling means, LLN & CLT',group:'概率与统计推断',summary:'把一次累积均值路径和多次独立样本均值分布分开观察，并用柯西分布检验定理边界。',question:'样本量很大时，任何分布的平均值都会越来越稳定吗？',
      formula:String.raw`\bar X_n=\frac1n\sum_{i=1}^nX_i,\quad E\bar X_n=\mu,\quad \mathrm{SE}(\bar X_n)=\frac\sigma{\sqrt n},\quad \frac{\sqrt n(\bar X_n-\mu)}\sigma\Rightarrow N(0,1)`,
      controls:[s('distribution','总体分布','exponential',[['normal','正态 N(0,1)'],['exponential','指数 Exp(rate=1)'],['cauchy','标准柯西（均值未定义）']]),r('n','每份样本量 n',1,200,1,30),r('reps','独立重复次数 B',100,2000,100,500),seed()],
      presets:[{label:'偏斜总体，小样本',params:{distribution:'exponential',n:3,reps:1000,seed:42}},{label:'偏斜总体，均值趋近正态',params:{distribution:'exponential',n:100,reps:1000,seed:42}},{label:'定理边界：柯西均值',params:{distribution:'cauchy',n:100,reps:1000,seed:42}}],
      steps:['保持同一个种子与 B，比较指数总体下 n=3、30、100 的均值直方图。','区分横轴上的“一个样本均值”和样本内部的单次观测。','只增加 B：预测图更平滑还是均值分布更窄，再观察。','换成柯西分布，查看理论均值和标准误为何显示未定义；检查分位数裁切说明与省略计数，并从导出查看极端值。'],
      explanation:'n 是每份样本里观测的个数，B 是把整个抽样过程重复几遍。增加 n 在有限方差的 IID 设置下把均值标准误缩小到 σ/√n；增加 B 只是更精确地画出同一个抽样分布。大数定律关心均值逼近总体期望，中心极限定理关心标准化波动的极限形状。柯西分布没有有限期望与方差，不能套用这里的公式。',
      assumptions:['正态和指数样本 IID；经典 CLT 使用有限均值和 0<σ²<∞。','正态总体的均值在每个 n 下都精确正态；指数总体的均值在有限 n 下仍有偏斜。','一条累积路径只能展示一种实现，不能证明或否定一个渐近定理；有限浮点模拟也不能穷尽重尾。'],
      pitfalls:['“n≥30 就够了”不是普适定理，近似质量还依赖总体形状与所求尾部。','柯西的对称中心 0 是位置参数，不是总体均值。','直方图会随分箱和种子变化；不要用一次好看的图替代假设检查。'],
      challenges:[challenge('指数总体方差为 1。n 从 25 增到 100，理论标准误变多少？B 从 500 增到 2000 呢？','标准误公式中只包含 n。','SE 从 1/5=0.2 降到 1/10=0.1，减半。只增加 B 不改变每份样本均值的理论标准误，只降低模拟估计与图形的随机误差。'),challenge('正态总体 n=1 与指数总体 n=1，谁的均值分布已经是正态？','n=1 的样本均值就是原观测。','正态总体在 n=1 时就精确正态，指数总体仍是指数。CLT 不需要原总体本来正态，但也不宣称小样本一定正态。'),challenge('柯西的图以 0 为中心，能否把“均值=0、标准误=某个有限数”写进报告？','对称性与期望存在是两件事。','不能。柯西一阶绝对矩发散，均值与标准差未定义；有限模拟的算术平均当然能算，但它不是一个有限总体均值的稳定估计。应写位置中心为 0，并明确该实验违反经典条件。')],
      sources:[source.clt,source.limitTheorems,source.cauchy],related:['confidence','testing','bootstrap']
    },
    confidence:{
      title:'置信区间的覆盖率',en:'Confidence intervals & coverage',group:'概率与统计推断',summary:'反复从同一正态总体抽样，观察区间宽度、漏盖次数和模拟覆盖率的波动。',question:'95% 置信区间意味着已经算出的这个区间有 95% 的概率包含 μ 吗？',
      formula:String.raw`I(\mathbf X)=\left[\bar X-z_{1-\alpha/2}\frac\sigma{\sqrt n},\ \bar X+z_{1-\alpha/2}\frac\sigma{\sqrt n}\right],\quad P_\mu\{\mu\in I(\mathbf X)\}=1-\alpha`,
      controls:[r('n','每份样本量 n',2,200,1,25),s('level','置信水平',.95,[[.8,'80%'],[.9,'90%'],[.95,'95%'],[.99,'99%']]),r('reps','重复区间数 B',30,300,10,100),r('sigma','已知总体标准差 σ',.25,3,.25,1),seed()],
      presets:[{label:'95% 覆盖基线',params:{n:25,level:.95,reps:100,sigma:1,seed:42}},{label:'样本量四倍，宽度减半',params:{n:100,level:.95,reps:100,sigma:1,seed:42}},{label:'提高置信水平',params:{n:25,level:.99,reps:100,sigma:1,seed:42}}],
      steps:['先预测 100 个 95% 区间中大约有几个漏掉 μ=0，再看模拟结果。','保持其余参数，比较 n=25 与 n=100 的宽度。','把置信水平改为 99%，说明覆盖与精度的交换。','当 B>100 时区间图仅展示前 100 个；使用指标中所有 B 个区间的覆盖率与分母。'],
      explanation:'参数 μ=0 在重复抽样中固定，区间随样本变化。95% 描述长期重复这套构造方法的覆盖比例；一次实现出来的区间要么覆盖，要么不覆盖。这里已知 σ 且总体正态，所以名义覆盖是精确的；有限 B 的模拟比例会波动，其量级可用覆盖指示变量的二项标准误衡量。',
      assumptions:['观测独立且同分布为 N(0,σ²)，σ 作为已知总体参数输入。','公式没有用样本标准差替换 σ；若 σ 未知且仍假设正态，通常需要 t 区间。','B 个区间的覆盖指示相互独立；模拟标准误描述覆盖率估计的精度，不是单个均值的标准误。'],
      pitfalls:['置信区间不表示包含 95% 的个体观测；那是不同的预测/容忍区间问题。','一次 B=100 模拟出现 92% 或 98% 覆盖，不足以判定 95% 方法错误。','提高 n 使区间更窄，按同一精确方法构造时名义覆盖仍不变。'],
      challenges:[challenge('σ=1、n=25、95% 时半宽约多少？n=100 呢？','z₀.₉₇₅≈1.959964。','半宽分别约 1.96/5=0.392 与 1.96/10=0.196；总宽约 0.784 与 0.392。样本量四倍才能让宽度减半。'),challenge('B=100、名义覆盖 0.95 时，模拟覆盖率的理论标准差约多少？','每次覆盖是一个 Bernoulli(0.95) 指示。','标准差 √(0.95×0.05/100)≈0.0218，即 2.18 个百分点。95% 左右相差几个百分点完全可能；B 增加四倍会将这个 Monte Carlo 标准差减半。'),challenge('得到区间 [−0.2,0.4] 后，能否作“μ 有 95% 概率在里面”的频率学解释？','区分抽样前的随机区间和抽样后的固定区间。','不能这样解释这个频率学区间。正确说法是：本方法在重复抽样中覆盖固定 μ 的比例为 95%。若要对参数作给定数据后的概率陈述，需要另外指定先验与模型，得到贝叶斯后验区间。')],
      sources:[source.ci,source.coverage],related:['sampling','testing','bayes','bootstrap']
    },
    testing:{
      title:'检验、p 值与统计功效',en:'Two-sided z tests & power',group:'概率与统计推断',summary:'分开查看零假设分布、指定备择分布与一次观测，理解误报、漏报和样本量。',question:'同一个观测均值，为什么样本量不同会改变 p 值？',
      formula:String.raw`Z=\sqrt n\,\bar X,\quad p=2\{1-\Phi(|Z|)\},\quad c=z_{1-\alpha/2},\quad \mathrm{Power}(\delta)=\Phi(-c-\delta\sqrt n)+1-\Phi(c-\delta\sqrt n)`,
      controls:[r('n','样本量 n',2,200,1,30),r('effect','备择真实均值 δ',-1.5,1.5,.05,.5),s('alpha','显著性水平 α',.05,[[.01,'1%'],[.05,'5%'],[.1,'10%']]),r('observed','本次观测均值 x̄',-1.5,1.5,.05,.3)],
      presets:[{label:'无效应：功效等于 α',params:{n:30,effect:0,alpha:.05,observed:.3}},{label:'中等效应，小样本',params:{n:10,effect:.5,alpha:.05,observed:.3}},{label:'同一效应，更大样本',params:{n:100,effect:.5,alpha:.05,observed:.3}}],
      steps:['先固定 n 与 α，移动 observed；它改变本次证据，不改变检验设计的功效。','再移动 effect；它改变功效和备择曲线，不改变同一个 observed 对 H₀ 的 p 值。','让 effect=0，检查拒绝概率是否等于 α。','比较 n=10 与 100，解释拒绝阈值与实际效应大小各扮演什么角色。'],
      explanation:'H₀ 设 μ=0，σ=1 已知。拒绝域放在零假设抽样分布的两端，总概率为 α。p 值使用 H₀ 下至少同样极端的统计量概率；功效则使用某个指定真实均值 δ 下落入同一拒绝域的概率。两者来自不同条件下的分布，不能把 p 值当作“无效应是真的概率”。',
      assumptions:['每个样本来自 IID N(μ,1)，双侧检验规则事先指定。','effect 是用来计算功效的假定真实均值；observed 是一次样本均值，两者可以独立设置。','单次检验的 α 控制这里的误报率；反复筛选、提前停试或多重比较需要另行处理。'],
      pitfalls:['没有拒绝 H₀ 不等于证明 μ=0；可能是样本太少，功效低。','p 值小不代表效应大，也不度量实际价值。','功效不是数据观察后“备择为真”的概率；更严格的 α 通常会降低同一效应下的功效。'],
      challenges:[challenge('n=100、observed=0.3、α=0.05，z 和 p 约是多少？','标准误为 0.1，双侧 p 使用两个尾部。','z=0.3/0.1=3，p≈0.00270，拒绝 H₀。若 n=10，z≈0.9487、p≈0.343，不能拒绝；观测效应相同，但估计精度不同。'),challenge('保持 n、α 和 observed 不变，把 effect 从 0.5 改成 −0.5。什么保持不变？','双侧拒绝域关于零对称。','p 值完全不变，因为它只由 observed、n 和 H₀ 决定；功效也不变，因为正负等幅的正态备择互为镜像。备择曲线向另一侧移动。'),challenge('设 effect=0。把 α 从 0.05 降到 0.01，功效为何也变成 0.01？','此时所谓备择分布就是零假设分布。','拒绝概率正是第一类错误率 α。此处显示的“功效函数在 δ=0 处的值”为检验大小；它不表示发现了真实的非零效应。')],
      sources:[source.test,source.pvalue,source.power],related:['sampling','confidence','bayes']
    },
    bayes:{
      title:'先验与数据怎样形成后验',en:'Beta–Binomial updating',group:'概率与统计推断',summary:'在成功率推断中并列观察先验、缩放似然和后验，解释收缩与可信区间。',question:'相同的七次成功、三次失败，为什么不同先验会给出不同后验？',
      formula:String.raw`p\sim\mathrm{Beta}(a,b),\quad L(p)\propto p^s(1-p)^f,\quad p\mid D\sim\mathrm{Beta}(a+s,b+f),\quad E[p\mid D]=\frac{a+s}{a+b+s+f}`, 
      controls:[r('a','先验参数 a',1,15,1,2),r('b','先验参数 b',1,15,1,2),r('success','成功次数 s',0,50,1,7),r('failure','失败次数 f',0,50,1,3)],
      presets:[{label:'温和对称先验',params:{a:2,b:2,success:7,failure:3}},{label:'较强对称先验',params:{a:15,b:15,success:7,failure:3}},{label:'尚无观测',params:{a:2,b:5,success:0,failure:0}},{label:'全部成功仍有不确定性',params:{a:1,b:1,success:10,failure:0}}],
      steps:['保持 s=7、f=3，比较 Beta(2,2) 与 Beta(15,15)；先预测后验均值更接近哪里。','设置 s=f=0，检查后验是否与先验一致。','使用全部成功预设，比较 MLE=1 与后验均值。','观察图例：似然只为比较形状而缩放，后验区间来自积分概率。'],
      explanation:'贝叶斯更新把关于同一参数的先验密度与数据似然相乘，再归一化。Beta 与二项似然的幂次相加，所以成功次数加到 a，失败次数加到 b。样本非空时，后验均值是先验均值 a/(a+b) 与成功比例 s/(s+f) 的加权平均，权重分别为 a+b 与 s+f。',
      assumptions:['给定固定成功概率 p，每次试验条件独立且同分布为 Bernoulli(p)。','这里 a,b≥1；所有后验都是适当分布，不涉及不适当先验。','95% 区间是后验 2.5% 与 97.5% 分位数构成的等尾可信区间，不一定是最短区间。'],
      pitfalls:['似然作为 p 的函数不是默认归一化的概率密度；图中缩放不改变后验更新。','先验强度影响有限样本结果；必须能说明为什么选择这一先验。','可信区间的概率解释以所给先验、数据模型成立为条件，不自动保证频率学覆盖。'],
      challenges:[challenge('先验 Beta(2,2)，观察 7 成功、3 失败。后验是什么？均值为什么不是 0.7？','把次数分别加到参数上。','后验 Beta(9,5)，均值 9/14≈0.642857；它等于 (4/14)×0.5+(10/14)×0.7。先验把样本比例向 0.5 收缩。'),challenge('先看 3 成功、1 失败，再看 4 成功、2 失败；是否等价于一次看到 7 成功、3 失败？','第一次后验成为第二次先验。','等价。Beta(2,2)→Beta(5,3)→Beta(9,5)，与一次更新一致。前提是同一个固定 p、同一条件独立模型，没有把旧数据重复计算。'),challenge('先验 Beta(1,1)，10 次全成功。后验均值与 MLE 各是多少？能断言以后不会失败吗？','后验为 Beta(11,1)，不是集中在 1 的点质量。','MLE=1，后验均值 11/12≈0.9167。后验仍在 0<p<1 上分布，因此不能断言没有失败。数据有限时，未知成功率的不确定性仍然存在。')],
      sources:[source.bayes,source.beta],related:['confidence','testing','sampling']
    },
    regression:{
      title:'共线性与岭回归',en:'Multicollinearity & ridge regression',group:'建模与表示',summary:'用两个相关解释变量比较 OLS 与岭回归，区分系数不稳定、预测拟合和正则化。',question:'两个变量几乎一样时，为什么预测看起来不错，系数却可能变化很大？',
      formula:String.raw`\hat\beta_\lambda=\arg\min_\beta\{\|y_c-X_c\beta\|_2^2+\lambda\|\beta\|_2^2\}=(X_c^TX_c+\lambda I)^{-1}X_c^Ty_c,\quad \mathrm{VIF}=\frac1{1-r^2}`, 
      controls:[r('rho','生成相关参数 ρ',-.98,.98,.02,.8),r('lambda','岭惩罚 λ（平方和约定）',0,10,.1,0),r('n','样本量 n',10,200,10,60),r('noise','噪声标准差',.1,2,.1,.7),seed()],
      presets:[{label:'近共线：OLS',params:{rho:.98,lambda:0,n:60,noise:.7,seed:42}},{label:'同一数据：加入岭惩罚',params:{rho:.98,lambda:2,n:60,noise:.7,seed:42}},{label:'低相关基线',params:{rho:0,lambda:0,n:60,noise:.7,seed:42}},{label:'强负相关',params:{rho:-.98,lambda:2,n:60,noise:.7,seed:42}}],
      steps:['保持种子和样本参数，只改变 λ，以确保比较同一份数据。','把 ρ 从 0 调到 0.98，检查样本相关、VIF 和条件数；它们由当前设计矩阵计算。','在高相关下更换几个种子，观察系数变化，再判断是否有重复样本层面的稳定现象。','观察残差与预测；本页训练拟合不能证明新数据上的泛化改进。'],
      explanation:'当两个标准化变量高度正相关，数据主要识别 β₁+β₂，较难区分各自的贡献。岭惩罚抬高 Gram 矩阵的小特征值，限制弱识别方向上的大系数。代价是引入偏差。这里先中心化并标准化解释变量，再按真实 β=(1,1) 生成 y；截距保留为不惩罚项。',
      assumptions:['y=Xβ+ε，真实 β=(1,1)，误差与设计变量独立、均值为 0，噪声控件给出其标准差。','ρ 是生成机制参数，有限样本的 r 不必等于 ρ；VIF=1/(1−r²) 只针对此处两个解释变量的设计。','目标使用残差平方和而非平均平方误差；若换为平均损失，同等惩罚强度需 λ_mean=λ_sum/n。'],
      pitfalls:['单次岭估计比 OLS 更接近真值不是必然；减少方差是重复抽样的性质。','强相关不必导致预测一定失败；系数可在弱识别方向不稳定而预测近似不变。','岭回归通常不把某个系数精确变为零，也不能把观测相关解释为因果效应。'],
      challenges:[challenge('两列标准化、样本相关 r=0.98。VIF 和 Gram 条件数分别约多少？','Gram 的特征值比例为 (1+r)/(1−r)，绝对值适用于负相关。','VIF=1/(1−0.98²)≈25.2525；κ(XᵀX)=(1+0.98)/(1−0.98)=99。设计矩阵 X 自身的二范数条件数为 √99≈9.9499，不要把两者混为一谈。'),challenge('若 x₁≈x₂，(β₁,β₂)=(1,1) 与 (2,0) 的预测为何可能接近？哪一组受岭惩罚更小？','比较预测之差与系数平方和。','预测之差为 x₁−x₂，接近 0；平方惩罚分别为 2 与 4。正相关时岭倾向避免沿“一个加、另一个减”的方向分配过大的系数。'),challenge('把同一数据集复制一遍，每行出现两次，仍用同一 λ，岭估计是否不变？','残差平方和翻倍，但原惩罚没有翻倍。','一般改变，相当于原数据使用 λ/2。要保持同一解，复制后必须把 λ 同时乘 2。这是平方和约定下样本量与惩罚尺度的关系。')],
      sources:[source.leastSquares,source.ridge],related:['projection','gradient','overfit','pca']
    },
    overfit:{
      title:'模型复杂度与泛化',en:'Polynomial fitting & generalization',group:'建模与表示',summary:'用同一组带噪训练数据拟合不同阶多项式，比较训练、独立验证与已知真实曲线。',question:'训练误差更低，为什么新数据预测可能更差？',
      formula:String.raw`y=\sin(\pi x)+\varepsilon,\quad \hat f(x)=\sum_{j=0}^{d}\beta_jP_j(x),\quad \min_\beta\sum_i\{y_i-\hat f(x_i)\}^2+\lambda\sum_{j=1}^{d}\beta_j^2`,
      controls:[r('degree','多项式阶数 d',1,15,1,5),r('n','训练样本量 n',10,80,5,20),r('noise','噪声标准差',0,1,.05,.25),r('lambda','岭惩罚 λ（平方和约定）',0,1,.001,.001),seed()],
      presets:[{label:'低阶欠拟合',params:{degree:1,n:20,noise:.25,lambda:0,seed:42}},{label:'适中复杂度',params:{degree:5,n:20,noise:.25,lambda:.001,seed:42}},{label:'高阶、少样本',params:{degree:15,n:20,noise:.25,lambda:0,seed:42}},{label:'相同高阶，加正则',params:{degree:15,n:20,noise:.25,lambda:.1,seed:42}}],
      steps:['固定种子、n 和噪声，只修改 degree，比较同一训练与验证数据。','从 d=1 开始预测偏差形状，再观察更高阶模型是否追随噪声。','固定高阶模型，逐渐提高 λ，比较训练误差、验证误差和曲线边界行为。','至少换三个种子，检查你观察到的规律是否只是某次数据的偶然；持续调参使用的验证集不能再当作最终测试集。'],
      explanation:'多项式在固定区间上能表示更弯曲的函数，但用有限带噪数据估计更多系数也可能放大采样波动。训练误差直接参与拟合，因此往往过于乐观；独立验证数据提供更有意义的比较。这里使用 Legendre 基改善数值尺度；基变换保留多项式函数空间，却会改变“系数平方和”惩罚的含义。',
      assumptions:['训练与验证样本从同一 x∼Uniform[−1,1]、y=sin(πx)+ε 机制独立生成。','ε 为独立零均值正态噪声；控件 noise 是标准差。噪声为零时没有不可约噪声，但仍可能拟合不足或数值不稳。','Pⱼ 为 Legendre 基；常数项不惩罚。λ=0 且列满秩时为普通最小二乘；参数多于信息时要报告秩或非唯一性。'],
      pitfalls:['验证误差在一次样本上不必是光滑 U 形，不能为了讲故事修改实际结果。','反复看验证结果选模型会对验证集过拟合；最终评估应使用尚未参与选择的数据。','区间 [−1,1] 内的拟合质量不保证区间外外推可靠。'],
      challenges:[challenge('固定训练数据且 λ=0，d 从 2 增到 3，理论最优训练平方误差能否增加？验证误差呢？','三阶模型包含所有二阶函数，只需新系数设为零。','在精确求解且同一数据下，最优训练误差不会增加，因为旧解仍可行；验证误差没有这个保证。若实现明显违反训练单调性，应检查数值秩或求解误差。'),challenge('n=10、degree=15、λ=0，有多少系数？为什么“训练误差接近零”仍不足以证明学到真函数？','包括常数项共有 d+1 个参数。','有 16 个系数，超过 10 个观测。设计矩阵最多秩 10，通常存在多个插值解；数据点间以及边界附近可能差异很大。唯一性和泛化都不能从训练误差为零推出。'),challenge('若真实 y=f(x)+ε，噪声标准差为 0.25，即使预测器知道真实 f，新样本的期望平方误差是多少？','误差只剩 ε，取 ε² 的期望。','为 Var(ε)=0.25²=0.0625。这是该生成模型下预测带噪 y 的不可约期望误差；有限验证集的实际 MSE 可高于或低于它，不是硬下界。')],
      sources:[source.overfit,source.ridge,source.selection],related:['regression','sampling','gradient','bootstrap']
    },
    pca:{
      title:'PCA、尺度与重构',en:'PCA, scaling & reconstruction',group:'建模与表示',summary:'把二维数据压到一条线，连接最大方差、特征向量和重构误差，并检验单位缩放的影响。',question:'仅把一个变量换单位，第一主成分为什么就可能转向？',
      formula:String.raw`S=\frac{X_c^TX_c}{n-1},\quad Sv_j=\lambda_jv_j,\quad z=X_cv_1,\quad \hat X_c=zv_1^T,\quad \mathrm{EVR}_1=\frac{\lambda_1}{\lambda_1+\lambda_2}`, 
      controls:[r('angle','生成旋转角度（度）',0,175,5,35),r('ratio','生成标准差比',1,10,.25,4),r('scale','观测第二坐标缩放',.25,4,.25,1),r('n','样本量 n',20,300,10,120),s('standardize','分析前标准化',0,[[0,'否：协方差 PCA'],[1,'是：相关矩阵 PCA']]),seed()],
      presets:[{label:'清晰的主方向',params:{angle:35,ratio:4,scale:1,n:120,standardize:0,seed:42}},{label:'改变第二变量单位',params:{angle:35,ratio:4,scale:4,n:120,standardize:0,seed:42}},{label:'标准化消除正单位缩放',params:{angle:35,ratio:4,scale:4,n:120,standardize:1,seed:42}},{label:'总体各向同性',params:{angle:0,ratio:1,scale:1,n:300,standardize:0,seed:42}}],
      steps:['固定种子、scale=1，增加标准差比，预测第一主成分解释率。','只改变 scale，比较协方差 PCA 的方向；再启用标准化，重复这个比较。','选择 ratio=1、scale=1，换几个种子观察方向：总体没有首选方向，但有限样本通常会有轻微差异。','用导出的得分与重构，验证投影残差正交，以及平方重构误差与第二特征值的关系。'],
      explanation:'PCA 在中心化数据中找最大投影方差方向；等价地，它用穿过样本均值的一条线最小化欧氏平方重构误差。协方差 PCA 对变量单位敏感。标准化后求的是相关矩阵的主方向，也就是在另一种度量下优化；此时重构误差的单位也随之改变。',
      assumptions:['先生成独立高斯变量，标准差比为 ratio:1，旋转后再对第二个观测坐标乘 scale。','样本协方差以 n−1 为分母；图中方向与指标基于样本，不等于生成角度或总体特征向量。','标准化开启时，方差、得分和重构误差针对标准化坐标计算；若回到原坐标展示，应恢复每列标准差与均值。'],
      pitfalls:['PCA 不使用 y，不保证保留对某个预测任务最有用的信息。','特征向量的正负号任意，v 与 −v 定义同一主轴；不能把符号翻转误判为结果改变。','两个特征值完全相等时方向不唯一；接近相等时方向可能对微小扰动很敏感。'],
      challenges:[challenge('scale=1、标准差比为 4:1，理论总体第一主成分解释率是多少？有限样本为何不必刚好等于它？','先把标准差平方成方差。','总体特征值为 16 与 1，所以解释率 16/17≈94.12%。样本协方差有抽样波动，因此样本解释率与主方向会偏离总体值。'),challenge('n=120，样本第二特征值 λ₂=0.8。保留一个主成分的总重构平方误差与逐行平均是多少？','协方差用了 n−1 分母。','总平方误差=(120−1)×0.8=95.2；按 n 平均为 95.2/120≈0.7933。若按 n−1 平均才等于 0.8；报告必须写清分母和坐标尺度。'),challenge('为什么启用逐列标准化后，第二列乘以正数 4 不改变标准化后的 PCA？','比较 (4x−4x̄)/(4s) 与原来的 z 分数。','正比例缩放同时乘到均值和标准差上，标准化值仍为 (x−x̄)/s，所以同一数据的相关矩阵和主轴保持相同（除浮点误差及符号选择）。标准化不是总更好，而是改变分析中各变量尺度的重要性。')],
      sources:[source.pca,source.pcaScale],related:['linear','projection','regression']
    },
    bootstrap:{
      title:'Bootstrap 能告诉我们什么',en:'Bootstrap uncertainty & its limits',group:'概率与统计推断',summary:'从同一份观测样本反复有放回抽取，估计均值的不确定性并检验百分位区间的边界。',question:'把重采样次数增加十倍，是否等价于收集十倍的新数据？',
      formula:String.raw`X_1^*,\ldots,X_n^*\overset{iid}{\sim}\hat F_n,\quad \bar X^*=\frac1n\sum_iX_i^*,\quad \widehat{\mathrm{SE}}_* =\mathrm{sd}(\bar X_1^*,\ldots,\bar X_B^*),\quad I_*=[q^*_{.025},q^*_{.975}]`,
      controls:[r('n','原始样本量 n',5,120,5,30),r('reps','重采样次数 B',100,2000,100,500),s('distribution','原始总体分布','exponential',[['normal','正态 N(0,1)'],['exponential','指数 Exp(rate=1)']]),seed()],
      presets:[{label:'小样本与右偏',params:{n:5,reps:1000,distribution:'exponential',seed:42}},{label:'更多真实观测',params:{n:100,reps:1000,distribution:'exponential',seed:42}},{label:'正态基线',params:{n:30,reps:1000,distribution:'normal',seed:42}}],
      steps:['先看原始样本均值与理论总体均值；Bootstrap 围绕当前样本，而非直接围绕总体真值。','只增加 B，确认原始样本与样本均值保持不变；观察直方图和分位数更稳定。','改变 n 或种子相当于更换原始样本，区间位置与宽度都可能改变。','比较 Bootstrap SE 与 s/√n；用下面的精确条件方差解释有限 n 差异。'],
      explanation:'我们只有一份样本，先把其中每个观测视作概率 1/n 的经验分布，再模拟从这个分布抽取 n 次的均值。重采样近似的是“如果经验分布能代表总体，估计量会怎样波动”。它不会生成原样本从未包含的尾部信息，也不能自动修正抽样偏差。这里的百分位区间取 Bootstrap 均值的两个分位数。',
      assumptions:['原始观测 IID；普通逐点 Bootstrap 不适合直接处理时间依赖、聚类或复杂抽样。','均值的常规 Bootstrap 近似需要适当的有限矩和代表性；本页提供正态与指数两种有限方差总体。','s 使用 n−1 分母；Bootstrap SE 是条件于原始样本的模拟标准差，分位数也随有限 B 有 Monte Carlo 误差。'],
      pitfalls:['增加 B 提高重采样计算精度，不增加真实样本的信息量。','单个百分位区间覆盖或漏掉真值，不足以估计该方法长期覆盖率。','小样本、偏斜与极端值可能使简单百分位区间覆盖不佳；“无需正态”不代表“无需条件”。'],
      challenges:[challenge('固定原始 n=30 的样本，把 B 从 100 增到 2000，为什么区间通常不会按 1/√20 缩短？','区分原始抽样不确定性和模拟误差。','Bootstrap 均值的条件分布保持不变；更大的 B 只更准确估计这个分布的分位数和标准差。区间可能轻微变化，但不因重复计算而产生更多观测信息。'),challenge('原样本标准差 s 用 n−1 分母。理想无限次 Bootstrap 的均值 SE 与 s/√n 有何精确关系？','经验分布的方差使用 1/n，而不是 1/(n−1)。','Var*(X*)=(n−1)s²/n；所以 Var*(x̄*)=(n−1)s²/n²，SE*=√((n−1)/n)×s/√n。有限 B 的模拟还会有波动，不能要求两种 SE 完全相等。'),challenge('五个指数样本恰好都较小，Bootstrap 能凭空恢复缺失的大尾部吗？','所有重采样值来自当前五个观测。','不能。每次 Bootstrap 均值都在原样本最小值与最大值之间。若大尾部未被采到，经验分布就遗漏它；增加 B 不会补回。这说明样本代表性是 Bootstrap 的实质前提。')],
      sources:[source.bootstrap],related:['sampling','confidence','overfit']
    }
  };
  root.LabContent = labs;
  if (typeof module === 'object' && module.exports) module.exports = labs;
})(typeof globalThis !== 'undefined' ? globalThis : window);
