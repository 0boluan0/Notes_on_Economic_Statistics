/* Teaching layer for explicit sequential and inference models. */
(function(root){
'use strict';
if(!root.LabContent&&typeof require==='function')root.LabContent=require('./content.js');
const r=(key,label,min,max,step,value)=>({key,label,type:'range',min,max,step,value});
const s=(key,label,value,options)=>({key,label,type:'select',value,options:options.map(([value,label])=>({value,label}))});
const c=(question,hint,answer)=>({question,hint,answer});
const src=(label,url,section,support)=>({label,url,section,support});
const source={
 rnn:src('Dive into Deep Learning','https://d2l.ai/chapter_recurrent-neural-networks/rnn.html','§9.4.2 Recurrent Networks with Hidden States','共享参数、隐藏状态递归；本页为手工指定的标量特例。'),
 bptt:src('Dive into Deep Learning','https://d2l.ai/chapter_recurrent-neural-networks/bptt.html','§9.7.1，式(9.7.7)；§9.7.2','时间路径梯度求和、完整反传与截断的区别。'),
 lstm:src('Dive into Deep Learning','https://d2l.ai/chapter_recurrent-modern/lstm.html','§10.1.1.2–10.1.1.5，式(10.1.1)–(10.1.4)','遗忘/输入/输出门、候选、细胞与隐藏状态；本页固定门值以分开检验路径和完整导数。'),
 contrastive:src('van den Oord, Li & Vinyals · CPC (2018)','https://arxiv.org/pdf/1807.03748','§2.3，式(4)','包含一个正样本及多个负样本的归一化对比损失；本页手工候选不用于声称互信息估计。'),
 jacobian:src('MIT 18.S096 · Matrix Calculus','https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/pages/lecture-notes-and-readings/','Lecture 1 Part 2；Lecture 2 Part 1；Lecture 3 Part 2','导数作为线性映射、Jacobian形状、局部扰动与有限差分；具体映射和二阶上界独立推导。'),
 uncertainty:src('Rasmussen & Williams · Gaussian Processes for Machine Learning','https://gaussianprocess.org/gpml/chapters/RW2.pdf','§2.1，式(2.8)–(2.11)；§2.2 noisy predictions','Gaussian线性模型权重后验、潜在函数预测及加上观测噪声的预测分布。'),
 causal:src('Pearl · Causal inference in statistics: An overview (2009)','https://ftp.cs.ucla.edu/pub/stat_ser/r350.pdf','§2.2；§3.2.1；§3.3.1，式(25)','观察和干预的区别、替换结构方程、满足后门条件时的协变量调整。具体二元SCM及所有概率为本实验设定。')
};
const labs={
 rnn:{animation:{key:'steps'},title:'RNN怎样在时间中保留状态与梯度',en:'Scalar recurrence & full backpropagation through time',group:'序列、表示与推断',summary:'实际展开一个共享参数的标量递归，查看时间路径乘积以及共享权重的梯度贡献之和。',question:'同一个权重在16步中重复出现，它的梯度为什么不能只看最后一步？',
 formula:String.raw`h_t=\phi(ah_{t-1}+u x_t),\quad L=\tfrac12(h_T-y)^2,\quad \frac{\partial L}{\partial a}=\sum_{t=1}^T\frac{\partial L}{\partial h_t}\phi'(z_t)h_{t-1}`,
 controls:[r('weight','递归权重 a',-1.5,1.5,.1,.8),r('inputWeight','输入权重 u',0,2,.1,.8),r('steps','序列长度 T',2,30,1,16),s('kind','隐藏激活','tanh',[['tanh','tanh'],['linear','线性']]),s('pattern','输入序列','pulse',[['pulse','首步脉冲'],['sine','正弦'],['alternating','交替正负'],['zero','全零']]),r('initial','初态 h₀',-1,1,.1,0),r('target','末步目标 y',-1,1,.1,.5)],
 presets:[{label:'脉冲逐渐被遗忘',params:{weight:.8,inputWeight:.8,steps:16,kind:'tanh',pattern:'pulse',initial:0,target:.5}},{label:'线性a=1保持初态',params:{weight:1,inputWeight:0,steps:20,kind:'linear',pattern:'zero',initial:.5,target:0}},{label:'线性乘积放大',params:{weight:1.2,inputWeight:0,steps:20,kind:'linear',pattern:'zero',initial:.1,target:0}},{label:'负权重带来符号交替',params:{weight:-.8,inputWeight:.8,steps:16,kind:'linear',pattern:'pulse',initial:0,target:.5}}],
 steps:['先逐步播放脉冲后的隐藏状态，区分有输入的首步与后续递归。','从第二图的最右端向左读，预测早期状态对末态还有多大影响。','切换线性激活，以a的幂核对梯度。','观察第三图各步贡献的符号，确认指标等于这些贡献的和。'],
 explanation:'同一参数a用于每个时间步，计算图展开后相当于多处使用同一个变量。一次使用对损失的局部影响，要乘上从该时刻到末端的全部状态导数；共享参数的总梯度再把这些路径贡献相加。tanh还会引入不超过1的局部导数，线性版本则能直接用a的幂检验。',
 assumptions:['单个标量状态、无偏置；权重固定，没有训练。','只有末时刻平方损失，完整反传全部T步；未采用截断BPTT或梯度裁剪。','敏感度对数图显示绝对值；符号必须结合数据和第三图读取。'],
 pitfalls:['状态看似平稳不保证反向梯度大小合适。','对共享参数求导时漏掉早期使用路径会得到错误梯度。','调参数演示长期依赖不等于证明RNN在真实序列任务上有效。'],
 challenges:[c('线性、u=0、a=0.5、T=4时，末态对初态的导数是多少？','每一步局部导数都为a。','∂h₄/∂h₀=0.5⁴=1/16=0.0625，与初态和目标无关。'),c('线性、u=0、a=1、h₀=0.5、T=4、y=0时，∂L/∂a是多少？','先写h_T=a^T h₀。','∂L/∂a=(a^T h₀−y)T a^(T−1)h₀=0.5×4×0.5=1。每步贡献0.25，不能只保留最后的0.25。'),c('线性a=−0.8时，从h_{T−3}到h_T的敏感度符号是什么？','数三次负号相乘。','(−0.8)³=−0.512。绝对值对数图不会显示负号，因此不能仅凭它判断扰动方向。')],sources:[source.rnn,source.bptt],related:['lstm','backprop','residual','initialization']},
 lstm:{animation:{key:'steps'},title:'LSTM记忆路径不等于完整梯度',en:'Gated cell memory & total derivatives',group:'序列、表示与推断',summary:'固定三个门值，实际递归细胞和隐藏状态；再打开候选反馈，比较直通路径与完整导数。',question:'遗忘门接近1，是否就能断言完整梯度等于1？',
 formula:String.raw`g_t=\tanh(x_t+r h_{t-1}),\quad c_t=f c_{t-1}+i g_t,\quad h_t=o\tanh(c_t),\quad \left.\frac{\partial c_T}{\partial c_0}\right|_{\text{direct path}}=f^T`,
 controls:[r('forget','遗忘门 f',0,1,.05,.95),r('input','输入门 i',0,1,.05,.4),r('output','输出门 o',0,1,.05,.8),r('recurrent','候选反馈权重 r',-2,2,.1,0),r('steps','时间步 T',2,40,1,20),s('pattern','输入序列','zero',[['zero','全零'],['pulse','首步脉冲'],['sine','正弦']]),r('initial','初始记忆 c₀',-2,2,.1,1)],
 presets:[{label:'只沿记忆路径遗忘',params:{forget:.95,input:.4,output:.8,recurrent:0,steps:20,pattern:'zero',initial:1}},{label:'记忆保持，输出仍受压缩',params:{forget:1,input:0,output:1,recurrent:0,steps:20,pattern:'zero',initial:1}},{label:'候选反馈增加额外路径',params:{forget:.9,input:.8,output:.9,recurrent:1.2,steps:20,pattern:'zero',initial:.1}},{label:'输出关闭但细胞仍可保存',params:{forget:1,input:0,output:0,recurrent:0,steps:20,pattern:'zero',initial:1}}],
 steps:['默认全零输入且r=0，核对c_t=f^t c₀。','将f设1且i设0，同时比较细胞状态、隐藏输出和两种导数。','打开r，使过去的h能经候选值回流到c，观察两条细胞梯度分离。','将o设0，说明“没有对外输出”与“内部没有记忆”不同。'],
 explanation:'遗忘门给细胞自连接提供乘f的路径。完整导数需要枚举所有依赖：只要候选值读取过去隐藏态，就多出c→h→候选→c路径。实际LSTM的门通常也依赖隐藏态，会再增加路径。本页把门固定，保留一条可控制的候选反馈，因此能准确指出f^T何时等于总导数、何时仅是一部分。',
 assumptions:['标量、三门固定、无训练；门0与1为Sigmoid的理想极限值。','h₀定义为o tanh(c₀)，对c₀求导时包含这一步。','图2使用保符号log₁₀(1+绝对值)，不是原始梯度刻度。'],
 pitfalls:['把直接记忆路径的f乘积直接写成实际LSTM总导数会漏路径。','f=1不消除输出端tanh的饱和。','门值固定的演示不能说明模型学会了何时记住和遗忘。'],
 challenges:[c('r=0、全零输入、f=0.5、c₀=1、T=4时，c_T和dc_T/dc₀各是多少？','候选值全为0。','两者均为0.5⁴=0.0625。这个特例中直通路径就是完整细胞导数。'),c('f=1、i=0、o=1、c₀=1时，细胞与隐藏输出的导数各是多少？','c_T=c₀，但h_T=tanh(c₀)。','dc_T/dc₀=1；dh_T/dc₀=1−tanh²(1)≈0.419974，保存细胞梯度不等于输出梯度为1。'),c('初态0、全零输入、f=0.5、i=o=1、r=0.5时，第一步完整细胞导数是多少？','在0处tanh′=1且dh₀/dc₀=1。','dc₁/dc₀=f+i r dh₀/dc₀=0.5+0.5=1，直通路径却只有0.5。差值来自候选反馈。')],sources:[source.lstm],related:['rnn','residual','backprop']},
 contrastive:{title:'InfoNCE如何让正负样本竞争',en:'Finite-vector InfoNCE & temperature',group:'序列、表示与推断',summary:'直接计算单位向量的点积、Softmax分母和单Query损失，比较温度与困难负样本的作用。',question:'正样本已很相似，加入一个同样相似的负样本会怎样？',
 formula:String.raw`s_j=q^Tk_j/\tau,\quad P_j=\frac{e^{s_j}}{\sum_\ell e^{s_\ell}},\quad L=-\log P_+,\quad \nabla_qL=\frac{\sum_jP_jk_j-k_+}{\tau}`,
 controls:[r('angle','Query角度（度）',-180,180,5,30),r('negativeAngle','第1负样本角度（度）',-180,180,5,60),r('temperature','温度 τ',.05,2,.05,.5),s('negatives','负样本数',3,[[1,'1个'],[2,'2个'],[3,'3个']])],
 presets:[{label:'两个近邻候选竞争',params:{angle:30,negativeAngle:60,temperature:.5,negatives:3}},{label:'正负向量完全重复',params:{angle:0,negativeAngle:0,temperature:.2,negatives:1}},{label:'降低温度放大错误排序',params:{angle:60,negativeAngle:60,temperature:.1,negatives:3}},{label:'相反负样本',params:{angle:0,negativeAngle:180,temperature:.5,negatives:1}}],
 steps:['第一图比较Query与各向量的夹角，预测哪个点积最大。','查看所有候选共用的分母，手算只有一个负样本的概率。','让第1负样本与正样本重合，再降低温度，观察不可区分性是否消失。','旋转Query阅读整条损失曲线；区分自由Query梯度与每度旋转导数。'],
 explanation:'对比损失把“在本候选集合里识别正样本”写成对数概率。温度缩放点积差异，既可强化正确排序，也会加重错误排序。负样本改变归一化分母，即使正样本点积不变，损失也会变化。本页的向量均固定在单位圆，只检查这项损失的实际几何和梯度。',
 assumptions:['二维单位向量，一个正样本、最多三个手工负样本；正样本固定0°，其余负样本固定120°和−120°。','没有训练、批次采样、增强、stop-gradient或表征坍塌的性能实验。','不能把固定向量单次损失当作对数据分布互信息的有效估计。'],
 pitfalls:['降低温度不保证减小损失；需看正样本是否排名第一。','向量重合却被给成相反角色是不可区分的标签约束，温度无法解决。','自由二维梯度和单位圆上的角度导数有不同单位。'],
 challenges:[c('只有一正一负且两向量完全相同，损失是多少？','两项分数总是相等。','各概率为1/2，L=log2≈0.693147，与Query和温度无关；两候选的自由Query梯度也为0。'),c('q=k+=(1,0)，唯一负样本(−1,0)，τ=1时，P+是多少？','分数分别为1和−1。','P+=e¹/(e¹+e⁻¹)=1/(1+e⁻²)≈0.880797；L≈0.126928。'),c('固定已有分数，增加一个有限分数的负样本，单Query损失会下降吗？','分子不变，分母增加一个正数。','不会；P+严格下降、L严格上升。不同负样本数量的原始损失不能直接作为公平训练质量排名。')],sources:[source.contrastive],related:['embeddings','attention','activations']},
 jacobian:{title:'非线性变形怎样被局部Jacobian近似',en:'Local Jacobian maps & Taylor remainder',group:'序列、表示与推断',summary:'把同一小网格分别送进真实非线性函数和局部仿射映射，再检验随着观察范围缩小的误差。',question:'Jacobian作用在坐标本身，还是作用在相对展开点的扰动上？',
 formula:String.raw`F(x,y)=\begin{pmatrix}x+\alpha\sin y\\ y+\beta x^2\end{pmatrix},\quad J(a)=\begin{pmatrix}1&\alpha\cos a_y\\2\beta a_x&1\end{pmatrix},\quad F(a+\Delta)\approx F(a)+J(a)\Delta`,
 controls:[r('alpha','正弦弯曲 α',-1.5,1.5,.1,.8),r('beta','二次弯曲 β',-1,1,.1,.4),r('x','展开中心 aₓ',-1,1,.1,.5),r('y','展开中心 aᵧ',-1,1,.1,.5),r('radius','网格半宽 ρ',.05,1.5,.05,.5)],
 presets:[{label:'局部网格的弯曲',params:{alpha:.8,beta:.4,x:.5,y:.5,radius:.5}},{label:'缩小观察范围',params:{alpha:.8,beta:.4,x:.5,y:.5,radius:.1}},{label:'中心处一阶压扁',params:{alpha:1,beta:.5,x:1,y:0,radius:.5}},{label:'完全线性的一致性',params:{alpha:0,beta:0,x:.5,y:.5,radius:1}}],
 steps:['先固定中心，把输入网格与输出网格的同一条线对应起来。','比较真实实线和仿射虚线在中心附近与边缘的误差。','缩小ρ，观察它们靠拢，并查看单位方向误差的二阶上界。','使用det J=0预设，辨别“导数压扁”与“整个非线性图像压扁”。'],
 explanation:'Jacobian把输入的微小扰动映成输出的一阶扰动。展开点本身需要先通过F映射，因此一般近似是F(a)+J(a)(x−a)。真实映射含曲率，网格越大越能看到实线偏离虚线。本例两个分量都有显式二阶导数，因此还能给出沿单位方向的保守二阶误差上界，直接核查图像的局部结论。',
 assumptions:['固定光滑二维函数，无神经网络拟合或数值微分替代真实映射。','网格ρ是各坐标半宽；误差图h是沿单位对角方向的欧氏位移长度，二者含义不同。','对数图把小于10⁻¹⁶的数显示在下限，原始0保留在导出数据。'],
 pitfalls:['J(a)x通常漏掉平移项，只有特殊中心/函数才成立。','det J=0只说明一阶导数奇异，不能单独判断非线性函数在邻域中单射与否。','直线近似误差小是局部结论，不能直接外推大范围。'],
 challenges:[c('α=1、β=0.5、a=(1,0)时J和行列式是什么？','分别计算αcos0与2βaₓ。','J=[[1,1],[1,1]]，det J=0；扰动(1,−1)的一阶输出为零，但仍可能有高阶变化。'),c('α=β=0时，任意中心的局部近似误差是多少？','此时F是恒等映射。','F(a)+I(x−a)=x=F(x)，误差严格0；对数图在显示下限的线不表示实际存在10⁻¹⁶误差。'),c('沿单位方向，误差上界为Ch²。把h减半，上界如何变化？','平方缩放。','上界变成原来的1/4。这是上界的关系，真实误差可能更小或具有更高阶抵消。')],sources:[source.jacobian],related:['linear','tangent','backprop','flow']},
 uncertainty:{title:'参数不确定性与新观测噪声怎样分开',en:'Bayesian linear posterior & predictive uncertainty',group:'序列、表示与推断',summary:'解析计算二维线性参数后验，同时显示潜在均值可信区间、新观测预测区间与噪声方差。',question:'数据越来越多以后，为什么新观测仍然不能被精确预测？',
 formula:String.raw`S=(\tau_0^{-2}I+\sigma^{-2}X^TX)^{-1},\quad m=\sigma^{-2}SX^Ty,\quad \mathrm{Var}(f_*\mid D)=\phi_*^TS\phi_*,\quad \mathrm{Var}(y_*\mid D)=\phi_*^TS\phi_*+\sigma^2`,
 controls:[r('n','观察数 n（同一池前n个）',2,30,1,12),r('sigma','已知观测噪声 σ',.1,1.5,.1,.4),r('priorSD','先验参数标准差 τ₀',.25,4,.25,2),r('span','设计输入半宽',.2,2,.1,1),r('query','查询输入 x*',-3,3,.1,2.5),r('seed','随机种子',1,99999,1,42)],
 presets:[{label:'区分两种95%区间',params:{n:12,sigma:.4,priorSD:2,span:1,query:2.5,seed:42}},{label:'少量集中设计，外推较不确定',params:{n:4,sigma:.4,priorSD:2,span:.3,query:2.5,seed:42}},{label:'增加同一池中的观察',params:{n:30,sigma:.4,priorSD:2,span:1,query:2.5,seed:42}},{label:'噪声较大的模型',params:{n:30,sigma:1.2,priorSD:2,span:1,query:2.5,seed:42}}],
 steps:['在同一x位置比较均值区间与新观测区间，确认它们中心相同。','阅读方差分解图，直接核对两条方差曲线之差等于σ²。','固定种子、σ和输入范围，增加n观察参数后验如何收缩。','把查询点移出设计范围，再看斜率不确定性如何放大预测均值的不确定性。'],
 explanation:'给定Gaussian先验和已知Gaussian噪声，参数后验仍是Gaussian。未知参数经过线性读出带来φᵀSφ的函数方差；新观测还包含一次独立噪声，因而方差额外增加σ²。更多信息可以收紧参数后验，却无法消除模型指定的新观测噪声。图中概率陈述都条件于这个模型与先验。',
 assumptions:['真实合成均值为0.4+1.2x，参数先验为N(0,τ₀²I)，噪声σ已知。','固定种子时n增加使用同一个30点样本池；改σ会同时改变生成数据与似然假设。','95%函数区间和预测区间都是逐点区间；参数图使用二维联合95%椭圆。'],
 pitfalls:['把均值的可信区间当作未来单次观测的预测区间会过窄。','后验概率依赖模型和先验，不是频率覆盖率的无条件保证。','参数与噪声的拆分只在明确模型下成立，不能涵盖现实中的结构错设风险。'],
 challenges:[c('某处参数方差0.09，σ=0.4，新观测预测方差与标准差是多少？','独立方差相加，再开平方。','预测方差0.09+0.16=0.25，标准差0.5；不能把0.3和0.4直接相加为标准差。'),c('参数后验协方差S趋近0时，新观测95%预测区间半宽趋向什么？','剩下已知噪声。','趋向z₀.₉₇₅σ≈1.96σ，仍非零；潜在均值区间半宽才趋近0。'),c('为什么二维联合95%参数椭圆不能直接用1.96作为半径？','标准化后两维平方和服从χ²₂。','需要半径平方χ²₂(0.95)=−2log(0.05)≈5.991465，半径约2.447747；1.96是单个标准正态双侧95%边界。')],sources:[source.uncertainty],related:['bayes','regression','confidence','bootstrap']},
 causal:{title:'混杂怎样让观察关联与干预效应反向',en:'Known structural model, Simpson reversal & intervention',group:'序列、表示与推断',summary:'精确枚举一个明确已知的二元结构模型，比较组内响应、观察混合比例、标准化和do干预。',question:'每个Z组中T都提高结果均值，总体观察均值差为什么仍可为负？',
 formula:String.raw`Z\sim\mathrm{Bernoulli}(q),\ e_0=.5+.4s,\ e_1=.5-.4s,\quad P(Y=1\mid T=t,Z=z)=.2+.5z+\tau t,\quad E[Y\mid do(t)]=\sum_z E[Y\mid t,z]P(z)`,
 controls:[r('selection','选择强度 s',-1,1,.1,.8),r('effect','组内加性效应 τ',-.15,.15,.01,.1),r('prevalence','总体P(Z=1) = q',.1,.9,.1,.5)],
 presets:[{label:'默认Simpson方向翻转',params:{selection:.8,effect:.1,prevalence:.5}},{label:'处理独立于Z',params:{selection:0,effect:.1,prevalence:.5}},{label:'零因果效应仍有关联',params:{selection:.8,effect:0,prevalence:.5}},{label:'反转选择机制',params:{selection:-.8,effect:.1,prevalence:.5}}],
 steps:['先分别比较两个Z组内T从0到1的变化。','再读总体观察曲线，并用第二图解释两个T组的组成差异。','将选择强度设0，使T独立于Z，检查观察差与干预差重合。','保持τ=0而改变选择强度，区分“存在关联”与“存在干预效应”。'],
 explanation:'在这个已知模型里，Z同时影响处理选择和结果基线。总体观察均值使用P(Z|T)加权，不同T组的权重不同；干预替换T的生成方程后仍面对同一个总体P(Z)，所以使用共同权重。默认情境中低结果基线组更常接受T，组成差异足以压过正的组内效应。调整等于干预依赖这里明确给定的因果结构。',
 assumptions:['三个相互独立的Uniform外生变量生成Z、T、Y；T由e_Z阈值产生，Y由0.2+0.5Z+τT阈值产生。','e_Z在0.1至0.9，结果概率在0.05至0.85；所有条件格都有正概率。','模型已知且不存在给定Z后剩余的共同原因；展示精确总体量，不从观察数据发现因果图。'],
 pitfalls:['观察表不能独自证明无遗漏混杂或do干预所需的机制稳定性。','调整所有可见变量不是普遍正确规则，本实验只调整已指定的共同原因Z。','Simpson方向翻转本身不是因果识别算法；没有翻转也不证明无混杂。'],
 challenges:[c('默认q=0.5、s=0.8、τ=0.1时，观察差与do差各是多少？','e₀=0.82、e₁=0.18，因此P(Z=1|T=1)=0.18、P(Z=1|T=0)=0.82。','观察均值为0.39和0.61，差−0.22；do均值为0.55和0.45，差+0.10。'),c('把s设0，在任意允许q下观察差为什么等于τ？','两组处理概率都为0.5。','T独立于Z，所以P(Z|T)=P(Z)。共同权重下基线抵消，观察差与do差都等于τ。'),c('τ=0、q=0.5、s=0.8时，观察差是多少？它表示干预有效吗？','保留组成差异，去掉组内效应。','观察差0.29−0.61=−0.32；do差仍为0。该关联完全来自本模型的选择机制。')],sources:[source.causal],related:['testing','regression','fairness','rl-policy']}
};
Object.assign(root.LabContent,labs);if(typeof module!=='undefined'&&module.exports)module.exports=labs;
})(typeof globalThis!=='undefined'?globalThis:this);
