(function(root){
'use strict';
root.LabGuides ||= {};
Object.assign(root.LabGuides,{
knn:{steps:[
 {title:'先问身边最近的一个点',caption:'查询点Q没有标签。k=1时，只用最近观测的类别作判断；k更大时，要看多个邻居的投票。',action:'移动 Q，看连线何时换到另一个邻居。',focus:['qx','qy'],plot:0,metrics:[1],params:{k:1,scale:1}},
 {title:'让更多邻居一起投票',caption:'邻居数变大，判断由一个点变成一群点的投票。局部细节可能被平均掉。',action:'保持 Q 不动，只改变 k，比较邻居连线和正类比例。',focus:['k'],plot:0,metrics:[0,3],params:{k:7}},
 {title:'距离的单位也会改变邻居',caption:'放大第二维在距离中的权重，会改变谁算“近”，即使样本和标签都没变。',action:'只调第二维距离尺度，看邻居和预测是否改变。',focus:['scale'],plot:0,metrics:[1,2],params:{scale:1}}
]},
logistic:{steps:[
 {title:'先看模型报了多大概率',caption:'曲线给每个 x 一个正类概率。这里的权重由你手动指定，还没有执行训练。',action:'只改变权重，观察概率曲线变陡、变平或反向。',focus:['weight'],plot:0,metrics:[0],params:{weight:1.8,intercept:-.3,threshold:.5}},
 {title:'概率相同，行动规则可以不同',caption:'阈值决定哪些概率被判成正类；只改变阈值，不会改变模型给出的概率曲线。',action:'把阈值从0.5降到0.2，比较分类边界和准确率。',focus:['threshold'],plot:0,metrics:[1,3],params:{threshold:.5}},
 {title:'用损失评价这条概率曲线',caption:'固定截距后，每个权重都有真实交叉熵。自信地报错会受到更大惩罚。',action:'沿权重方向移动，看当前位置离损失低点有多远。',focus:['weight'],plot:1,metrics:[0],params:{intercept:0,weight:-3}}
]},
svm:{steps:[
 {title:'先看一条分类线',caption:'法向非零时，中间线决定分类，两侧线标出间隔；法向为零时，不存在这组几何间隔线。',action:'先平移分界线，再旋转方向，观察被错分的点。',focus:['offset','angle'],plot:0,metrics:[3],params:{norm:1,angle:0,offset:0,C:1}},
 {title:'分对了，也可能仍有惩罚',caption:'函数间隔在0到1之间的点分类正确，却还没有达到要求的间隔，因此 hinge 损失仍为正。',action:'改变法向长度，找出“分类正确但有损失”的样本。',focus:['norm'],plot:1,metrics:[1,2]},
 {title:'C 改变惩罚的价格',caption:'本页保持手动参数。改变 C 只改当前目标值，不会自动寻找新的最优分界线。',action:'只调 C，核对目标值变化而样本间隔保持不变。',focus:['C'],plot:1,metrics:[0],params:{C:.1}}
]},
tree:{steps:[
 {title:'一次只能沿一个坐标切一刀',caption:'深度上限为1时最多切一刀。各区域用内部正类比例预测，边界沿坐标轴；深度0时不分裂。',action:'从深度0移到1，观察第一条分割线如何产生。',focus:['depth'],plot:0,metrics:[0,1],params:{dataset:'axis',depth:0,minleaf:4,noise:0}},
 {title:'再切几刀，区域更细',caption:'后续划分在已有区域内部进行。小叶节点限制会阻止样本太少的区域继续分裂。',action:'先看深度3，再提高最小叶节点样本数，比较留下的区域。',focus:['minleaf'],plot:0,metrics:[0,2],params:{depth:3}},
 {title:'没有眼前收益，贪心树就停住',caption:'严格对称 XOR 的每个单次划分都不能降低不纯度。即使允许很深，根节点仍可能不分裂。',action:'比较对称 XOR 与轴向规则的根节点收益曲线。',focus:['dataset'],plot:1,metrics:[0,3],params:{dataset:'balancedxor',depth:5,minleaf:4,n:100,noise:0}}
]},
ensemble:{steps:[
 {title:'一个预测器也会左右波动',caption:'目标固定为0。一个预测器的重复结果会波动；分布宽度表示方差，共同偏差会移动它的中心。',action:'改变误差尺度，看预测分布怎样变宽。',focus:['sigma'],plot:1,metrics:[0],params:{members:1,rho:0,bias:0,sigma:1}},
 {title:'独立的波动可以互相抵消',caption:'误差独立时，平均能减少方差；若误差相关，只靠增加成员数未必能让方差接近零。',action:'只增加成员数，对照实际方差与理论曲线。',focus:['members'],plot:0,metrics:[0,1],params:{members:20,rho:0}},
 {title:'大家共同犯的错，平均不掉',caption:'相关性升高意味着共享误差更多。成员再多，方差也会受到共享部分的限制。',action:'保持成员数100，把相关性从0调到1，观察平台。',focus:['rho'],plot:0,metrics:[0,3],params:{members:100,rho:.7}}
]},
boosting:{steps:[
 {title:'先只报一个平均数',caption:'还没有加入树桩时，所有输入得到同一个预测。曲线与样本之间留下需要修正的误差。',action:'先看0轮预测，找出明显偏高和偏低的位置。',focus:['steps'],plot:0,metrics:[3,1],params:{steps:0,rate:.2}},
 {title:'下一棵树只补当前的差',caption:'每棵树桩拟合当前残差，再乘学习率加入原预测。此图是最后一次增量；0轮时还没有树桩。',action:'改变学习率，查看最后这次修正的方向和幅度。',focus:['rate'],plot:2,metrics:[0,1],params:{steps:1,rate:.2}},
 {title:'许多小修正累积成曲线',caption:'每一轮都根据更新后的残差再拟合。训练误差下降不代表独立验证误差也一定下降。',action:'逐步增加轮数，比较最终拟合与两种误差。',focus:['steps'],plot:0,metrics:[1,2],params:{steps:12,rate:.2}}
]},
kmeans:{steps:[
 {title:'先放中心，再认最近的中心',caption:'0轮时菱形是初始中心，之后是更新后的中心。每个观测归给最近中心，初始中心未必在簇中间。',action:'比较分散和集中的初始化，先不执行更新。',focus:['initialization'],plot:0,metrics:[0],params:{k:3,steps:0,initialization:'spread'}},
 {title:'把中心移到成员的平均位置',caption:'每轮先按旧分配求均值，再把样本分给更新后最近的中心。重复这两个动作。',action:'从0轮走到1轮、2轮，跟踪菱形中心的移动。',focus:['steps'],plot:0,metrics:[0,2],params:{steps:1}},
 {title:'下降不等于找到了唯一答案',caption:'不同初始化会走不同的路径。目标值不增加，只说明这些更新改善了当前目标。',action:'固定数据和簇数，切换初始化，比较完整下降轨迹。',focus:['initialization'],plot:1,metrics:[0,3],params:{steps:10}}
]},
gmm:{steps:[
 {title:'一个点可以部分属于两个分量',caption:'责任度从0到1，表示当前模型给两个分量分配的权重，并用于下一轮加权更新。',action:'先改变初始分离程度，看中间观测的软分配。',focus:['initial'],plot:1,metrics:[0,1],params:{steps:0,initial:1}},
 {title:'用软分配重新估计分量',caption:'责任度成为均值更新的权重。更新后又重新计算责任度；本例共同方差保持固定。',action:'增加 EM 轮数，看两个均值与混合密度怎样变化。',focus:['steps'],plot:0,metrics:[0,1,2],params:{steps:6,initial:1}},
 {title:'完全重合的起点不会自动分开',caption:'两个分量从同一均值出发，会获得相同责任度并保持对称。迭代本身不会打破这种对称。',action:'先看初始分离为0，再给它一点分离，比较10轮后的结果。',focus:['initial'],plot:0,metrics:[0,1,3],params:{initial:0,steps:10}}
]},
kde:{steps:[
 {title:'每个观测放一个小山包',caption:'这里先画前5个加权核及其部分和。每个核面积是1/n，这还不是完整的密度估计。',action:'改变带宽，看每个小山包如何变宽、变矮。',focus:['bandwidth'],plot:1,metrics:[3],params:{n:60,distribution:'mixture',bandwidth:.35}},
 {title:'把所有观测的小山包加起来',caption:'完整 KDE 使用全部观测。窄核会保留很多样本细节，也可能把随机波动画成尖峰。',action:'从很窄的带宽慢慢加宽，比较估计曲线与真实密度。',focus:['bandwidth'],plot:0,metrics:[0],params:{bandwidth:.05}},
 {title:'更平滑，也可能丢掉真实结构',caption:'混合分布情境的真实密度有两个峰。带宽过大时，两峰可能被抹平；正态情境本来只有一个峰。',action:'在0.35和1.5附近比较，指出平滑过程中消失了什么。',focus:['bandwidth'],plot:0,metrics:[0,2],params:{bandwidth:1.5}}
]},
modelselection:{steps:[
 {title:'先从两个候选中挑低分',caption:'这里所有候选的真实风险完全相同。验证误差仍会因有限样本而不同，我们挑最小的一个。',action:'找到被选中的点，比较它与共同真实风险的距离。',focus:['candidates'],plot:0,metrics:[2,0],params:{candidates:2,nval:30,scale:1}},
 {title:'多挑几次，更容易碰到幸运低分',caption:'候选增加没有提升真实能力，却给了搜索更多偶然偏低的验证分数。',action:'只增加候选数，比较最低验证分数与真实风险。',focus:['candidates'],plot:0,metrics:[0,3],params:{candidates:80}},
 {title:'到新数据上重新检查',caption:'每个点是一轮重新选择后的验证与全新测试误差。选出来的低验证分数可能过于乐观。',action:'增加验证样本数，比较这种乐观偏差是否减弱。',focus:['nval'],plot:1,metrics:[0,1,2],params:{nval:30}}
]},
metrics:{steps:[
 {title:'先看谁被判成正类',caption:'列是真实类别，行是预测类别。阈值改变四格人数，从而改变查准率和查全率。',action:'提高阈值，找到减少的预测正类去了哪一格。',focus:['threshold'],plot:0,metrics:[0,1],params:{prevalence:.5,threshold:.5}},
 {title:'移动阈值，只在同一条 ROC 上走',caption:'ROC 已经遍历所有阈值。当前点会移动，整条曲线和 AUC 不会因为选阈值而改变。',action:'来回拖动阈值，观察当前点与 AUC。',focus:['threshold'],plot:1,metrics:[1,2]},
 {title:'正类很少时，误报会更显眼',caption:'在稀有正类场景，误报容易占据许多预测正类，precision 因而可能很低。',action:'保持分离程度不变，比较不同正类比例下的 PR 曲线。',focus:['prevalence'],plot:2,metrics:[0,3],params:{prevalence:.05,threshold:.5}}
]},
likelihood:{steps:[
 {title:'数据固定，比较不同参数解释',caption:'固定这份成功与失败记录，移动候选p，比较各参数对同一份数据给出的相对似然。',action:'把候选 p 从0.5移到0.7，看哪里达到峰值。',focus:['candidateP'],plot:0,metrics:[0,1],params:{n:40,observedRate:.7,candidateP:.5}},
 {title:'更多同样比例的数据，让曲线更尖',caption:'增加同样比例的观测通常使曲线更尖；成功次数先取整，峰值可能略移动。峰高统一缩放为1。',action:'比较 n=40与100的曲线宽度，不把峰高当参数概率。',focus:['n'],plot:0,metrics:[0],params:{n:100,candidateP:.7}},
 {title:'最优值也可能在边界',caption:'全部成功时，最大似然估计是 p=1。若候选 p=0，这份观测序列的概率就是0。',action:'把候选从1移向0，观察相对似然与对数似然的边界。',focus:['candidateP'],plot:0,metrics:[0,1],params:{n:40,observedRate:1,candidateP:1}}
]},
svd:{steps:[
 {title:'先找输入里的两个特殊方向',caption:'单位圆上的两个箭头是右奇异方向。它们告诉我们：沿哪两个正交方向分别观察变换。',action:'旋转右方向，保持两个方向互相垂直。',focus:['right'],plot:0,metrics:[0],params:{left:30,right:20,scale:2,ratio:.35,retained:2,shared:0}},
 {title:'两个方向可以被拉伸不同倍数',caption:'单位圆的像一般是椭圆，也可能退化成线或点。两条半轴长度对应两个奇异值。',action:'只改变奇异值比例，比较圆、椭圆和塌成线段的边界。',focus:['ratio'],plot:1,metrics:[0,1]},
 {title:'只保留一个方向，会丢掉什么',caption:'保留一项时，重构的像至多是一条线段。误差来自被丢弃方向；保留两项则恢复原映射。',action:'在保留1项和2项之间切换，比较形状与误差。',focus:['retained'],plot:1,metrics:[2,3],params:{ratio:.35,retained:1}}
]},
biasvariance:{steps:[
 {title:'同样的训练位置，重采噪声再拟合',caption:'灰线来自重复训练。噪声非零时，同样训练位置也可能得到不同预测；噪声为零时这些拟合重合。',action:'改变噪声强度，比较灰线围绕平均预测的散开程度。',focus:['noise'],plot:0,metrics:[1],params:{degree:3,n:20,noise:.4,lambda:.01,x0:.5}},
 {title:'在一个固定位置，分清偏离与分散',caption:'平均预测离真值多远是偏差；不同拟合围绕平均预测有多散是方差。两件事不能混为一谈。',action:'只改变评估位置 x₀，比较两条标记与预测分布。',focus:['x0'],plot:2,metrics:[0,1,2]},
 {title:'预测新响应，还要面对新的噪声',caption:'同一设置下，预测独立新响应的期望误差，比估计真函数多出 σ²；图中明确画出这项。',action:'比较两种 MSE 的差，再与当前噪声方差核对。',focus:['noise'],plot:1,metrics:[2,3]}
]},
kernel:{steps:[
 {title:'先看核怎样衡量附近的相似性',caption:'使用RBF时，附近位置的相似值较大。锚点只用于观察核截面，不是新增训练样本。',action:'先移动锚点，再改变带宽，区分位置与相似范围。',focus:['anchor','bandwidth'],plot:1,metrics:[],params:{kernel:'rbf',anchor:0,bandwidth:.4,lambda:.05}},
 {title:'把许多核组合成实际预测',caption:'训练实际求出核的组合系数。带宽只改变RBF核及其拟合；线性核和二次核不受该控件影响。',action:'固定样本，只改变 RBF 带宽，比较拟合与真实曲线。',focus:['bandwidth'],plot:0,metrics:[0,1]},
 {title:'贴近训练点，也要付复杂度代价',caption:'增大正则惩罚会压低函数范数，却不保证对真函数的误差一定变小。',action:'只增加 λ，比较训练误差、函数范数与曲线。',focus:['lambda'],plot:0,metrics:[0,2],params:{bandwidth:.4,lambda:3}}
]},
entropy:{steps:[
 {title:'真实会发生什么，你又报了什么',caption:'P 是真实类别分布，Q 是报告概率。先让两者相同，再只改变报告，保持真实分布不动。',action:'改变 Q 的两个分配参数，比较两组柱子的差别。',focus:['q1','q2'],plot:0,metrics:[0,2],params:{p1:.5,p2:.6,q1:.5,q2:.6,unit:'nats'}},
 {title:'报告偏离，平均对数损失会增加',caption:'各类别的损失按真实出现频率加权。交叉熵由真实分布熵加上 KL 偏离组成。',action:'比较逐类贡献与总数，核对交叉熵减熵等于 KL。',focus:['q1','q2'],plot:1,metrics:[0,1,2],params:{q1:.8,q2:.5}},
 {title:'把可能事件报成零，并非有限大错',caption:'若A真实概率大于0却被报告为0，交叉熵与KL为无穷；若真实概率也为0，该项贡献为0。',action:'保持A真实概率为正，将报告概率设为0，再稍微增加，观察无穷边界。',focus:['q1'],plot:1,metrics:[1,2],params:{q1:0}}
]},
softmax:{steps:[
 {title:'一个分数变大，会挤走其他类别的份额',caption:'三个分数相同时各占三分之一。只提高A的分数，无论原先是否相等，其他两类概率都会下降。',action:'只拖动 z₁，看三根概率柱怎样共同变化。',focus:['z1'],plot:0,metrics:[0,2],params:{z1:0,z2:0,z3:0,temperature:1,shift:0,target:0}},
 {title:'共同加一个数，份额不变',caption:'Softmax 取决于相对分数。所有分数同时加100，不会改变概率，也不会改变同一目标的损失。',action:'只改变共同平移量，核对概率和损失保持不变。',focus:['shift'],plot:0,metrics:[0,1],params:{shift:100}},
 {title:'很自信，也可能错得很远',caption:'概率保持固定，只换真实类别。若目标是低概率类别，损失更大，梯度会要求提高它的分数。',action:'在 A 和 C 之间切换目标，比较损失与梯度方向。',focus:['target'],plot:1,metrics:[0,1],params:{z1:3,z2:0,z3:-1,temperature:.5,shift:0,target:2}}
]},
calibration:{steps:[
 {title:'分类结果相同，信心可以不同',caption:'固定零偏移，锐化会把概率推向0或1，但不改变0.5阈值的分类结果。',action:'只提高锐化程度，比较报告概率与准确率。',focus:['sharpness'],plot:2,metrics:[2],params:{sharpness:1,offset:0}},
 {title:'说八成会发生，实际有几成',caption:'把相近概率放在一起，比较平均报告与实际频率。偏离对角线可能是失配，也可能有有限样本波动。',action:'从锐化3回到1，比较可靠性图和 Brier 分数。',focus:['sharpness'],plot:0,metrics:[0,1],params:{sharpness:3,offset:0}},
 {title:'先看每个箱里有多少证据',caption:'箱内样本很少时，频率会抖动；空箱没有可报告的频率。分箱数也会影响经验 ECE。',action:'只改变箱数，先读每箱人数，再回看校准误差。',focus:['bins'],plot:1,metrics:[0],params:{sharpness:1,offset:0}}
]}
});
})(typeof globalThis!=='undefined'?globalThis:this);
