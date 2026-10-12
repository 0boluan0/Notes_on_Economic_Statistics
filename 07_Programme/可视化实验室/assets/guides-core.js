/* Curated reading order. Each step chooses a specific observable mechanism. */
(()=>{'use strict';const g=globalThis.LabGuides||(globalThis.LabGuides={});Object.assign(g,{
linear:{steps:[
{title:'追踪两条基向量',caption:'矩阵的两列，正是单位横箭头和单位竖箭头的新位置。先认清这两根箭头，再看整个网格。',action:'只拖动 b。竖直基向量向哪里倾斜？水平基向量为什么不动？',focus:['b'],plot:0,metrics:[0,1]},
{title:'看面积怎样改变',caption:'单位正方形变成平行四边形。行列式的绝对值给出面积倍率，符号记录取向是否翻转。',action:'改变 a 或 d，寻找行列式接近零的组合；a=0本身不一定使面积为零。',focus:['a','d'],plot:0,metrics:[0,1]},
{title:'压成线，或压成点',caption:'两列线性相关时，平面会压成线或点，面积为零；不同原位置重合，无法唯一反向恢复。',action:'点击下方“秩一：压成一条线”情境，再用矩阵参数把两列重新分开。',focus:['a','b','c','d'],plot:0,metrics:[0,2]}
]},
projection:{steps:[
{title:'把向量投到直线上',caption:'从向量终点向直线作垂线，垂足就是投影。剩下的那段箭头叫残差。',action:'旋转投影方向。向量保持不动，观察垂足和残差怎样移动。',focus:['angle'],plot:0,metrics:[0,1]},
{title:'为什么一定垂直',caption:'沿直线移动垂足，只会让连接终点的线段更长。正交投影选的是距离最短的位置。',action:'先用非零向量，让直线接近其方向，再转到近乎垂直；比较残差长度。',focus:['angle','vx','vy'],plot:0,metrics:[1]},
{title:'检查长度的分解',caption:'投影与残差正交，非退化时组成直角三角形；即使一项为零，平方长度分解仍成立。',action:'改变向量的一个分量，核对下面两项是否始终相等。',focus:['vx','vy'],plot:0,metrics:[2,3]}
]},
gradient:{steps:[
{title:'一步朝哪里走',caption:'每一步沿当前负梯度移动。椭圆较陡的方向产生更大的梯度，轨迹可能左右摆动。',action:'从少量步数开始，逐步增加。对照等高线观察每一步的方向。',focus:['steps'],plot:0,metrics:[0]},
{title:'步子大也会走远',caption:'越过稳定界的方向若含非零起点分量，误差会放大；起点已在最小值时仍不动。完整损失可核对。',action:'先用非零起点分量，再提高学习率跨过稳定上界；检查实际损失而不只看轨迹。',focus:['rate','x0','y0'],plot:1,metrics:[1,3],params:{x0:3,y0:2,condition:4}},
{title:'更窄的谷地',caption:'最陡方向限制步长。对该二次模型，0<η<2/L 保证任意起点收敛；特定起点可能避开不稳定方向。',action:'固定学习率，提高曲率比。原来稳定的步子还适合吗？',focus:['condition','rate'],plot:0,metrics:[1,2]}
]},
sampling:{steps:[
{title:'一次均值仍会波动',caption:'直方图汇总许多独立样本均值，纵轴是密度。它描述重复抽样后的均值，不是原始观测分布。',action:'用指数总体，先把每份样本量设小。观察均值分布的偏斜。',focus:['n','distribution'],plot:0,metrics:[0,1]},
{title:'哪些总体的均值会更稳',caption:'对于这里的有限方差总体，样本均值的标准误按 1/√n 缩小；中心极限定理解释标准化后的形状。',action:'增加 n，比较柱子的宽度。再换成 Cauchy，总体均值和这条结论还成立吗？',focus:['n','distribution'],plot:0,metrics:[1,2]},
{title:'重复次数不是样本量',caption:'重复次数 B 增加，只让模拟更充分地描述同一抽样分布；每份样本量仍是 n，理论标准误不变。',action:'固定 n，只增加 B，比较直方图并核对理论标准误；再换种子观察模拟波动。',focus:['reps','seed'],plot:0,metrics:[1,3]}
]},
confidence:{steps:[
{title:'每条线是一份样本',caption:'竖线是真实均值。每次重新抽样得到一条新区间；单次模拟可能有区间错过，也可能全部覆盖。',action:'点击“换一组”，若有未跨过真值的区间就指出它们；全覆盖也是可能结果。',focus:['seed'],plot:0,metrics:[0,1]},
{title:'更有把握，要付出什么',caption:'提高置信水平，构造出的每条区间都会更宽。95% 指这套方法的长期覆盖率，不是固定真值会随机移动。',action:'把置信水平从 90% 提高到 99%，比较区间宽度和覆盖率。',focus:['level'],plot:0,metrics:[0,2]},
{title:'收窄区间的另一条路',caption:'增加样本量能收窄区间，同时保留名义覆盖水平。有限次模拟得到的覆盖率仍会围绕名义值波动。',action:'固定置信水平，增加 n；然后增加重复次数，观察覆盖率的模拟波动。',focus:['n','reps'],plot:0,metrics:[0,3]}
]},
testing:{steps:[
{title:'两种世界，两种分布',caption:'零假设把真实均值设为零；备择世界的均值由 δ 指定。拒绝边界由 H₀ 和显著性水平决定。',action:'让 δ 远离零，观察两种分布分开；固定 n、α 时拒绝边界会随 δ 移动吗？',focus:['effect'],plot:0,metrics:[2]},
{title:'p 值数哪一块面积',caption:'p 值始终在 H₀ 分布下计算：出现至少与本次观测同样极端结果的两侧尾部面积。',action:'让观测均值远离零，再靠近零，比较尾部面积；它不是“H₀ 为真的概率”。',focus:['observed'],plot:1,metrics:[0,1]},
{title:'更容易发现多大的效应',caption:'功效是指定备择下落入拒绝域的概率。固定非零效应时，增大 n 会提高功效；效应为零时功效等于 α。',action:'固定 δ，提高样本量；再提高 α，观察功效与误报约束的取舍。',focus:['n','alpha'],plot:0,metrics:[2,3]}
]},
bayes:{steps:[
{title:'先表达原有的不确定性',caption:'Beta 先验把不同成功概率的相对可信程度画成密度。更高的密度不是某个单点拥有更大的概率质量。',action:'先只追踪先验线，分别提高 a 或 b，看它向哪侧移动；后验还使用当前数据。',focus:['a','b'],plot:0,metrics:[2]},
{title:'把这次数据乘进来',caption:'成功 s 次、失败 f 次，把 Beta(a,b) 更新成 Beta(a+s,b+f)。后验同时使用先验与这份数据。',action:'固定先验，增加成功次数；再增加失败次数，观察后验如何移动。',focus:['success','failure'],plot:0,metrics:[0,1]},
{title:'形状相似，身份不同',caption:'似然把已观测数据固定，比较不同参数的解释能力。图中只按峰值归一化，它不是参数的概率密度。',action:'保持成功比例，比较不同数据量的似然宽度；再读后验均值与区间指标。',focus:['success','failure'],plot:1,metrics:[2,3]}
]},
regression:{steps:[
{title:'两个变量长得太像',caption:'当两个解释变量高度相关，数据较难区分“谁贡献了多少”。拟合可以不错，单个系数却可能不稳定。',action:'提高 ρ，再换种子。比较系数的跳动和训练误差。',focus:['rho','seed'],plot:0,metrics:[1,2,3]},
{title:'收缩系数的代价',caption:'岭惩罚限制系数的幅度，让近似共线时的解更稳定，也会引入收缩偏差。这里用的是平方和损失约定。',action:'固定数据，提高 λ，比较各系数变化；不要求每一个系数都单调趋零。',focus:['lambda'],plot:0,metrics:[2,3]},
{title:'预测好，不等于解释可靠',caption:'拟合值靠近对角线，说明当前样本上的预测接近响应；这不能单独证明每个系数都有稳定的解释。',action:'在高相关情境下比较拟合图与系数图，并检查完整图表中的残差。',focus:['rho','noise'],plot:1,metrics:[0,3]}
]},
overfit:{steps:[
{title:'曲线开始追逐观测点',caption:'阶数增加允许更复杂的弯曲。噪声强度非零时，圆点也包含噪声；更贴训练点不保证更贴真函数。',action:'固定样本和种子，逐渐提高多项式阶数。先预测哪里会出现大幅弯曲。',focus:['degree'],plot:0,metrics:[0,1]},
{title:'用另一批数据检验',caption:'同一训练数据上分别拟合各个阶数，并在固定的独立验证集上评分。看训练误差和验证误差何时分离。',action:'比较图中两条误差曲线。验证误差最低处一定是最高阶数吗？',focus:['degree','noise'],plot:1,metrics:[0,1]},
{title:'约束复杂度',caption:'岭惩罚会收缩系数；更多训练样本也会改变合适的复杂度。这份有限样本的最佳点不代表普遍最优。',action:'先提高 λ，再恢复并增加 n。观察拟合曲线和验证误差的变化。',focus:['lambda','n'],plot:0,metrics:[0,1]}
]},
pca:{steps:[
{title:'寻找最分散的方向',caption:'第一主方向穿过样本中心，最大化投影方差；最大特征值重合时，方向可能不唯一。',action:'旋转生成数据，观察主方向怎样随点云转动。',focus:['angle','ratio'],plot:0,metrics:[0]},
{title:'留下一个坐标',caption:'把每个点投到第一主方向再搬回二维，得到秩一重构。垂直方向的信息被丢弃，线段显示丢掉了多少。',action:'改变标准差比，分别比较解释方差占比与绝对重构误差；两者不必同向变化。',focus:['ratio'],plot:1,metrics:[2,3]},
{title:'单位改变，主方向也可能变',caption:'第二坐标的尺度会改变协方差。标准化改用标准差单位，相当于改变衡量距离与重构误差的坐标系。',action:'先放大第二坐标，再切换标准化。比较两次分析使用的坐标。',focus:['scale','standardize'],plot:0,metrics:[0,3]}
]},
bootstrap:{steps:[
{title:'手里只有这一份样本',caption:'Bootstrap 把这份已观测样本当成经验分布，再有放回地重复抽取同样大小的样本。',action:'先看每个原始观测。更换种子后，手里的样本和后续估计都会改变。',focus:['seed','n'],plot:1,metrics:[0]},
{title:'重新抽样，再重新算均值',caption:'每次重采样得到一个新均值，它们的分散程度估计均值的不确定性。重复观测可以出现多次。',action:'增加重采样次数 B，看直方图更平滑；它能补出原样本从未观察到的尾部吗？',focus:['reps'],plot:0,metrics:[1,2]},
{title:'区间取决于原样本',caption:'百分位区间取重采样均值的分位点。小样本或偏态会限制可靠性；增加 B 不能补足原样本缺少的信息。',action:'比较小样本与更大样本，再换总体分布。不要把某次区间当作覆盖率保证。',focus:['n','distribution'],plot:0,metrics:[0,3]}
]}
});})();
