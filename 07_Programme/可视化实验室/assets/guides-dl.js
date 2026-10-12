/* Hand-authored reading routes for ten neural-network and six inference labs. */
(function(root){
  'use strict';
  root.LabGuides ||= {};
  Object.assign(root.LabGuides,{
    activations:{steps:[
      {title:'一个输入，变成什么输出？',caption:'观察点沿输入轴移动，曲线给出变换后的值。先分清输入位置、内部 z 与输出。',action:'移动 x，查看观察点如何沿曲线走；再切换激活函数。',focus:['x','kind'],plot:0,metrics:[0,1]},
      {title:'放大输入，梯度反而变小？',caption:'这里对原输入 x 求导，既要乘增益 g，也要看当前点是否进入 Sigmoid 饱和区。',action:'固定 x=2，把 g 从 1 增到 4，读当前点导数的变化。',focus:['gain','x'],plot:1,metrics:[0,2],params:{kind:'sigmoid',x:2,bias:0,gain:1}},
      {title:'拐点处，为什么断开？',caption:'ReLU 在 z=0 的左右斜率不同，普通导数不存在。空缺有数学含义。',action:'从 x=0 向两边移动，比较左右导数；换 Leaky ReLU 看负侧斜率。',focus:['x','kind'],plot:1,metrics:[0,2],params:{kind:'relu',gain:1,bias:0,x:0}}
    ]},
    neuron:{steps:[
      {title:'权重怎样改变输入的贡献？',caption:'每个输入先乘自己的权重，再与偏置相加。追踪这一项的正负与大小，最后看 Sigmoid 输出。',action:'分别改变 w₁、w₂，指出中间哪一项变化；检查三项之和是否等于 z。',focus:['w1','w2'],plot:0,metrics:[0,1]},
      {title:'偏置怎样改变最后的输出？',caption:'保持输入和权重，只改变求和中的 b。z 随偏置平移，Sigmoid 将新的 z 转为输出。',action:'把 b 从负值移到正值，追踪第三项、z 和输出；完整响应面可在下方展开。',focus:['bias'],plot:0,metrics:[0,1],params:{w1:1,w2:-1,bias:0}},
      {title:'同一个点，哪个输入更敏感？',caption:'w₂=0 时，第二个输入完全不影响输出。第一输入的敏感度还取决于是否饱和。',action:'移动 x₁ 与 x₂；解释为何第二根导数柱始终为 0。',focus:['x1','x2'],plot:1,metrics:[0,1],params:{w1:2,w2:0,bias:-1,x1:.5,x2:1}}
    ]},
    mlp:{steps:[
      {title:'先认出各个隐藏单元',caption:'各条曲线来自不同偏置。ReLU 情形下，一个单元提供一个拐点。',action:'选 ReLU 后移动全部拐点，观察各单元一起平移。',focus:['kind','shift'],plot:0,metrics:[0,2]},
      {title:'带符号相加，拼出帐篷',caption:'本步三单元示例使用权重 1、−2、1。虚线是各项贡献，粗线才是含偏置的最终输出。',action:'载入本步示例后，在 x=2 核对三项贡献和为 0；再改变输出偏置。',focus:['offset','gain'],plot:1,metrics:[1],params:{width:3,kind:'relu',gain:1,shift:0,offset:0}},
      {title:'去掉非线性，拐点还在吗？',caption:'本对称构造在线性激活下抵消为常数。一般而言，仿射层复合仍是仿射映射。',action:'在 linear 和 ReLU 之间切换，对照粗线的拐点是否存在。',focus:['kind','width'],plot:1,metrics:[0,2],params:{width:3,kind:'linear',gain:1,shift:0,offset:0}}
    ]},
    initialization:{steps:[
      {title:'信号过一层，会剩下多少？',caption:'从左往右读层号。纵轴每变化 1，统计量相差十倍；二阶矩与中心方差是两条不同曲线。',action:'保持种子，切换小尺度、He 和大尺度，观察前向幅度。',focus:['mode','gain'],plot:0,metrics:[0,1]},
      {title:'输出有界，梯度就安全吗？',caption:'改用大尺度 tanh。激活可以有界，反传却可能因饱和而衰减；梯度图要从右往左读。',action:'调低增益，观察传回输入的梯度是否改变。',focus:['gain','depth'],plot:1,metrics:[2],params:{mode:'large',kind:'tanh',gain:2}},
      {title:'理论尺度，不是逐层守恒',caption:'回到 ReLU 与 He。有限宽度会波动，且 ReLU 的正均值让方差小于二阶矩。',action:'换一个种子，对照两条曲线；不要把一次平稳曲线当成保证。',focus:['seed','depth'],plot:0,metrics:[1,3],params:{mode:'he',kind:'relu',gain:1}}
    ]},
    optimizers:{steps:[
      {title:'同一起点，沿哪条路走？',caption:'三种规则都从 (3,2) 出发。轨迹只画当前窗口内的部分，断开不代表停止更新。',action:'增加更新次数，比较三条路如何靠近原点或离开窗口。',focus:['steps','rate'],plot:0,metrics:[0,1,2]},
      {title:'没有历史，动量退化成什么？',caption:'将 β 设为 0 后，Momentum 每步只用当前梯度，轨迹应与 GD 重合。',action:'先看 β=0 的重合，再逐渐增大 β，追踪累积历史造成的变化。',focus:['momentum','steps'],plot:0,metrics:[0,1],params:{momentum:0}},
      {title:'离开画面之后，损失怎样了？',caption:'完整损失保留所有更新。GD 的稳定步长取决于最大曲率，而非画面的大小。',action:'曲率比 20 时把 η 从 0.05 提到 0.25，观察越过 0.1 后的 GD 损失。',focus:['rate','condition'],plot:1,metrics:[0,1,2],params:{condition:20,rate:.05,momentum:.9}}
    ]},
    regularization:{steps:[
      {title:'每次随机保留哪些贡献？',caption:'一行是一次掩码，1 表示保留。保留项还要除以 1−p，才是倒置 Dropout 的读出。',action:'改变丢弃概率，观察四列中的 0 与 1；p=0 时所有贡献都保留。',focus:['dropout','seed'],plot:0,metrics:[0,1]},
      {title:'均值保持，单次输出会散开',caption:'菱形是精确概率质量，柱子是模拟频率。输出只有有限种可能，不是连续密度。',action:'提高 p 看输出分散；增加重复次数通常降低模拟波动，单次误差未必下降。',focus:['dropout','reps'],plot:1,metrics:[1,2]},
      {title:'均值正确，损失为什么仍大于零？',caption:'目标等于无掩码输出 2 时，平均预测没有偏差，随机波动仍贡献一半方差的损失。',action:'让 p 从 0 增加，追踪方差附加项与总损失一起增长。',focus:['dropout','target'],plot:2,metrics:[3],params:{scale:1,target:2,dropout:.5}}
    ]},
    normalization:{steps:[
      {title:'先看输入的行与列',caption:'行是样本，列是特征。这里的偏移与缩放只作用于第三列，其余输入保持。',action:'改变第三特征的偏移，指出输入矩阵中实际变化的位置。',focus:['shift','scale'],plot:0,metrics:[]},
      {title:'换统计轴，会牵动谁？',caption:'BN 每列跨样本统计；LN 每行跨特征统计。同样改第三列，受影响的标准化值不同。',action:'先用 BN 改偏移，再换 LN 重做，观察同一行其他特征是否变化。',focus:['mode','shift'],plot:1,metrics:[0,1]},
      {title:'标准化之后，还要做仿射变换',caption:'输出为 γz+β，所以最终均值未必为 0。γ=0 时，所有输出都等于 β。',action:'先把 γ 设为 0，再移动 β，检查每个格子的实际数值。',focus:['gamma','beta'],plot:2,metrics:[0,2],params:{gamma:0,beta:1}}
    ]},
    convolution:{steps:[
      {title:'一个输出，只看一个窗口',caption:'框出的 3×3 区域决定当前输出。行列控制的是输出位置，输入窗口跟随步幅移动。',action:'移动输出行与列，追踪输入上被框住的九个数。',focus:['row','col'],plot:0,metrics:[0,1]},
      {title:'九项乘积，怎样合成一个数？',caption:'窗口与核在同位置逐项相乘；图里是这九项乘积，相加得到当前输出。本页不翻转核。',action:'切换核，再把九个数相加，核对下方的输出与乘积之和。',focus:['kernel','col'],plot:3,metrics:[1,2,3]},
      {title:'步幅与补零，改变输出形状',caption:'形状是 floor((7+2p−3)/s)+1。改变 s 或 p，改变采样位置和输出网格大小。',action:'保持核与图案，分别改变步幅和补零，先预测再核对尺寸。',focus:['stride','padding'],plot:2,metrics:[0,1]}
    ]},
    pooling:{steps:[
      {title:'这一格输出，看了哪些输入？',caption:'方框给出当前池化窗口。先找到最大值，再比较它与其他位置的数值。',action:'移动行与列，观察窗口扫过尖峰时，当前输出怎样改变。',focus:['row','col'],plot:0,metrics:[1,2]},
      {title:'最大值与平均值，留下什么？',caption:'Max 只保留窗口最大值；Mean 汇总所有位置。同样输入会得到不同输出。',action:'切换 Max 与 Mean，再改变窗口大小，比较输出与形状。',focus:['mode','size'],plot:1,metrics:[0,1]},
      {title:'梯度分给谁？',caption:'Mean 均分到窗口各点；唯一最大值收到全部梯度。并列时所画的是一种次梯度选择。',action:'对比两种模式，再选择全部相同的输入，检查并列最大值。',focus:['mode','pattern'],plot:2,metrics:[2,3]}
    ]},
    residual:{steps:[
      {title:'每层多加一个原输入',caption:'普通链是 aφ(h)，残差链是 h+aφ(h)。比较同一分支下，输出怎样逐层改变。',action:'增加层数或改变分支系数，追踪两条输出曲线。',focus:['depth','weight'],plot:0,metrics:[0,1]},
      {title:'分支为零，哪条路还在？',caption:'a=0 时普通链从第一层变成 0；残差链保持输入，输入梯度为 1。',action:'从 a=0 开始轻调系数，观察两种链的梯度开始怎样分离。',focus:['weight','depth'],plot:1,metrics:[2,3],params:{weight:0,kind:'tanh'}},
      {title:'捷径也可能被抵消',caption:'线性分支 a=−1 时，每层 h−h=0，残差局部导数也为 0。恒等路径不是稳定保证。',action:'把 a 从 −1 稍微移开，再与正系数比较梯度的衰减或放大。',focus:['weight','depth'],plot:1,metrics:[2,3],params:{kind:'linear',weight:-1,depth:5}}
    ]},
    rnn:{steps:[
      {title:'输入过去之后，状态还剩多少？',caption:'输入序列与递归状态是两条不同曲线。旧状态通过同一个共享权重影响下一时刻。',action:'选择脉冲输入，再改变递归权重，比较后续状态的遗忘或符号交替。',focus:['weight','pattern'],plot:0,metrics:[0,1]},
      {title:'早期状态还能影响末态吗？',caption:'从右向左读敏感度。线性递归中，每倒退一步就多乘一次共享权重 a。',action:'在 a=0.8、1、1.2 间比较，观察乘积的衰减、保持与放大。',focus:['weight','steps'],plot:1,metrics:[3],params:{kind:'linear',inputWeight:0,pattern:'zero',initial:.5,weight:.8}},
      {title:'同一个参数，要累加所有时间路径',caption:'a 在每个时刻都被使用，所以参数梯度是各时间贡献的和，不能只取最后一根柱。',action:'改变目标，比较每根贡献柱的符号与共享参数梯度总和。',focus:['target','weight'],plot:2,metrics:[1,2]}
    ]},
    lstm:{steps:[
      {title:'内部记忆与对外输出不是同一个量',caption:'遗忘门保留旧细胞状态；隐藏输出还经过 tanh 和输出门。记住了，不等于全部输出。',action:'改变遗忘门与输出门，分别追踪 c 和 h 两条状态曲线。',focus:['forget','output'],plot:0,metrics:[0,3]},
      {title:'直通记忆，只是梯度的一条路径',caption:'候选无递归反馈时，两条细胞梯度重合。打开反馈，完整导数还包括绕经隐藏状态的路径。',action:'从递归反馈 r=0 逐渐增加，观察完整导数与 fᵀ 是否分离。',focus:['recurrent','forget'],plot:1,metrics:[1,2],params:{recurrent:0,input:.8,output:.9,forget:.9,initial:.1,pattern:'zero'}},
      {title:'关闭输出，记忆仍可保留',caption:'f=1、i=0、o=0 时，细胞保留初态，但对外隐藏输出为 0，不能把两者混同。',action:'只打开输出门，观察 h 重新出现，而 c 仍保持。',focus:['output'],plot:0,metrics:[0,2,3],params:{forget:1,input:0,output:0,recurrent:0,initial:1,pattern:'zero'}}
    ]},
    contrastive:{steps:[
      {title:'哪个候选与 Query 更对齐？',caption:'向量都在单位圆上，点积由夹角决定。先比较几何位置，再讨论归一化后的概率。',action:'旋转 Query 或第一个负样本，预测哪个候选得到更大点积。',focus:['angle','negativeAngle'],plot:0,metrics:[0]},
      {title:'所有候选竞争同一个分母',caption:'正负样本一起进入 Softmax。低温强化已有分数差异，也可能强化错误排序。',action:'先判断当前谁领先，再降低温度，查看正样本分配概率和损失。',focus:['temperature','negatives'],plot:1,metrics:[0,1]},
      {title:'正负向量相同，低温能救吗？',caption:'只有一个负样本且与正样本重合时，任何 Query 角度都无法把二者分开。',action:'旋转 Query 并降低温度，检查损失为何一直是 log 2。',focus:['angle','temperature'],plot:2,metrics:[0,1],params:{negativeAngle:0,negatives:1,angle:0,temperature:.5}}
    ]},
    jacobian:{steps:[
      {title:'先圈定一个局部邻域',caption:'网格围绕当前中心。Jacobian 描述的是这里的小位移，不是整个平面的固定变换。',action:'移动中心 x、y，指出接下来要近似的是哪一块输入。',focus:['x','y'],plot:0,metrics:[0,1]},
      {title:'弯曲网格与直线近似差在哪里？',caption:'实线来自真实非线性映射，虚线来自中心处的仿射近似；二者在中心具有相同一阶行为。',action:'保持中心，缩小半宽 ρ，比较两组网格在边缘的偏差。',focus:['radius','alpha'],plot:1,metrics:[2,3]},
      {title:'剩下的误差怎样随距离缩小？',caption:'这里沿固定单位对角方向比较真实值和一阶近似。距离趋小时，余项由二阶项控制。',action:'改变邻域半宽与 β，比较当前距离的误差和二阶上界。',focus:['radius','beta'],plot:2,metrics:[3]}
    ]},
    uncertainty:{steps:[
      {title:'同一个均值，为什么有两种区间？',caption:'窄区间描述潜在线性均值；宽区间预测一次新观测，还必须包含观测噪声。',action:'移动查询点，比较同一位置的两种区间，尤其看设计范围之外。',focus:['query'],plot:0,metrics:[1,3]},
      {title:'更多数据，能减少哪一部分？',caption:'预测方差等于参数方差加 σ²。同一数据池增加观测会降低参数不确定性，噪声项保持。',action:'固定种子、噪声与设计范围，增加 n，比较两条方差曲线。',focus:['n','query'],plot:1,metrics:[1,2,3]},
      {title:'截距与斜率也有联合不确定性',caption:'椭圆是两个参数的联合 95% 后验区域；它不是上一图的逐点均值区间。',action:'改变先验标准差或样本数，观察椭圆中心、大小与倾斜方向。',focus:['priorSD','n'],plot:2,metrics:[0]}
    ]},
    causal:{steps:[
      {title:'组内趋势与总体趋势可能反向',caption:'每个 Z 组有自己的结果基线。总体观察曲线还取决于两个处理组各混入了多少人。',action:'改变选择强度，比较组内斜率和总体观察斜率是否同向。',focus:['selection','effect'],plot:0,metrics:[0,2]},
      {title:'总体比较，究竟换了哪些人？',caption:'观察 T=0 与 T=1 时，Z 的组成可能不同；已知模型中的 do 干预保留总体 Z 分布。',action:'把选择强度调到 0，检查观察组构成怎样接近同一总体构成。',focus:['selection','prevalence'],plot:1,metrics:[0,1]},
      {title:'有关联，也可能没有干预效应',caption:'在给定合成模型里令真实效应 τ=0。选择机制仍可制造观察差，而 do 效应保持 0。',action:'改变选择强度，追踪观察差与两个调整后效应量；结论只针对这个已知模型。',focus:['selection'],plot:2,metrics:[0,1,2],params:{effect:0,selection:.8,prevalence:.5}}
    ]}
  });
})(typeof globalThis!=='undefined'?globalThis:this);
