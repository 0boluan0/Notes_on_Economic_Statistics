/* Each guide follows the mechanism of its own model and existing plots. */
(function(root){
  'use strict';
  root.LabGuides ||= {};
  Object.assign(root.LabGuides, {
    positional:{steps:[
      {title:'先看一对相位',caption:'两根箭头表示两个位置的第一对 sin/cos。固定位置差后共同移动，箭头转动，但夹角不变。',action:'固定 Δ，只拖动起点 p；核对余弦相似度保持不变。',focus:['position','shift'],plot:2,metrics:[2,3]},
      {title:'距离更远，相似度未必更低',caption:'只有一对频率时，相似度随位移周期性升降；多频也不保证单调，不能直接当作距离。',action:'先只保留一对频率，再把 Δ 从0移到12，寻找先下降、后上升的区间。',focus:['shift','pairs'],plot:1,metrics:[0,2],params:{pairs:1}},
      {title:'多种频率一起记录位置',caption:'同一行是一个位置。增加频率对数后，可比较不同变化速度；每对 sin/cos 的平方和始终为1。',action:'比较底数 10 与 10000；找出随位置变化最慢的列。',focus:['pairs','base'],plot:0,metrics:[0,1],params:{pairs:4,base:10000}}
    ]},
    gnn:{steps:[
      {title:'一层能把信息送多远？',caption:'节点旁的数是当前特征。初始只有节点 0 为 1；每层聚合最多沿一条边把影响传出去。',action:'把层数归零，再增到 1、2、3；找出最远的非零节点。',focus:['layers','selfLoop'],plot:0,metrics:[0]},
      {title:'为什么会来回振荡？',caption:'无自环、无残差的星形图做邻居平均时，中心与叶节点交替接到脉冲；加自环会改变这条轨迹。',action:'依次看 4、5、6 层；再加自环，观察奇偶交替怎样改变。',focus:['layers','selfLoop'],plot:1,metrics:[0,2],params:{graph:'star',normalization:'mean',selfLoop:0,residual:0,layers:4}},
      {title:'归一化改变最终形状',caption:'对称归一化以度数平方根为固定方向，邻居平均以常数为固定方向；有限层还不一定达到极限。',action:'在两种归一化之间切换；比较 20 层后中心与叶节点的值。',focus:['normalization','layers'],plot:0,metrics:[1,2],params:{graph:'star',normalization:'symmetric',selfLoop:1,residual:0,layers:20}}
    ]},
    embeddings:{steps:[
      {title:'方向和长度，分别影响什么',caption:'内积同时使用长度和方向；余弦只比较非零向量的方向。端点距离还会受到长度差影响。',action:'先让两向量都非零，再固定夹角、只改变 b 的长度；比较三个指标。',focus:['normA','normB'],plot:0,metrics:[0,1,2]},
      {title:'把长度因素拿掉',caption:'两向量都为单位长度时，内积与余弦曲线重合；若改了长度，这种重合不再得到保证。',action:'把夹角从 0 转到 180°；在 90° 检查内积与余弦均为 0。',focus:['angle'],plot:1,metrics:[0,1,2],params:{normA:1,normB:1}},
      {title:'零向量为什么是边界？',caption:'任一向量长度为0时，内积仍可计算；余弦和夹角因方向不存在而未定义。',action:'令 a=0、b非零，再把 a 的长度升到0.1，观察方向比较何时恢复。',focus:['normA'],plot:0,metrics:[0,1,3],params:{normA:0,normB:2}}
    ]},
    autoencoder:{steps:[
      {title:'一个数怎样重构一个点？',caption:'增益为 1 时，二维点被投影到一条直线；点与重构之间的连线就是被丢掉的部分。',action:'先设 g=1，再旋转编码方向，观察每条残差如何改变。',focus:['gain','angle'],plot:0,metrics:[0,2]},
      {title:'寻找最少丢失的方向',caption:'g=1时，损失最低处对应当前样本的主方向；虚线是最优秩一投影误差，其他增益须另看。',action:'沿损失曲线移动方向角，找到接近最低处的位置；比较两个误差。',focus:['angle'],plot:1,metrics:[0,1],params:{gain:1}},
      {title:'放大解码，为什么也会出错？',caption:'固定方向时，g=1保留投影长度。g=0与g=2分别少取和多取同样一份，平方误差相同。',action:'保持方向和种子不动，依次比较 g=0、1、2。',focus:['gain'],plot:0,metrics:[0,3]}
    ]},
    vae:{steps:[
      {title:'样本移动，分布未必改变',caption:'点是 z=μ+σε 的一次采样。只改变 ε 会移动点，不会改变后验曲线、KL 或解析期望目标。',action:'只拖动 ε，核对样本值改变而 KL 与期望重构项不变。',focus:['epsilon'],plot:0,metrics:[0,1,2]},
      {title:'KL 在惩罚哪种差异？',caption:'μ=0且对数方差=0时，近似后验等于标准正态先验，KL为0；移动任一参数会偏离这个起点。',action:'分别移动 μ 与 log variance，观察偏移和变宽怎样产生 KL。',focus:['mu','logvar'],plot:0,metrics:[1],params:{mu:0,logvar:0}},
      {title:'重构与先验怎样折中？',caption:'当x=2、解码权重为1时，最优μ=2/(1+β)。β=1得到μ=1；其他条件要读实际目标曲线。',action:'比较 β=0、1、4 时总目标的最低处；β≠1 是加权目标。',focus:['beta','mu'],plot:1,metrics:[2,3,4],params:{mu:1,logvar:0,decoder:1,x:2,beta:1}}
    ]},
    gan:{steps:[
      {title:'判别器怎样评价生成点？',caption:'曲线是冻结的判别器。D(θ) 接近 0 表示它强烈拒绝当前假样本；这还不是生成器梯度。',action:'移动 θ，看评分能否改变；若判别器斜率为0，移动点还有效吗？',focus:['theta'],plot:0,metrics:[0]},
      {title:'同样被拒绝，两种梯度不同',caption:'判别器斜率非零且D接近0时，饱和梯度接近零，非饱和梯度仍可能明显；斜率为0时两者都为0。',action:'把 θ 向右移，比较两条梯度曲线的绝对大小与方向。',focus:['theta'],plot:1,metrics:[1,2],params:{theta:-4,weight:2,bias:-2}},
      {title:'沿梯度更新会走到哪里？',caption:'指标比较起点梯度，曲线记录之后的更新。判别器保持冻结，生成点不一定停在真实示例处。',action:'比较 20 与 50 次更新；观察两种目标的轨迹，而不只看终点。',focus:['steps','rate'],plot:2,metrics:[1,2]}
    ]},
    flow:{steps:[
      {title:'先追踪一个样本',caption:'横轴是原始 z，纵轴是变换后的 y。曲线单调递增，因此每个输出位置都有唯一的原像。',action:'拖动 z 追踪点；只改变平移 b，观察曲线整体上下移动。',focus:['z','shift'],plot:0,metrics:[0]},
      {title:'附近的小区间被拉多长？',caption:'s=1、a=0时，Jacobian处处为1；加入非线性后，不同位置的拉伸量会不同。平移不改变拉伸。',action:'保持 s=1，把 a 从 0 改为 0.8，比较中心与两端的 Jacobian。',focus:['warp','z'],plot:2,metrics:[1],params:{scale:1,shift:0,warp:0,z:0}},
      {title:'质量不变，密度要反向调整',caption:'同一个小区间被拉长，单位长度内的概率就变少。密度必须除以 Jacobian 才能保留概率质量。',action:'固定 z 与 a，把 s 从 1 改为 2；对应密度应减半。',focus:['scale','z'],plot:1,metrics:[1,2,3]}
    ]},
    diffusion:{steps:[
      {title:'先分清观测在哪条轴上',caption:'这条曲线是加噪后的 xₜ 分布，点是已经看到的 xₜ；它不是原始样本 x₀。',action:'固定分布参数，只移动观测点 xₜ，找它在分布中的位置。',focus:['observed'],plot:1,metrics:[3]},
      {title:'同一个观测对应哪些原值？',caption:'这次横轴换成 x₀。实线是看到 xₜ 后的条件分布；它有一个最优均值，也仍有不确定性。',action:'固定噪声比例，只改变 xₜ；观察后验均值移动、条件方差不变。',focus:['observed'],plot:0,metrics:[0,1]},
      {title:'噪声越多，能恢复的信息越少',caption:'固定原始高斯分布，降低信号比例会增加条件方差。这里是精确条件推断，不是网络的多步降噪轨迹。',action:'比较 ᾱ=0.1 与 0.9；用方差而非单个估计值判断剩余不确定性。',focus:['alpha'],plot:2,metrics:[1]}
    ]},
    'rl-bellman':{steps:[
      {title:'一次选择会发生什么？',caption:'安全动作立即得 1 并终止。冒险先得 c，再到 S₁ 或回到 S₀；返回后还可以重新选择。',action:'沿箭头分别读即时奖励和转移概率，别把两者相乘后当成全部回报。',focus:['success','cost'],plot:0,metrics:[1]},
      {title:'未来价值逐轮传回来',caption:'从零价值开始，第1轮得到V(S₁)=3；第2轮可把它计入S₀的冒险价值，但最优动作未必选择冒险。',action:'将迭代次数从 1 增到 2，再增到 12；比较当前值与最优值。',focus:['iterations'],plot:1,metrics:[0,1],params:{gamma:.9,success:.5,cost:-.2,iterations:1}},
      {title:'停止时离答案还有多远？',caption:'实线是对精确解的误差，虚线是由残差算出的上界。上界可以很松，不能把它当成实际误差。',action:'增加迭代，再把 γ 调近 1；观察收敛速度与上界怎样改变。',focus:['iterations','gamma'],plot:2,metrics:[2,3]}
    ]},
    'rl-policy':{steps:[
      {title:'先把参数变成动作概率',caption:'θ 经 sigmoid 决定冒险概率。曲线显示从 S₀ 出发的期望折扣回报，不是某次冒险的奖励。',action:'移动 θ，先读当前梯度的符号：它建议增大还是减小 θ？',focus:['theta'],plot:0,metrics:[0,1,2]},
      {title:'好动作的概率逐步增加',caption:'冒险有利时，精确梯度提高它的概率；不利时则降低。曲线使用解析期望，没有采样造成的抖动。',action:'比较少量与更多更新；概率接近边界时，每步变化是否仍一样大？',focus:['steps','rate'],plot:2,metrics:[0,4],params:{theta:0,gamma:.9,success:.5,cost:-.2,rate:.5,steps:30}},
      {title:'环境变了，更新方向也该变',caption:'成功率0.1、即时奖励−1、折扣0.9时，冒险不利。改变环境后，应重新检查梯度符号与回报。',action:'若初始梯度为负，核对更新后的冒险概率是否下降、期望回报是否上升。',focus:['steps','rate'],plot:1,metrics:[2,3,4],params:{theta:0,gamma:.9,success:.1,cost:-1,rate:.5,steps:30}}
    ]},
    'double-descent':{steps:[
      {title:'训练误差何时可能归零？',caption:'m 个特征还要加一个截距。虚线 m=n−1 表示方形设计；只有无惩罚且满秩时才保证精确插值。',action:'将 λ 设为 0，固定 n 与种子，把 m 移到 n−1 附近。',focus:['width','lambda','n'],plot:0,metrics:[0,2]},
      {title:'只看这次实际的验证风险',caption:'先看无惩罚时的实际验证曲线。插值附近是否有峰、之后是否下降，都取决于这次样本与模型。',action:'让 m 跨过当前的 n−1，再换种子；检查实际曲线，不预设它一定双降。',focus:['width','seed'],plot:1,metrics:[1,3],params:{width:23,n:24,noise:.3,lambda:0,seed:42}},
      {title:'误差背后是哪一些预测？',caption:'散点比较预测与无噪声真值；指标里的验证MSE还包含响应噪声。训练插值不保证这两种误差都小。',action:'固定数据，比较 λ=0 与 0.2；观察预测散布和验证误差的变化。',focus:['lambda','width'],plot:2,metrics:[0,1]}
    ]},
    fairness:{steps:[
      {title:'先找到 PPV 的分母',caption:'有正预测时，PPV=真阳性/(真阳性+假阳性)，对应“预测1”列；若这列合计为0，PPV未定义。',action:'分别用两组右侧两格计算 PPV，核对指标；四格是联合概率。',focus:['baseA','baseB'],plot:0,metrics:[0,1]},
      {title:'相同错误率，为何 PPV 不同？',caption:'TPR、FPR均为正且两组基准率不同时，相同条件错误率仍会给出不同PPV；零错误率边界另看。',action:'把 A 的基准率设为与 B 相同，核对两组 PPV 和选择率是否一致。',focus:['baseA','baseB'],plot:1,metrics:[0,1,2,3],params:{baseA:.2,baseB:.6,tpr:.8,fpr:.2}},
      {title:'边界要单独判断',caption:'FPR=0且TPR>0时，两组PPV都为1；若TPR也为0，则没有正预测，PPV未定义。',action:'保持 FPR=0，将 TPR 从正值降到0；核对无正预测时的未定义边界。',focus:['tpr','fpr'],plot:2,metrics:[0,1],params:{baseA:.2,baseB:.6,tpr:.8,fpr:0}}
    ]}
  });
})(typeof globalThis!=='undefined'?globalThis:this);
