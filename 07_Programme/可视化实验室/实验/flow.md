<!-- visual-lab:generated:start -->
# 可逆变换如何搬运概率密度

<!-- bilingual-en:start -->
*Normalizing flows & Jacobian mass conservation*
<!-- bilingual-en:end -->

对标准正态施加严格单调的非线性变换，同时核对逆映射、局部拉伸与密度积分。

> [!question] 先预测，再操作
> 把样本位置拉伸两倍，密度高度为什么要除以二？

<!-- visual-lab:film:start -->
## 动画：分步观察

<!-- bilingual-en:start -->
*Animation: follow the mechanism*
<!-- bilingual-en:end -->

![[07_Programme/可视化实验室/media/flow.mp4]]

约 36 秒，中英字幕。可暂停、点击进度条定位或全屏观看。

<!-- bilingual-en:start -->
About 36 seconds, with Chinese and English captions. Pause, seek on the timeline or watch full screen.
<!-- bilingual-en:end -->

- **0:00** 先搬运位置，再问密度怎样变
- **0:10** 同一小段概率，铺到更长的区间
- **0:27** Jacobian 补偿的是局部体积变化

<!-- bilingual-en:start -->
- **0:00** First move positions, then account for density.
- **0:10** Stretching a small interval spreads its probability mass.
- **0:27** The Jacobian corrects local volume change.
<!-- bilingual-en:end -->

## 亲手实验

<!-- bilingual-en:start -->
*Try it yourself*
<!-- bilingual-en:end -->

在下方展开“影片参数与延伸对照”，选择“载入本片的模型与数据”，先核对影片中的一个对象或运算，再改变一个条件。其他对照用于继续探索。

<!-- bilingual-en:start -->
Expand the film inputs below and load the model and data used in this film. Check one object or calculation, then change one condition. The other comparisons provide further exploration.
<!-- bilingual-en:end -->

<!-- visual-lab:film:end -->
```visual-lab
id: flow
height: 980
```

## 观察路线

1. 从恒等变换开始，确认输出仍是标准正态。
2. 设置纯缩放 s=2，比较中心密度和位置尺度。
3. 固定 s=1，分别设 a=−0.8 与 2，查看中心 Jacobian 与密度反向变化。
4. 移动追踪点 z，逐个核对 p_Y(g(z))g′(z)=φ(z)。

## 公式与适用条件

$$
y=g(z)=b+s[z+a\tanh z],\quad g'(z)=s[1+a\,\mathrm{sech}^2z]>0,\quad p_Y(g(z))=\frac{\phi(z)}{g'(z)}
$$

- z~N(0,1)，s>0，a>−1；本页范围保证 g′严格为正。
- 变换在整条实线上单调且满射，数值逆使用括区间二分。
- 绘图区间 z∈[−5,5]，标准正态尾部约 5.73×10⁻⁷ 未显示。

## 回到知识点

- [[00_Knowledge/随机变量的分布变换|随机变量的分布变换]]：严格单调变换以逆 Jacobian 校正局部质量；展示有限绘图区间、逆函数与积分近似。

[[07_Programme/可视化实验室/00_实验库入口|回到实验库]] · [[07_Programme/可视化实验室/实验/vae|VAE 的重参数化与 KL 代价]] · [[07_Programme/可视化实验室/实验/diffusion|扩散加噪与精确 Gaussian 去噪]] · [[07_Programme/可视化实验室/实验/linear|矩阵怎样改变空间]]

## 来源与核验

- [Rezende & Mohamed · Variational Inference with Normalizing Flows](https://arxiv.org/pdf/1505.05770) — §3 Normalizing Flows; change-of-variables formula。支持：可逆映射的密度变换和 Jacobian 修正；本页自定义单调一维 tanh 变换。

实验使用明确给定的公式、合成数据或固定小型示例。代码、可复现检查和逐轮修订记录见 [[07_Programme/可视化实验室/01_使用与维护|使用与维护]]。
<!-- visual-lab:generated:end -->
