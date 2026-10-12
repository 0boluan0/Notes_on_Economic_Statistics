<!-- visual-lab:generated:start -->
# Softmax 温度与交叉熵梯度

<!-- bilingual-en:start -->
*Multiclass softmax, temperature & Jacobian*
<!-- bilingual-en:end -->

在真正的三类别向量上计算概率、交叉熵和全部 Jacobian，分开观察共同平移、温度与目标标签。

> [!question] 先预测，再操作
> 把所有 logits 同时加 100，为什么模型概率和梯度都不会变？

<!-- visual-lab:film:start -->
## 动画：分步观察

<!-- bilingual-en:start -->
*Animation: follow the mechanism*
<!-- bilingual-en:end -->

![[07_Programme/可视化实验室/media/softmax.mp4]]

约 34 秒，中英字幕。可暂停、点击进度条定位或全屏观看。

<!-- bilingual-en:start -->
About 34 seconds, with Chinese and English captions. Pause, seek on the timeline or watch full screen.
<!-- bilingual-en:end -->

- **0:00** 三个分数，要共同分一份概率
- **0:04** 先指数化，再使用同一个分母
- **0:14** 一个概率变化，其他概率也要响应
- **0:25** 标签影响损失，分数决定预测

<!-- bilingual-en:start -->
- **0:00** Three scores compete for one unit of probability.
- **0:04** Exponentiate, then divide by a shared denominator.
- **0:14** Normalization couples the three outputs.
- **0:25** Changing the label changes loss, not this prediction.
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
id: softmax
height: 980
```

## 观察路线

1. 把三个 logits 设相同，预测概率与交叉熵；之后只改变真实类别看输出概率是否变化。
2. 保持 logits，改变温度，观察概率尖锐程度、损失与梯度。
3. 共同 shift 加 100 或减 100，核对概率、损失与 Jacobian 不变。
4. 读 Jacobian 第 j 列：提高 zⱼ 会提高自己的概率，并压低另外两类。

## 公式与适用条件

$$
p_i=\frac{e^{z_i/T}}{\sum_j e^{z_j/T}},\quad\ell=-\log p_y,\quad\frac{\partial p_i}{\partial z_j}=\frac{p_i(\delta_{ij}-p_j)}T,\quad\frac{\partial\ell}{\partial z_j}=\frac{p_j-\mathbb1\{j=y\}}T
$$

- 三类别、单样本、有限 logits、正温度；没有训练网络或估计温度。
- 梯度和 Jacobian 均对原始未除温度的 z 求导。
- 真实标签为 one-hot；未加入 label smoothing、类别权重或掩码。

## 回到知识点

- [[00_Knowledge/Softmax|Softmax]]：真实三类指数归一化、共同平移不变性、概率比和相互耦合；新增温度与导数由同一定义推出。

[[07_Programme/可视化实验室/00_实验库入口|回到实验库]] · [[07_Programme/可视化实验室/实验/entropy|熵、交叉熵与 KL 的区别]] · [[07_Programme/可视化实验室/实验/logistic|概率、阈值与交叉熵]] · [[07_Programme/可视化实验室/实验/attention|注意力从分数到读取向量]] · [[07_Programme/可视化实验室/实验/calibration|准确率之外的概率校准]]

## 来源与核验

- [Blanchard, Higham & Higham · Stable Softmax](https://webhomes.maths.ed.ac.uk/~dhigham/Publications/P149.pdf) — §1 equations (1.1)–(1.2); shifted algorithms。支持：核对 log-sum-exp、softmax 与减最大分数的稳定计算。
- [Dive into Deep Learning · Softmax Regression](https://d2l.ai/chapter_linear-classification/softmax-regression.html) — §4.1.2; §4.1.3 Loss Function and likelihood/gradient subsections。支持：核对多类交叉熵及 p−y 导数；温度 1/T 因子由链式法则独立核算。

实验使用明确给定的公式、合成数据或固定小型示例。代码、可复现检查和逐轮修订记录见 [[07_Programme/可视化实验室/01_使用与维护|使用与维护]]。
<!-- visual-lab:generated:end -->
