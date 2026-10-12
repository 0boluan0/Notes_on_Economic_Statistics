<!-- visual-lab:generated:start -->
# 熵、交叉熵与 KL 的区别

<!-- bilingual-en:start -->
*Entropy, cross-entropy & support boundaries*
<!-- bilingual-en:end -->

改变三类别的真实分布 P 与报告分布 Q，逐项计算熵、交叉熵、双向 KL，并保留支持不匹配产生的无穷值。

> [!question] 先预测，再操作
> 给真实会发生的类别报告概率零，为什么不是一个“很大但有限”的损失？

<!-- visual-lab:film:start -->
## 动画：分步观察

<!-- bilingual-en:start -->
*Animation: follow the mechanism*
<!-- bilingual-en:end -->

![[07_Programme/可视化实验室/media/entropy.mp4]]

约 37 秒，中英字幕。可暂停、点击进度条定位或全屏观看。

<!-- bilingual-en:start -->
About 37 seconds, with Chinese and English captions. Pause, seek on the timeline or watch full screen.
<!-- bilingual-en:end -->

- **0:00** 预测概率，决定遇到事件时付出多少损失
- **0:11** 真正经常发生的事件，占更大的平均权重
- **0:27** 真实不确定性，加上报告错误的代价

<!-- bilingual-en:start -->
- **0:00** A reported probability sets the surprise of an event.
- **0:11** True event frequencies weight the average loss.
- **0:27** Cross-entropy includes irreducible uncertainty and mismatch.
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
id: entropy
height: 980
```

## 观察路线

1. 保持 P，只改变 Q；先预测哪项只取决于 P，哪两项会变化。
2. 设 Q=P，核对 KL=0、交叉熵等于熵。
3. 给一个 P>0 的类别设置 Q=0，检查指标与图中明确的 ∞ 标记。
4. 切换 nats/bits，区分数值单位变换与分布本身变化。

## 公式与适用条件

$$
H(P)=-\sum_i p_i\log p_i,\quad H(P,Q)=-\sum_i p_i\log q_i,\quad D_{KL}(P\|Q)=\sum_i p_i\log\frac{p_i}{q_i}=H(P,Q)-H(P)
$$

- 有限三类别分布；采用两步分配保证概率非负且总和为 1。
- 0 log 0 采用极限约定为 0；P>0、Q=0 时保留扩展实数 ∞。
- 这里只讨论离散熵；不能把非负性直接照搬到 differential entropy。

## 回到知识点

- [[00_Knowledge/KL参考约束|KL参考约束]]：只关联“约束读取概率比”段中正向离散 KL 的公式与方向：本页真实计算 ∑P log(P/Q)，帮助读懂该惩罚项；没有实现策略优化、奖励或 β 调节。

[[07_Programme/可视化实验室/00_实验库入口|回到实验库]] · [[07_Programme/可视化实验室/实验/softmax|Softmax 温度与交叉熵梯度]] · [[07_Programme/可视化实验室/实验/logistic|概率、阈值与交叉熵]] · [[07_Programme/可视化实验室/实验/likelihood|似然曲线与最大似然估计]] · [[07_Programme/可视化实验室/实验/calibration|准确率之外的概率校准]]

## 来源与核验

- [Dive into Deep Learning · Information Theory](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/information-theory.html) — §22.11.2.2; §22.11.4.1–4.2; §22.11.5.1–5.2。支持：核对离散熵、KL、交叉熵定义、恒等式与正质量对零支持产生∞；未复制示例代码中的 abs/nansum 处理。

实验使用明确给定的公式、合成数据或固定小型示例。代码、可复现检查和逐轮修订记录见 [[07_Programme/可视化实验室/01_使用与维护|使用与维护]]。
<!-- visual-lab:generated:end -->
