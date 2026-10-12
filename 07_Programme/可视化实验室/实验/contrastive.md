<!-- visual-lab:generated:start -->
# InfoNCE如何让正负样本竞争

<!-- bilingual-en:start -->
*Finite-vector InfoNCE & temperature*
<!-- bilingual-en:end -->

直接计算单位向量的点积、Softmax分母和单Query损失，比较温度与困难负样本的作用。

> [!question] 先预测，再操作
> 正样本已很相似，加入一个同样相似的负样本会怎样？

<!-- visual-lab:film:start -->
## 动画：分步观察

<!-- bilingual-en:start -->
*Animation: follow the mechanism*
<!-- bilingual-en:end -->

![[07_Programme/可视化实验室/media/contrastive.mp4]]

约 38 秒，中英字幕。可暂停、点击进度条定位或全屏观看。

<!-- bilingual-en:start -->
About 38 seconds, with Chinese and English captions. Pause, seek on the timeline or watch full screen.
<!-- bilingual-en:end -->

- **0:00** 先把查询与各个候选方向摆出来
- **0:09** 相似度除以温度，再在候选间归一化
- **0:21** 目标只奖励分给正样本的概率
- **0:28** 温度放大已有排序，不保证排序正确

<!-- bilingual-en:start -->
- **0:00** Compare a query with positive and negative keys.
- **0:09** Scale similarities, then normalize across candidates.
- **0:21** The loss rewards probability assigned to the positive key.
- **0:28** Temperature sharpens the existing ranking, including mistakes.
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
id: contrastive
height: 980
```

## 观察路线

1. 第一图比较Query与各向量的夹角，预测哪个点积最大。
2. 查看所有候选共用的分母，手算只有一个负样本的概率。
3. 让第1负样本与正样本重合，再降低温度，观察不可区分性是否消失。
4. 旋转Query阅读整条损失曲线；区分自由Query梯度与每度旋转导数。

## 公式与适用条件

$$
s_j=q^Tk_j/\tau,\quad P_j=\frac{e^{s_j}}{\sum_\ell e^{s_\ell}},\quad L=-\log P_+,\quad \nabla_qL=\frac{\sum_jP_jk_j-k_+}{\tau}
$$

- 二维单位向量，一个正样本、最多三个手工负样本；正样本固定0°，其余负样本固定120°和−120°。
- 没有训练、批次采样、增强、stop-gradient或表征坍塌的性能实验。
- 不能把固定向量单次损失当作对数据分布互信息的有效估计。

## 回到知识点

- [[00_Knowledge/Softmax|Softmax]]：正负候选共用指数归一化分母；温度改变分数差异，不自动构成校准概率。

[[07_Programme/可视化实验室/00_实验库入口|回到实验库]] · [[07_Programme/可视化实验室/实验/embeddings|嵌入向量的角度、长度与相似度]] · [[07_Programme/可视化实验室/实验/attention|注意力从分数到读取向量]] · [[07_Programme/可视化实验室/实验/activations|激活函数与局部梯度]]

## 来源与核验

- [van den Oord, Li & Vinyals · CPC (2018)](https://arxiv.org/pdf/1807.03748) — §2.3，式(4)。支持：包含一个正样本及多个负样本的归一化对比损失；本页手工候选不用于声称互信息估计。

实验使用明确给定的公式、合成数据或固定小型示例。代码、可复现检查和逐轮修订记录见 [[07_Programme/可视化实验室/01_使用与维护|使用与维护]]。
<!-- visual-lab:generated:end -->
