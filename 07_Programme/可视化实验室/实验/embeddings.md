<!-- visual-lab:generated:start -->
# 嵌入向量的角度、长度与相似度

<!-- bilingual-en:start -->
*Embedding geometry without semantic shortcuts*
<!-- bilingual-en:end -->

单独控制两个向量的长度和夹角，看内积、余弦与欧氏距离分别保留什么信息。

> [!question] 先预测，再操作
> 把向量放大两倍，哪一种相似度会保持不变？

<!-- visual-lab:film:start -->
## 动画：分步观察

<!-- bilingual-en:start -->
*Animation: follow the mechanism*
<!-- bilingual-en:end -->

![[07_Programme/可视化实验室/media/embeddings.mp4]]

约 32 秒，中英字幕。可暂停、点击进度条定位或全屏观看。

<!-- bilingual-en:start -->
About 32 seconds, with Chinese and English captions. Pause, seek on the timeline or watch full screen.
<!-- bilingual-en:end -->

- **0:00** 相似，是指方向还是长度？
- **0:04** 先除以各自长度，只保留方向
- **0:15** 把长度放回来，才得到内积
- **0:23** 零向量没有方向

<!-- bilingual-en:start -->
- **0:00** Alignment and magnitude are different properties.
- **0:04** Normalize both vectors to isolate direction.
- **0:15** The dot product restores both magnitudes.
- **0:23** Cosine is undefined for a zero vector.
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
id: embeddings
height: 980
```

## 观察路线

1. 固定夹角只改变 b 的长度，先预测哪个指标不变。
2. 把两个长度都设为 1，比较距离平方与 2−2cos θ。
3. 把夹角从 0 转到 180°，看内积符号和距离变化。
4. 设置零长度，确认余弦显示未定义，再解释为何不能把它记作 0。

## 公式与适用条件

$$
a^Tb=\|a\|\|b\|\cos\theta,\quad \mathrm{cos}(a,b)=\frac{a^Tb}{\|a\|\|b\|},\quad\|a-b\|^2=\|a\|^2+\|b\|^2-2a^Tb
$$

- a=(‖a‖,0)，b=‖b‖(cos θ,sin θ)，长度非负。
- 没有降维投影误差；图直接显示完整二维向量。
- 零向量没有方向，其余弦分母为零。

## 回到知识点

- [[00_Knowledge/Token嵌入|Token嵌入]]：查表得到表示向量后，才可以比较其几何；实验不把 token ID 距离或合成箭头当作训练语义证据。
- [[00_Knowledge/正交|正交]]：内积为零的代数定义；区分非零向量的 90° 夹角和零向量的余弦未定义。

[[07_Programme/可视化实验室/00_实验库入口|回到实验库]] · [[07_Programme/可视化实验室/实验/projection|投影、残差与最短距离]] · [[07_Programme/可视化实验室/实验/attention|注意力从分数到读取向量]] · [[07_Programme/可视化实验室/实验/positional|位置编码如何表达相对距离]]

## 来源与核验

- [Mikolov et al. · Efficient Estimation of Word Representations](https://arxiv.org/pdf/1301.3781) — §4.1 Task Description。支持：学习到的词向量可用余弦寻找邻近表示；本页仅解释合成向量的几何，不重现词向量训练或语义评价。
- [CMU · Information Retrieval and Similarity Search](https://stat.cmu.edu/~cshalizi/dm/20/lectures/24/lecture-24.html) — §4 Measuring Similarity; §4.1 normalizing vectors。支持：向量距离、长度归一化及表示选择的限制。

实验使用明确给定的公式、合成数据或固定小型示例。代码、可复现检查和逐轮修订记录见 [[07_Programme/可视化实验室/01_使用与维护|使用与维护]]。
<!-- visual-lab:generated:end -->
