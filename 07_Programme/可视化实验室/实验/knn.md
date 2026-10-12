<!-- visual-lab:generated:start -->
# 最近邻如何决定分类

<!-- bilingual-en:start -->
*KNN, local votes & distance*
<!-- bilingual-en:end -->

在二维 XOR 数据上移动查询点，查看实际邻居和局部投票；再改变 k 与距离尺度。

> [!question] 先预测，再操作
> 只给一个变量换单位，最近邻分类为何就可能改变？

<!-- visual-lab:film:start -->
## 动画：分步观察

<!-- bilingual-en:start -->
*Animation: follow the mechanism*
<!-- bilingual-en:end -->

![[07_Programme/可视化实验室/media/knn.mp4]]

约 33 秒，中英字幕。可暂停、点击进度条定位或全屏观看。

<!-- bilingual-en:start -->
About 33 seconds, with Chinese and English captions. Pause, seek on the timeline or watch full screen.
<!-- bilingual-en:end -->

- **0:00** 一个新点，先找谁离它最近
- **0:04** 按距离，选出同一批 K 个邻居
- **0:16** 把邻居标签平均，得到局部正类比例
- **0:23** 距离的定义，决定谁有投票权

<!-- bilingual-en:start -->
- **0:00** A new query first needs a neighborhood.
- **0:04** Select the K nearest observations by distance.
- **0:16** Average the selected labels into a local class proportion.
- **0:23** The distance metric determines who votes.
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
id: knn
height: 980
```

## 观察路线

1. 固定种子与 k，移动查询点，先预测哪几条连接会改变，再对照邻居标签。
2. 只改变 k=1→5→15，比较局部突变与平滑；保持同一份观测。
3. 只改变第二维权重，解释邻居次序变化；图的坐标仍是原单位。
4. 增加标签噪声并换几个种子，区分真实规律、采样空白和错误标签。

## 公式与适用条件

$$
d_s(x,z)^2=(x_1-z_1)^2+s^2(x_2-z_2)^2,\qquad\hat p(x)=\frac1k\sum_{i\in N_k(x)}y_i
$$

- 使用有限二维样本、加权欧氏距离、均匀投票；不是距离加权或近似搜索。
- k≤n；本页 k≤15、n≥20。两类标签由象限 XOR 加独立翻转噪声生成。
- 固定数据的网格预测仅作可视化，不能证明泛化；邻居比例也不自动校准为真实后验。

## 回到知识点

- [[00_Knowledge/聚类尺度|聚类尺度]]：本页距离权重 s 的平方进入第二坐标距离，演示原子“缩放等价于重新加权”的同一恒等式；它跨任务应用于分类，未将 KNN 称为聚类。

[[07_Programme/可视化实验室/00_实验库入口|回到实验库]] · [[07_Programme/可视化实验室/实验/tree|决策树怎样划分空间]] · [[07_Programme/可视化实验室/实验/kde|带宽怎样改变密度估计]] · [[07_Programme/可视化实验室/实验/pca|PCA、尺度与重构]]

## 来源与核验

- [scikit-learn · Nearest Neighbors](https://scikit-learn.org/stable/modules/neighbors.html) — §1.6.2 Nearest Neighbors Classification; §1.6.1 tie warning。支持：近邻多数投票、距离选择与等距边界；本项目独立实现穷举距离计算。

实验使用明确给定的公式、合成数据或固定小型示例。代码、可复现检查和逐轮修订记录见 [[07_Programme/可视化实验室/01_使用与维护|使用与维护]]。
<!-- visual-lab:generated:end -->
