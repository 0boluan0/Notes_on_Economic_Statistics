<!-- visual-lab:generated:start -->
# 线性自编码器与秩一重构

<!-- bilingual-en:start -->
*Linear bottlenecks & PCA reconstruction*
<!-- bilingual-en:end -->

把二维样本编码成一个数再解码，直接旋转瓶颈方向与调整增益，核对 PCA 最优误差。

> [!question] 先预测，再操作
> 降低维数后丢失的，具体是哪一部分平方误差？

<!-- visual-lab:film:start -->
## 动画：分步观察

<!-- bilingual-en:start -->
*Animation: follow the mechanism*
<!-- bilingual-en:end -->

![[07_Programme/可视化实验室/media/autoencoder.mp4]]

约 34 秒，中英字幕。可暂停、点击进度条定位或全屏观看。

<!-- bilingual-en:start -->
About 34 seconds, with Chinese and English captions. Pause, seek on the timeline or watch full screen.
<!-- bilingual-en:end -->

- **0:00** 二维点，要经过一个一维瓶颈
- **0:09** 解码器只能沿一条方向画回来
- **0:16** 对所有点重复，才得到重建误差
- **0:25** 在线性约束下，瓶颈与 PCA 相遇

<!-- bilingual-en:start -->
- **0:00** A two-dimensional point passes through a one-dimensional code.
- **0:09** The decoder reconstructs on a line.
- **0:16** Repeat over the dataset to measure reconstruction error.
- **0:25** A tied linear unit-gain projection connects to PCA.
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
id: autoencoder
height: 980
```

## 观察路线

1. 设 g=1，看每个点与它在直线上的投影之间的残差。
2. 旋转 u，找当前样本的最小损失；它不必恰好等于生成方向 35°。
3. 固定 u 改变 g，观察增益为 0、1、2 的误差关系。
4. 把生成标准差比例设为 1，区分总体各向同性与有限样本仍有方向差异。

## 公式与适用条件

$$
z_i=u^Tx_i,\quad\hat x_i=gu z_i,\quad\|u\|=1,\quad L=\frac1n\sum_i\|x_i-\hat x_i\|^2,\quad L_{\min}=\frac{n-1}{n}\lambda_2(S)
$$

- 80 个 Gaussian 样本先按样本均值中心化，生成方向固定为 35°。
- 编码与解码共享单位方向 u，解码另乘 g；平方误差按每个观测的两坐标平方和平均。
- 样本协方差用 n−1 作分母，图中重构损失用 n；两者要乘 (n−1)/n。

## 回到知识点

- [[00_Knowledge/PCA低秩重构|PCA低秩重构]]：线性平方损失下的最优秩一子空间；核对样本协方差 n−1 与每点重构误差 n 的分母。

[[07_Programme/可视化实验室/00_实验库入口|回到实验库]] · [[07_Programme/可视化实验室/实验/pca|PCA、尺度与重构]] · [[07_Programme/可视化实验室/实验/projection|投影、残差与最短距离]] · [[07_Programme/可视化实验室/实验/vae|VAE 的重参数化与 KL 代价]]

## 来源与核验

- [Goodfellow, Bengio & Courville · Deep Learning](https://www.deeplearningbook.org/contents/autoencoders.html) — §14.1 Undercomplete Autoencoders。支持：线性解码、平方误差下的 PCA 子空间关系；模型容量过大可能只记忆输入。

实验使用明确给定的公式、合成数据或固定小型示例。代码、可复现检查和逐轮修订记录见 [[07_Programme/可视化实验室/01_使用与维护|使用与维护]]。
<!-- visual-lab:generated:end -->
