<!-- visual-lab:generated:start -->
# 带宽怎样改变密度估计

<!-- bilingual-en:start -->
*Kernel density & bandwidth*
<!-- bilingual-en:end -->

对固定样本实际求和 Gaussian 核，比较单个核、整体估计和已知总体密度。

> [!question] 先预测，再操作
> 更平滑的曲线一定更接近真实分布吗？

<!-- visual-lab:film:start -->
## 动画：分步观察

<!-- bilingual-en:start -->
*Animation: follow the mechanism*
<!-- bilingual-en:end -->

![[07_Programme/可视化实验室/media/kde.mp4]]

约 39 秒，中英字幕。可暂停、点击进度条定位或全屏观看。

<!-- bilingual-en:start -->
About 39 seconds, with Chinese and English captions. Pause, seek on the timeline or watch full screen.
<!-- bilingual-en:end -->

- **0:00** 一个观测，先放上一座小山
- **0:15** 在同一个 x，把所有小山的高度相加
- **0:23** 带宽改变的是每座小山的宽窄
- **0:29** 平滑曲线，不等于已经发现真实结构

<!-- bilingual-en:start -->
- **0:00** Place one kernel at each observation.
- **0:15** Sum the kernel heights at the same evaluation point.
- **0:23** Bandwidth changes the scale of smoothing.
- **0:29** A smooth estimate is still a smoothing choice.
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
id: kde
height: 980
```

## 观察路线

1. 固定种子和 n，只调整 h，核对原观测没有移动。
2. 对照第二张图前 5 个核及其部分和，解释每个面积 1/n、部分和面积 5/n；它不是完整 KDE。
3. 切到双峰分布，寻找过小与过大带宽各自的问题。
4. 增加 n，再比较估计与真实密度；有限样本曲线仍不会完全重合。

## 公式与适用条件

$$
\hat f_h(x)=\frac1{nh}\sum_{i=1}^n\phi\!\left(\frac{x-X_i}{h}\right),\qquad\int_{\mathbb R}\hat f_h(x)dx=1
$$

- 一维 IID 样本、Gaussian 核、统一正带宽、Euclidean 距离。
- 双峰生成密度为 0.5N(−1.5,0.5²)+0.5N(1.5,0.5²)。
- 画面和数值积分在有限区间，理论归一化针对整个实线。

## 回到知识点

- [[00_Knowledge/概率密度函数|概率密度函数]]：完整 KDE 在实线积分为 1，点高度不等于点概率，带宽可改变峰高。
- [[00_Knowledge/密度换元的尺度因子|密度换元的尺度因子]]：每个核的 1/h 来自线性缩放的逆导数，使各加权核面积固定 1/n。
- [[00_Knowledge/密度的局部概率解释|密度的局部概率解释]]：挑战通过 f(x)Δx 区分局部区间概率、密度高度与量纲。

[[07_Programme/可视化实验室/00_实验库入口|回到实验库]] · [[07_Programme/可视化实验室/实验/sampling|抽样均值、大数与中心极限定理]] · [[07_Programme/可视化实验室/实验/gmm|EM 怎样做软分配]] · [[07_Programme/可视化实验室/实验/knn|最近邻如何决定分类]] · [[07_Programme/可视化实验室/实验/likelihood|似然曲线与最大似然估计]]

## 来源与核验

- [scikit-learn · Density Estimation](https://scikit-learn.org/stable/modules/density.html) — §2.8.2 Kernel Density Estimation。支持：核平均与带宽平滑；本项目使用一维 Euclidean Gaussian 核的解析归一化。

实验使用明确给定的公式、合成数据或固定小型示例。代码、可复现检查和逐轮修订记录见 [[07_Programme/可视化实验室/01_使用与维护|使用与维护]]。
<!-- visual-lab:generated:end -->
