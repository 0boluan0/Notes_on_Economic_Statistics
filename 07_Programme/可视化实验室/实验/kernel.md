<!-- visual-lab:generated:start -->
# 核矩阵如何产生非线性拟合

<!-- bilingual-en:start -->
*Trained kernel ridge regression*
<!-- bilingual-en:end -->

实际形成 Gram 矩阵并求解核岭回归，比较线性、多项式和 RBF 的函数空间、带宽与正则化。

> [!question] 先预测，再操作
> 预测是核函数的线性组合，为什么仍能对原始输入形成非线性曲线？

<!-- visual-lab:film:start -->
## 动画：分步观察

<!-- bilingual-en:start -->
*Animation: follow the mechanism*
<!-- bilingual-en:end -->

![[07_Programme/可视化实验室/media/kernel.mp4]]

约 67 秒，中英字幕。可暂停、点击进度条定位或全屏观看。

<!-- bilingual-en:start -->
About 67 seconds, with Chinese and English captions. Pause, seek on the timeline or watch full screen.
<!-- bilingual-en:end -->

- **0:00** 把一个观测，当作相似度的中心
- **0:18** 每座小山，乘上一个拟合系数
- **0:39** 把所有贡献，在同一个 x 上相加
- **0:48** 系数不是随手定的，来自同一组线性方程
- **0:57** 局部相似度，经组合产生全局预测

<!-- bilingual-en:start -->
- **0:00** An observation anchors a similarity function.
- **0:18** Each similarity function is scaled by a fitted coefficient.
- **0:39** Sum all weighted kernels at the same query location.
- **0:48** The coefficients solve a regularized linear system.
- **0:57** Local similarities combine into a global predictor.
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
id: kernel
height: 980
```

## 观察路线

1. 固定数据切换核，比较拟合曲线和 Gram 矩阵结构。
2. 在 RBF 模式移动锚点，解释一个局部核截面与最终带正负权重的组合不同。
3. 保持核带宽，提高 λ，比较训练误差与函数范数；不要把 λ 当概率。
4. 改变带宽后解释相邻训练点的核值如何改变，再观察实际重拟合。

## 公式与适用条件

$$
K_{ij}=k(x_i,x_j),\quad(K+\lambda I)\alpha=y,\quad\hat f(x)=\sum_i\alpha_i k(x,x_i),\quad\min_f\sum_i(y_i-f(x_i))^2+\lambda\|f\|_{\mathcal H}^2
$$

- 一维输入与平方损失；λ严格正，用 Cholesky 解正定系统，不显式求逆。
- 线性核为 1+xz，二次核为 (1+xz)²，RBF 为 exp[−(x−z)²/(2h²)]。
- 没有单独免惩罚截距；常数特征也进入核范数。λ采用损失求和口径，不含 n 因子。

## 回到知识点

- [[00_Knowledge/岭正则化|岭正则化]]：核空间中的平方误差加二范数正则，通过实际训练演示 λ 改变目标与拟合；线性核有显式特征空间对照，常数特征也受罚。
- [[00_Knowledge/Gram矩阵正定判据|Gram矩阵正定判据]]：线性/二次核矩阵是特征向量的 Gram，半正定不自动可逆；实际用 K+λI 保证正定，避免把方阵当可逆。

[[07_Programme/可视化实验室/00_实验库入口|回到实验库]] · [[07_Programme/可视化实验室/实验/svm|软间隔与 hinge 损失]] · [[07_Programme/可视化实验室/实验/regression|共线性与岭回归]] · [[07_Programme/可视化实验室/实验/svd|奇异值分解与低秩重构]] · [[07_Programme/可视化实验室/实验/kde|带宽怎样改变密度估计]]

## 来源与核验

- [Stanford CS229 · Kernel methods](https://cs229.stanford.edu/notes2022fall/main_notes.pdf) — §5.1–§5.4, printed pp.49–59。支持：核作为特征内积、kernel trick 与半正定 Gram 条件。
- [scikit-learn · Kernel ridge regression](https://scikit-learn.org/stable/modules/kernel_ridge.html) — §1.3 Kernel ridge regression。支持：支持平方损失加 L2/RKHS 正则、核空间线性与闭式训练；本页求和口径线性系统独立推导。

实验使用明确给定的公式、合成数据或固定小型示例。代码、可复现检查和逐轮修订记录见 [[07_Programme/可视化实验室/01_使用与维护|使用与维护]]。
<!-- visual-lab:generated:end -->
