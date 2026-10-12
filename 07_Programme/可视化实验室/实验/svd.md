<!-- visual-lab:generated:start -->
# 奇异值分解与低秩重构

<!-- bilingual-en:start -->
*SVD geometry & low-rank reconstruction*
<!-- bilingual-en:end -->

实际构造 UΣVᵀ，把单位圆映成椭圆，再逐项截断；重复奇异值时共同转动两组基，直接检查矩阵不变。

> [!question] 先预测，再操作
> 如果两个奇异值相等，图中的“第一方向”还是矩阵唯一决定的吗？

<!-- visual-lab:film:start -->
## 动画：分步观察

<!-- bilingual-en:start -->
*Animation: follow the mechanism*
<!-- bilingual-en:end -->

![[07_Programme/可视化实验室/media/svd.mp4]]

约 48 秒，中英字幕。可暂停、点击进度条定位或全屏观看。

<!-- bilingual-en:start -->
About 48 seconds, with Chinese and English captions. Pause, seek on the timeline or watch full screen.
<!-- bilingual-en:end -->

- **0:00** 一个矩阵，可以拆成三次动作
- **0:04** 先用 Vᵀ 改变输入方向
- **0:12** 再按两个奇异值分别伸缩
- **0:20** 最后用 U 放到输出方向
- **0:28** 秩一近似，把第二个方向压到零
- **0:38** 截断误差，对应被舍弃的方向

<!-- bilingual-en:start -->
- **0:00** A matrix acts as rotate, stretch, rotate.
- **0:04** Rotate the input directions with Vᵀ.
- **0:12** Stretch the aligned axes by the singular values.
- **0:20** Finally rotate the result with U.
- **0:28** Rank-one truncation removes the second singular direction.
- **0:38** Truncation error comes from the discarded direction.
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
id: svd
height: 980
```

## 观察路线

1. 先看输入 v₁、v₂，再核对 A vⱼ=σⱼuⱼ；两组坐标方向属于不同空间。
2. 保持矩阵，切换 k=2→1→0，检查像从椭圆降为线段再成为原点。
3. 将 ratio 设为 1，改变 shared，核对矩阵元素不变而显示的基方向改变。
4. 将 ratio 设为 0 或 scale 设为 0，区分截断近似与原矩阵自身秩降低。

## 公式与适用条件

$$
A=U\Sigma V^T=\sum_{j=1}^{2}\sigma_j u_jv_j^T,\qquad A_k=\sum_{j\le k}\sigma_j u_jv_j^T,\qquad \|A-A_k\|_F^2=\sum_{j>k}\sigma_j^2
$$

- 构造 U、V 为二维旋转，σ₁≥σ₂≥0；示例覆盖非负行列式的实 2×2 族，不代表完整矩形或反射情形。
- retained 是保留项数；若某项奇异值为零，保留项数可以大于实际秩。
- shared 同时旋转两组基；仅在两奇异值相等时保证 A 不变。

## 回到知识点

- [[00_Knowledge/奇异值分解|奇异值分解]]：实际 UΣVᵀ 与输入/输出正交基对应，矩阵元素随控件重新计算。
- [[00_Knowledge/奇异值与秩|奇异值与秩]]：σ₂=0 或两奇异值全零直接改变原矩阵秩；保留项数与秩显式区分。
- [[00_Knowledge/截断SVD最佳低秩近似|截断SVD最佳低秩近似]]：实际 A−Aₖ Frobenius 误差与尾部奇异值公式核对，同时显示谱误差。
- [[00_Knowledge/重奇异值下方向不唯一|重奇异值下方向不唯一]]：两奇异值相等时共同旋转 U、V，基改变而 A 不变；该不唯一性有直接控件。

[[07_Programme/可视化实验室/00_实验库入口|回到实验库]] · [[07_Programme/可视化实验室/实验/linear|矩阵怎样改变空间]] · [[07_Programme/可视化实验室/实验/pca|PCA、尺度与重构]] · [[07_Programme/可视化实验室/实验/projection|投影、残差与最短距离]] · [[07_Programme/可视化实验室/实验/kernel|核矩阵如何产生非线性拟合]]

## 来源与核验

- [LAPACK Users’ Guide · SVD](https://www.netlib.org/lapack/lug/node32.html) — Singular Value Decomposition。支持：核对 UΣV* 的尺寸、正交性与非负奇异值。
- [Murray et al. · Randomized Numerical Linear Algebra](https://www.netlib.org/utk/people/JackDongarra/PAPERS/rand-templates.pdf) — §4.1.1 Spectral decompositions, eqs.(4.2)–(4.6), printed pp.65–66。支持：支持截断 SVD 的谱/Frobenius 误差与最佳秩约束近似；二维算例独立复算。

实验使用明确给定的公式、合成数据或固定小型示例。代码、可复现检查和逐轮修订记录见 [[07_Programme/可视化实验室/01_使用与维护|使用与维护]]。
<!-- visual-lab:generated:end -->
