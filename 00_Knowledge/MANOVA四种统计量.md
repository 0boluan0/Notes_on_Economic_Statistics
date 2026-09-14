---
aliases:
  - "Wilks、Pillai、Hotelling–Lawley 与 Roy 以不同函数汇总同一组 MANOVA 特征根"
  - "four classical MANOVA statistics"
  - "Wilks、Pillai、Hotelling-Lawley 与 Roy"
student_os: knowledge-atom
atom_id: STAT-MAN-003
atom_set: manova
atom_type: statistic-family
status: source-checked
mastery_state: unassessed
part_of:
  - "[[MANOVA 多元方差分析.canvas|MANOVA 多元方差分析]]"
requires:
  - "[[MANOVA SSCP分解]]"
  - "[[误差SSP秩边界]]"
implies:
  - "[[MANOVA检验校准]]"
related:
  - "[[Pillai稳健性边界]]"
  - "[[多响应联合显著边界]]"
  - "[[响应尺度与线性变换]]"
---

# Wilks、Pillai、Hotelling–Lawley 与 Roy 以不同函数汇总同一组 MANOVA 特征根
<!-- bilingual-en:start -->
*Wilks, Pillai, Hotelling–Lawley, and Roy aggregate the same MANOVA roots in different ways*
<!-- bilingual-en:end -->

> [!summary] 原子统计量族
> 当误差 SSCP $E\succ0$ 时，令 $\lambda_1,\ldots,\lambda_p\ge0$ 为 $E^{-1}H$ 的全部特征根；它们也是对称半正定矩阵 $E^{-1/2}HE^{-1/2}$ 的特征值，因此为非负实数，其中至多 $\operatorname{rank}(H)$ 个非零。四个经典统计量是
> $$
> \Lambda=\prod_{j=1}^p\frac{1}{1+\lambda_j},\quad
> V=\sum_{j=1}^p\frac{\lambda_j}{1+\lambda_j},\quad
> U=\sum_{j=1}^p\lambda_j,\quad
> \Theta=\max_j\lambda_j.
> $$
> 它们依次是 Wilks' lambda、Pillai trace、Hotelling–Lawley trace 和 Roy's greatest root。
> <!-- bilingual-en:start -->
> The four classical criteria are different functions of the same roots of $E^{-1}H$: a product, a bounded sum, an unbounded sum, and the largest root.
> <!-- bilingual-en:end -->

矩阵表达与方向是：

- **Wilks：** $\Lambda=|E|/|E+H|$；越小越反对零假设；
- **Pillai：** $V=\operatorname{tr}\{H(H+E)^{-1}\}$；越大越反对零假设；
- **Hotelling–Lawley：** $U=\operatorname{tr}(E^{-1}H)$；越大越反对零假设；
- **Roy：** $\Theta$ 只取 $E^{-1}H$ 的最大特征根；越大越反对零假设。

四者检验同一个预先指定的多元零假设，却强调不同的根结构。Roy 只看最强方向；Hotelling–Lawley 把全部根线性相加；Pillai 先把每根压到 $[0,1)$；Wilks 把所有 $1+\lambda_j$ 的倒数相乘。因此多个非零根时，它们给出的数值、近似 $F$ 和 $p$ 值可以不同，不能看完结果再挑最显著的一种。

实际选择统计量时，还会用到有范围的比较稳健性证据；这部分由 [[Pillai稳健性边界]] 单独说明，不能混进四个统计量的数学定义。

若只有一个非零根，四个统计量都只是该根的单调变换，排序信息相同。若 $E$ 奇异，上述 $E^{-1}H$ 和普通行列式比的经典推导不能直接使用；广义逆、降维或高维方法会改变定义或校准，不能仍把结果无条件称作这套经典全空间 MANOVA，见 [[误差SSP秩边界]]。

> [!question]- 自检
> 哪个统计量只保留最强的一个响应方向？Wilks 越大还是越小越反对零假设？
>
> **答案：** Roy's greatest root 只取最大根；Wilks' lambda 越小越反对零假设。

## 来源与核验

- [[01_Math/04_多元统计分析/06_比较多个均值向量comparisons of multivariate mean vectors.md#1.5.4. Wilks Lambda 检验|本地多元统计课程 §1.5.4]]：核对 Wilks 行列式比和方向。
- [Penn State STAT 505, Lesson 8](https://online.stat.psu.edu/stat505/Lesson08)：核对四种统计量的矩阵表达与拒绝方向。
- [SAS GLM, Multivariate Analysis of Variance](https://support.sas.com/documentation/cdl/en/statug/66103/HTML/default/statug_glm_details45.htm)：核对四者都是 $E^{-1}H$ 或 $(E+H)^{-1}H$ 特征值的函数，以及各自的 trace/determinant/root 定义。
- [[响应尺度与线性变换]]：核对可逆响应同余变换下完整响应空间的广义特征根不变。
<!-- bilingual-en:start -->
- Penn State and the SAS GLM documentation verify the four criteria and their directions; comparative robustness is owned by the linked boundary atom.
<!-- bilingual-en:end -->
