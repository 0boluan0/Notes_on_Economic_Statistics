---
aliases:
  - "2SLS 只用工具空间投影出的内生变量变异估计结构系数"
  - "两阶段最小二乘的工具空间投影"
  - "Two-stage least squares as an instrument-space projection"
student_os: knowledge-atom
atom_id: ECON-IV-009
atom_set: endogeneity-iv-gmm
atom_type: estimator-mechanism
status: source-checked
mastery_state: unassessed
part_of: "[[内生性与工具变量.canvas|内生性与工具变量]]"
requires:
  - "[[工具变量有效条件]]"
  - "[[识别阶数与秩]]"
related:
  - "[[IV比率估计]]"
  - "[[样本正交与总体外生性]]"
leads_to:
  - "[[2SLS标准误]]"
  - "[[弱工具推断]]"
  - "[[GMM矩条件]]"
---

# 2SLS 只用工具空间投影出的内生变量变异估计结构系数
<!-- bilingual-en:start -->
*Two-stage least squares estimates structural coefficients using only regressor variation projected onto the instrument space*
<!-- bilingual-en:end -->

> [!summary] 原子方法
> 在线性结构方程
> $$
> y=X\beta+u
> $$
> 中，令 $Z$ 包含排除工具、结构方程中的全部外生控制变量和常数项（若有）。删去完全共线的列后，设 $\operatorname{rank}(Z)=L$，并令
> $$
> P_Z=Z(Z'Z)^{-1}Z'.
> $$
> 若 $X$ 有 $K$ 列且
> $$
> \operatorname{rank}(P_ZX)=K,
> $$
> 即样本中的 $X'P_ZX$ 可逆，则 2SLS 估计量为
> $$
> \hat\beta_{2SLS}=(X'P_ZX)^{-1}X'P_Zy.
> $$
> 对应的总体识别条件是 $\operatorname{rank}E[Z_iX_i']=K$。$P_ZX$ 是 $X$ 在工具空间上的投影；对本身已在 $Z$ 中的外生控制列，$P_ZX_j=X_j$，真正被工具投影替换的是内生解释变量列。
>
> <!-- bilingual-en:start -->
> With $Z$ containing the excluded instruments, every included exogenous regressor, and any constant, 2SLS exists when $X'P_ZX$ has full rank. Included exogenous columns project to themselves; the endogenous columns are replaced by their instrument-space projections. Population identification requires $E[Z_iX_i']$ to have full column rank.
> <!-- bilingual-en:end -->

“两阶段”给出直觉，但实际估计应被看成一个整体。第一阶段把每个内生解释变量对 $Z$ 做投影，得到 $\hat X=P_ZX$；第二阶段的系数等于用 $\hat X$ 回归 $y$ 的系数。等价地，2SLS 让下面的投影正交条件成立：

$$
X'P_Z(y-X\hat\beta_{2SLS})=0.
$$

这不是说 $\hat X$ 已变成“没有误差的真实处理”，也不是说第一阶段有预测力就自动产生因果解释。因果解释仍依赖 [[工具变量有效条件|工具的相关性、外生性和排除限制]]；矩阵可逆还依赖 [[识别阶数与秩|足够的识别秩]]。
<!-- bilingual-en:start -->
The two stages are an interpretation of one estimator. Projecting $X$ onto $Z$ does not turn fitted values into an error-free treatment or validate the instrument. Causal interpretation still comes from the maintained instrument assumptions, while existence of the estimator comes from the rank condition.
<!-- bilingual-en:end -->

所有进入结构方程的外生控制变量都应同时进入 $Z$。否则投影可能遗漏本应保留的外生方向，手工两阶段也不再对应目标结构方程的标准 2SLS。若 $Z$ 与 $X$ 张成同一列空间，则 $P_ZX=X$，2SLS 退化为 OLS；这也说明二者的差别来自工具提供的不同变异，而不是“跑了两次回归”本身。
<!-- bilingual-en:start -->
Every included exogenous regressor belongs in the instrument matrix as its own instrument. If $Z$ and $X$ span the same column space, $P_ZX=X$ and 2SLS collapses to OLS; the distinction is the source of variation used, not the number of regressions typed into software.
<!-- bilingual-en:end -->

在排除工具数量多于内生解释变量数量的过度识别情形中——等价地，在 $Z$ 与 $X$ 都包含同一外生控制集且满列秩时，$Z$ 的列数大于 $X$ 的列数——样本里的每个工具矩条件一般无法同时精确为零。2SLS 以 $P_Z$ 隐含的权重组合这些条件；更一般的组合方式进入 [[GMM矩条件]] 与 [[GMM权重矩阵]]。

> [!question]- 最小自检
> 第一阶段对内生变量的拟合值很准，是否已经证明 2SLS 系数有因果含义？
>
> **答案：** 没有。拟合好只支持相关性和可能的秩；仍须论证工具与结构误差正交，且不会绕过处理直接影响结果。

## 来源与核验

- [[02_Economy/01_Econometrics/09_联立方程模型(内生性).md#4.2.2. 两阶段最小二乘法|本地课程：两阶段步骤]] 与 [[02_Economy/01_Econometrics/09_联立方程模型(内生性).md#4.2.4. IV 估计量公式|本地课程：矩阵公式]]：核对第一阶段、第二阶段和 $P_Z$ 形式。
- MIT OpenCourseWare 14.310x, [Lecture 21: Endogeneity and Instrument Variables](https://ocw.mit.edu/courses/14-310x-data-analysis-for-social-scientists-spring-2023/mit14_310x_s23_week10_lec21.pdf)：核验恰好识别公式、工具矩阵必须包含控制变量，以及过度识别时“把 $X$ 投影到 $Z$”的解释。
- MIT OpenCourseWare 14.382, [Lecture 3: Structural Equations Models and GMM](https://ocw.mit.edu/courses/14-382-econometrics-spring-2017/f6b8cd36eee8df2b259c979a1fd2673e_MIT14_382S17_lec3.pdf)：核验总体矩阵满列秩与线性 IV 识别条件。
- StataCorp, [Two-stage least-squares regression FAQ](https://www.stata.com/support/faqs/statistics/instrumental-variables-regression/)：核验结构方程中的外生变量须作为自身工具进入第一阶段。
