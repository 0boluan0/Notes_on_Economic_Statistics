---
aliases:
  - "手工第二阶段 OLS 可以复现 2SLS 点估计，却不能把普通 OLS 标准误当作 2SLS 推断"
  - "手工两阶段回归的标准误边界"
  - "Why manual second-stage OLS standard errors are not 2SLS standard errors"
student_os: knowledge-atom
atom_id: ECON-IV-010
atom_set: endogeneity-iv-gmm
atom_type: inference-boundary
status: source-checked
mastery_state: unassessed
part_of: "[[内生性与工具变量.canvas|内生性与工具变量]]"
requires:
  - "[[2SLS投影]]"
related:
  - "[[标准误口径匹配]]"
  - "[[OLS经典协方差估计]]"
  - "[[弱工具推断]]"
---

# 手工第二阶段 OLS 可以复现 2SLS 点估计，却不能把普通 OLS 标准误当作 2SLS 推断
<!-- bilingual-en:start -->
*A manual second-stage OLS regression can reproduce the 2SLS point estimate, but its ordinary OLS standard errors are not valid 2SLS inference*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> 在同一样本、同一结构方程和完整工具矩阵下，先算 $\hat X=P_ZX$，再用 $\hat X$ 回归 $y$，可以得到
> $$
> \hat\beta=(X'P_ZX)^{-1}X'P_Zy,
> $$
> 即标准 2SLS 点估计。但第二阶段软件默认计算的是“普通 OLS 回归 $y$ 对 $\hat X$”的方差；它使用的拟合残差和抽样公式都不是结构 2SLS 所需对象，因此对应的标准误、$t$ 检验和置信区间不能直接采用。
>
> <!-- bilingual-en:start -->
> Regressing $y$ on $\hat X=P_ZX$ can reproduce the 2SLS coefficient. Ordinary second-stage OLS inference nevertheless uses the wrong residual and sampling formula for the structural estimator, so its standard errors and tests cannot be used as 2SLS inference.
> <!-- bilingual-en:end -->

关键区别可以从残差看清。手工第二阶段 OLS 把

$$
r=y-P_ZX\hat\beta_{2SLS}
$$

当成回归残差；结构方程要估计的扰动却对应

$$
\hat u=y-X\hat\beta_{2SLS}.
$$

2SLS 是用工具给出的变异估计 $X$ 的结构系数，估计完成后仍要把系数放回原结构方程 $y=X\beta+u$。所以点估计可以由投影回归复现，残差平方和和普通 OLS 方差公式却不能照搬。
<!-- bilingual-en:start -->
The projected regression has residual $y-P_ZX\hat\beta$, whereas the structural residual is $y-X\hat\beta$. The latter belongs to the structural equation whose coefficient is being estimated. Equality of point estimates therefore does not imply equality of residual variance estimates or covariance formulas.
<!-- bilingual-en:end -->

在条件同方差且观测独立的线性模型中，经典形式为

$$
\widehat{\operatorname{Var}}(\hat\beta_{2SLS})
=\hat\sigma^2(X'P_ZX)^{-1},
$$

其中 $\hat\sigma^2$ 应由结构残差 $\hat u=y-X\hat\beta_{2SLS}$ 估计。若存在异方差，典型稳健夹心形式把中间项改成

$$
X'P_Z\hat\Omega P_ZX,
\qquad
\hat\Omega=\operatorname{diag}(\hat u_1^2,\ldots,\hat u_n^2),
$$

并在两侧乘以 $(X'P_ZX)^{-1}$。聚类或时间相关还需要相应 cluster/HAC 中间项；这与 [[标准误口径匹配]] 的原则相同。

上面的对角 $\hat\Omega$ 只对应跨观测独立、允许条件异方差的 HC 口径。若观测按 $g=1,\ldots,G$ 聚类，允许簇内任意相关而要求簇间独立，令
$$
A=X'P_ZX,
\qquad Q=(Z'Z)^{-1},
\qquad
\hat B_{\mathrm{cl}}
=\sum_{g=1}^{G}(Z_g'\hat u_g)(Z_g'\hat u_g)'.
$$
相应的 cluster sandwich 为
$$
\widehat{\operatorname{Var}}_{\mathrm{cl}}(\hat\beta)
=A^{-1}X'ZQ\hat B_{\mathrm{cl}}QZ'XA^{-1},
$$
再按所用软件与样本设计说明有限样本修正。常规 cluster 渐近依赖独立簇数量增长；簇很少时需专门校准，聚类层级还应覆盖结构冲击或工具赋值共同变化的层级。HAC 则另外依赖时间排序、核函数和带宽选择。
<!-- bilingual-en:start -->
Under homoskedastic independent errors, the classical covariance is $\hat\sigma^2(X'P_ZX)^{-1}$ with variance estimated from structural residuals. Heteroskedasticity, clustering, or serial dependence requires a matching sandwich middle term rather than ordinary OLS output from the generated regressor. Cluster formulas permit within-cluster dependence only under an appropriate between-cluster asymptotic design; few clusters need separate calibration.
<!-- bilingual-en:end -->

“第一阶段是估出来的”是提醒，但不能被简化成随便给第二阶段标准误加一个修正量。最稳妥的做法是把结构方程、内生变量、工具变量和协方差口径一次性交给正规的 IV/2SLS 实现。即便标准误公式正确，[[弱工具推断|弱识别]] 仍可能使常规正态或 Wald 近似失真；那不是换一个常规协方差估计就能修好的问题。

> [!question]- 最小自检
> 手工两阶段与 `ivregress 2sls` 给出相同系数，是否说明手工第二阶段表里的 $t$ 值也相同且有效？
>
> **答案：** 不说明。系数相同来自投影代数；手工第二阶段按普通 OLS 处理 $P_ZX$ 并使用错误残差，标准误和 $t$ 值没有同样的等价关系。

## 来源与核验

- MIT OpenCourseWare 14.310x, [Lecture 22 transcript](https://ocw.mit.edu/courses/14-310x-data-analysis-for-social-scientists-spring-2023/1go0TK95EP8iDSJ8EKVNbcUp5-UY3r96-_transcript.pdf)：明确说明机械地跑两阶段可得到系数，但标准误和检验会错误，实际应由 IV 程序联合处理。
- StataCorp, [Negative and missing $R^2$ for 2SLS/IV](https://www.stata.com/support/faqs/statistics/two-stage-least-squares/index.html)：核验手工投影回归的残差 $y-P_ZX\hat\beta$ 与正确结构残差 $y-X\hat\beta$ 的区别。
- StataCorp, [Two-stage least-squares regression FAQ](https://www.stata.com/support/faqs/statistics/instrumental-variables-regression/)：核验手工两阶段可复现系数但默认标准误错误，以及经典同方差修正的实现逻辑。
- StataCorp, [`ivregress` Methods and formulas](https://www.stata.com/manuals/rivregress.pdf)：核验 2SLS 的 robust 与 cluster-robust sandwich、结构残差、跨簇独立及有限样本修正口径。
- [[02_Economy/01_Econometrics/09_联立方程模型(内生性).md#4.2.4. IV 估计量公式|本地课程：2SLS 矩阵公式]]：核对本地符号与投影矩阵。
