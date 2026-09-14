---
aliases:
  - "Wald 检验按 R V̂(β̂) R' 标准化 Rβ̂-r 来联合检验线性限制"
  - Wald joint test for linear restrictions
  - 回归 Wald 检验
student_os: knowledge-atom
atom_id: ECON-OLS-021
atom_set: regression-inference
atom_type: procedure
status: source-checked
mastery_state: unassessed
part_of:
  - "[[回归推断.canvas|回归推断]]"
  - "[[回归模型比较与选择.canvas|回归模型比较与选择]]"
requires:
  - "[[线性组合推断]]"
  - "[[标准误口径匹配]]"
leads_to:
  - "[[回归联合显著边界]]"
related:
  - "[[回归t检验]]"
  - "[[嵌套模型F检验]]"
  - "[[Chow结构稳定性检验]]"
---

# Wald 检验按 R V̂(β̂) R' 标准化 Rβ̂-r 来联合检验线性限制
<!-- bilingual-en:start -->
*A Wald test jointly tests linear restrictions by standardizing Rβ̂-r with R V̂(β̂) R'*
<!-- bilingual-en:end -->

> [!summary] 原子方法
> 对 $q$ 个独立线性限制 $H_0:R\beta=r$，令 $\widehat V=\widehat{\operatorname{Var}}(\hat\beta)$，Wald 统计量为
> $$
> W=(R\hat\beta-r)'[R\widehat VR']^{-1}(R\hat\beta-r).
> $$
> 它用限制估计误差自身的协方差来衡量无约束估计离原假设有多远，因此会同时使用所有限制之间的相关性。
>
> <!-- bilingual-en:start -->
> The Wald statistic measures the distance between the unrestricted estimate and the null in the covariance geometry of the restrictions. It therefore uses both marginal uncertainty and covariance among restrictions.
> <!-- bilingual-en:end -->

矩阵 $R$ 必须代表 $q$ 个线性独立限制；否则 $R\widehat VR'$ 可能不可逆，实际自由度小于写出的行数。若检验“$\beta_2=0$ 且 $\beta_3=0$”，$R$ 选择相应两个坐标；若检验“$\beta_1+\beta_2=1$”，一行 $R$ 同时给两个系数权重。

在常见正则条件下，原假设下 $W\xrightarrow{d}\chi_q^2$。软件也可能报告缩放后的 F 形式，例如 $F=W/q$ 再配合有限样本或聚类自由度修正；此时必须一起报告统计量形式、分子/分母自由度和使用的 $\widehat V$，不能只抄一个“Wald F”。
<!-- bilingual-en:start -->
Under standard regularity conditions, $W$ is asymptotically $\chi_q^2$. Software may instead report a scaled and degrees-of-freedom-adjusted F form; the covariance estimator and calibration must be reported with it.
<!-- bilingual-en:end -->

Wald 的优势是只需无约束估计，并且可以直接使用异方差稳健、HAC 或聚类协方差。这个灵活性也意味着协方差口径不能省略：同一 $R\hat\beta-r$ 配上不同 $\widehat V$ 会得到不同检验。若标准误结构选错，联合统计量也会错；若系数本身因内生性而不识别，Wald 计算得再精确也没有修复识别问题。

当 $q=1$ 且使用与单限制 t 检验相同的 $\widehat V$ 时，$W=t^2$。若把 $W$ 改报为 F，是否数值相同还取决于软件的缩放和自由度约定。

> [!question]- 自检
> 两个限制的各自标准误不变，但它们的估计协方差改变。Wald 联合检验会不会改变？
>
> **答案：** 会。$R\widehat VR'$ 的非对角元改变了联合方向上的距离；逐项 t 值相同不代表联合统计量相同。

## 来源与核验

- [[02_Economy/01_Econometrics/05_多元回归模型的矩阵表达.md#4.2. Wald 检验|本地课程：Wald 检验]]：核对矩阵统计量与渐近 $\chi_q^2$ 参考分布。
- [[线性组合推断]]：核对 $R\widehat VR'$ 的协方差传播和独立限制条件。
- [[标准误口径匹配]]：核对 Wald 检验必须与所选协方差估计口径一起解释。
<!-- bilingual-en:start -->
- The local matrix-regression section verifies the Wald form and asymptotic reference distribution; the linked atoms supply covariance propagation and dependence-structure boundaries.
<!-- bilingual-en:end -->
