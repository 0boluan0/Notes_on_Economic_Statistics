---
aliases:
  - "二元结果的条件方差为 p(X)[1-p(X)]，所以 LPM 通常异方差但在特殊概率结构下未必异方差"
  - Heteroskedasticity in the LPM
student_os: knowledge-atom
atom_id: ECON-BIN-005
atom_set: binary-outcome-models
atom_type: inference-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[二元结果条件方差]]"
  - "[[LPM的OLS估计]]"
related:
  - "[[标准误口径匹配]]"
  - "[[OLS经典协方差估计]]"
part_of:
  - "[[二元结果模型.canvas|二元结果模型]]"
---

# 二元结果的条件方差为 p(X)[1-p(X)]，所以 LPM 通常异方差但在特殊概率结构下未必异方差

<!-- bilingual-en:start -->
*A binary outcome has conditional variance $p(X)[1-p(X)]$, so the LPM is usually heteroskedastic but need not be under special probability structures.*
<!-- bilingual-en:end -->

> [!summary] 精确边界
> 对正确刻画条件均值的 LPM，误差 $u=Y-p(X)$ 满足
> $$
> \operatorname{Var}(u\mid X)=p(X)[1-p(X)].
> $$
> 当 $p(X)$ 随 $X$ 以一般方式变化时，这个量也变化，所以 LPM 通常异方差；但“二元结果必然异方差”过强，因为不同概率可能给出相同的 $p(1-p)$。
> <!-- bilingual-en:start -->
> For an LPM that correctly specifies the conditional mean, the error $u=Y-p(X)$ has the conditional variance shown above. When $p(X)$ varies with $X$ in a general way, this variance also varies, so an LPM is usually heteroskedastic. The stronger claim that “a binary outcome must be heteroskedastic,” however, is false because different probabilities can produce the same value of $p(1-p)$.
> <!-- bilingual-en:end -->

最简单的同方差特例是 $p(X)$ 恒定。另一个边界例子是样本支持上 $p(X)$ 只取 $0.2$ 与 $0.8$：两处方差都为 $0.16$，尽管条件概率不同。因此异方差是 LPM 的典型结构，不是仅由 $Y\in\{0,1\}$ 推出的逻辑必然。

<!-- bilingual-en:start -->
The simplest homoskedastic special case is a constant $p(X)$. Another boundary case occurs when $p(X)$ takes only the values $0.2$ and $0.8$ on the sample support: the variance is $0.16$ at both points even though the conditional probabilities differ. Heteroskedasticity is therefore typical of the LPM, not a logical necessity implied by $Y\in\{0,1\}$ alone.
<!-- bilingual-en:end -->

实务中不应靠这一特殊可能性使用经典同方差标准误。独立横截面通常报告异方差稳健标准误；同一人、公司或地区内误差相关时，标准误还要按数据依赖和处理分配层级聚类。稳健或聚类标准误只改变不确定性估计，不会把越界拟合值拉回单位区间，也不会修复错误的概率曲线。

<!-- bilingual-en:start -->
In practice, this special possibility does not justify using classical homoskedastic standard errors. Independent cross-sectional data are usually reported with heteroskedasticity-robust standard errors. When errors are correlated within a person, firm, or region, standard errors must also be clustered at a level consistent with the data dependence and treatment assignment. Robust or clustered standard errors change only the uncertainty estimate; they neither pull out-of-range fitted values back into the unit interval nor repair a misspecified probability curve.
<!-- bilingual-en:end -->

> [!question]- 自检
> 若两类人的成功概率分别为 0.3 和 0.7，二元误差的条件方差是否不同？
>
> **答案：** 不同概率却有相同方差：$0.3\times0.7=0.7\times0.3=0.21$。这正是“通常异方差”不能写成“逻辑上必然异方差”的原因。
> <!-- bilingual-en:start -->
> If two groups have success probabilities of 0.3 and 0.7, do their binary errors have different conditional variances?
>
> **Answer:** No. The probabilities differ, but the variances are equal: $0.3\times0.7=0.7\times0.3=0.21$. This is precisely why “usually heteroskedastic” cannot be replaced by “logically must be heteroskedastic.”
> <!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- Wooldridge, *Introductory Econometrics: A Modern Approach*, 6th ed., §7.5：核对 LPM 中 $\operatorname{Var}(u\mid X)=p(X)[1-p(X)]$ 及稳健推断建议。
<!-- bilingual-en:start -->
- Wooldridge, *Introductory Econometrics: A Modern Approach*, 6th ed., §7.5: verifies $\operatorname{Var}(u\mid X)=p(X)[1-p(X)]$ for the LPM and the recommendation to use robust inference.
<!-- bilingual-en:end -->
- [[二元结果条件方差]]：复用二元变量方差公式；本卡据公式给出 $p$ 与 $1-p$ 方差相同的边界反例。
<!-- bilingual-en:start -->
- [[二元结果条件方差|Conditional variance of a binary outcome]]: supplies the binary-variable variance formula; this card uses it to construct the boundary counterexample in which $p$ and $1-p$ yield the same variance.
<!-- bilingual-en:end -->
- [[标准误口径匹配]]：复用独立、聚类与其他依赖结构下标准误选择的通用规则。
<!-- bilingual-en:start -->
- [[标准误口径匹配|Matching the standard-error method to the dependence structure]]: supplies the general rules for choosing standard errors under independence, clustering, and other dependence structures.
<!-- bilingual-en:end -->
