---
aliases:
  - "LPM 通常用 OLS 估计；若条件概率不是线性函数，OLS 仍给出对所列回归量的最佳线性投影"
  - Estimating the LPM by OLS
student_os: knowledge-atom
atom_id: ECON-BIN-023
atom_set: binary-outcome-models
atom_type: estimator
status: source-checked
mastery_state: unassessed
requires:
  - "[[线性概率模型]]"
  - "[[OLS正规方程]]"
related:
  - "[[条件均值与线性投影]]"
  - "[[标准误口径匹配]]"
leads_to:
  - "[[LPM系数]]"
  - "[[LPM预测越界]]"
  - "[[LPM异方差]]"
part_of:
  - "[[二元结果模型.canvas|二元结果模型]]"
---

# LPM 通常用 OLS 估计；若条件概率不是线性函数，OLS 仍给出对所列回归量的最佳线性投影

<!-- bilingual-en:start -->
*The LPM is usually estimated by OLS; if the conditional probability is not linear, OLS still gives its best linear projection on the included regressors.*
<!-- bilingual-en:end -->

> [!summary] 估计器入口
> 对样本 $(Y_i,X_i)$，LPM 使用与普通线性回归相同的平方损失：
> $$
> \hat\beta=\arg\min_b\sum_{i=1}^n(Y_i-X_i'b)^2.
> $$
> 因此它共享 OLS 的正规方程和秩条件。“用 OLS 求系数”是估计规则，不是 [[线性概率模型|线性概率形状]] 本身的定义。
> <!-- bilingual-en:start -->
> For a sample $(Y_i,X_i)$, the LPM uses the same squared-loss criterion as an ordinary linear regression. It therefore shares the OLS normal equations and rank conditions. Using OLS to obtain the coefficients is an estimation rule, not the definition of the [[线性概率模型|linear probability shape]] itself.
> <!-- bilingual-en:end -->

若真实条件概率确实满足 $P(Y=1\mid X)=X'\beta$，并且所需的外生性和样本条件成立，OLS 针对这个线性条件均值参数。若真实 $P(Y=1\mid X)$ 是弯曲的，总体 OLS 系数改为解

$$
\beta^{LP}=\arg\min_b E[(Y-X'b)^2],
$$

即在已列回归量和其实际分布上，对条件概率的最佳线性投影。这个投影仍可以是有用的平均线性摘要，但它不声称 $X'\beta^{LP}$ 在每个协变量位置都是真实概率。

<!-- bilingual-en:start -->
If the true conditional probability satisfies $P(Y=1\mid X)=X'\beta$ and the required exogeneity and sampling conditions hold, OLS targets that linear conditional-mean parameter. If the true probability is curved, the population coefficient instead solves the best-linear-projection problem above. This projection can remain a useful average linear summary over the included regressors and their actual distribution, but it does not claim that $X'\beta^{LP}$ is the true probability at every covariate value.
<!-- bilingual-en:end -->

> [!example] 同一个估计规则，两种总体含义
> 若就业概率在经验较低时快速上升、随后逐渐变平，用 OLS 拟合一条直线仍会得到唯一的样本斜率。该斜率摘要样本支持上的平均线性关系，却不是每一年经验都共享的真实概率效应。
> <!-- bilingual-en:start -->
> Suppose employment probability rises quickly at low experience and then flattens. OLS still fits a unique sample line. Its slope summarizes an average linear relationship over the supported sample, but it is not the true probability effect of every additional year of experience.
> <!-- bilingual-en:end -->

估计出 $\hat\beta$ 以后，还要分别处理系数解释、预测越界和误差方差。[[LPM系数]] 规定何时能直读概率斜率；[[LPM预测越界]] 检查拟合值是否仍能当作概率；[[LPM异方差]] 处理二元误差的推断边界。一项的修复不会自动修复另外两项。

<!-- bilingual-en:start -->
After obtaining $\hat\beta$, coefficient interpretation, out-of-range predictions, and the binary error variance remain separate questions. [[LPM系数|LPM coefficients]] states when a slope can be read directly on the probability scale; [[LPM预测越界|the range boundary]] checks whether a fitted value can still be treated as a probability; and [[LPM异方差|LPM heteroskedasticity]] handles inference. Repairing one does not repair the other two.
<!-- bilingual-en:end -->

## 来源与核验

- Wooldridge, *Introductory Econometrics: A Modern Approach*, 6th ed., §7.5：核对 LPM 的 OLS 估计以及线性条件均值未必正确时的最佳线性投影口径。
<!-- bilingual-en:start -->
- Wooldridge, *Introductory Econometrics: A Modern Approach*, 6th ed., §7.5: verifies OLS estimation of the LPM and the best-linear-projection interpretation when the conditional probability is not exactly linear.
<!-- bilingual-en:end -->
- [[OLS正规方程]] 与 [[条件均值与线性投影]]：分别核对样本估计代数与总体投影对象。
<!-- bilingual-en:start -->
- [[OLS正规方程|The OLS normal equations]] and [[条件均值与线性投影|the conditional-mean versus linear-projection distinction]] separately verify the sample algebra and population target.
<!-- bilingual-en:end -->
