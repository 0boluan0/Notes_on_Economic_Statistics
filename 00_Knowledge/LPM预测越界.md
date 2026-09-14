---
aliases:
  - "LPM 的线性拟合值可能落在 [0,1] 之外且事后截断不能修复条件均值函数"
  - Out-of-range LPM predictions
student_os: knowledge-atom
atom_id: ECON-BIN-004
atom_set: binary-outcome-models
atom_type: model-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[LPM的OLS估计]]"
related:
  - "[[二元预测概率]]"
leads_to:
  - "[[Logit模型]]"
  - "[[Probit模型]]"
  - "[[二元模型选择]]"
part_of:
  - "[[二元结果模型.canvas|二元结果模型]]"
---

# LPM 的线性拟合值可能落在 [0,1] 之外且事后截断不能修复条件均值函数

<!-- bilingual-en:start -->
*An LPM's linear fitted values may fall outside $[0,1]$, and clipping them afterwards cannot repair the conditional-mean function.*
<!-- bilingual-en:end -->

> [!summary] 失败边界
> LPM 的拟合值 $X'\hat\beta$ 是线性投影值，代数上可以小于 0 或大于 1；事件概率却必须位于 $[0,1]$。把越界值硬截成 0 或 1 只改变输出，不会重新估计系数，也不会恢复正确的条件均值形状或有效推断。
> <!-- bilingual-en:start -->
> The LPM fitted value $X'\hat\beta$ is a linear-projection value and can algebraically be less than 0 or greater than 1, whereas an event probability must lie in $[0,1]$. Hard-clipping an out-of-range value to 0 or 1 changes only the reported output: it neither re-estimates the coefficients nor restores the correct shape of the conditional mean or valid inference.
> <!-- bilingual-en:end -->

例如

<!-- bilingual-en:start -->
For example, consider
<!-- bilingual-en:end -->

$$
\widehat P(default=1)=-0.10+0.015\,debtRatio
$$

在 `debtRatio=5` 时给出 $-0.025$，在 `debtRatio=80` 时给出 $1.10$。这两个数可作为线性拟合值存在，却不能按概率解释。若样本关注的局部区间内预测均合理，LPM 仍可能是有用的平均关系近似；越界风险要按实际支持范围检查，而不是只看均值处。

<!-- bilingual-en:start -->
This equation gives $-0.025$ at `debtRatio=5` and $1.10$ at `debtRatio=80`. Both numbers can exist as linear fitted values, but neither can be interpreted as a probability. If predictions remain reasonable throughout the local range relevant to the sample, the LPM may still be a useful approximation to an average relationship. The risk of out-of-range predictions must therefore be checked over the actual support, not only at the mean.
<!-- bilingual-en:end -->

事后设 $\tilde p=\min(1,\max(0,X'\hat\beta))$ 会在人为拐点制造平坦区，并且原 OLS 标准误不再描述这个非光滑后处理量。若合法概率或尾部预测是任务核心，应考虑直接使用把指数映射到 $(0,1)$ 的 [[Logit模型]] 或 [[Probit模型]]，再做样本外检查。

<!-- bilingual-en:start -->
Defining $\tilde p=\min(1,\max(0,X'\hat\beta))$ after estimation creates flat regions at artificial kinks, and the original OLS standard errors no longer describe this nonsmooth post-processed quantity. If valid probabilities or tail predictions are central to the task, consider using a [[Logit模型|logit model]] or [[Probit模型|probit model]], which maps the index directly into $(0,1)$, and then evaluate it out of sample.
<!-- bilingual-en:end -->

> [!question]- 自检
> 某 LPM 只有 2% 的训练样本预测越界，是否足以宣布模型可以用于个体风险定价？
>
> **答案：** 不足。还要检查越界出现在哪些支持区域、部署样本的协变量范围、概率校准和样本外损失；训练内比例本身不是用途许可。
> <!-- bilingual-en:start -->
> Only 2% of an LPM's training-sample predictions fall outside the unit interval. Is that enough to conclude that the model can be used to price individual risk?
>
> **Answer:** No. One must also examine where on the support the violations occur, the covariate range in the deployment population, probability calibration, and out-of-sample loss. The in-sample percentage alone does not establish fitness for that use.
> <!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- Wooldridge, *Introductory Econometrics: A Modern Approach*, 6th ed., §7.5：核对 LPM 拟合值不受单位区间约束这一结构性限制。
<!-- bilingual-en:start -->
- Wooldridge, *Introductory Econometrics: A Modern Approach*, 6th ed., §7.5: verifies the structural limitation that LPM fitted values are not constrained to the unit interval.
<!-- bilingual-en:end -->
- [[分类概率校准]] 与 [[分类验证与泄漏]]：复用概率模型必须在未参与拟合的数据和目标部署总体中验证的规则。
<!-- bilingual-en:start -->
- [[分类概率校准|Probability calibration for classifiers]] and [[分类验证与泄漏|classification validation and leakage]]: supply the rule that a probability model must be evaluated on data not used for fitting and in the target deployment population.
<!-- bilingual-en:end -->
