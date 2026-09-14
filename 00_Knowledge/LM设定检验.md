---
aliases:
  - "LM 设定检验从受限模型残差检查预先指定的遗漏方向"
  - "LM specification test"
  - "拉格朗日乘数设定检验"
student_os: knowledge-atom
atom_id: ECON-SPEC-006
atom_type: method
status: source-checked
mastery_state: unassessed
part_of:
  - "[[回归模型设定与函数形式.canvas|回归模型设定与函数形式]]"
requires:
  - "[[残差图边界]]"
contrasts_with:
  - "[[RESET检验边界]]"
---

# LM 设定检验从受限模型残差检查预先指定的遗漏方向
<!-- bilingual-en:start -->
*An LM specification test uses restricted-model residuals to check prespecified omitted directions*
<!-- bilingual-en:end -->

> [!summary] 原子方法
> 先估计受限模型并保存残差，再用原回归变量和事先提出的候选遗漏项作辅助回归。在经典同方差线性模型的常见形式下，
> $$
> LM=nR^2\overset{a}{\sim}\chi_q^2,
> $$
> 其中 $q$ 是把候选项对原回归量残差化后仍线性独立的新增方向数，也就是有效限制的秩，而不是机械数一遍候选列。
> <!-- bilingual-en:start -->
> Estimate the restricted model, save its residuals, and regress them on the original regressors plus prespecified omitted candidates. In the common homoskedastic linear-model form, $LM=nR^2\overset{a}{\sim}\chi_q^2$, where $q$ is the rank of the added directions after residualising them on the original regressors, not merely the nominal number of candidate columns.
> <!-- bilingual-en:end -->

## 为什么它比通用探针更具体
<!-- bilingual-en:start -->
*Why it is more specific than a generic probe*
<!-- bilingual-en:end -->

若理论和残差图共同提出“可能遗漏 $X^2$ 与 $XZ$”，LM 直接检验这两个方向。拒绝说明受限模型沿这些方向仍有系统改进空间；不拒绝只表示当前样本没有提供足够证据反对相应限制，不能证明候选项真实系数严格为零。
<!-- bilingual-en:start -->
If theory and residual plots suggest omitted $X^2$ and $XZ$, the LM test targets those directions directly. Rejection indicates systematic room for improvement along them. Failure to reject only means that the current sample provides insufficient evidence against the restrictions; it does not prove the candidate coefficients are exactly zero.
<!-- bilingual-en:end -->

$nR^2$ 不是任何数据结构下都可机械使用的万能公式。异方差、聚类或更一般似然模型需要与该协方差结构匹配的 score/LM 实现和参考分布；候选项若在控制原回归量后共线，还要按有效秩减少自由度。具体软件输出应确认辅助回归、有效样本、协方差估计和自由度。
<!-- bilingual-en:start -->
$nR^2$ is not a universal formula for every data structure. Heteroskedastic, clustered, or more general likelihood models require a score or LM implementation matched to the covariance structure and its reference distribution. Candidate terms that are collinear after controlling for the original regressors reduce the effective degrees of freedom. Software output must therefore be checked for the auxiliary regression, effective sample, covariance estimator, and degrees of freedom.
<!-- bilingual-en:end -->

> [!question]- 回忆提示
> LM 检验拒绝“未遗漏 $X^2$ 与 $XZ$”后，能否推出还应加入另一个未检验变量 $W$？
>
> **答案：** 不能。结论只针对预先放入检验的方向；它不枚举所有可能遗漏结构。

## 继续

- [[RESET检验边界]]：对照通用拟合值幂探针的定位能力。
- [[设定修复循环]]：拒绝后把候选项放回机制和验证流程。

## 来源与核验

- [[02_Economy/01_Econometrics/04_模型设定.md#3.2. 拟合不足的检验：LM 拉格朗日乘数检验]]：核对受限残差辅助回归、$nR^2$ 和自由度的课程形式。
- [[02_Economy/01_Econometrics/05_多元回归模型的矩阵表达.md#4.3. LM 检验]]：核对 LM 作为受限模型 score 方向检验的直觉。
- Wooldridge, *Introductory Econometrics: A Modern Approach*, 6th ed., Chapters 5 and 9：核验经典 LM/score 检验及稳健实现边界。
