---
aliases:
  - "AIC 用最大化对数似然加两倍参数数目的惩罚，比较候选模型的相对预期信息损失"
  - "AIC adds a twice-parameter-count penalty to maximised negative log-likelihood to compare candidates by relative expected information loss"
  - "Akaike information criterion"
student_os: knowledge-atom
atom_id: ECON-SEL-005
atom_set: regression-model-selection
atom_type: definition
status: source-checked
mastery_state: unassessed
part_of:
  - "[[回归模型比较与选择.canvas|回归模型比较与选择]]"
requires:
  - "[[模型比较可比性]]"
contrasts_with:
  - "[[BIC]]"
related:
  - "[[ARMA信息准则]]"
leads_to:
  - "[[选择后推断]]"
---

# AIC 用最大化对数似然加两倍参数数目的惩罚，比较候选模型的相对预期信息损失
<!-- bilingual-en:start -->
*AIC adds a twice-parameter-count penalty to maximised negative log-likelihood to compare candidates by relative expected information loss*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> 对用最大似然估计的候选模型 $m$，常用定义是
> $$
> AIC_m=-2\ell_m(\hat\theta_m)+2k_m,
> $$
> 其中 $\ell_m(\hat\theta_m)$ 是最大化后的完整对数似然，$k_m$ 是该 likelihood 中独立估计的参数数目。第一项奖励拟合，第二项补偿因增加自由参数而产生的乐观偏差；在同一可比候选集合中，AIC 较小者估计具有较小的相对预期信息损失。
> <!-- bilingual-en:start -->
> For a maximum-likelihood candidate $m$, $AIC_m=-2\ell_m(\hat\theta_m)+2k_m$, where $\ell_m(\hat\theta_m)$ is the maximised full log-likelihood and $k_m$ counts independently estimated parameters in that likelihood. The first term rewards fit and the second corrects optimism from additional free parameters. Within one comparable candidate set, a lower AIC estimates smaller relative expected information loss.
> <!-- bilingual-en:end -->

## 它给的是相对排序
<!-- bilingual-en:start -->
*It provides a relative ranking*
<!-- bilingual-en:end -->

通常先计算

$$
\Delta_m=AIC_m-\min_j AIC_j.
$$

$\Delta_m=0$ 只表示该模型在当前候选集中最小；它不证明模型绝对正确，也不给“真模型概率”。如果所有候选都漏掉关键结构，AIC 只能在这些不充分选项中排序。增加或删除候选模型还会改变“最佳”所相对的集合。

<!-- bilingual-en:start -->
The usual comparison uses $\Delta_m=AIC_m-\min_jAIC_j$. A zero difference means only that the model is smallest among the candidates considered. It neither proves absolute adequacy nor reports a probability that the model is true. If every candidate omits an important feature, AIC merely ranks inadequate options.
<!-- bilingual-en:end -->

## 回归课程公式为什么看起来不同
<!-- bilingual-en:start -->
*Why regression-course formulas can look different*
<!-- bilingual-en:end -->

在同方差 Gaussian 回归、同一响应和样本下，把共同常数删去并除以 $n$，AIC 可写成与

$$
\log(\hat\sigma^2)+\frac{2k}{n}
$$

同序的 scaled form。软件对截距、方差参数和共同常数的计数或显示可能不同；应在同一 convention 内比较差值，不能跨软件或不同 outcome scales 横比裸数值。

<!-- bilingual-en:start -->
For homoskedastic Gaussian regression on the same outcome and observations, dropping constants common to all candidates and dividing by $n$ yields a scaled criterion ordered like $\log(\hat\sigma^2)+2k/n$. Software may count or display the intercept, variance parameter, and common constants differently. Compare differences under one convention rather than raw values across software or outcome scales.
<!-- bilingual-en:end -->

AIC 的用途也有边界。它不检查变量是否具备因果身份，不保护看过许多模型后再报告普通 p 值，也不替代残差诊断或真正的部署验证。样本很小而参数多时，AIC 的有限样本修正 AICc 可能更合适，但仍须满足同一比较契约。

<!-- bilingual-en:start -->
AIC does not determine causal admissibility, protect ordinary p-values after extensive search, replace residual diagnostics, or substitute for deployment validation. When the sample is small relative to parameter count, a finite-sample correction such as AICc may be preferable, while retaining the same comparability requirements.
<!-- bilingual-en:end -->

> [!question]- 自检
> 一个模型的 AIC 是 120，另一个是 125。为什么不能只凭 120 这个绝对数值判断模型拟合“很好”？
>
> **答案：** AIC 的共同常数和尺度依实现而异，目标也是候选间的相对信息损失。必须看同一次可比比较中的差值，并另行检查模型充分性。

## 来源与核验

- Akaike（1974），[A New Look at the Statistical Model Identification](https://doi.org/10.1109/TAC.1974.1100705)：核验 $-2$ 倍最大对数似然加 $2k$ 的准则及其以信息损失进行模型识别的目标。
- Penn State STAT 501, [Lesson 10.5: Information Criteria and PRESS](https://online.stat.psu.edu/stat501/Lesson10)：核对 Gaussian regression 中的等价 scaled form、参数惩罚和较小值排序。
- [[ARMA信息准则]]：复用 AIC 在 ARMA 阶数选择中的专门计数与诊断边界。
