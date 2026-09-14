---
aliases:
  - "BIC 用最大化对数似然加参数数目乘 log n 的惩罚，近似比较正则候选模型的贝叶斯证据"
  - "BIC adds a parameter-count times log n penalty to maximised negative log-likelihood as a large-sample approximation to Bayesian model evidence"
  - "Bayesian information criterion"
  - "Schwarz criterion"
  - "SBC"
student_os: knowledge-atom
atom_id: ECON-SEL-006
atom_set: regression-model-selection
atom_type: definition
status: source-checked
mastery_state: unassessed
part_of:
  - "[[回归模型比较与选择.canvas|回归模型比较与选择]]"
requires:
  - "[[模型比较可比性]]"
contrasts_with:
  - "[[AIC]]"
related:
  - "[[ARMA信息准则]]"
---

# BIC 用最大化对数似然加参数数目乘 log n 的惩罚，近似比较正则候选模型的贝叶斯证据
<!-- bilingual-en:start -->
*BIC adds a parameter-count times log n penalty to maximised negative log-likelihood as a large-sample approximation to Bayesian model evidence*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> 对样本量为 $n$ 的候选模型 $m$，常用定义是
> $$
> BIC_m=-2\ell_m(\hat\theta_m)+k_m\log n.
> $$
> 在正则、大样本条件下，它来自模型边际似然的渐近展开：忽略候选模型间共同项后，较小的 BIC 对应较大的近似贝叶斯证据。它仍然是同一候选集合内的相对准则，而不是脱离先验与候选集的“真模型分数”。
> <!-- bilingual-en:start -->
> For a candidate $m$ fitted to $n$ observations, $BIC_m=-2\ell_m(\hat\theta_m)+k_m\log n$. Under regular large-sample conditions, it arises from an asymptotic expansion of the model marginal likelihood: after dropping terms common to candidates, a smaller BIC corresponds to larger approximate Bayesian evidence. It remains a relative criterion within a candidate set, not a prior-free probability that a model is true.
> <!-- bilingual-en:end -->

## 与 AIC 的差别不是“一个更高级”
<!-- bilingual-en:start -->
*Its difference from AIC is not that one is universally superior*
<!-- bilingual-en:end -->

[[AIC]] 的参数惩罚是 $2k$；BIC 是 $k\log n$。当 $n$ 较大时，BIC 对新增参数通常惩罚更强，因此常选择更小模型。更重要的是，两者的推导目标不同：AIC 面向候选模型的相对预期信息损失；BIC 面向一个贝叶斯模型比较问题的渐近证据。在包含固定维度真模型并满足正则条件的经典情形，BIC 具有模型选择一致性；如果真机制不在候选集中、维度随样本快速增长或目标是最低预测损失，这个保证不能机械外推。

<!-- bilingual-en:start -->
[[AIC]] uses a $2k$ penalty, whereas BIC uses $k\log n$. BIC therefore tends to penalise added parameters more strongly as $n$ grows. More importantly, the derivation targets differ: AIC addresses relative expected information loss, while BIC approximates evidence in a Bayesian model-comparison problem. Classical selection consistency requires a fixed-dimensional true candidate and regularity conditions; it should not be extrapolated mechanically when every candidate is misspecified, dimensionality grows rapidly, or the goal is minimum predictive loss.
<!-- bilingual-en:end -->

与 AIC 一样，BIC 比较要求完整 likelihood、同一数据与一致参数计数。Gaussian 回归课程中常见

$$
\log(\hat\sigma^2)+\frac{k\log n}{n}
$$

是删去共同常数并按 $n$ 缩放后的同序形式。SBC、SC 和 BIC 在许多教材中指同一 Schwarz criterion；实际使用时仍要检查软件怎样计算 $k$ 与 likelihood。

<!-- bilingual-en:start -->
Like AIC, BIC requires a full comparable likelihood, the same data, and consistent parameter counting. The Gaussian-regression expression $\log(\hat\sigma^2)+k\log(n)/n$ is an order-equivalent scaled form after common constants are removed. SBC, SC, and BIC commonly name the same Schwarz criterion, but software conventions for $k$ and the likelihood must still be checked.
<!-- bilingual-en:end -->

> [!question]- 自检
> 大样本中 BIC 比 AIC 更常选择简单模型。能否因此说 BIC 对任何预测任务都更好？
>
> **答案：** 不能。更强惩罚来自不同的推导目标；预测任务应按部署损失验证，BIC 的经典一致性也依赖真模型在固定候选集和正则条件等假设。

## 来源与核验

- Schwarz（1978），[Estimating the Dimension of a Model](https://doi.org/10.1214/aos/1176344136)：核验从 Bayes 解的渐近展开得到按模型维度与 $\log n$ 惩罚的选择准则。
- Penn State STAT 501, [Lesson 10.5: Information Criteria and PRESS](https://online.stat.psu.edu/stat501/Lesson10)：核对回归中的 BIC/SBC 公式、参数计数和相对排序。
- [[ARMA信息准则]]：复用时间序列阶数选择中 likelihood、参数数目与残差诊断的专门边界。
