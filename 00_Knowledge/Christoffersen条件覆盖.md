---
aliases:
  - "Christoffersen 条件覆盖联合检验例外率正确且一阶独立"
  - Christoffersen conditional-coverage test
  - VaR 条件覆盖检验
student_os: knowledge-atom
atom_id: RM-VAR-009
atom_set: var-es-backtesting
atom_type: diagnostic-test
status: source-checked
mastery_state: unassessed
requires:
  - "[[VaR回测损益口径]]"
  - "[[Kupiec无条件覆盖]]"
  - "[[Christoffersen独立性]]"
related:
  - "[[GARCH选择与预测评估]]"
  - "[[风险模型验证边界]]"
leads_to:
  - "[[VaR-ES联合识别]]"
part_of:
  - "[[VaR、ES与回测.canvas|VaR、ES与回测]]"
---

# Christoffersen 条件覆盖联合检验例外率正确且一阶独立
<!-- bilingual-en:start -->
*The Christoffersen conditional-coverage test jointly tests correct exception frequency and first-order independence*
<!-- bilingual-en:end -->

> [!summary] 条件覆盖是两项约束的联合检验
> Christoffersen 条件覆盖要求例外序列既有正确的名义概率，又不由上一期例外状态预测。它把 [[Kupiec无条件覆盖|无条件覆盖]] 与 [[Christoffersen独立性|一阶独立性]] 的似然比合并，而不是把两项测试混成同一个定义。
>
> <!-- bilingual-en:start -->
> Conditional coverage requires both the correct nominal exception probability and no predictability from the previous hit. It combines the likelihood ratios for [[Kupiec无条件覆盖|unconditional coverage]] and [[Christoffersen独立性|first-order independence]].
> <!-- bilingual-en:end -->

令名义尾概率为 $p_0=1-\alpha$，例外序列为 $I_t$。条件覆盖原假设是

<!-- bilingual-en:start -->
Let $p_0=1-\alpha$ be the nominal tail probability and $I_t$ the exception sequence. The conditional-coverage null is
<!-- bilingual-en:end -->

$$
H_{\mathrm{cc}}:\quad
P(I_t=1\mid I_{t-1}=0)
=P(I_t=1\mid I_{t-1}=1)
=p_0.
$$

在一阶 Markov 备择下，令 $L_1$ 为分别估计两种转移概率的似然，令

<!-- bilingual-en:start -->
Under a first-order Markov alternative, let $L_1$ be the likelihood with separate transition probabilities and define
<!-- bilingual-en:end -->

$$
L_{\mathrm{cc},0}
=(1-p_0)^{n_{00}+n_{10}}p_0^{n_{01}+n_{11}}.
$$

则

$$
LR_{\mathrm{cc}}
=-2\log\!\left(\frac{L_{\mathrm{cc},0}}{L_1}\right)
\overset{a}{\sim}\chi_2^2.
$$

若覆盖项和独立项使用同一组 $t=2,\ldots,T$ 转移观测并以相同方式处理首个指标，则

<!-- bilingual-en:start -->
When the coverage and independence components use the same transition sample $t=2,\ldots,T$ and treat the first hit consistently,
<!-- bilingual-en:end -->

$$
LR_{\mathrm{cc}}
=LR_{\mathrm{uc}}^{(2:T)}
+LR_{\mathrm{ind}}.
$$

若 $LR_{\mathrm{uc}}$ 使用全部 $T$ 个指标，而转移似然只有 $T-1$ 项，这个加法不再是机械恒等式；报告时必须说明样本处理。

<!-- bilingual-en:start -->
If $LR_{\mathrm{uc}}$ uses all $T$ hits while the transition likelihood uses only $T-1$, the displayed decomposition is no longer a mechanical identity; sample treatment must be reported.
<!-- bilingual-en:end -->

> [!question]- 自检
> 通过纯独立性检验是否等于通过条件覆盖？
>
> **答案：** 不等于。例外可以互相独立但平均频率错误；条件覆盖还要求共同例外概率等于名义尾概率。
>
> <!-- bilingual-en:start -->
> **Self-check:** Does passing the pure independence test imply passing conditional coverage?
>
> **Answer:** No. Hits can be independent while occurring at the wrong average rate; conditional coverage additionally fixes their common probability at the nominal tail probability.
> <!-- bilingual-en:end -->

## 边界

- 简单等号原假设需要连续分位或预先规定的 tie rule；分位点有概率质量时必须相应调整。
- 该检验只针对一阶 Markov 备择，不穷尽高阶、持续期或协变量依赖。
- 极少例外会使转移估计落在边界并削弱卡方近似。

<!-- bilingual-en:start -->
- The equality null requires a continuous quantile or a pre-specified tie rule.
- The test uses a first-order Markov alternative and does not exhaust higher-order or covariate dependence.
- Very few hits create boundary estimates and weaken the chi-square approximation.
<!-- bilingual-en:end -->

## 来源与核验

- Christoffersen, [*Evaluating Interval Forecasts*](https://doi.org/10.2307/2527341)：逐式核对条件覆盖、无条件覆盖与独立性分解及渐近自由度。
- Christoffersen, [公开教学镜像全文](https://users.ssc.wisc.edu/~behansen/718/Christoffersen1998.pdf)：核对转移似然与共同样本处理。
