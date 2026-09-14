---
aliases:
  - "Kupiec 比例失败检验只检验 VaR 例外率是否等于名义尾概率"
  - "Kupiec 无条件覆盖检验"
  - "Kupiec proportion-of-failures test"
  - "VaR unconditional coverage test"
student_os: knowledge-atom
atom_id: RM-VAR-008
atom_set: var-es-backtesting
atom_type: diagnostic-test
status: source-checked
mastery_state: unassessed
requires:
  - "[[VaR回测损益口径]]"
related:
  - "[[独立同分布]]"
  - "[[GARCH选择与预测评估]]"
  - "[[VaR采样误差]]"
  - "[[风险模型验证边界]]"
leads_to:
  - "[[Christoffersen独立性]]"
  - "[[Christoffersen条件覆盖]]"
  - "[[VaR-ES联合识别]]"
part_of:
  - "[[VaR、ES与回测.canvas|VaR、ES与回测]]"
---

# Kupiec 比例失败检验只检验 VaR 例外率是否等于名义尾概率
*The Kupiec proportion-of-failures test asks only whether the VaR exception rate equals the nominal tail probability*

> [!summary] 它看例外总数，不看例外顺序和严重程度
> 在损益与预测口径已经对齐后，Kupiec 比例失败检验把每期例外视为一次 Bernoulli 结果，检验长期例外概率是否等于 $p_0=1-\alpha$。同样多的例外，无论均匀分散还是连续聚集，都会给出同一个统计量。

设回测样本有 $n$ 期，例外指标为 $I_t$，则

$$
x=\sum_{t=1}^{n}I_t,
\qquad
\widehat p=\frac{x}{n},
\qquad
H_0:p=p_0=1-\alpha.
$$

这个等号原假设需要“正确 VaR 的严格例外概率恰为 $p_0$”。例如，条件分布在 $q_t$ 处连续，或者已预先规定与分位数定义一致的随机化 tie rule（阈值相等时的处理规则）。若采用左分位数与 $L_t>q_t$，且阈值处有点质量，正确 VaR 一般只保证

$$
P(I_t=1\mid\mathcal F_{t-1})\le p_0.
$$

此时不能把由分位点并列值造成的少例外，直接解释为模型过度保守。

原假设与不受限 Bernoulli 似然分别为

$$
L_0=(1-p_0)^{n-x}p_0^x,
\qquad
L_1=(1-\widehat p)^{n-x}\widehat p^x.
$$

比例失败似然比统计量是

$$
LR_{\mathrm{uc}}
=-2\log\!\left(\frac{L_0}{L_1}\right)
=-2\log\!\left[
\frac{(1-p_0)^{n-x}p_0^x}
{(1-\widehat p)^{n-x}\widehat p^x}
\right]
\overset{a}{\sim}\chi_1^2.
$$

$x=0$ 或 $x=n$ 时，似然项按连续极限解释。这个标准检验是双侧的：例外过多和过少都会偏离 $H_0$。若研究问题明确只关心风险低估，即 $H_1:p>p_0$，可以在独立 Bernoulli 原假设下报告精确二项上尾

$$
X\sim\operatorname{Binomial}(n,p_0),
\qquad
p_{\mathrm{upper}}=P_{p_0}(X\ge x),
$$

或另行构造带单侧约束的似然比检验；不能把这个单侧 $p$ 值称为标准双侧 $LR_{\mathrm{uc}}$ 的同一个 $p$ 值。

## 能回答与不能回答

- 它能检验样本例外频率是否与名义尾概率相容。
- 它把路径压缩成 $x$，所以看不见例外聚集，也不使用每次超越的损失幅度。
- 精确二项分布与上面的标准似然校准都依赖独立 Bernoulli 结构。若例外相关，统计量仍能描述总频率偏离，但所写参考分布不再自动有效。
- 罕见尾部和短样本通常使检验力很低；当 $np_0$ 很小，或出现 $x=0$、$x=n$ 等边界计数时，$\chi_1^2$ 校准的有限样本 size（一类错误率）和 $p$ 值也可能很差，不只是检验力低。应按事前固定的单双侧规则报告精确二项或模拟校准结果。未拒绝只能说没有发现频率失配，不能证明 VaR 模型正确。
- 例外很少也未必是好消息：它可能表示模型过度保守，而标准双侧检验同样会把这种偏离计入。

> [!question]- 自检
> 两个 250 期回测都出现 3 次例外，其中一组连续三天发生，另一组分散在全年。Kupiec 检验会区分它们吗？
>
> **答案：** 不会。两组的 $n$ 与 $x$ 相同，因此 $LR_{\mathrm{uc}}$ 相同；连续聚集要由独立性或条件覆盖检验处理，而且聚集还会削弱独立 Bernoulli 校准的依据。
>
> <!-- bilingual-en:start -->
> **Self-check:** Two 250-period backtests each contain three exceptions. One has three consecutive exceptions and the other spreads them through the year. Does the Kupiec test distinguish them?
>
> **Answer:** No. Both samples have the same $n$ and $x$, so they have the same $LR_{\mathrm{uc}}$. Clustering requires an independence or conditional-coverage test and also weakens the independent-Bernoulli calibration.
> <!-- bilingual-en:end -->

## 来源与核验

- Kupiec, [Techniques for Verifying the Accuracy of Risk Measurement Models, FEDS 95-24](https://fedinprint.org/item/fedgfe/34596/original)：核对比例失败似然比、双侧覆盖检验及有限样本检验力问题。
- Kupiec, [Techniques for Verifying the Accuracy of Risk Measurement Models](https://doi.org/10.3905/jod.1995.407942)：原始期刊版本。
- Christoffersen, [Evaluating Interval Forecasts](https://doi.org/10.2307/2527341)：核对无条件覆盖只使用边际例外率，不能代替独立性条件。
