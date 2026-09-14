---
aliases:
  - "EGARCH 用中心化幅度与有符号冲击递推对数方差"
  - EGARCH model
  - Exponential GARCH
  - 指数 GARCH
student_os: knowledge-atom
atom_id: TS-VOL-022
atom_set: conditional-volatility
atom_type: definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[波动不对称与杠杆]]"
  - "[[条件尺度与标准化冲击]]"
related:
  - "[[GJR-GARCH模型]]"
  - "[[TARCH命名边界]]"
  - "[[ARCH-GARCH正性条件]]"
part_of:
  - "[[条件异方差：ARCH 与 GARCH.canvas]]"
---

# EGARCH 用中心化幅度与有符号冲击递推对数方差
<!-- bilingual-en:start -->
*EGARCH recursively models log variance with a centred magnitude term and a signed shock*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> 一种常见 EGARCH(1,1) 记号是
> $$\log h_t=\omega+\beta\log h_{t-1}
> +\alpha\bigl(|z_{t-1}|-E|z_t|\bigr)+\gamma z_{t-1},$$
> 其中 $z_t=\varepsilon_t/\sqrt{h_t}$。幅度项先减去 $E|z_t|$，让它均值为零；有符号项单独表示正负冲击差异。

因为 $h_t=\exp(\log h_t)>0$，$\alpha,\gamma$ 不需要像线性 GARCH 系数那样全非负。在当前 $z=\varepsilon/\sqrt h$ 约定下，$\gamma<0$ 意味着同幅负冲击提高的 log variance 多于正冲击；若作者把冲击符号反过来，系数解释也会反过来。

中心化不是装饰。删去 $E|z|$ 会把幅度项的非零均值吸入截距，使 $\omega$ 与长期 log variance 的解释改变。对简单 AR(1) 式 log variance，$|\beta|<1$ 是常见稳定条件；一般 EGARCH 的严格平稳与矩存在仍需对应模型条件，不能只看正性。

> [!question]- 自检
> Gaussian 标准化冲击下，幅度中心应减去什么？
>
> **答案：** $E|Z|=\sqrt{2/\pi}$。换创新分布后该期望也要随之改变。

## 来源与核验

- [Nelson (1991), *Conditional Heteroskedasticity in Asset Returns: A New Approach*](https://doi.org/10.2307/2938260)：核对 EGARCH 的对数方差、中心化幅度与有符号创新。
- [Engle & Ng (1993)](https://doi.org/10.1111/j.1540-6261.1993.tb05127.x)：核对 EGARCH news-impact curve 与不对称比较。
