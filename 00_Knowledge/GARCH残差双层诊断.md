---
aliases:
  - "GARCH 诊断应分别检查标准化残差与其平方"
  - GARCH standardized residual diagnostics
  - Volatility-model residual diagnostics
  - GARCH 模型诊断
student_os: knowledge-atom
atom_id: TS-VOL-014
atom_set: conditional-volatility
atom_type: diagnostic-workflow
status: source-checked
mastery_state: unassessed
requires:
  - "[[GARCH条件似然]]"
  - "[[ARMA残差诊断]]"
related:
  - "[[ARCH-LM检验]]"
  - "[[McLeod-Li检验]]"
  - "[[波动不对称与杠杆]]"
part_of:
  - "[[条件异方差：ARCH 与 GARCH.canvas]]"
---

# GARCH 诊断应分别检查标准化残差与其平方
<!-- bilingual-en:start -->
*GARCH diagnostics should separately check standardized residuals and their squares*
<!-- bilingual-en:end -->

> [!summary] 原子流程
> 拟合后构造
> $$\hat z_t=\frac{\hat\varepsilon_t}{\sqrt{\hat h_t}}.$$
> 对 $\hat z_t$ 的 ACF/Ljung–Box 检查剩余条件均值相关；对 $\hat z_t^2$ 的 ACF、McLeod–Li 或 ARCH-LM 检查剩余条件方差动态。两组都需要，因为原残差不相关不代表平方残差不相关。

若采用 Gaussian 或 Student-$t$ 完整分布，还应检查 QQ/PIT、尾部、偏态与异常点。假定 Gaussian 时，理想的 $\hat z_t$ 才接近 i.i.d. $N(0,1)$；只通过相关检验不能证明独立、正态或模型真实。

标准化残差诊断还要看时间图、滚动尺度和断点。若某一时段持续出现 $|\hat z_t|$ 过大，问题可能是结构变化而不是再加一个 GARCH 滞后。模型充分性是多项证据的交集，不是一个 p 值。

> [!question]- 自检
> $\hat z_t$ 的 Ljung–Box 未拒绝，但 $\hat z_t^2$ 的检验显著。哪一部分最可能仍有遗漏？
>
> **答案：** 条件均值的线性相关可能已处理，但条件方差仍有动态遗漏；应重查波动阶数、不对称、断点或其他方差规格。

## 来源与核验

- [Engle & Ng (1993), *Measuring and Testing the Impact of News on Volatility*](https://doi.org/10.1111/j.1540-6261.1993.tb05127.x)：核对平方标准化残差与 sign/size-bias 诊断。
- [McLeod & Li (1983)](https://doi.org/10.1111/j.1467-9892.1983.tb00373.x)：核对平方残差 portmanteau 诊断。
- [[01_Math/06_时间序列分析/lecture.pdf#page=171|课程讲义 p. 171]]：核对课程的标准化残差两层检查。
