---
aliases:
  - "McLeod-Li 用平方残差的联合自相关诊断非线性依赖"
  - McLeod-Li test
  - Squared-residual portmanteau test
  - 平方残差联合检验
student_os: knowledge-atom
atom_id: TS-VOL-010
atom_set: conditional-volatility
atom_type: diagnostic-test
status: source-checked
mastery_state: unassessed
requires:
  - "[[ARMA残差诊断]]"
  - "[[Ljung-Box检验]]"
related:
  - "[[ARCH-LM检验]]"
  - "[[平方创新ARMA表示]]"
part_of:
  - "[[条件异方差：ARCH 与 GARCH.canvas]]"
---

# McLeod-Li 用平方残差的联合自相关诊断非线性依赖
<!-- bilingual-en:start -->
*McLeod-Li uses joint autocorrelations of squared residuals to diagnose nonlinear dependence*
<!-- bilingual-en:end -->

> [!summary] 原子检验
> McLeod–Li 诊断把已拟合 ARMA 等均值模型的残差平方 $\hat\varepsilon_t^2$ 当作检查对象，联合检验若干滞后自相关为零。常见实现对平方残差使用 Ljung–Box 型 portmanteau 统计量；显著结果说明线性均值残差中仍有幅度依赖。

它与原残差 Ljung–Box 回答不同问题：原残差检验剩余线性均值相关，平方残差检验二阶幅度结构。一个模型完全可能前者不显著、后者显著，这正是进入 ARCH/GARCH 的常见证据路线。

原论文推导考虑了拟合 ARMA 残差对平方自相关分布的影响。实际软件的自由度、有限样本校正和是否减去均值模型参数并不完全统一，所以应报告实现、滞后集合与样本，而不是把 $\chi_m^2$ 当作任何情形下的精确分布。

平方残差的总体自相关至少要求原创新有有限四阶矩；经典 portmanteau 渐近推导还需更强的正则与矩条件，常见版本要求到八阶矩。若重尾使这些条件可疑，名义 $\chi^2$ 校准可能失真，应明确实现并考虑 bootstrap 或其他稳健校准。

> [!question]- 自检
> 平方残差 portmanteau 检验拒绝后，是否已经证明正确模型就是 GARCH(1,1)？
>
> **答案：** 没有。它只发现平方相关；ARCH、GARCH、非对称模型、断点或其他非线性过程都可能产生该结果。

## 来源与核验

- [McLeod & Li (1983), *Diagnostic Checking ARMA Time Series Models Using Squared-Residual Autocorrelations*](https://doi.org/10.1111/j.1467-9892.1983.tb00373.x)：核对平方残差相关及其拟合后渐近分布。
- [[01_Math/06_时间序列分析/lecture.pdf#page=166|课程讲义 p. 166]]：核对课程中的平方残差 portmanteau 实施路线。
