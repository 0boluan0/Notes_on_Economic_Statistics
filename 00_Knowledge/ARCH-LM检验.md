---
aliases:
  - "ARCH-LM 用平方残差辅助回归检验所选阶数的 ARCH 效应"
  - Engle ARCH LM test
  - ARCH-LM test
  - ARCH 效应检验
student_os: knowledge-atom
atom_id: TS-VOL-009
atom_set: conditional-volatility
atom_type: diagnostic-test
status: source-checked
mastery_state: unassessed
requires:
  - "[[ARCH(q)模型]]"
  - "[[ARMA残差诊断]]"
related:
  - "[[McLeod-Li检验]]"
  - "[[GARCH残差双层诊断]]"
part_of:
  - "[[条件异方差：ARCH 与 GARCH.canvas]]"
---

# ARCH-LM 用平方残差辅助回归检验所选阶数的 ARCH 效应
<!-- bilingual-en:start -->
*The ARCH-LM test uses an auxiliary squared-residual regression to test an ARCH effect at selected lags*
<!-- bilingual-en:end -->

> [!summary] 原子检验
> 先拟合条件均值并取得残差 $\hat\varepsilon_t$，再估计
> $$\hat\varepsilon_t^2=c+\sum_{j=1}^{q}a_j\hat\varepsilon_{t-j}^2+u_t.$$
> 原假设是 $H_0:a_1=\cdots=a_q=0$。在相应正则条件下，辅助回归有效样本量乘 $R^2$ 的 LM 统计量渐近服从 $\chi_q^2$。

拒绝原假设表示：在所选 $q$ 个滞后上，均值残差平方仍含 ARCH 型可预测结构。它不是“所有条件异方差”的万能检验，也不告诉你 GARCH 阶数、创新分布或经济原因。

检验依赖均值模型。若均值中仍有遗漏自相关、断点或异常值，平方残差辅助回归可能把这些设定错误吸收为 ARCH。$q$ 也必须事先说明；未拒绝只表示当前样本与阶数证据不足。

> [!question]- 自检
> ARCH-LM($5$) 的 p 值很大，能否直接宣布残差同方差且 i.i.d.？
>
> **答案：** 不能。它只未检测到前五阶的 ARCH 型平方依赖；其他滞后、非线性方差形式、分布尾部和独立性仍未被证明。

## 来源与核验

- [Engle (1982)](https://doi.org/10.2307/1912773)：核对 ARCH LM 辅助回归、联合零限制与渐近统计量。
- [[01_Math/06_时间序列分析/lecture.pdf#page=167|课程讲义 p. 167]]：核对本课的 $TR^2$ 实施步骤。
