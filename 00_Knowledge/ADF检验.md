---
aliases:
  - "ADF 用滞后差分吸收短期动态而滞后阶数权衡检验大小与功效"
  - Augmented Dickey-Fuller test
  - ADF
  - ADF 检验
student_os: knowledge-atom
atom_id: TS-UR-011
atom_set: trends-unit-roots-differencing
atom_type: test-principle
status: source-checked
mastery_state: unassessed
requires:
  - "[[DF检验]]"
  - "[[单位根确定项规格]]"
related:
  - "[[ADF未拒绝的含义]]"
  - "[[PP检验]]"
part_of:
  - "[[趋势、单位根与差分.canvas]]"
---

# ADF 用滞后差分吸收短期动态而滞后阶数权衡检验大小与功效
<!-- bilingual-en:start -->
*ADF absorbs short-run dynamics with lagged differences, so lag order trades size against power*
<!-- bilingual-en:end -->

> [!summary] 原子命题
> ADF 在 DF 回归中加入 $\Delta y_{t-1},\ldots,\Delta y_{t-p}$，用有限阶参数化近似吸收创新的序列相关，使剩余扰动更接近白噪声；它检验的仍是滞后水平系数 $\gamma=0$。

典型回归为
$$
\Delta y_t=\alpha+\beta t+\gamma y_{t-1}
 +\sum_{i=1}^{p}\delta_i\Delta y_{t-i}+u_t.
$$
滞后差分控制短期动态，并不把单位根“差掉后再检验”。在适当条件和随样本增长的滞后阶序列下，ADF 可覆盖比原始 DF 更一般的 ARMA 型误差。

$p$ 太小会把序列相关留在 $u_t$ 中，造成检验大小失真；$p$ 太大则损失有效样本和估计精度，降低功效。因此信息准则、逐步删减和残差诊断只能帮助选择 $p$，不存在对所有样本都正确的固定滞后数。

> [!question]- 自检
> 为什么“多加一些滞后总不会错”是错误的？
>
> **答案：** 过多滞后会消耗样本与自由度并降低拒绝错误原假设的能力；滞后数需要同时控制剩余相关和有限样本功效。

## 来源与核验

- [Said & Dickey (1984), *Testing for Unit Roots in Autoregressive-Moving Average Models of Unknown Order*](https://doi.org/10.1093/biomet/71.3.599)：核对扩展回归及其适用的 ARMA 误差边界。
- [[01_Math/06_时间序列分析/lecture.pdf]]：对照课程的 ADF 回归与滞后选择。
