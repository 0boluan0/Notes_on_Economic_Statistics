---
aliases:
  - "TARCH 标签可能指 GJR 方差递推或 Zakoïan 条件标准差递推"
  - TARCH naming boundary
  - TGARCH naming ambiguity
  - Threshold ARCH naming
student_os: knowledge-atom
atom_id: TS-VOL-026
atom_set: conditional-volatility
atom_type: naming-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[GJR-GARCH模型]]"
related:
  - "[[EGARCH模型]]"
  - "[[ARCH-GARCH正性条件]]"
part_of:
  - "[[条件异方差：ARCH 与 GARCH.canvas]]"
---

# TARCH 标签可能指 GJR 方差递推或 Zakoïan 条件标准差递推
<!-- bilingual-en:start -->
*The TARCH label may denote either a GJR variance recursion or a Zakoïan conditional-standard-deviation recursion*
<!-- bilingual-en:end -->

> [!summary] 命名边界
> “TARCH”或“TGARCH”不是一条足以确定模型的名称。有的来源用它指 [[GJR-GARCH模型|GJR 型方差递推]]
> $$h_t=\omega+\alpha\varepsilon_{t-1}^2+\gamma I\{\varepsilon_{t-1}<0\}\varepsilon_{t-1}^2+\beta h_{t-1},$$
> 另一些来源沿用 Zakoïan 的思路，直接递推条件标准差，例如
> $$\sigma_t=\omega+\alpha_+\varepsilon_{t-1}^{+}+\alpha_-\varepsilon_{t-1}^{-}+\beta\sigma_{t-1},
> \qquad h_t=\sigma_t^2,$$
> 其中 $\varepsilon^+=\max(\varepsilon,0)$、$\varepsilon^- =\max(-\varepsilon,0)$。

<!-- bilingual-en:start -->
> [!summary] Naming boundary
> TARCH or TGARCH does not uniquely determine an equation. Some sources mean the GJR variance recursion, while others follow Zakoïan and model the conditional standard deviation directly. The two equations operate on different scales and therefore do not share coefficients or restrictions merely because they share a label.
<!-- bilingual-en:end -->

两类规格都能表示正负冲击的不同影响，却不是换个符号就完全相同。前者的冲击项以平方为单位并直接更新 $h_t$；后者在线性尺度上更新 $\sigma_t$，平方后会产生不同的交叉项和非线性。因此，GJR 的 $\alpha+\gamma$、有限矩条件或 news-impact 解释，不能原封不动搬到 Zakoïan 规格。

<!-- bilingual-en:start -->
Both families can express sign asymmetry, but they are not interchangeable notation. The GJR shock enters in squared units and updates $h_t$ directly; the Zakoïan shock updates $\sigma_t$ linearly, so squaring creates a different nonlinear structure. GJR coefficient sums, moment conditions, and news-impact interpretations cannot be copied verbatim to the other specification.
<!-- bilingual-en:end -->

实际阅读或使用软件时，至少记录四件事：左边是 $h_t$、$\sigma_t$ 还是 $\log h_t$；正负部分怎样定义；冲击是否标准化；阶数顺序由哪个滞后决定。模型名只帮助搜索，公式才决定估计与解释。

<!-- bilingual-en:start -->
When reading a paper or using software, record the left-hand-side scale, the definitions of positive and negative parts, whether innovations are standardized, and the package's lag-order convention. The label is a search aid; the equation governs estimation and interpretation.
<!-- bilingual-en:end -->

> [!question]- 自检
> 一篇论文只写“估计 TARCH(1,1)”而不给方程。能否据此判断其非对称参数的正性条件？
>
> **答案：** 不能。必须先确认它递推的是条件方差还是条件标准差，以及正负冲击部分的定义。

## 来源与核验

- [Glosten, Jagannathan & Runkle (1993)](https://doi.org/10.1111/j.1540-6261.1993.tb05128.x)：核对以冲击符号改变平方项系数的条件方差递推。
- [Zakoïan (1994), *Threshold Heteroskedastic Models*](https://doi.org/10.1016/0165-1889(94)90039-6)：核对以过去创新的分段线性函数递推条件标准差的 threshold 规格。
- [[01_Math/06_时间序列分析/lecture.pdf#page=176|课程讲义 p. 176]]：核对本课 “TARCH” 标签实际对应 GJR 型方差递推。
