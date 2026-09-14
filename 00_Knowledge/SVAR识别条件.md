---
aliases:
  - "从简约型 VAR 恢复结构冲击需要归一化和有效识别限制"
  - SVAR identification
  - Structural VAR identification
student_os: knowledge-atom
atom_id: TS-VAR-011
atom_set: vector-autoregression
atom_type: condition
status: source-checked
mastery_state: unassessed
requires:
  - "[[结构VAR]]"
  - "[[简约型创新不是结构冲击]]"
related:
  - "[[Cholesky递归识别]]"
  - "[[长期识别限制]]"
  - "[[符号限制集合识别]]"
  - "[[Proxy SVAR]]"
  - "[[外部工具识别条件]]"
part_of:
  - "[[VAR、脉冲响应与 Granger 因果.canvas]]"
---

# 从简约型 VAR 恢复结构冲击需要归一化和有效识别限制
<!-- bilingual-en:start -->
*Recovering structural shocks from a reduced-form VAR requires normalizations and valid identifying restrictions*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> 简约型创新协方差只给出冲击混合后的二阶矩。要恢复结构冲击，必须先固定尺度、符号等归一化，再施加数量足够且在模型中确实独立、经济上可信的识别限制。

采用冲击映射
$$
u_t=B\varepsilon_t,\qquad E(\varepsilon_t\varepsilon_t')=I_K,
$$
则简约型协方差满足
$$
\Sigma_u=BB'.
$$
$B$ 有 $K^2$ 个元素，而对称 $\Sigma_u$ 只有 $K(K+1)/2$ 个独立元素。在单位方差归一化下，常见冲击矩阵参数化要达到精确识别，通常还需至少
$$
K^2-\frac{K(K+1)}2=\frac{K(K-1)}2
$$
个独立限制。更多限制构成过度识别并可能允许检验；更少限制通常只能得到一个等价结构集合。

这个计数不是脱离参数化的万能规则。若写成 $A_0u_t=C\varepsilon_t$、允许非单位结构冲击方差，或同时限制当期矩阵和长期矩阵，未知量与归一化会改变。还必须确认限制的秩条件：写了足够多的等式，不代表它们彼此独立或能在局部、全局唯一确定所需结构。

最后，统计上的唯一解不等于经济上的有效冲击。零限制、长期限制、符号限制或工具变量都需要由制度、时间顺序或理论解释支持；否则只是把任意旋转命名为经济冲击。

> [!question]- 自检
> 在 $K=3$、$E(\varepsilon_t\varepsilon_t')=I$ 的冲击矩阵参数化下，为什么常见精确识别至少还需 3 个独立限制？
>
> **答案：** $B$ 有 9 个未知元素，$\Sigma_u$ 给出 6 个独立协方差矩，差额为 $9-6=3=K(K-1)/2$。

## 来源与核验

- [Kilian & Lütkepohl (2017), *Structural Vector Autoregressive Analysis*](https://doi.org/10.1017/9781108164818)，第 8–9 章：核对冲击归一化、阶条件、秩条件与短期限制。
- [Kilian & Lütkepohl, Cambridge excerpt](https://assets.cambridge.org/97811071/96575/excerpt/9781107196575_excerpt.pdf)：核对递归限制的经济含义与识别边界。
- [Rubio-Ramírez, Waggoner & Zha (2010), *Structural Vector Autoregressions: Theory of Identification and Algorithms for Inference*](https://doi.org/10.1111/j.1467-937X.2009.00578.x)：核对“限制数量够”之外仍需秩与正则条件，且局部与全局唯一性不能仅靠阶条件推出。
- [[01_Math/06_时间序列分析/lecture.pdf]]：对照课程二变量“9 对 10”例子，并把计数限定到明确参数化与方向。
