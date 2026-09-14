---
aliases:
  - "因果 ARMA 能展开为收敛的无限 MA 冲击响应"
  - Infinite MA representation of ARMA
  - Causal ARMA expansion
  - ARMA impulse-response coefficients
  - ARMA 的 MA 无穷表示
student_os: knowledge-atom
atom_id: TS-ARMA-009
atom_set: arma-modeling
atom_type: representation
status: source-checked
mastery_state: unassessed
requires:
  - "[[ARMA(p,q)模型]]"
  - "[[AR因果根条件]]"
  - "[[滞后多项式求逆]]"
related:
  - "[[宽平稳定义]]"
  - "[[均方收敛]]"
  - "[[ARMA多步预测]]"
part_of:
  - "[[ARMA 模型：识别、估计、诊断与预测.canvas]]"
  - "[[差分方程与滞后算子.canvas]]"
---

# 因果 ARMA 能展开为收敛的无限 MA 冲击响应
<!-- bilingual-en:start -->
*A causal ARMA expands into a convergent infinite-MA impulse response*
<!-- bilingual-en:end -->

> [!summary] 原子表示
> 若 $\phi(z)$ 在闭单位圆内没有零点，则
> $$\frac{\theta(B)}{\phi(B)}=\psi(B)=\sum_{j=0}^{\infty}\psi_jB^j,$$
> 从而
> $$y_t-\mu=\sum_{j=0}^{\infty}\psi_j\varepsilon_{t-j}.$$
> 系数 $\psi_j$ 是创新对未来观测的冲击响应。
> <!-- bilingual-en:start -->
> When the AR polynomial has no zero on or inside the unit circle, the transfer function $\theta(B)/\phi(B)$ has a one-sided power-series expansion. Its coefficients are the dynamic effects of an innovation on current and future observations.
> <!-- bilingual-en:end -->

对有限阶因果 ARMA，根与单位圆保持正距离，使 $\psi_j$ 以几何速度衰减，因此绝对可和并且平方可和。若创新有有限方差，部分和就在 [[均方收敛|均方]]（$L^2$）意义下收敛；平方可和保证
$\operatorname{Var}(y_t)=\sigma^2\sum_j\psi_j^2<\infty$，重叠创新项又使自协方差只依赖滞后。这不是仅凭形式幂级数便自动得到的逐路径收敛声明。

展开方向很重要：它用**当前及过去**创新生成 $y_t$，所以可用于预测和冲击解释。若只是形式上写出一个发散级数，或需要未来创新，便不是通常的因果 MA($\infty$) 表示。一般无限线性过程只要求平方可和即可有有限方差；“几何衰减/绝对可和”是有限阶因果 ARMA 的更强结论。
<!-- bilingual-en:start -->
For a finite-order causal ARMA, the coefficients decay geometrically and are both absolutely and square summable. With finite-variance innovations, the partial sums converge in mean square ($L^2$), giving finite variance; this is not automatically a pathwise-convergence claim. The expansion is one-sided in current and past innovations; a divergent formal series or a representation using future innovations is not the usual causal MA($\infty$) solution.
<!-- bilingual-en:end -->

> [!question]- 自检
> 为什么 $\sum_j\psi_j^2<\infty$ 与平稳性有关？
>
> **答案：** 白噪声跨期不相关，所以线性和的方差是 $\sigma^2\sum_j\psi_j^2$；平方和有限才给出有限、恒定方差。

## 来源与核验

- [[01_Math/06_时间序列分析/lecture.pdf#page=90|课程讲义 pp. 90–96]]：核对 ARMA($2,1$) 的 $c_i$ 递推、平方可和与宽平稳证明。
- [MIT OCW 18.S096, Lecture 8](https://ocw.mit.edu/courses/18-s096-topics-in-mathematics-with-applications-in-finance-fall-2013/1926c83ecd7ea700f7cb63914c6d7c0f_MIT18_S096F13_lecnote8.pdf)：核对 causal ARMA 的无限 MA 表示。
