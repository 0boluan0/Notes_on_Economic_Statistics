---
aliases:
  - "VAR(p) 可用伴随形式改写为 VAR(1)"
  - VAR companion form
  - VAR 伴随形式
student_os: knowledge-atom
atom_id: TS-VAR-005
atom_set: vector-autoregression
atom_type: derivation
status: source-checked
mastery_state: unassessed
requires:
  - "[[VAR(p)模型]]"
related:
  - "[[VAR稳定根条件]]"
  - "[[矩阵幂趋零判据]]"
  - "[[谱半径]]"
  - "[[离散系统谱稳定性]]"
part_of:
  - "[[VAR、脉冲响应与 Granger 因果.canvas]]"
---

# VAR(p) 可用伴随形式改写为 VAR(1)
<!-- bilingual-en:start -->
*A VAR(p) can be rewritten as a first-order companion system*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> 把当前向量及其前 $p-1$ 期堆叠为一个 $Kp$ 维状态，就能把 VAR($p$) 改写成 VAR(1)；这一伴随形式统一处理稳定性、预测、VMA 递推与状态协方差。

对去均值后的
$$
y_t=A_1y_{t-1}+\cdots+A_py_{t-p}+u_t,
$$
定义状态与状态创新
$$
Y_t=
\begin{pmatrix}
y_t\\y_{t-1}\\ \vdots\\y_{t-p+1}
\end{pmatrix},
\qquad
U_t=
\begin{pmatrix}
u_t\\0\\ \vdots\\0
\end{pmatrix}.
$$
则
$$
Y_t=F Y_{t-1}+U_t,
\qquad
F=
\begin{pmatrix}
A_1&A_2&\cdots&A_{p-1}&A_p\\
I&0&\cdots&0&0\\
0&I&\cdots&0&0\\
\vdots&&\ddots&&\vdots\\
0&0&\cdots&I&0
\end{pmatrix}.
$$
$F$ 是 $Kp\times Kp$ 的伴随矩阵。原 VAR 的当前观测由选择矩阵 $J=(I_K,0,\ldots,0)$ 从状态取出：$y_t=JY_t$。

伴随形式不是新模型，而是同一动态系统的状态扩维。它让 $F^h$ 同时编码 $h$ 期传播；VAR(1) 时 $F=A_1$，但在 $p>1$ 时，不能再把原始 $K\times K$ 的某个 $A_i$ 的幂当作完整的多期传播矩阵。确定项或外生项可相应堆叠进状态方程的输入部分，不改变这一核心重写。

> [!question]- 自检
> 三变量 VAR(4) 的伴随状态和伴随矩阵分别是什么维度？
>
> **答案：** 状态是 $Kp=12$ 维，伴随矩阵是 $12\times12$。

## 来源与核验

- [Lütkepohl (2005), *New Introduction to Multiple Time Series Analysis*](https://doi.org/10.1007/978-3-540-27752-1)，第 2 章：核对有限阶 VAR 的伴随表示。
- [[离散系统谱稳定性]]、[[矩阵幂趋零判据]]与[[谱半径]]：复用伴随矩阵的线性系统接口。
