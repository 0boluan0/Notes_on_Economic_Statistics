---
aliases:
  - "递归 Cholesky 识别等于变量排序加同期零限制"
  - Recursive Cholesky identification
  - Cholesky ordering in VAR
student_os: knowledge-atom
atom_id: TS-VAR-012
atom_set: vector-autoregression
atom_type: definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[SVAR识别条件]]"
  - "[[结构VAR]]"
  - "[[Cholesky 正定判据]]"
related:
  - "[[简约型VAR创新]]"
  - "[[结构脉冲响应]]"
  - "[[预测误差方差分解]]"
part_of:
  - "[[VAR、脉冲响应与 Granger 因果.canvas]]"
---

# 递归 Cholesky 识别等于变量排序加同期零限制
<!-- bilingual-en:start -->
*Recursive Cholesky identification is a variable ordering plus contemporaneous zero restrictions*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> 对排好顺序的变量做 $\Sigma_u=PP'$ 且令 $P$ 下三角，不只是数值分解；在 $u_t=P\varepsilon_t$ 的方向下，它规定前序变量在当期不响应后序冲击，因而等价于一组递归同期零限制。

给定排序 $(y_{1t},\ldots,y_{Kt})$ 和正定 $\Sigma_u$，Cholesky 分解给出唯一正对角下三角矩阵 $P$：
$$
\Sigma_u=PP',\qquad u_t=P\varepsilon_t,\qquad
E(\varepsilon_t\varepsilon_t')=I.
$$
下三角意味着
$$
P_{ij}=0\quad\text{当 }j>i.
$$
因此变量 $i$ 在冲击发生当期不响应排在它之后的结构冲击 $j$；较早的冲击却可以当期影响同序或更晚的变量。共有 $K(K-1)/2$ 个上三角零，正好满足常见冲击矩阵参数化的递归精确识别计数。

改变变量顺序就改变哪些同期响应被设为零，也会改变 $P$、结构 IRF 和 FEVD。排序只有在制度时序、信息到达或经济理论能支持这些当期排除限制时，才有结构含义。Cholesky 算法本身只保证对给定正定协方差的代数分解，不保证递归因果链真实。

还要明确变换方向：若教材写 $\varepsilon_t=B u_t$ 而不是 $u_t=P\varepsilon_t$，矩阵的三角方向与“谁不响应谁”的口头表述会相应改变。判断零限制前必须先写清方程。

> [!question]- 自检
> 排序为 $(y_1,y_2,y_3)$ 且 $P$ 下三角时，$y_1$ 能否在当期响应 $\varepsilon_3$？
>
> **答案：** 不能，因为 $P_{13}=0$；这正是排序施加的同期排除限制，不是数据自动给出的事实。

## 来源与核验

- [Kilian & Lütkepohl (2017), *Structural Vector Autoregressive Analysis*](https://doi.org/10.1017/9781108164818)，第 8–9 章：核对递归短期识别与变量排序。
- [Kilian & Lütkepohl, Cambridge excerpt](https://assets.cambridge.org/97811071/96575/excerpt/9781107196575_excerpt.pdf)：核对递归排序是一条需要经济论证的同期因果链。
- [[Cholesky 正定判据]]：复用 Cholesky 存在唯一性的线性代数接口。
