---
aliases:
  - 大O记号表示一个函数最终被另一个函数的固定常数倍从上方控制
  - Big O notation
student_os: knowledge-atom
atom_id: CS-ASYM-001
atom_type: definition
status: source-checked
part_of:
  - "[[渐近记号与算法复杂度.canvas]]"
related:
  - "[[渐近比较的符号与零点边界]]"
leads_to:
  - "[[大Θ记号]]"
  - "[[渐近界的统一常数]]"
---

# 大O记号表示一个函数最终被另一个函数的固定常数倍从上方控制
<!-- bilingual-en:start -->
*Big O notation means eventual domination by a fixed constant multiple*
<!-- bilingual-en:end -->

对定义在同一个趋向无穷的输入域上、且最终严格为正的函数 $f,g$，**大 O 记号** $f=O(g)$ 表示存在与输入无关的常数 $C>0$ 和阈值 $N$，使所有 $n\ge N$ 都满足 $f(n)\le Cg(n)$。输入可以是正整数，也可以是实数；“最终”允许忽略有限的初始区间。
<!-- bilingual-en:start -->
For functions $f,g$ on the same input domain tending to infinity and eventually strictly positive, **big O notation** $f=O(g)$ means that constants $C>0$ and $N$, independent of the input, exist such that $f(n)\le Cg(n)$ for every $n\ge N$. Inputs may be integers or real numbers. “Eventually” allows an initial bounded interval to be ignored.
<!-- bilingual-en:end -->

$$
f=O(g)
\quad\Longleftrightarrow\quad
\exists C>0\;\exists N\;\forall n\ge N:\ f(n)\le Cg(n).
$$

例如 $f(n)=3n^2+7n+4$：对 $n\ge1$ 有 $f(n)\le14n^2$，所以 $f=O(n^2)$。同一函数也满足 $O(n^3)$；大 O 给的是上界，不声称这个上界紧。要同时保留上下界，用 [[大Θ记号]]。
<!-- bilingual-en:start -->
For example, $f(n)=3n^2+7n+4\le14n^2$ for $n\ge1$, so $f=O(n^2)$. The same function is also $O(n^3)$: big O gives an upper bound, not necessarily a tight one. Use [[大Θ记号|big Theta notation]] to retain both upper and lower bounds.
<!-- bilingual-en:end -->

这里的等号是约定写法；更明确的集合写法是 $f\in O(g)$，因为 $O(g)$ 表示满足该约束的一类函数。它不是普通数值等式，不能据此反写 $g=O(f)$。常数与阈值的量词顺序见 [[渐近界的统一常数]]；带符号或有零点的函数见 [[渐近比较的符号与零点边界]]。
<!-- bilingual-en:start -->
The equals sign is conventional; the explicit set notation is $f\in O(g)$ because $O(g)$ is a class of functions satisfying the bound. This is not numerical equality and does not imply $g=O(f)$. See [[渐近界的统一常数|uniform constants in asymptotic bounds]] for the quantifier order and [[渐近比较的符号与零点边界|sign and zero boundaries]] for extensions beyond positive functions.
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [[01_Math/07-Mathematics for Computer Science/MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session24.pdf#page=3|MIT Session 24，印刷页 530，PDF 页 3]]，Definition 13.7.9：支持常数与阈值定义、最终控制及多项式上界；印刷页 532（PDF 页 5）支持关系记号不能按普通等式倒写。本文采用最终正函数的共同语境，幅度扩展另列；数值例由不等式直接核验。
<!-- bilingual-en:start -->
- [[01_Math/07-Mathematics for Computer Science/MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session24.pdf#page=3|MIT Session 24, printed p. 530, PDF p. 3]], Definition 13.7.9, supports the constant-and-threshold definition, eventual domination and polynomial bounds. Printed p. 532 (PDF p. 5) supports treating the notation as a relation rather than reversible equality. This note uses a common eventually-positive domain; magnitude extensions are separate. The numerical example is checked directly.
<!-- bilingual-en:end -->
