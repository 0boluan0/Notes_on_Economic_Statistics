---
aliases:
  - "复方阵可由 unitary 矩阵对角化当且仅当它是 normal 矩阵"
  - "复矩阵可 unitary 对角化当且仅当它是 normal 矩阵"
  - Spectral theorem for normal matrices
  - Unitary diagonalization criterion
student_os: knowledge-atom
atom_id: LA-SPD-006
atom_set: symmetric-positive-definite
atom_type: theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[Normal 矩阵]]"
  - "[[Unitary 矩阵]]"
related:
  - "[[Hermitian 谱定理]]"
  - "[[可对角化判据]]"
part_of:
  - "[[对称矩阵与正定二次型.canvas]]"
---

# 复方阵可由 unitary 矩阵对角化当且仅当它是 normal 矩阵
<!-- bilingual-en:start -->
*A complex square matrix is unitarily diagonalizable if and only if it is normal*
<!-- bilingual-en:end -->

> [!summary] Normal 谱定理
> 对 $A\in\mathbb C^{n\times n}$，下列条件等价：
> 1. $A^*A=AA^*$；
> 2. $\mathbb C^n$ 有一组由 $A$ 的特征向量构成的标准正交基；
> 3. 存在 unitary $U$ 与复对角矩阵 $\Lambda$，使 $A=U\Lambda U^*$。
> <!-- bilingual-en:start -->
> A complex square matrix is normal if and only if it has an orthonormal eigenbasis, equivalently if and only if $A=U\Lambda U^*$ for a unitary $U$ and a complex diagonal $\Lambda$.
> <!-- bilingual-en:end -->

“若可由 unitary 矩阵对角化，则 normal”可直接验算：对角矩阵 $\Lambda$ 与 $\Lambda^*$ 可交换，unitary 相似变换保留这个等式。反向是 normal 谱定理的实质。
<!-- bilingual-en:start -->
The easy direction starts from a unitary diagonalization: diagonal matrices commute with their adjoints, and unitary similarity preserves this identity. The converse is the substantive normal spectral theorem.
<!-- bilingual-en:end -->

边界有两层。第一，一般可对角化 $A=S\Lambda S^{-1}$ 不保证 $S$ unitary，也不保证 $A$ normal。第二，在实数域上 normal 矩阵未必有实特征向量；平面旋转 normal，却只能在 $\mathbb C$ 上 unitary 对角化。实对称矩阵则因谱为实数而有实正交对角化。
<!-- bilingual-en:start -->
Ordinary diagonalizability does not imply unitary diagonalizability. Also, a real normal matrix may lack real eigenvectors: a planar rotation diagonalizes unitarily only after passing to $\mathbb C$. Real symmetric matrices are the special case whose spectrum and orthonormal eigenbasis can both be taken real.
<!-- bilingual-en:end -->

> [!question]- 自检
> 为什么一般的可对角化不能直接推出 normal？
>
> **答案：** 一般对角化所用的特征基未必标准正交；normal 要求的是 unitary 对角化。
>
> <!-- bilingual-en:start -->
> **Question:** Why does ordinary diagonalizability not imply that a matrix is normal?
>
> **Answer:** The eigenbasis used in an ordinary diagonalization need not be orthonormal; normality requires a unitary diagonalization.
> <!-- bilingual-en:end -->

## 来源与核验

- [MIT Quantum Physics II Lecture Notes 5, pp. 12–13](https://ocw.mit.edu/courses/8-05-quantum-physics-ii-fall-2013/005979fa741c3ea2e0430456b70caf93_MIT8_05F13_Chap_05.pdf#page=12)：核对“unitarily diagonalizable iff normal”的双向谱定理。
- [MIT RES.18-011 Algebra I Lecture 28](https://ocw.mit.edu/courses/res-18-011-algebra-i-student-notes-fall-2021/mit18_701f21_full_lec_new.pdf#page=138)：交叉核对标准正交特征基与实数域边界。
<!-- bilingual-en:start -->
- MIT Quantum Physics II Lecture Notes 5 were checked for the if-and-only-if normal spectral theorem.
- MIT RES.18-011 Lecture 28 was cross-checked for the orthonormal eigenbasis formulation and the real-field boundary.
<!-- bilingual-en:end -->
