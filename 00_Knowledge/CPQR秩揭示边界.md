---
aliases:
  - "普通列主元 QR 的对角衰减不自动给出强秩揭示保证"
  - CPQR is not automatically strong RRQR
  - Column-pivoted QR rank-revealing boundary
student_os: knowledge-atom
atom_id: LA-PROJ-023
atom_set: orthogonal-projection-least-squares
atom_type: numerical-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[QR分解]]"
  - "[[数值秩]]"
leads_to:
  - "[[CPQR回代不保最小范数]]"
  - "[[秩亏最小二乘算法]]"
related:
  - "[[条件性与算法稳定性]]"
part_of:
  - "[[正交投影与最小二乘.canvas]]"
  - "[[广义逆与最小范数解.canvas]]"
---

# 普通列主元 QR 的对角衰减不自动给出强秩揭示保证
<!-- bilingual-en:start -->
*Diagonal decay in ordinary column-pivoted QR does not automatically provide a strong rank-revealing guarantee*
<!-- bilingual-en:end -->

> [!summary] 数值边界
> 列主元 QR（CPQR）选择置换矩阵 $P$，使
> $$
> AP=QR
> $$
> 的前部三角块尽量先收集独立列。它是常用的有效秩估计工具，但仅凭 $R$ 的对角元衰减，不能无条件推出领先块与原矩阵的奇异值一一对应。
> <!-- bilingual-en:start -->
> Column-pivoted QR chooses a permutation $P$ so that a leading triangular block of $AP=QR$ tends to collect independent columns first. It is a common effective-rank estimator, but diagonal decay in $R$ alone does not unconditionally guarantee that the leading block tracks the singular values of the original matrix.
> <!-- bilingual-en:end -->

## “强秩揭示”多要求了什么
<!-- bilingual-en:start -->
*What a strong rank-revealing guarantee adds*
<!-- bilingual-en:end -->

强 RRQR 不只要求对角元看起来从大到小。它还要给出可检查的界：领先块应良态，尾块应足够小，而未选列对已选列的依赖系数也应受控。Gu–Eisenstat 算法通过额外的列交换来获得这些显式界。
<!-- bilingual-en:start -->
A strong RRQR factorization requires more than visually decreasing diagonal entries. It provides explicit bounds ensuring that the leading block is well conditioned, the trailing block is small, and the dependence of unselected columns on selected columns is controlled. The Gu-Eisenstat algorithms use additional column exchanges to obtain such bounds.
<!-- bilingual-en:end -->

## 实际使用边界
<!-- bilingual-en:start -->
*Practical use boundary*
<!-- bilingual-en:end -->

这不意味着 CPQR 不能用。它仍是一种重要的秩揭示启发式，LAPACK `xGELSY` 也从 CPQR 出发，再用 `RCOND` 和领先块条件估计选择有效秩。但如果论证依赖强 RRQR 的奇异值边界，就必须实际使用具有该保证的分解，不能把普通 CPQR 的经验表现当成定理。
<!-- bilingual-en:start -->
This does not make CPQR useless. It remains an important rank-revealing heuristic, and LAPACK `xGELSY` begins with CPQR before using `RCOND` and a condition estimate of the leading block to choose an effective rank. But an argument that needs strong RRQR singular-value bounds must use a factorization that actually provides them; empirical diagonal decay from ordinary CPQR is not the theorem.
<!-- bilingual-en:end -->

> [!question]- 回忆提示
> CPQR 的 $R$ 对角元在某处明显变小，是否已经证明前部块对原矩阵的主要奇异值有强保证？
> <!-- bilingual-en:start -->
> If the diagonal of $R$ from CPQR drops sharply, does that prove that the leading block has strong guarantees for the dominant singular values of the original matrix?
> <!-- bilingual-en:end -->
>
> **答案：** 不能。这可以是有效秩的实用证据，但强秩揭示还需要领先块、尾块和列耦合的显式界。
> <!-- bilingual-en:start -->
> **Answer:** No. It may be useful evidence for an effective-rank choice, but a strong rank-revealing result also requires explicit bounds on the leading block, trailing block, and column coupling.
> <!-- bilingual-en:end -->

## 来源与核验

- Gu and Eisenstat, [“Efficient Algorithms for Computing a Strong Rank-Revealing QR Factorization”](https://doi.org/10.1137/0917055), 1996：核验强 RRQR 对领先块、尾块及列耦合给出显式界，以及其与普通列主元 QR 的区别。
- [LAPACK `DGELSY`](https://www.netlib.org/lapack/explore-html/d6/d4b/dgelsy_8f_source.html)：核验 CPQR、`RCOND`、领先块条件估计和有效秩的具体用法。
<!-- bilingual-en:start -->
- Gu and Eisenstat (1996) support the explicit leading-block, trailing-block, and coupling bounds of strong RRQR and distinguish them from ordinary column-pivoted QR.
- LAPACK `DGELSY` supports the practical use of CPQR, `RCOND`, and a condition estimate of the leading block to select an effective rank.
<!-- bilingual-en:end -->
