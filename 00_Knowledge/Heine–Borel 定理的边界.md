---
aliases:
  - "欧氏空间中紧致当且仅当闭且有界但该逆命题不适用于一般度量空间"
  - Heine-Borel holds in finite-dimensional Euclidean space
  - Closed and bounded need not imply compact in general metric spaces
  - Heine-Borel 定理的边界
student_os: knowledge-atom
atom_id: TOPO-MET-013
atom_set: topology-foundations
atom_type: theorem-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[紧致性]]"
  - "[[闭集]]"
part_of:
  - "[[拓扑基础：开集、闭集与连续.canvas]]"
related:
  - "[[极值定理]]"
---

# 欧氏空间中紧致当且仅当闭且有界但该逆命题不适用于一般度量空间
<!-- bilingual-en:start -->
*Compactness is equivalent to closedness and boundedness in Euclidean space, but not in a general metric space*
<!-- bilingual-en:end -->

> [!summary] 原子定理与边界
> Heine–Borel 定理：对有限维欧氏空间中的 $K\subseteq\mathbb R^n$，
> $$K\text{ 紧致}\iff K\text{ 闭且有界}.$$
> 在任意度量空间中，紧致仍推出闭且有界；反向“闭且有界推出紧致”一般不成立。
> <!-- bilingual-en:start -->
> In $\mathbb R^n$, compactness is equivalent to being closed and bounded. General metric spaces retain only the forward implication without additional hypotheses.
> <!-- bilingual-en:end -->

一个最短反例是把 $X=(0,1)$ 本身作为带通常距离的环境空间。$X$ 相对于自身既闭又有界，但不紧致：相对开集
$$
U_n=(1/n,1)\qquad(n\ge2)
$$
覆盖 $X$，任何有限子族却都会漏掉足够接近 0 的点。

在无限维 Hilbert 空间里，闭单位球也不紧致：可取一列正交单位向量，它们两两距离为 $\sqrt2$，所以不存在收敛子序列。这说明有限维性不是装饰条件。经济学或优化中写“可行域闭且有界，所以最优解存在”时，必须同时确认可行域位于有限维 $\mathbb R^n$，或另有能够推出紧致的定理。

> [!question]- 自检
> 为什么“$X$ 在自身中总是闭”不会让所有有界度量空间都紧致？
>
> **答案：** 闭只排除在环境空间内遗漏极限点；它不提供从无限覆盖抽取有限覆盖的全局有限性。

## 来源与核验

- LSE EC400，*Revision Mathematics Notes*（合订本，§8.8，PDF p.90）：核对 $\mathbb R^n$ 中闭、有界与极值存在的课程语境。
- [MIT 18.100B Real Analysis, Lecture 14](https://live.ocw.mit.edu/courses/18-100b-real-analysis-spring-2025/mit18_100b_s25_lec_full.pdf)：明确核对一般度量空间中逆命题失败。
- [MIT 18.102 Functional Analysis, full lecture notes, Example 193](https://ocw.mit.edu/courses/18-102-introduction-to-functional-analysis-spring-2021/8fb8d5c170f1613151aca71de21027bc_MIT18_102s21_full_lec.pdf)：核对无限维 Hilbert 空间的闭单位球非紧。
