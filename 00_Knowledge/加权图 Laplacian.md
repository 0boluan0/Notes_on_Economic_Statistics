---
aliases:
  - "加权图 Laplacian 把边传导与节点守恒合成半正定系统"
  - Weighted graph Laplacian
  - 加权图拉普拉斯矩阵
  - Graph Laplacian
student_os: knowledge-atom
atom_id: MCS-GRAPH-022
atom_type: model
status: source-checked
mastery_state: unassessed
part_of:
  - "[[图的基本结构、路径与遍历.canvas]]"
requires:
  - "[[关联矩阵势差映射]]"
---

# 加权图 Laplacian 把边传导与节点守恒合成半正定系统
<!-- bilingual-en:start -->
*The weighted graph Laplacian combines edge conductance with vertex conservation in a positive-semidefinite system*
<!-- bilingual-en:end -->

> [!summary] 原子模型
> 设 $A\in\mathbb R^{m\times n}$ 是 oriented incidence matrix，$C=\operatorname{diag}(c_1,\ldots,c_m)$ 给每条 edge 非负 conductance。weighted graph Laplacian 定义为
> $$
> L_G=A^TCA.
> $$
> 它先用 $Ax$ 从 vertex potentials 得到 edge differences，以 $C$ 转成带权 edge responses，再用 $A^T$ 把这些量汇总回 vertices。
> <!-- bilingual-en:start -->
> Let $A\in\mathbb R^{m\times n}$ be an oriented incidence matrix and $C=\operatorname{diag}(c_1,\ldots,c_m)$ contain nonnegative edge conductances. The weighted graph Laplacian is
> $$
> L_G=A^TCA.
> $$
> It maps vertex potentials to edge differences through $Ax$, weights edge responses through $C$, and accumulates them back at vertices through $A^T$.
> <!-- bilingual-en:end -->

## 对称、半正定与方向无关
<!-- bilingual-en:start -->
*Symmetry, positive semidefiniteness, and orientation independence*
<!-- bilingual-en:end -->

$L_G$ 对称，而且对任意 $x$，
$$
x^TL_Gx=(Ax)^TC(Ax)=\sum_{e=1}^m c_e(\Delta x_e)^2\ge0.
$$
所以它 positive semidefinite。反转任一参考方向只把 $A$ 的对应行和 $Ax$ 的对应坐标同时反号，平方能量与 $A^TCA$ 不变；Laplacian 属于 underlying weighted graph，不属于随意画出的箭头方向。
<!-- bilingual-en:start -->
$L_G$ is symmetric, and for every $x$,
$$
x^TL_Gx=(Ax)^TC(Ax)=\sum_{e=1}^m c_e(\Delta x_e)^2\ge0.
$$
It is therefore positive semidefinite. Reversing a reference orientation changes the sign of one row of $A$ and the corresponding coordinate of $Ax$, leaving both the squared energy and $A^TCA$ unchanged. The Laplacian belongs to the underlying weighted graph, not to the arbitrary arrows.
<!-- bilingual-en:end -->

若所有 retained edges 的 $c_e>0$，$x^TL_Gx=0$ 当且仅当每条 edge 的两端势相同。因此 $N(L_G)$ 由“在每个 positive-weight component 上为常数”的 vectors 组成；connected graph 时 $N(L_G)=\operatorname{span}\{\mathbf1\}$。若某些 $c_e=0$，这些边对 $L_G$ 不起连接作用，component 应按 positive-weight edges 重新判断。
<!-- bilingual-en:start -->
If every retained edge has $c_e>0$, then $x^TL_Gx=0$ exactly when every edge has equal endpoint potentials. Thus $N(L_G)$ consists of vectors constant on each positive-weight component; for a connected graph, $N(L_G)=\operatorname{span}\{\mathbf1\}$. Edges with $c_e=0$ do not connect the Laplacian system, so components must be recomputed using positive-weight edges.
<!-- bilingual-en:end -->

## 守恒方程与可解性
<!-- bilingual-en:start -->
*Conservation equation and solvability*
<!-- bilingual-en:end -->

一种一致符号约定是 edge flow $y=-CAx$、external injection $f=-A^Ty$，合并后得到
$$
L_Gx=f.
$$
若把正方向定义成 external outflow 或令 $y=CAx$，等式中的负号会改变；物理内容不变，关键是从同一 orientation convention 推导到底。
<!-- bilingual-en:start -->
One consistent convention is edge flow $y=-CAx$ and external injection $f=-A^Ty$, giving
$$
L_Gx=f.
$$
Defining positive external outflow instead, or using $y=CAx$, changes signs. The physical content is unchanged; every equation must follow one orientation convention consistently.
<!-- bilingual-en:end -->

connected graph 中 $L_G\mathbf1=0$，所以 potentials 只能确定到加一个常数；把一个 vertex ground 为 $0$，或加上零均值约束，可选出唯一代表。$L_Gx=f$ 可解必须有 $\mathbf1^Tf=0$，即 total injection 为零；图不连通时，每个 component 上都必须分别净注入为零。
<!-- bilingual-en:start -->
For a connected graph, $L_G\mathbf1=0$, so potentials are determined only up to an additive constant. Grounding one vertex or imposing a zero-mean condition selects a unique representative. Solvability of $L_Gx=f$ requires $\mathbf1^Tf=0$, meaning zero total injection. In a disconnected graph, the injection must sum to zero separately on every component.
<!-- bilingual-en:end -->

按 vertex coordinates 展开时，$L_{ii}$ 是 incident conductances 的和，$L_{ij}$ 是 $i,j$ 之间 total conductance 的负值（无边为 $0$）。因此每行和为 $0$。这与 adjacency matrix 不同：Laplacian 的 diagonal 编码 weighted degree，并把相邻耦合写成负的 off-diagonal entries。
<!-- bilingual-en:start -->
In vertex coordinates, $L_{ii}$ is the sum of conductances incident to $i$, while $L_{ij}$ is the negative total conductance between $i$ and $j$ and is zero when they are nonadjacent. Every row therefore sums to zero. Unlike an adjacency matrix, the Laplacian places weighted degree on the diagonal and negative coupling off the diagonal.
<!-- bilingual-en:end -->

> [!question]- 回忆与应用
> 为什么 connected graph 的 $L_G$ 通常不可逆，却仍能在 ground 一个 vertex 后求出唯一 potentials？
>
> **答案：**所有 constant vectors 都在核中，给每个 potential 同加常数不改变 edge differences，故原系统没有唯一坐标代表。grounding 固定一个坐标，消除这一个 gauge freedom；在 positive connected weights 且注入总和为零时，剩余 reduced system 唯一可解。
> <!-- bilingual-en:start -->
> Why is $L_G$ of a connected graph singular, yet grounding one vertex can make the potential solution unique?
>
> **Answer:** Constant vectors lie in the nullspace because adding one constant to every potential changes no edge difference. Grounding fixes one coordinate and removes this gauge freedom. With positive connected weights and zero total injection, the reduced system then has a unique solution.
> <!-- bilingual-en:end -->

## 来源与核验

- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U1_S13_Lecture_Graphs_Networks_Incidence_Matrices.pdf#page=22|MIT 18.06SC Session 1.13 lecture transcript, pp. 22–24]]：核验 $Ax$、conductance law、$A^Ty$ 守恒与组合矩阵 $A^TCA$。
  <!-- bilingual-en:start -->
  The Session 1.13 transcript verifies $Ax$, the conductance law, conservation through $A^Ty$, and the combined matrix $A^TCA$.
  <!-- bilingual-en:end -->
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U1_S13_Lecture_Graphs_Networks_Incidence_Matrices.pdf#page=24|MIT 18.06SC Session 1.13 lecture transcript, symmetry]]：核验 $A^TCA$ 与 $A^TA$ 的对称结构。
  <!-- bilingual-en:start -->
  The end of the lecture verifies the symmetric structure of $A^TCA$ and $A^TA$.
  <!-- bilingual-en:end -->
- [[01_Math/02_linear algebra/01_Ax = b and the Four Subspaces.md#Session 1.13 Graphs, networks, and incidence matrices|Session 1.13 course note]]：核对符号约定、energy identity、nullspace、grounding 与注入总和边界。
  <!-- bilingual-en:start -->
  The course note verifies the sign convention, energy identity, nullspace, grounding, and zero-total-injection boundary.
  <!-- bilingual-en:end -->
