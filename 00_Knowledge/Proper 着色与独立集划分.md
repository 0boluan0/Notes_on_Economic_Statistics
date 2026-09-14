---
aliases:
  - "冲突图的 proper coloring 等价于把顶点划分成 independent sets"
  - Proper vertex coloring and independent-set partition
  - 冲突图着色
student_os: knowledge-atom
atom_id: MCS-COLOR-001
atom_type: definition
status: source-checked
mastery_state: unassessed
part_of:
  - "[[图着色与色数.canvas]]"
requires:
  - "[[图模型语义]]"
implies:
  - "[[精确色数的上下界]]"
---

# 冲突图的 proper coloring 等价于把顶点划分成 independent sets
<!-- bilingual-en:start -->
*A proper coloring of a conflict graph is exactly a partition into independent sets*
<!-- bilingual-en:end -->

> [!summary] 原子定义
> 给 finite simple undirected graph $G=(V,E)$ 和正整数 $k$。proper $k$-coloring 是函数
> $$
> c:V\to\{1,\ldots,k\},\qquad uv\in E\Rightarrow c(u)\ne c(v).
> $$
> 实际被使用的每一种颜色对应一个 independent set；反过来，把全部顶点划分进至多 $k$ 个 independent sets，也就得到一个 proper $k$-coloring。这里不要求 $k$ 种颜色全部出现。
>
> <!-- bilingual-en:start -->
> For a finite simple undirected graph $G=(V,E)$ and a positive integer $k$, a proper $k$-coloring is a map
> $$
> c:V\to\{1,\ldots,k\},\qquad uv\in E\Rightarrow c(u)\ne c(v).
> $$
> Every color that is actually used has an independent color class, and conversely any partition of all vertices into at most $k$ independent sets gives a proper $k$-coloring. All $k$ colors need not appear.
> <!-- bilingual-en:end -->

颜色只是组标签。交换“红色”和“蓝色”的名字不会改变分组，也不会产生新的资源结构。真正有意义的是：哪些顶点被放进同一个无冲突组，以及每条冲突边的两端是否被分开。
<!-- bilingual-en:start -->
Colors are only group labels. Swapping the names “red” and “blue” changes neither the partition nor the resource structure. What matters is which vertices share a conflict-free class and whether every conflict edge has separated endpoints.
<!-- bilingual-en:end -->

对非空图，chromatic number 定义为
$$
\chi(G)=\min\{k:G\text{ admits a proper }k\text{-coloring}\}.
$$
因此“$G$ 可用 $k$ 色”只表示 $\chi(G)\le k$，而不是已经算出了 $\chi(G)$。若图非空但没有边，所有顶点可以共享一种颜色，所以 $\chi(G)=1$。上面的 minimum 特意只对非空图下定义；对没有顶点的 null graph，常见约定是 $\chi(G)=0$，但使用其他约定时必须明说。
<!-- bilingual-en:start -->
For a nonempty graph,
$$
\chi(G)=\min\{k:G\text{ admits a proper }k\text{-coloring}\}.
$$
Thus $k$-colorability says only $\chi(G)\le k$; it does not by itself determine $\chi(G)$. A nonempty edgeless graph has chromatic number one. The displayed minimum is intentionally defined only for nonempty graphs; the common convention for the null graph is $\chi(G)=0$, and any different convention must be stated explicitly.
<!-- bilingual-en:end -->

## 为什么它能表达资源冲突

考试排期中，顶点代表考试；两门考试有共同学生时连边；颜色代表时段。proper coloring 保证任何共同学生都不会同时参加两场考试。相同结构也可表达重叠航班分配登机口、相互干扰的电台分配频率，或 live ranges 重叠的变量分配寄存器。
<!-- bilingual-en:start -->
In exam scheduling, vertices are exams, an edge joins two exams sharing a student, and colors are time slots. A proper coloring prevents every represented conflict. The same structure models overlapping flights assigned to gates, interfering stations assigned frequencies, or variables with overlapping live ranges assigned registers.
<!-- bilingual-en:end -->

但这个解释成立的前提是：一条边始终表示“这两个对象不能共享同一类资源”。如果边表示允许配对，问题属于 [[二分图配对关系|matching 的允许关系模型]]；如果还存在容量、时长或偏好，普通顶点着色也不会自动表达那些约束。
<!-- bilingual-en:start -->
This interpretation requires every edge to mean that its endpoints cannot share one resource class. If edges encode feasible pairings, the problem belongs to [[二分图配对关系|the allowed-edge model for matching]]. Capacity, duration, and preference constraints are likewise not represented automatically by ordinary vertex coloring.
<!-- bilingual-en:end -->

> [!question]- 回忆与辨别
> 为什么一个 color class 必然是 independent set？
>
> **答案：**proper coloring 禁止任何相邻顶点同色，所以同一颜色内不存在边；这正是 independent set 的定义。反方向也一样：若每一组内部都没有边，那么每条边的两个端点必落在不同组。

## 来源与核验

- [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf#page=523|MIT Mathematics for Computer Science, Section 12.6.1]]：给出 exam conflict graph、valid coloring、$k$-colorable 与 chromatic number 的定义。
  <!-- bilingual-en:start -->
  Section 12.6.1 gives the exam conflict graph and the definitions of valid coloring, $k$-colorability, and chromatic number.
  <!-- bilingual-en:end -->
- [[01_Math/07-Mathematics for Computer Science/MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_Coloring.pdf|MIT 6.042J Coloring slides]]，slides 5–15：交叉核验冲突图、资源颜色与最少颜色的解释。
