---
aliases:
  - 解 RREF 齐次方程并逐个激活自由变量可构造原矩阵零空间的一组基
  - Reading a nullspace basis from RREF
  - Special solutions of the nullspace
student_os: knowledge-atom
atom_id: LA-SYS-027
atom_set: linear-systems-four-subspaces
atom_type: method
status: source-checked
mastery_state: unassessed
requires:
  - "[[行最简形]]"
  - "[[基]]"
  - "[[主元与自由变量]]"
  - "[[自由变量]]"
  - "[[零空间]]"
related:
  - "[[RREF读取子空间基]]"
  - "[[秩零度定理]]"
part_of:
  - "[[线性方程组与四个基本子空间.canvas]]"
---

# 解 RREF 齐次方程并逐个激活自由变量可构造原矩阵零空间的一组基
<!-- bilingual-en:start -->
*Solving the homogeneous RREF system and activating one free variable at a time constructs a basis of the original nullspace*
<!-- bilingual-en:end -->

> [!summary] 方法
> 设 $R$ 是 $A$ 的 RREF。先解 $Rx=0$，把主元变量写成自由变量的线性组合。然后对每个自由变量依次令它等于 $1$、其余自由变量等于 $0$；所得特殊解构成 $N(A)$ 的一组基。
> <!-- bilingual-en:start -->
> Express pivot variables in terms of free variables, then set one free variable at a time to one and the others to zero. The resulting special solutions form a nullspace basis.
> <!-- bilingual-en:end -->

若 $R=EA$ 且 $E$ 可逆，那么
$$
Ax=0\iff EAx=0\iff Rx=0,
$$
所以 $N(A)=N(R)$。每个自由变量提供一个独立方向；若共有 $n-r$ 个自由变量，就会得到 $n-r$ 个特殊解，这与[[秩零度定理|秩—零度定理]]给出的零空间维数一致。

“逐个激活”只是从参数化通解中选取一组最自然的基。零空间还有无数其他基；把这些特殊解换成它们的任意可逆线性组合，仍是同一零空间的基。

若没有自由变量，算法不会产生任何特殊解；此时 $N(A)=\{0\}$，它的一组基是空向量组。

> [!question]- 自检
> 若通解为 $x=s(1,0,-2)^T+t(0,1,3)^T$，零空间的一组特殊解基是什么？
>
> **答案：** $(1,0,-2)^T$ 与 $(0,1,3)^T$；它们分别对应 $(s,t)=(1,0)$ 和 $(0,1)$。

## 来源与核验

- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.9sum.pdf|MIT 18.06SC Session 1.9 summary]]：核对自由变量、特殊解与零空间基的构造。
