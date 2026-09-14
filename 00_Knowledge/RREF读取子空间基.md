---
aliases:
  - 从 RREF 获取四个基本子空间的基必须采用四种不对称规则
  - Four asymmetric routes from RREF to fundamental-subspace bases
  - RREF 用原矩阵主元列给列空间基并用非零行给行空间基
  - 从 RREF 读四个基本子空间
student_os: knowledge-atom
atom_id: LA-SYS-014
atom_set: linear-systems-four-subspaces
atom_type: method-distinction
status: source-checked
mastery_state: unassessed
requires:
  - "[[行最简形]]"
  - "[[基]]"
  - "[[行操作与四个子空间]]"
  - "[[主元与自由变量]]"
  - "[[零空间]]"
  - "[[四个基本子空间]]"
leads_to:
  - "[[RREF读取列空间基]]"
  - "[[RREF读取行空间基]]"
  - "[[RREF读取零空间基]]"
  - "[[消元追踪左零空间基]]"
part_of:
  - "[[线性方程组与四个基本子空间.canvas]]"
---

# 从 RREF 获取四个基本子空间的基必须采用四种不对称规则
<!-- bilingual-en:start -->
*Obtaining bases of the four fundamental subspaces from RREF requires four asymmetric routes*
<!-- bilingual-en:end -->

> [!summary] 先选对路线
> 设 $A\in\mathbb R^{m\times n}$，并由可逆行操作得到 $R=EA$。四个基本子空间不能靠一条“从 $R$ 直接抄向量”的规则统一处理：
>
> | 目标 | 正确方法 | 为什么 |
> |---|---|---|
> | $C(A)$ | [[RREF读取列空间基|用 $R$ 找主元列位置，再回到 $A$ 取原列]] | 行操作一般改变列空间 |
> | $C(A^T)$ | [[RREF读取行空间基|取 $R$ 的非零行]] | 行操作保持行空间 |
> | $N(A)$ | [[RREF读取零空间基|解 $Rx=0$ 并构造特殊解]] | $N(R)=N(A)$ |
> | $N(A^T)$ | [[消元追踪左零空间基|在消元时同步追踪 $E$]]；若保留 $A$，也可直接解 $A^Ty=0$ | 仅凭 $R$ 无法恢复原来的左零方向 |
> <!-- bilingual-en:start -->
> RREF provides four different routes: pivot locations select original columns, nonzero rows supply a row-space basis, special solutions supply a nullspace basis, and the left nullspace requires either solving $A^Ty=0$ or retaining the row-operation matrix.
> <!-- bilingual-en:end -->

不对称来自行操作的真实作用。$R=EA$ 中，左乘可逆矩阵 $E$ 保持 $Ax=0$ 的解，也只是在原有各行之间做可逆重组；但它会把每个输出向量 $Ax$ 变成 $EAx$，因而通常把 $C(A)$ 变成另一个子空间 $C(R)=E C(A)$。同理，$R$ 的左零空间记录的是消元后矩阵的约束方向，而不是原矩阵的约束方向。

## 使用顺序

实际计算时，先确认目标子空间及其环境空间，再选择对应的方法。列空间回到原矩阵取主元列，行空间读取 RREF 的非零行，零空间构造特殊解；左零空间若要复用同一次消元，就必须同步追踪行操作矩阵，若保留原矩阵，也可以另行解转置齐次系统。分清这些证据来自哪里，才能避免把“主元列”“非零行”“特殊解”和“零行追踪”混成一句口诀。
<!-- bilingual-en:start -->
First identify the target subspace and its ambient space, then use the corresponding construction; the four routes are deliberately different.
<!-- bilingual-en:end -->

> [!question]- 自检
> 已知 $R=EA$。若题目要的是 $N(A^T)$，为什么不能直接取 $N(R^T)$ 的一组基？
>
> **答案：** 因为行操作会改变左零空间的具体方向；实际上 $N(R^T)=E^{-T}N(A^T)$。要恢复 $N(A^T)$，必须直接解 $A^Ty=0$，或在消元时保留 $E$。

## 来源与核验

- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.9sum.pdf|MIT 18.06SC Session 1.9 summary]]：核对原矩阵主元列与特殊解两条路线。
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.10sum.pdf|MIT 18.06SC Session 1.10 summary]]：核对 $A$ 与 $R$ 的行空间相同、列空间不同，以及用 $[A\mid I_m]$ 追踪左零空间基。
