---
aliases:
  - "Rosenblatt 变换按条件分布递推，把正确的连续联合模型映射为独立均匀变量"
  - "Rosenblatt transform"
student_os: knowledge-atom
atom_id: RM-DEP-012
atom_set: dependence-and-copulas
atom_type: method
status: source-checked
mastery_state: unassessed
requires:
  - "[[累积分布函数]]"
related:
  - "[[Copula验证]]"
  - "[[Copula]]"
part_of:
  - "[[依赖建模.canvas]]"
---

# Rosenblatt 变换按条件分布递推，把正确的连续联合模型映射为独立均匀变量
<!-- bilingual-en:start -->
*The Rosenblatt transform recursively uses conditional distributions to map a correct continuous joint model to independent uniforms*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> 给定变量顺序 $X_1,\ldots,X_d$，Rosenblatt 变换依次使用边际 CDF 与条件 CDF：
> $$U_1=F_1(X_1),\qquad U_j=F_{j\mid1:(j-1)}(X_j\mid X_1,\ldots,X_{j-1}).$$
> 在合适的连续性条件下，若联合模型正确，$U_1,\ldots,U_d$ 相互独立且都服从 Uniform$(0,1)$。
> <!-- bilingual-en:start -->
> Under suitable continuity conditions, the ordered conditional-CDF transform maps a correctly specified joint model to independent Uniform$(0,1)$ variables.
> <!-- bilingual-en:end -->

## 为什么可用于诊断

变换后的每一维都应均匀，而且各维之间不应残留依赖。因此可以分别检验边际均匀性和联合独立性。任何系统性偏离都说明原联合模型至少有一层没有解释数据。

## 顺序与离散边界

从二维开始，第二个坐标就取决于先条件在哪个变量上；高维时顺序更多。不同顺序可能暴露不同方向的错设，所以单一顺序下“未拒绝”不能证明模型正确。

若联合分布不满足所需的绝对连续性，或边际有离散值与 ties，普通条件 CDF 变换不再自动给出连续独立均匀变量，需使用随机化变换或与离散模型相匹配的诊断。

Rosenblatt 变换只是 [[Copula验证]]中的一种诊断工具，不替代边际、尾部、时间稳定性与样本外用途检查。

> [!question]- 自检
> 某个变量顺序下的 Rosenblatt 检验没有拒绝模型，能否宣布模型正确？
>
> **答案：** 不能。检验力、变量顺序和未覆盖的尾部或样本外目标都可能留下盲点。

## 来源与核验

- Murray Rosenblatt (1952), [“Remarks on a Multivariate Transformation”](https://doi.org/10.1214/aoms/1177729394)：核对顺序条件分布递推与独立均匀结论。
- Christian Genest, Bruno Rémillard and David Beaudoin (2009), [“Goodness-of-fit tests for copulas: A review and a power study”](https://doi.org/10.1016/j.insmatheco.2007.10.005)：核对检验力与顺序边界。
