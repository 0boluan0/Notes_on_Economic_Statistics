---
aliases:
  - "稳定 VAR 要求伴随特征值在单位圆内而滞后多项式根在圆外"
  - VAR stability roots
  - VAR 稳定性根条件
student_os: knowledge-atom
atom_id: TS-VAR-006
atom_set: vector-autoregression
atom_type: condition
status: source-checked
mastery_state: unassessed
requires:
  - "[[VAR伴随形式]]"
  - "[[离散系统谱稳定性]]"
related:
  - "[[AR因果根条件]]"
  - "[[谱半径]]"
  - "[[协整与差分边界]]"
part_of:
  - "[[VAR、脉冲响应与 Granger 因果.canvas]]"
---

# 稳定 VAR 要求伴随特征值在单位圆内而滞后多项式根在圆外
<!-- bilingual-en:start -->
*A stable VAR has companion eigenvalues inside the unit circle and lag-polynomial roots outside it*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> VAR 稳定性有两种等价口径：伴随矩阵的全部特征值模小于 1；或滞后多项式行列式的全部根模大于 1。两者所说对象互为倒数，不能把“内”和“外”混用。

定义矩阵滞后多项式
$$
A(z)=I_K-A_1z-\cdots-A_pz^p.
$$
VAR($p$) 的稳定条件可写为
$$
\det A(z)\ne0\quad\text{对所有 }|z|\le1.
$$
也就是说，方程 $\det A(z)=0$ 的根全部位于单位圆外。若 $F$ 是对应伴随矩阵，同一条件也等价于
$$
\rho(F)<1,
$$
即 $F$ 的全部特征值位于单位圆内。对非零根而言，伴随特征值与滞后多项式根采用互为倒数的记号，所以两个方向并不矛盾。

稳定性使初始状态的作用 $F^hY_t$ 随期限消失，并支持收敛的因果 VMA。恰在单位圆上的根对应单位根边界；此时水平 VAR 不是稳定平稳系统，但变量可能通过协整组合保留稳定关系。模大于 1 的伴随特征值则表示爆炸方向。检查软件输出时必须先确认它报告的是 companion eigenvalues、inverse roots 还是 polynomial roots。

> [!question]- 自检
> 软件报告“inverse roots 全在单位圆内”，能否与教材所说“AR roots 必须在单位圆外”同时为真？
>
> **答案：** 可以。inverse roots 对应根的倒数；必须先辨认软件输出的对象，再判断方向。

## 来源与核验

- [Lütkepohl (2005), *New Introduction to Multiple Time Series Analysis*](https://doi.org/10.1007/978-3-540-27752-1)，第 2 章：核对稳定 VAR 的行列式根条件与伴随形式。
- [[AR因果根条件]]、[[离散系统谱稳定性]]：复用标量 AR 与线性系统的同一根—特征值边界。
- [[01_Math/06_时间序列分析/lecture.pdf]]：核对课程二变量示例，并纠正根在单位圆“内/外”的混写。
