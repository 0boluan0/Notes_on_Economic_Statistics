---
aliases:
  - "VAR(p) 用全部系统变量的滞后联合描述线性动态"
  - VAR Model
  - Vector Autoregression
  - VAR(p)
student_os: knowledge-atom
atom_id: TS-VAR-001
atom_set: vector-autoregression
atom_type: definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[AR(p)模型]]"
related:
  - "[[VAR参数量]]"
  - "[[简约型VAR创新]]"
  - "[[结构VAR]]"
  - "[[VAR到VECM重参数化]]"
part_of:
  - "[[VAR、脉冲响应与 Granger 因果.canvas]]"
---

# VAR(p) 用全部系统变量的滞后联合描述线性动态
<!-- bilingual-en:start -->
*A VAR(p) jointly describes linear dynamics using lags of every system variable*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> VAR($p$) 的核心是：一个 $K$ 维变量向量的每个分量，都由同一系统中全部变量的前 $p$ 期线性预测。它首先是简约型动态模型，不因变量被共同建模就自动成为结构因果模型。

令 $y_t=(y_{1t},\ldots,y_{Kt})'$。带确定项或外生变量的 VAR($p$) 可写为
$$
y_t=c+A_1y_{t-1}+\cdots+A_py_{t-p}+D x_t+u_t,
$$
其中每个 $A_i$ 都是 $K\times K$ 矩阵。矩阵元素 $(A_i)_{rs}$ 表示：在控制系统的其他所列滞后后，变量 $s$ 的第 $i$ 阶滞后对变量 $r$ 的线性预测所作的贡献。只有再把 VAR 指定为条件均值模型，例如假定 $E(u_t\mid\mathcal F_{t-1})=0$，这个线性预测才同时等于完整条件均值。$x_t$ 可容纳趋势、季节虚拟变量或真正外生的控制变量；是否纳入这些项是规格选择，不属于 VAR 动态系数本身。

在只给出二阶条件的简约型解释下，$u_t$ 是相对于过去系统变量张成的线性预测空间的一步创新；在上述条件均值假定下，它才还可写成完整信息集的条件预测误差。VAR 把多条动态方程放在一个共同系统里，适合联合预测、检验滞后预测关系和构造动态响应。但是，“系统变量都出现在方程中”只说明联合建模；它既没有说明同期作用的方向，也没有把 $u_t$ 的各分量识别为经济冲击。要作结构解释，还需另行给出可辩护的识别限制。

若变量为 $I(1)$ 且协整，直接在差分 VAR 中删除水平信息可能丢掉误差修正通道；此时应转向 [[VAR到VECM重参数化|VAR–VECM 重参数化]]，而不是把所有序列机械差分后仍沿用同一解释。

> [!question]- 自检
> 把利率、通胀和产出放进同一个 VAR，是否已经把三个简约型创新识别成货币、供给和需求冲击？
>
> **答案：** 没有。VAR 先给出联合动态与简约型创新；经济冲击的名称和方向需要额外的结构识别。

## 来源与核验

- [Lütkepohl (2005), *New Introduction to Multiple Time Series Analysis*](https://doi.org/10.1007/978-3-540-27752-1)，第 2 章：核对有限阶 VAR 的定义、稳定性与估计框架。
- [Sims (1980), *Macroeconomics and Reality*](https://doi.org/10.2307/1912017)：核对 VAR 作为联合动态系统的经典出发点。
- [[01_Math/06_时间序列分析/lecture.pdf]]：对照课程中的二变量 VAR 记号。
