---
aliases:
  - "多Greek对冲须联立匹配各工具的全部目标敏感度且检查线性系统可行性"
student_os: knowledge-atom
atom_id: FI-HEDGE-007
atom_type: method
status: source-checked
requires:
  - "[[持仓Greeks聚合]]"
  - "[[Delta-Gamma对冲]]"
  - "[[列空间]]"
related:
  - "[[关键利率对冲]]"
part_of:
  - "[[市场风险、Greeks 与动态对冲.canvas]]"
---

# 多Greek对冲须联立匹配各工具的全部目标敏感度且检查线性系统可行性
<!-- bilingual-en:start -->
*Multi-Greek hedging jointly matches all target sensitivities of each instrument and checks feasibility of the linear system*
<!-- bilingual-en:end -->

先固定每一行所指的因子、导数阶数与报价尺度。将原组合的目标 Greeks 排成列向量 $g$，将每种工具一多头单位的全部目标敏感度排成矩阵 $A$ 的一列，则工具数量 $h$ 必须同时满足
<!-- bilingual-en:start -->
First fix the factor, derivative order, and quotation scale of each row. Put the portfolio's target Greeks in vector $g$, and place every target sensitivity of one long unit of each tool in a column of $A$. Hedge quantities $h$ must satisfy all rows together:
<!-- bilingual-en:end -->

$$g+Ah=0.$$

在不限制实数头寸的代数问题中，精确解存在当且仅当 $-g$ 属于 $A$ 的 [[列空间]]。方阵可逆时解唯一；工具数量多不保证列独立，也不保证覆盖目标。若加入整数、卖空或头寸限制，原本有代数解也可能无法实施。
<!-- bilingual-en:start -->
With unrestricted real-valued quantities, an exact solution exists precisely when $-g$ lies in the [[列空间|column space]] of $A$. An invertible square matrix gives a unique solution. More tools do not guarantee independent columns or coverage, and integer, short-sale, or position constraints can make an algebraic solution infeasible.
<!-- bilingual-en:end -->

例如目标是同时消去 Gamma 与 Vega，原组合为 $(-6,-4)$；工具 A 的 $(\Gamma,\nu)=(1.5,0.8)$，工具 B 为 $(0.5,0.6)$。各行使用同一报价单位，联立得到
<!-- bilingual-en:start -->
Suppose the portfolio's gamma and vega are $(-6,-4)$, tool A contributes $(1.5,0.8)$ per unit, and tool B contributes $(0.5,0.6)$. With a common quotation convention within each row, solve:
<!-- bilingual-en:end -->

$$
\begin{pmatrix}1.5&0.5\\0.8&0.6\end{pmatrix}
\begin{pmatrix}h_A\\h_B\end{pmatrix}
=\begin{pmatrix}6\\4\end{pmatrix}
\quad\Rightarrow\quad (h_A,h_B)=(3.2,2.4).
$$

若原 Delta 为 $-30$，A、B 的 Delta 分别为 $0.6,0.1$，上述交易使 Delta 变为 $-27.84$，须再买入 $27.84$ 股同一标的。不能先用 A 单独对冲 Gamma、再用 B 单独对冲 Vega：第二笔期权也会改变 Gamma，第一条约束可能重新失效。
<!-- bilingual-en:start -->
If original delta is −30 and A and B have deltas 0.6 and 0.1, these trades leave delta −27.84, requiring 27.84 underlying shares. Hedging gamma with A and then independently hedging vega with B fails in general because the second option trade changes gamma again.
<!-- bilingual-en:end -->

Gamma 与 Vega 行的单位不同没有妨碍“各自等于零”，但若精确抵销做不到而改做最小二乘，就必须解释残差的尺度与权重；不能任意把不同单位的平方相加作为风险目标。矩阵近奇异时，输入小误差还会放大成巨大交易。即使成功匹配，仍只消去所选状态下的目标 Greeks，不覆盖全部曲面节点或全部市场风险。
<!-- bilingual-en:start -->
Different units across gamma and vega rows do not prevent setting each row to zero. If exact matching is replaced by least squares, however, residual scales and weights must be justified; unweighted squares of unlike units are not an automatic risk objective. Near-singularity can amplify small input errors into large trades. A successful match still removes only selected Greeks at the chosen state, not every surface node or market risk.
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [NYU Kohn／Allen，Section 5，PDF 第 5–6 页](https://math.nyu.edu/~kohn/derivative.securities/2007/section5.pdf#page=5)：已重开组合 Delta、Vega 与更多工具所提供约束的讨论。矩阵写法、列空间条件与 $3.2,2.4,27.84$ 算例由这些同时约束独立推导并复算。
- [[关键利率对冲]]：复用“工具一列、目标一行”的联合敏感度匹配机制；本卡处理不同阶数／种类的 Greeks，不把关键期限节点误作 Vega 因子。
<!-- bilingual-en:start -->
- The reopened NYU notes, pp. 5–6, support simultaneous portfolio delta/vega constraints and the role of additional instruments. The matrix form, column-space criterion, and numerical solution were independently derived and checked.
- [[关键利率对冲|Key-rate hedging]] uses the same column-by-instrument matching mechanism. Here rows can instead represent different Greek types and orders; curve nodes are not treated as volatility factors.
<!-- bilingual-en:end -->
