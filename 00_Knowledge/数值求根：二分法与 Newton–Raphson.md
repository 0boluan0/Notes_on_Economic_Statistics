---
aliases:
  - "Numerical Root-Finding"
  - "Bisection and Newton-Raphson"
  - "数值求根"
status: source-checked
---

# 数值求根：二分法与 Newton–Raphson
<!-- bilingual-en:start -->
*Numerical Root-Finding: Bisection and Newton–Raphson*
<!-- bilingual-en:end -->

> [!summary] 快速恢复
> **它解决什么：** 在无法解析求解 $f(x)=0$ 时，用不变量与误差标准产生可验证的近似根。
> **具体锚点：** 二分法始终保留一个端点异号区间；Newton 法用当前切线预测下一点。
> **核心难点：** 二分法的稳健来自 bracket 条件，Newton 的速度来自局部导数；前提不满足时结论不能交换。
> **为什么重要：** 求根是优化、方程求解与模型校准的基础，但“迭代停止”不等于“找到目标根”。
> **继续：** 先写连续性、区间和误差要求，再选择稳健、快速或混合方案。
> <!-- bilingual-en:start -->
> **Problem addressed:** When $f(x)=0$ has no practical analytic solution, use invariants and error criteria to produce a verifiable approximate root.
> **Concrete anchor:** Bisection preserves an interval with opposite endpoint signs, while Newton's method predicts the next point from the current tangent.
> **Central difficulty:** Bisection derives robustness from a bracketing condition and Newton derives speed from local derivative behavior; their conclusions cannot be exchanged when assumptions fail.
> **Why it matters:** Root-finding underlies optimization, equation solving, and calibration, but stopping an iteration does not by itself mean that the desired root was found.
> **Continue with:** State continuity, interval, and error requirements before choosing a robust, fast, or hybrid method.
> <!-- bilingual-en:end -->

## guess-and-check
<!-- bilingual-en:start -->
*Guess and Check*
<!-- bilingual-en:end -->

枚举候选并验证，适合小离散空间或建立基准。算法必须定义候选范围、步长和“不存在”处理；连续问题固定步长只能给离散近似。
<!-- bilingual-en:start -->
Enumerating and checking candidates suits a small discrete space or a baseline. The algorithm must define its range, step, and behavior when no answer exists; a fixed step over a continuous problem provides only a discrete approximation.
<!-- bilingual-en:end -->

## 二分搜索
<!-- bilingual-en:start -->
*Bisection and Binary Search*
<!-- bilingual-en:end -->

有序列表搜索保持目标若存在则位于当前半开区间的不变量，每步减半。求根的 bisection 需要连续函数端点异号，保持根被 bracket。两者都为 $O(\log n)$ / 误差几何缩小，但前提不同。
<!-- bilingual-en:start -->
Binary search on a sorted list preserves the invariant that an existing target lies inside the current half-open interval and halves it each step. Root-finding bisection requires a continuous function with opposite endpoint signs and keeps a root bracketed. Both halve uncertainty geometrically, but their preconditions differ.
<!-- bilingual-en:end -->

数组查找的完整算法边界在 [[渐近记号与算法复杂度#搜索复杂度入口|搜索复杂度入口]]；本卡只保留这段原有比较，并集中解释连续函数求根。
<!-- bilingual-en:start -->
The complete boundary for array search is in [[渐近记号与算法复杂度#搜索复杂度入口|Search Complexity]]. This note retains the original comparison but focuses on root-finding for continuous functions.
<!-- bilingual-en:end -->

## Newton–Raphson
<!-- bilingual-en:start -->
*Newton–Raphson*
<!-- bilingual-en:end -->

$x_{k+1}=x_k-f(x_k)/f'(x_k)$ 用切线交点迭代，近简单根时可二次收敛。导数近零、初值差或函数形状复杂时可发散/跑到别的根；结合 bracket 或步长控制更稳。
<!-- bilingual-en:start -->
$x_{k+1}=x_k-f(x_k)/f'(x_k)$ iterates to the tangent's intercept and can converge quadratically near a simple root. A near-zero derivative, poor initial value, or complicated function shape can cause divergence or convergence to another root; a bracket or step control improves robustness.
<!-- bilingual-en:end -->

## 终止与误差
<!-- bilingual-en:start -->
*Termination and Error*
<!-- bilingual-en:end -->

按目标使用残差 $|f(x)|$、区间宽度或相邻迭代差，但每种指标与真实解误差关系需条件。设置最大迭代并对 NaN/overflow 失败显式报告。
<!-- bilingual-en:start -->
Use residual $|f(x)|$, bracket width, or successive-step size according to the goal, but each metric requires conditions to bound true solution error. Set a maximum iteration count and report NaN or overflow explicitly.
<!-- bilingual-en:end -->

## 算法选择
<!-- bilingual-en:start -->
*Choosing an Algorithm*
<!-- bilingual-en:end -->

二分慢而稳，Newton 快而局部，混合法先 bracket 再加速。选择取决于单调、导数、成本和可靠性，不是只比较理论收敛阶。
<!-- bilingual-en:start -->
Bisection is slower but robust, Newton is fast but local, and a hybrid first brackets then accelerates. The choice depends on monotonicity, derivative availability, evaluation cost, and reliability rather than convergence order alone.
<!-- bilingual-en:end -->

## Worked example：求 $x^2-2=0$
<!-- bilingual-en:start -->
*Worked Example: Solve $x^2-2=0$*
<!-- bilingual-en:end -->

在区间 $[1,2]$ 上端点异号，二分法每轮保留异号半区间。若容许根位置误差至多 $10^{-6}$，当区间宽度不超过 $2\times10^{-6}$ 时中点误差已被控制。
<!-- bilingual-en:start -->
On $[1,2]$, the endpoint signs differ, so bisection retains the half-interval with opposite signs on each iteration. If root-position error must be at most $10^{-6}$, a midpoint taken after the bracket width reaches $2\times10^{-6}$ satisfies that bound.
<!-- bilingual-en:end -->

Newton 从 $x_0=1.5$ 开始使用 $x_{k+1}=(x_k+2/x_k)/2$，数步即接近 $\sqrt2$；但实现仍需检查 $x_k\neq0$、有限值和最大迭代。
<!-- bilingual-en:start -->
Newton's method from $x_0=1.5$ uses $x_{k+1}=(x_k+2/x_k)/2$ and approaches $\sqrt2$ in a few steps. An implementation must still check that $x_k\neq0$, values remain finite, and the iteration limit is respected.
<!-- bilingual-en:end -->

## 失败诊断
<!-- bilingual-en:start -->
*Failure Diagnosis*
<!-- bilingual-en:end -->

- 二分区间不缩向根：核对连续性、端点异号与更新哪一侧的逻辑。
  <!-- bilingual-en:start -->
  A bisection interval does not converge to a root: verify continuity, opposite endpoint signs, and the branch that updates each endpoint.
  <!-- bilingual-en:end -->
- Newton 震荡或发散：记录导数和步长，换初值、阻尼或退回 bracket 方法。
  <!-- bilingual-en:start -->
  Newton oscillates or diverges: record derivatives and step sizes, change the initial value, damp steps, or fall back to a bracketed method.
  <!-- bilingual-en:end -->
- 残差小但位置误差大：检查根附近导数是否很小或问题是否病态，不要把残差直接当距离。
  <!-- bilingual-en:start -->
  Residual is small but position error is large: inspect a small derivative or ill-conditioning near the root; a residual is not automatically a distance.
  <!-- bilingual-en:end -->

## 最小自检
<!-- bilingual-en:start -->
*Minimum Self-Check*
<!-- bilingual-en:end -->

### 二分求根保持什么不变量？
<!-- bilingual-en:start -->
*What invariant does bisection root-finding preserve?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 连续函数在当前区间两端异号，因此区间内至少有一个根。
<!-- bilingual-en:start -->
> [!answer]- Answer
> The continuous function has opposite signs at the current endpoints, so the interval contains at least one root.
<!-- bilingual-en:end -->

### Newton 法快，为什么仍需最大迭代和失败处理？
<!-- bilingual-en:start -->
*Why does Newton's speed not remove the need for an iteration cap and failure handling?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 初值、零导数或非良性函数可使其发散、震荡或溢出。
<!-- bilingual-en:start -->
> [!answer]- Answer
> A poor initial value, zero derivative, or ill-behaved function can make it diverge, oscillate, or overflow.
<!-- bilingual-en:end -->

### 为什么停止条件要匹配最终误差目标？
<!-- bilingual-en:start -->
*Why must the stopping criterion match the final error objective?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 小步长、小残差和小位置误差并不无条件等价；只有在相应数学条件下，一个指标才能控制另一个。
<!-- bilingual-en:start -->
> [!answer]- Answer
> A small step, residual, and position error are not unconditionally equivalent; one controls another only under appropriate mathematical conditions.
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and Verification*
<!-- bilingual-en:end -->

- MIT 6.100L 本地官方 slides、transcripts、finger exercises 与 problem sets：支持 guess-and-check、bisection、Newton 与终止条件。
  <!-- bilingual-en:start -->
  Local official MIT 6.100L slides, transcripts, finger exercises, and problem sets support guess-and-check, bisection, Newton's method, and stopping conditions.
  <!-- bilingual-en:end -->
- [[03_Computer_Science/03_MIT 6.100L/Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf]]：交叉核验数值近似算法和失败边界。
  <!-- bilingual-en:start -->
  [[03_Computer_Science/03_MIT 6.100L/Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Introduction to Computation and Programming Using Python]] cross-checks numerical approximation algorithms and their failure boundaries.
  <!-- bilingual-en:end -->
