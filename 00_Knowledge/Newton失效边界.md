---
aliases:
  - "Newton failure modes"
  - "Newton method failure boundaries"
  - "Newton 法失效"
  - "牛顿法失效边界"
student_os: knowledge-atom
atom_id: CS-NR-009
atom_set: numerical-root-finding
atom_type: failure-boundary
status: needs-review
requires:
  - "[[Newton迭代]]"
related:
  - "[[Newton局部收敛]]"
  - "[[数值迭代停止条件]]"
  - "[[条件性与算法稳定性]]"
  - "[[求根残差]]"
  - "[[残差控制根误差]]"
leads_to:
  - "[[混合求根]]"
  - "[[求根策略]]"
part_of:
  - "[[数值求根.canvas|数值求根]]"
  - "[[导数的应用.canvas]]"
---

<!-- bilingual-en:start -->
# Failure modes of Newton iteration
<!-- bilingual-en:end -->
*Newton 迭代的失效边界*

<!-- bilingual-en:start -->
A Newton step follows the tangent where the current guess happens to be. If that tangent poorly describes the curve over the proposed distance, the step can point somewhere unhelpful even when the desired root exists. Diagnosing a run means identifying what failed: the update formula, the path toward the desired root, the function domain, or the numerical computation.
<!-- bilingual-en:end -->

Newton 的一步只沿着当前猜测处的切线走。如果这条切线在候选步长范围内不能代表原曲线，即使目标根存在，新点也可能没有帮助。诊断运行时，要分清坏掉的是更新公式、通向目标根的路径、函数定义域，还是数值计算本身。

<!-- bilingual-en:start -->
First recall the [[Newton迭代|tangent update]]. Its [[Newton局部收敛|local convergence theorem]] provides a small invariant neighborhood of a smooth simple root. A **basin of attraction** means the set of starting guesses whose valid iterates converge to a particular root. Plain Newton does not maintain the sign-changing interval certificate used by [[二分求根|bisection]].
<!-- bilingual-en:end -->

先回顾[[Newton迭代]]的切线更新。[[Newton局部收敛]]定理在光滑简单根附近给出一个小的不变邻域。**吸引域**指合法迭代最终收敛到某个指定根的所有初值组成的集合。普通 Newton 不像[[二分求根]]那样持续维护异号括区间证书。

> [!abstract] The central idea · 核心思路
> <!-- bilingual-en:start -->
> Each Newton step uses only the current local tangent; it does not preserve a certificate that a root remains inside a trusted interval. A finite next point therefore proves neither that a root exists nor that the point is approaching the intended root. Failure can appear as an undefined update, a huge jump, divergence, a cycle, a domain escape, or stable convergence to a different root.
> <!-- bilingual-en:end -->
>
> Newton 每轮只依赖当前局部切线，没有持续保存“某个根仍在可信区间内”的证书。于是，算出一个有限的下一点不代表根存在，也不代表它靠近原先想找的根。失败可能表现为公式无定义、巨大跳步、发散、周期、越界，或稳定地收敛到另一个根。

<!-- bilingual-en:start -->
## Separate the four mechanisms
<!-- bilingual-en:end -->

## 四类机制要分开诊断

<!-- bilingual-en:start -->
&nbsp;
**1. Zero or near-zero derivative at a nonroot.** If $f(x_k)\ne0$ but $f'(x_k)=0$, the update is undefined. A small derivative may make $-f(x_k)/f'(x_k)$ so large that the local model is no longer useful. If $f(x_k)=0$, return the root before division.<br>
**2. A different basin of attraction.** Different starting guesses may converge to different roots. Reaching another root is not divergence, but may fail the specified task.<br>
**3. Cycles or divergence.** For the cubic displayed below, starting at zero gives one, then returns to zero: a two-cycle.<br>
**4. Domain or machine-number failure.** A candidate may leave the function domain, produce NaN or overflow, or stop moving through rounding. Continuing to print numbers is not evidence of convergence.
<!-- bilingual-en:end -->

1. **非根处的零或近零导数。** 若 $f(x_k)\ne0$ 而 $f'(x_k)=0$，更新式没有定义；若导数绝对值很小，$-f(x_k)/f'(x_k)$ 可能成为远离局部模型的巨大步长。若 $f(x_k)=0$，则应先按已找到根返回，不再做除法。
2. **初值落在别的吸引域。** 多根函数会把不同初值送往不同根。收敛到另一个根在数值上未必发散，却可能没有回答任务指定的问题。
3. **周期或发散。** 对

   $$
   f(x)=x^3-2x+2,
   $$

   从 $x_0=0$ 出发有 $x_1=1$，再由 $x_1=1$ 得 $x_2=0$，形成 $0\leftrightarrow1$ 的二周期。
4. **定义域与机器数失败。** 候选可能越出函数定义域，或产生 NaN、overflow 和舍入停滞。程序继续吐出数值不能把它们改造成收敛证据。

<!-- bilingual-en:start -->
One decrease in residual does not guarantee monotone progress: Newton does not generally minimize $|f(x)|$. Similarly, $x_{k+1}\approx x_k$ may only mean rounding has stalled the update. Monitor the [[求根残差|residual]], derivative, step, finite values, iteration limit and failure status through [[数值迭代停止条件|stopping criteria]]. Use [[残差控制根误差|residual-to-root-error conditions]] before interpreting a function-value defect as a position error.
<!-- bilingual-en:end -->

单轮残差变小也不是单调保证；Newton 并不普遍最小化 $|f(x)|$。同理，$x_{k+1}\approx x_k$ 可能只是舍入后走不动。运行时应把 [[求根残差|残差]]、导数、步长、有限值、迭代上限和失败状态交给 [[数值迭代停止条件]]，并用 [[残差控制根误差]] 判断函数值缺口能否解释为位置误差。

> [!example] A flat tangent can launch a long step · 平切线造成长跳步
> <!-- bilingual-en:start -->
> For $f(x)=x^2-1$, the roots exist at $\pm1$. At $x_0=0$, $f=-1$ but $f'=0$, so no Newton step exists. At $x_0=0.01$, both evaluations are finite and the derivative is nonzero, yet $x_1=(0.01+100)/2=50.005$. The residual magnitude jumps from $0.9999$ to $2499.500025$. A defined step need not improve the residual.
> <!-- bilingual-en:end -->
>
> 对 $f(x)=x^2-1$，根确实存在于 $\pm1$。在 $x_0=0$，$f=-1$ 而 $f'=0$，不能执行 Newton 步。改取 $x_0=0.01$，函数与导数都有限、导数也非零，但 $x_1=(0.01+100)/2=50.005$，残差绝对值从 $0.9999$ 增为 $2499.500025$。步骤能算出来，不代表残差会改善。

<!-- bilingual-en:start -->
A derivative’s absolute size alone is not a scale-independent failure test. Replacing $f$ by $10^{-12}f$ shrinks its derivative by the same factor but leaves the exact quotient $f/f'$ unchanged. Inspect the step, domain, evaluation accuracy and problem units rather than applying a universal numerical threshold to the derivative.
<!-- bilingual-en:end -->

导数的绝对大小本身不是与尺度无关的失败判据。把 $f$ 换成 $10^{-12}f$，导数也缩小同样倍数，但精确商 $f/f'$ 不变。需要结合步长、定义域、评价精度与问题单位判断，不能给导数随意套一个通用数值阈值。

<!-- bilingual-en:start -->
The cycle above has nonzero denominators at both points: $f(0)=2$, $f'(0)=-2$, $f(1)=1$ and $f'(1)=1$, so $N(0)=1$ and $N(1)=0$. Meanwhile $f(-2)=-2$ and $f(-1)=3$, so continuity ensures a root in $(-2,-1)$. The iteration fails to reach it despite a real root and well-defined steps; even its initial residual decrease from $2$ to $1$ is misleading.
<!-- bilingual-en:end -->

上面的二周期在两点都有非零分母：$f(0)=2$、$f'(0)=-2$、$f(1)=1$、$f'(1)=1$，故 $N(0)=1$、$N(1)=0$。另一方面，$f(-2)=-2$、$f(-1)=3$，连续性保证 $(-2,-1)$ 内存在根。根存在、每步也合法，仍然到达不了；第一步残差从 $2$ 降到 $1$ 也会造成误导。

<!-- bilingual-en:start -->
For a domain failure, take the natural logarithm $f(x)=\log x$ on $x>0$, whose root is $1$. Its Newton map is $N(x)=x(1-\log x)$. Starting at $x_0=e^2$ produces $x_1=-e^2$, where the next real logarithm is undefined. A positive starting point did not preserve the positive domain.
<!-- bilingual-en:end -->

定义域失败可以用 $x>0$ 上的自然对数 $f(x)=\log x$ 看清，根为 $1$。Newton 映射是 $N(x)=x(1-\log x)$；从 $x_0=e^2$ 出发，得到 $x_1=-e^2$，下一次实对数已经没有定义。初值在正半轴，并不保证下一点仍在正半轴。

<!-- bilingual-en:start -->
Wrong-root convergence is different. For $f(x)=x^2-1$, $x_0=-2$ gives $-1.25$, then $-1.025$. Negative inputs stay negative under $N(x)=(x+1/x)/2$. After one step their magnitudes are at least $1$; above $1$ those magnitudes decrease and their limit solves $u=(u+1/u)/2$, hence $u=1$. The run converges to $-1$, which succeeds for “find any root” but fails if the required root is $+1$.
<!-- bilingual-en:end -->

收敛到错误目标是另一回事。对 $f(x)=x^2-1$，$x_0=-2$ 依次给出 $-1.25$、$-1.025$。在 $N(x)=(x+1/x)/2$ 下，负输入始终为负；一步之后绝对值至少为 $1$，在 $1$ 以上又单调下降，极限满足 $u=(u+1/u)/2$，故 $u=1$。所以迭代收敛到 $-1$：对“找任意一个根”成功，对“找正根 $+1$”则失败。

> [!warning] Smoothness alone does not prevent divergence · 光滑也可能发散
> <!-- bilingual-en:start -->
> For $f(x)=\arctan x$ (inverse tangent), the unique root is the simple root $0$, but $N(x)=x-(1+x^2)\arctan x$. If $x\ge2$, then $\arctan x>1$ and $N(x)<-(x^2-x+1)$. Thus the next point changes sign and its magnitude exceeds $x+1$. The map is odd, so the same magnitude growth holds for $x\le-2$. Starting with $|x_0|\ge2$ therefore alternates signs and diverges in magnitude, with every step defined. The inequality $\arctan2>1$ follows from $\arctan2>\pi/3>1$, since $\tan(\pi/3)=\sqrt3<2$.
> <!-- bilingual-en:end -->
>
> 对 $f(x)=\arctan x$（反正切），唯一的根 $0$ 是简单根，但 $N(x)=x-(1+x^2)\arctan x$。若 $x\ge2$，则 $\arctan x>1$，因此 $N(x)<-(x^2-x+1)$；下一点换号，绝对值还超过 $x+1$。映射是奇函数，所以 $x\le-2$ 也有相同的绝对值增长。从 $|x_0|\ge2$ 出发便会正负交替、绝对值发散，而且每一步都有定义。这里 $\arctan2>1$ 可由 $\tan(\pi/3)=\sqrt3<2$ 推出 $\arctan2>\pi/3>1$。

<!-- bilingual-en:start -->
## Safeguards change the path, not the local theorem
<!-- bilingual-en:end -->

## 保护机制改变的是路径，不是定理前提

<!-- bilingual-en:start -->
Changing the initial guess, damping or limiting the step may help a particular problem but does not automatically supply a global certificate. Given continuity and a sign-changing bracket, a [[混合求根|hybrid solver]] can reject Newton candidates that escape or fail its progress test and fall back to bisection. Its safeguard is the maintained bracket invariant, not a conversion of Newton’s local theorem into a global one.
<!-- bilingual-en:end -->

换初值、阻尼或限制步长可能改善某个问题，却不自动提供全局证书。若有连续异号括区间，[[混合求根]] 可以拒绝越界或缺乏进展的 Newton 候选并退回二分；它之所以更稳，是因为括区间不变量一直保留，而不是因为 Newton 的局部定理被扩成了全局定理。

> [!question-] Try the distinction · 检验理解
> <!-- bilingual-en:start -->
> Newton stably converges to root $r_2$, but the task requested a different root $r_1$ of the same function. Is that success?
>
> **Answer:** It may succeed under “find any root”, but fails under “find $r_1$”. Specify how an interval, a sign or another condition identifies the desired root; a stable-looking sequence alone is not enough.
> <!-- bilingual-en:end -->
>
> Newton 从某个初值稳定收敛到根 $r_2$，而任务原先要找同一函数的另一个根 $r_1$。这算“算法收敛成功”吗？
>
> **答案：** 对“找到任意一个根”的规格可以成功；对“找到 $r_1$”的规格则失败。必须先写清目标根怎样由区间、符号或其他条件识别，不能只看数值序列是否稳定。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- MIT 18.330, Chapter 4: Nonlinear equations, PDF pp. 2–4: starting-point dependence, local convergence and the inverse-tangent divergence example.
<!-- bilingual-en:end -->

- MIT 18.330, [*Introduction to Numerical Analysis, Chapter 4: Nonlinear equations*](https://ocw.mit.edu/courses/18-330-introduction-to-numerical-analysis-spring-2012/5b325bfa56a599794c7196de926844b0_MIT18_330S12_Chapter4.pdf)：核对 Newton 的初值依赖、局部性和收敛失败。

<!-- bilingual-en:start -->
- SciPy’s linked newton documentation: initial/derivative interface, step-based stopping limitation and the bracketed-algorithm alternative.
<!-- bilingual-en:end -->

- SciPy, [`newton`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.newton.html)：核对初值、导数、步长停止不保证根以及有括区间时更安全算法的官方边界。

<!-- bilingual-en:start -->
- The linked calculus exam discussion supplies the horizontal-tangent failure context.
<!-- bilingual-en:end -->

- [[01_Math/01_calculus/02_Applications_of_Differentiation.md#Exam 2 第 5 题：Newton 法为什么失败|微积分课程反例]]：核对水平切线和迭代失败的课程语境。

<!-- bilingual-en:start -->
The inverse-tangent growth estimate, cycle, wrong-root sequence, scaling and logarithm-domain examples are checked directly above. MIT’s statement that Newton converges only when already close is not used as a necessary condition for every successful initial guess. The local theorem supplies a sufficient neighborhood; other starting points may also succeed.
<!-- bilingual-en:end -->

反正切的增长估计、二周期、错误根序列、尺度与对数定义域例子均在正文直接验证。MIT 关于“只有已经足够接近才收敛”的措辞不当成所有成功初值的必要条件：局部定理给出充分邻域，其他初值也可能成功。
