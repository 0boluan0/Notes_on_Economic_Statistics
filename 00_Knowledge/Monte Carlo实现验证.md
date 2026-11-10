---
aliases:
  - "Monte Carlo implementation verification"
  - "模拟实现验证"
student_os: knowledge-atom
atom_id: PROB-MC-006
atom_set: monte-carlo-methods
atom_type: procedure
status: source-checked
part_of:
  - "[[Monte Carlo.canvas|Monte Carlo]]"
requires:
  - "[[Monte Carlo估计]]"
  - "[[Monte Carlo误差分层]]"
  - "[[伪随机复现]]"
  - "[[并行伪随机子流]]"
related:
  - "[[路径数不修模型]]"
  - "[[风险模型验证边界]]"
---

<!-- bilingual-en:start -->
# Verifying a Monte Carlo implementation
<!-- bilingual-en:end -->

*Monte Carlo 实现验证*

<!-- bilingual-en:start -->
Verifying a Monte Carlo implementation means checking that the program carries out the probability experiment and calculation we intended. A plausible-looking simulated answer is weak evidence by itself: correct code can miss the target by chance, and wrong code can land near it. Useful tests make failures traceable to a particular part of the computation.
<!-- bilingual-en:end -->

Monte Carlo 实现验证，是检查程序是否执行了原本打算进行的概率实验和计算。一个“看起来合理”的模拟答案，单独作为证据很弱：正确代码可能碰巧偏离目标，错误代码也可能碰巧靠近它。有效的测试应让失败能追溯到计算中的具体部分。

<!-- bilingual-en:start -->
We will use one test problem throughout. Draw $U$ [[连续均匀分布|uniformly]] from $[0,1]$, compute the output $Y=U^2$, and estimate its [[期望|mean]] $\mu=E[Y]=1/3$. The exact value follows from [[LOTUS|integrating the squared input]], $\int_0^1u^2\,du=1/3$. This is a small, analytically solvable example of [[Monte Carlo估计|Monte Carlo estimation]], not a reason to simulate an integral that is already easy to calculate.
<!-- bilingual-en:end -->

下面始终使用一个测试问题：从 $[0,1]$ [[连续均匀分布|均匀抽取]] $U$，计算输出 $Y=U^2$，估计它的[[期望]] $\mu=1/3$。精确值由 [[LOTUS|对输入的平方积分]]得到。选择这个简单问题，是为了得到已知答案来检查 [[Monte Carlo估计]]程序。

<!-- bilingual-en:start -->
## A minimal executable sequence
<!-- bilingual-en:end -->

## 最小可执行顺序

<!-- bilingual-en:start -->
### State the calculation before testing the answer
For this example, one trial draws one number and squares it. A run fixes a positive integer $N$, takes $N$ [[独立同分布|iid]] trials, and averages the $N$ squared values. Record that rule, the input distribution, and any units or conditioning information. In a path model, also specify the time horizon and what one complete path contributes to the final average. This separates the target from implementation choices such as storage layout or parallel workers.
<!-- bilingual-en:end -->

*先声明要算什么。* 本例一次实验抽一个数并平方；一轮运行先固定正整数 $N$，做 $N$ 次[[独立同分布|iid]]实验，再平均这些平方值。记录这条规则、输入分布，以及涉及的单位或条件信息。路径模型还要说明时间范围，以及一条完整路径向最终均值贡献什么。这样才不会把目标量与存储布局、并行 worker 等实现选择混在一起。

<!-- bilingual-en:start -->
### Feed known inputs to each component
Temporarily bypass random generation and feed the transformation and averaging functions a short list. On inputs $0$ and $1$, squaring then averaging must return $1/2$. On inputs $1/4$ and $3/4$, it must return $5/16$. These are deterministic input/output checks: repeated runs should not need a statistical tolerance for these exactly representable small arithmetic values.
<!-- bilingual-en:end -->

*用已知输入逐段检查。* 暂时绕开随机生成器，给变换与平均函数输入一个短列表。输入 $0,1$ 时，先平方再平均必须得到 $1/2$；输入 $1/4,3/4$ 时必须得到 $5/16$。这些小数值在二进制浮点下可精确表示，这里的简单运算检查不需要借助统计容差解释差异。

> [!example] A test that catches the order-of-operations bug · 抓住运算顺序错误
> <!-- bilingual-en:start -->
> Squaring the mean of $0,1$ gives $1/4$, so the first test detects that bug immediately. A test using only identical inputs, such as $1/2,1/2$, would not: both rules return $1/4$. Choose cases that separate plausible wrong implementations, not only convenient cases on which they agree.
> <!-- bilingual-en:end -->
>
> 先平均 $0,1$ 再平方会得到 $1/4$，因此第一个测试立即发现错误。若只测试完全相同的输入，例如 $1/2,1/2$，两条规则都返回 $1/4$，就发现不了。测试样例需要能区分可能写错的实现，而不只是挑选大家恰好一致的方便情况。

<!-- bilingual-en:start -->
Check generation separately: values should lie in the intended range, and distributional diagnostics should match the specified law. In a stateful simulation, check the update rule on controlled states and any constraints the model is supposed to preserve. A range check alone cannot distinguish uniform inputs from inputs concentrated near zero, and correct one-variable distributions do not establish the required joint independence.
<!-- bilingual-en:end -->

随机生成也要单独检查：取值应落在规定范围，分布诊断应与指定规律相符。对于带状态的模拟，应在受控状态上核对更新规则，以及模型本应保持的约束。仅检查范围，不能区分均匀输入和集中在零附近的输入；每个变量各自的分布正确，也不能证明联合独立性。

<!-- bilingual-en:start -->
### Check sampling fluctuations under a fixed implementation
Keep the model and numerical settings fixed. Repeat whole runs using a documented random-stream scheme, and compare their behaviour with the predicted [[Monte Carlo均值标准误|standard error]]. For our iid uniform example, $E[U^4]=1/5$ and hence $\operatorname{Var}(U^2)=1/5-1/9=4/45$. The exact repeated-run variance of the average is $4/(45N)$. Repeated results should be assessed against this sampling variability, rather than required to equal $1/3$ exactly.
<!-- bilingual-en:end -->

*在固定实现下检查抽样波动。* 固定模型与数值设置，使用有文档依据的随机流方案重复整轮运行，并与理论[[Monte Carlo均值标准误|标准误]]比较。本例 $E[U^4]=1/5$，因此 $\operatorname{Var}(U^2)=4/45$，iid 条件下模拟均值的方差为 $4/(45N)$。评价重复结果时要考虑这种抽样变动，而不是要求每轮都恰好等于 $1/3$。

<!-- bilingual-en:start -->
Statistical checks have false alarms and limited ability to detect errors. Choose their tolerances and replication plan before inspecting the results; do not keep rerunning until one answer passes. A good mean alone can also miss duplicated streams. If every trial reuses a single uniform draw, each input still has the right marginal distribution, but averaging its repeated square does not reduce the true variance with $N$.
<!-- bilingual-en:end -->

统计检查可能误报，也只能发现部分错误。应在查看结果前确定容差与重复方案，不能一直重跑直到某个答案“通过”。只看均值也可能漏掉随机流复制：若每次都重用同一个均匀随机数，单次输入的边际分布仍正确，但反复平均它的平方不会让真实方差随 $N$ 缩小。

<!-- bilingual-en:start -->
For parallel work, follow the generator's documented [[并行伪随机子流|substream construction]], such as a spawn tree, supported jump-ahead scheme, or distinct keys where specified. Different integer seeds alone are not a general independence proof. These constructions support the intended sampling design; empirical checks for duplicated or abnormal streams remain useful.
<!-- bilingual-en:end -->

并行运行时，使用生成器有文档支持的[[并行伪随机子流|子流构造]]，例如 spawn 树、受支持的跳跃方案，或规定机制下的不同 key。整数 seed 不同，本身不是通用的独立性证明。这些构造为抽样设计提供依据，同时仍应检查重复流或异常流。

<!-- bilingual-en:start -->
### Change numerical accuracy separately from trial count
When the calculation uses a grid or solver, compare refinements while preserving the same target. The midpoint example in [[Monte Carlo误差分层|the error-layer card]] has expectation $5/16$ on a two-cell grid and $21/64$ on a four-cell grid. More random trials at the first resolution cannot remove that difference. Separate changes to $N$ from changes to the grid or solver tolerance so their effects can be interpreted.
<!-- bilingual-en:end -->

*分别改变数值精度与重复次数。* 使用网格或求解器时，在目标不变的前提下比较精化设置。[[Monte Carlo误差分层]]中的中点例子，在两个网格下期望为 $5/16$，四个网格下为 $21/64$。在第一种精度下增加随机重复，不能消除这个差别。把 $N$ 的变化与网格或求解容差的变化分开，才能解释结果。

<!-- bilingual-en:start -->
Shared underlying random inputs can make such comparisons easier to interpret, but the correspondence must be constructed deliberately. Reusing the same seed does not guarantee matched paths when a finer grid consumes random numbers in a different order. A systematic shift under refinement warrants investigation; neither one shift nor one apparent plateau proves numerical convergence. Also test numerical edge cases, such as variance calculation when outputs have a large common offset and very small spread: subtraction of nearly equal large numbers can lose precision.
<!-- bilingual-en:end -->

共用底层随机输入有时更便于比较，但需要有意构造它们的对应关系。细网格若按不同顺序消耗随机数，仅重复使用同一个 seed 不保证路径一一匹配。精化后的系统性变化值得调查，但一次变化或一次看似稳定，都不能证明数值收敛。还应测试数值边界，例如输出具有很大共同偏移、波动却很小时的方差计算；两个几乎相等的大数相减，可能损失精度。

<!-- bilingual-en:start -->
### Keep enough evidence to replay a failure
Record the code and library versions, generator algorithm and initialisation, stream-to-task mapping, parameters, input data, call protocol, numerical settings, and outputs used in the check. [[伪随机复现|Pseudorandom reproducibility]] depends on more than a seed: NumPy's stream-compatibility conditions also include the generator, call sequence and arguments, build, environment, and machine. State what was checked and which parts or regimes remain untested.
<!-- bilingual-en:end -->

*保留能够重放失败的证据。* 记录代码与库版本、生成器算法和初始化、随机流到任务的分配、参数、输入数据、调用协议、数值设置，以及检查所用输出。[[伪随机复现]]不只依赖 seed：NumPy 的流兼容条件还包括生成器、调用顺序与参数、构建版本、环境和机器。记录已经检查的范围，也明确哪些部分或运行区间尚未测试。

> [!warning] Verification has an intended scope · 验证有明确的范围
> <!-- bilingual-en:start -->
> These checks build evidence that code implements a stated model and achieves an assessed numerical accuracy. They do not establish that the model represents the real system well enough for a decision. That requires validation against relevant external evidence. Passing a few test cases cannot certify every input, and a reproducible wrong answer is still wrong.
> <!-- bilingual-en:end -->
>
> 这些检查为“代码执行了指定模型，并达到已评估的数值精度”积累证据；模型是否足以支持现实决策，还需要与相关外部证据比较。少量测试通过不能保证所有输入都正确，可复现的错误答案仍然是错误答案。

## 来源与核验

<!-- bilingual-en:start -->
- [[03_Computer_Science/03_MIT 6.100L/MIT 6.100L-slides/mit6_100l_lec26.pdf#page=21|MIT 6.100L Lecture 26, slides 21–28]]: the course framework of specifying an experiment, repeating, recording, and summarising; includes tractable dice cases and a transformed-output mean.
- Art B. Owen, [*Monte Carlo Theory, Methods and Examples*, Chapter 2, §§2.1–2.3, pp. 15–21](https://artowen.su.domains/mc/Ch-intro.pdf#page=15): sampling variability, estimated standard errors, and numerical stability of variance computation. The uniform-square tests are direct calculations.
- NASA, [*NASA-STD-7009B* (2024), §§4.2.2–4.2.6, pp. 27–30](https://standards.nasa.gov/sites/default/files/standards/NASA/B/1/NASA-STD-7009B-Final-3-5-2024.pdf#page=27): benchmark input/output cases, code versus solution verification, numerical-error records, and the separate responsibility of real-system validation. The sequence here applies those responsibilities to Monte Carlo; it is not a prescribed NASA checklist.
- NumPy Developers, [Parallel random number generation](https://numpy.org/doc/stable/reference/random/parallel.html) and [Compatibility policy](https://numpy.org/doc/stable/reference/random/compatibility.html): documented stream derivation and the conditions for reproducible streams, including limits on seed-only claims.
<!-- bilingual-en:end -->

- MIT 6.100L 同上第 21–28 张：支持课程中的实验定义、重复、记录与汇总框架，包括可直接核算的骰子问题与变换后输出均值。
- Owen 同上第 15–21 页：支持抽样波动、估计标准误与方差计算的数值稳定性；平方均匀输入的测试是直接计算。
- NASA 同上第 27–30 页：支持基准输入/输出用例、代码与解验证、数值误差记录，以及现实系统有效性验证的独立责任。这里把这些责任用于 Monte Carlo，没有冒充 NASA 规定的操作清单。
- NumPy 官方并行生成与兼容文档：支持随机流派生方法与复现条件，明确仅凭 seed 不能作出哪些保证。
