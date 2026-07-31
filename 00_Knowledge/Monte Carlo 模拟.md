---
aliases:
  - "Monte Carlo Simulation"
  - "随机模拟"
  - "蒙特卡罗模拟"
status: source-checked
---

# Monte Carlo 模拟
<!-- bilingual-en:start -->
*Monte Carlo Simulation*
<!-- bilingual-en:end -->

> [!summary] 快速恢复
> **它解决什么：** 用伪随机重复实验近似难以解析求出的概率、期望和分布。
> **具体锚点：** 估计复杂事件概率时重复模拟，样本比例会随次数增加稳定，但 Monte Carlo 标准误只按 $1/\sqrt n$ 缩小。
> **核心难点：** 必须把模型误设、实现错误与 Monte Carlo 抽样误差分开；设 seed 只保证特定生成器和环境下可复现。
> **为什么重要：** 当解析计算太难时，模拟仍能比较方案、传播不确定性并量化估计精度。
> **继续：** 先用已知小例子验证模拟器，再增加次数并报告 Monte Carlo 标准误；结果表达见 [[数据可视化与不确定性表达]]。
> <!-- bilingual-en:start -->
> **Problem addressed:** Approximate probabilities, expectations, and distributions that are difficult to derive analytically through repeated pseudorandom experiments.
> **Concrete anchor:** Repeated simulation can estimate a complicated event probability, but Monte Carlo standard error shrinks only as $1/\sqrt n$.
> **Central difficulty:** Model misspecification, implementation errors, and Monte Carlo sampling error must be separated; a seed only reproduces a sequence for a particular generator and environment.
> **Why it matters:** When analytic calculation is impractical, simulation can still compare alternatives, propagate uncertainty, and quantify estimation precision.
> **Continue with:** Validate the simulator against a known small case, then increase the run count and report Monte Carlo standard error; present results with [[数据可视化与不确定性表达|Data Visualization and Uncertainty Communication]].
> <!-- bilingual-en:end -->

> [!source] 本节依据
> - MIT 6.100L 本地官方 slides、transcripts、finger exercises 与 problem sets：支持每个概念、示例和课程范围。
> - [[03_Computer_Science/03_MIT 6.100L/Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf]]：交叉核验 Python、算法、复杂度、OOP 与模拟。
> <!-- bilingual-en:start -->
> - Local official MIT 6.100L slides, transcripts, finger exercises, and problem sets support the concepts, examples, and course scope.
> - [[03_Computer_Science/03_MIT 6.100L/Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Introduction to Computation and Programming Using Python]] cross-checks Python, algorithms, complexity, object-oriented programming, and simulation.
> <!-- bilingual-en:end -->

## 伪随机数
<!-- bilingual-en:start -->
*Pseudorandom Numbers*
<!-- bilingual-en:end -->

PRNG 从 seed 确定性产生看似随机序列，适合模拟但非所有密码用途。可复现工作记录 seed、生成器、版本和参数；并行模拟需避免重复子流。
<!-- bilingual-en:start -->
A PRNG deterministically generates an apparently random sequence from a seed. It suits simulation but not every cryptographic purpose. Reproducible work records the seed, generator, version, and parameters; parallel simulations must avoid overlapping substreams.
<!-- bilingual-en:end -->

## Monte Carlo 设计
<!-- bilingual-en:start -->
*Monte Carlo Design*
<!-- bilingual-en:end -->

明确随机机制、一次 trial、统计量和重复数。独立重复下估计均值标准误为样本标准差除以 $\sqrt n$。想把误差减半通常需约四倍样本，计算预算应据此规划。
<!-- bilingual-en:start -->
Specify the random mechanism, one trial, the statistic, and the repetition count. Under independent repetition, the estimated mean has standard error equal to sample standard deviation divided by $\sqrt n$. Halving error normally requires about four times as many trials.
<!-- bilingual-en:end -->

## 验证模拟
<!-- bilingual-en:start -->
*Validating a Simulation*
<!-- bilingual-en:end -->

对可解析特例检查均值、方差和分布；逐层测试生成、状态更新和汇总。bug 可稳定地产生漂亮结果，增加重复次数只降低错误模型的随机噪声。
<!-- bilingual-en:start -->
Check mean, variance, and distribution on analytically tractable cases. Test generation, state transitions, and aggregation separately. A bug can produce beautiful stable output; more repetitions only reduce random noise around the wrong model.
<!-- bilingual-en:end -->

## variance reduction 入口
<!-- bilingual-en:start -->
*Entry Point to Variance Reduction*
<!-- bilingual-en:end -->

共同随机数比较方案、antithetic variables、control variates 和 importance sampling 可提效，但需保持无偏/正确权重。稀有事件直接模拟可能几乎观察不到目标。
<!-- bilingual-en:start -->
Common random numbers, antithetic variables, control variates, and importance sampling can improve efficiency, but unbiasedness or correct weighting must be preserved. Direct simulation of a rare event may observe almost no target events.
<!-- bilingual-en:end -->

## Worked example：估计圆周率
<!-- bilingual-en:start -->
*Worked Example: Estimate Pi*
<!-- bilingual-en:end -->

在单位正方形均匀采样 $(x,y)$，落入四分之一圆的概率为 $\pi/4$，所以四倍命中比例估计 $\pi$。可解析答案让我们同时验证生成范围、事件判定和标准误趋势。
<!-- bilingual-en:start -->
Sample $(x,y)$ uniformly in the unit square. The probability of landing inside the quarter circle is $\pi/4$, so four times the hit proportion estimates $\pi$. The analytic answer validates the sampling range, event predicate, and standard-error trend together.
<!-- bilingual-en:end -->

```python
import random

def estimate_pi(trials, seed=0):
    rng = random.Random(seed)
    hits = sum(rng.random() ** 2 + rng.random() ** 2 <= 1 for _ in range(trials))
    return 4 * hits / trials
```

## 失败诊断
<!-- bilingual-en:start -->
*Failure Diagnosis*
<!-- bilingual-en:end -->

- 加倍次数但误差未明显下降：一次运行波动很大，应比较重复批次的标准误趋势，并检查 trial 是否独立。
  <!-- bilingual-en:start -->
  Doubling trials does not visibly reduce error: one run is noisy, so compare repeated batches and inspect whether trials are independent.
  <!-- bilingual-en:end -->
- seed 相同却结果不同：记录生成器、版本、并行分流、数据处理顺序和浮点环境。
  <!-- bilingual-en:start -->
  The same seed gives different results: record the generator, version, parallel stream allocation, processing order, and floating-point environment.
  <!-- bilingual-en:end -->
- 结果非常稳定但偏离真值：优先查模型与代码偏差，而不是继续增加 trials。
  <!-- bilingual-en:start -->
  The result is stable but biased: inspect model and implementation bias before adding more trials.
  <!-- bilingual-en:end -->

## 最小自检
<!-- bilingual-en:start -->
*Minimum Self-Check*
<!-- bilingual-en:end -->

### Monte Carlo 次数增加四倍，标准误大约怎样变化？
<!-- bilingual-en:start -->
*How does Monte Carlo standard error change when the trial count is quadrupled?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 在独立同分布条件下减半，因为标准误按 $1/\sqrt n$ 缩小。
> <!-- bilingual-en:start -->
> It halves under independent, identically distributed trials because standard error scales as $1/\sqrt n$.
> <!-- bilingual-en:end -->

### 设置 seed 能保证什么，不能保证什么？
<!-- bilingual-en:start -->
*What does setting a seed guarantee, and what does it not guarantee?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 帮助复现伪随机序列；不能保证随机模型、代码或结论正确。
> <!-- bilingual-en:start -->
> It helps reproduce a pseudorandom sequence; it does not guarantee that the stochastic model, code, or conclusion is correct.
> <!-- bilingual-en:end -->

### 为什么模拟前先做解析小例子？
<!-- bilingual-en:start -->
*Why use an analytic small case before a large simulation?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 能验证生成和汇总逻辑；否则大量重复只会更精确地估计错误程序。
> <!-- bilingual-en:start -->
> It validates generation and aggregation logic; otherwise many repetitions merely estimate an incorrect program more precisely.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and Verification*
<!-- bilingual-en:end -->

- MIT 6.100L 本地官方 slides、transcripts、finger exercises 与 problem sets：支持每个概念、示例和课程范围。
  <!-- bilingual-en:start -->
  Local official MIT 6.100L slides, transcripts, finger exercises, and problem sets support the concepts, examples, and course scope.
  <!-- bilingual-en:end -->
- [[03_Computer_Science/03_MIT 6.100L/Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf]]：交叉核验 Python、算法、复杂度、OOP 与模拟。
  <!-- bilingual-en:start -->
  [[03_Computer_Science/03_MIT 6.100L/Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Introduction to Computation and Programming Using Python]] cross-checks simulation design, error, and validation.
  <!-- bilingual-en:end -->
