---
aliases:
  - "Solow Growth Model, Steady State, and Convergence"
  - "Solow Model"
  - "索洛增长模型"
status: source-checked
---

# Solow 增长模型、稳态与收敛
<!-- bilingual-en:start -->
*The Solow Growth Model, Steady State, and Convergence*
<!-- bilingual-en:end -->

> [!summary] 快速恢复
> **它解决什么：** 给定储蓄、人口、折旧和外生技术增长时，解释每单位有效劳动的资本如何走向稳态，以及政策改变增长率还是收入水平。
> **具体锚点：** 储蓄率上升后，投资短期超过稀释所需，资本加深使人均产出较快增长；到更高新稳态后，没有更快技术进步便不会永久保持更高人均增长率。
> **核心难点：** “每名劳动者”和“每单位有效劳动”不是同一变量；稳态中后者不变，前者仍可随技术增长。
> **为什么重要：** 它是区分资本积累的水平效应、过渡增长和长期技术增长的基准。
> **继续：** 若要解释技术进步为什么发生，进入 [[内生增长理论]]；若要比较固定比例的不稳定，回到 [[Harrod—Domar 增长模型]]。
<!-- bilingual-en:start -->
> [!summary] Quick recovery
> **What it solves:** Given saving, population, depreciation, and exogenous technological growth, it explains how capital per unit of effective labor approaches a steady state and whether policy changes a growth rate or an income level.
> **Concrete anchor:** After the saving rate rises, investment temporarily exceeds the amount needed to offset dilution. Capital deepening accelerates output-per-person growth, but once the economy reaches a higher steady state, growth per person is not permanently faster without faster technology.
> **Central difficulty:** “Per worker” and “per unit of effective labor” are different variables. The latter is constant in steady state while the former can still grow with technology.
> **Why it matters:** It is the benchmark for distinguishing the level effect of capital accumulation, transitional growth, and long-run technological growth.
> **Continue:** To explain why technology advances, see [[内生增长理论|Endogenous Growth Theory]]. To compare fixed-proportion instability, return to [[Harrod—Domar 增长模型|The Harrod–Domar Growth Model]].
<!-- bilingual-en:end -->

> [!source] 本节依据
> - Solow（1956）《[A Contribution to the Theory of Economic Growth](https://doi.org/10.2307/1884513)》：核验可替代生产函数、资本动态与稳定调整。
> - Solow（1957）《[Technical Change and the Aggregate Production Function](https://doi.org/10.2307/1926047)》：核验增长核算与技术残差的原始口径。
<!-- bilingual-en:start -->
> [!source] Basis for this section
> - Solow (1956), “[A Contribution to the Theory of Economic Growth](https://doi.org/10.2307/1884513),” verifies substitutable production, capital dynamics, and stable adjustment.
> - Solow (1957), “[Technical Change and the Aggregate Production Function](https://doi.org/10.2307/1926047),” verifies the original growth-accounting and technical-residual convention.
<!-- bilingual-en:end -->

## 基本动态方程
<!-- bilingual-en:start -->
*The fundamental dynamic equation*
<!-- bilingual-en:end -->

每有效劳动资本 $k$ 满足 $\dot k=s f(k)-(n+g+\delta)k$。储蓄投资增加 k，人口、技术有效劳动增长和折旧稀释 k。稳态由两者相等，资本递减回报带来稳定调整。
<!-- bilingual-en:start -->
Capital per effective worker $k$ satisfies $\dot k=s f(k)-(n+g+\delta)k$. Saving and investment raise $k$, while population growth, growth in effective labor through technology, and depreciation dilute it. The steady state equates the two, and diminishing returns to capital produce stable adjustment.
<!-- bilingual-en:end -->

当 $sf(k)>(n+g+\delta)k$ 时，新投资超过维持现有 $k$ 所需的 break-even investment，$k$ 上升；反之下降。相图的稳定性来自 $f(k)$ 凹而稀释线线性，不是“经济总会自动最优”。
<!-- bilingual-en:start -->
When $sf(k)>(n+g+\delta)k$, new investment exceeds the break-even amount needed to maintain $k$, so $k$ rises; otherwise it falls. Stability in the phase diagram comes from concave $f(k)$ against a linear dilution line, not from a claim that the economy is automatically optimal.
<!-- bilingual-en:end -->

## 稳态、比较静态与黄金律
<!-- bilingual-en:start -->
*Steady state, comparative statics, and the Golden Rule*
<!-- bilingual-en:end -->

更高 s 提高稳态 k 和 y 的水平、在过渡期提高增长；更高 n 或 $\delta$ 降低人均稳态。黄金律选择使稳态消费最大，满足净资本边际产出与增长/折旧条件相配。
<!-- bilingual-en:start -->
A higher $s$ raises steady-state $k$ and $y$ and produces faster transitional growth. Higher $n$ or $\delta$ lowers the per-capita steady state. The Golden Rule chooses the steady state that maximizes consumption, matching net marginal product of capital to growth and depreciation requirements.
<!-- bilingual-en:end -->

若 $f(k)=k^\alpha$，则稳态满足 $sk^\alpha=(n+g+\delta)k$，因此

$$k^*=\left(\frac{s}{n+g+\delta}\right)^{\frac{1}{1-\alpha}}.$$

例如 $\alpha=1/3$、$s=0.24$、$n+g+\delta=0.06$，则 $k^*=(4)^{3/2}=8$，$y^*=8^{1/3}=2$。若 $s$ 上升，$k^*$ 和 $y^*$ 上升，但在新稳态中每有效劳动产出仍不再增长。
<!-- bilingual-en:start -->
If $f(k)=k^\alpha$, steady state requires $sk^\alpha=(n+g+\delta)k$, so

$$k^*=\left(\frac{s}{n+g+\delta}\right)^{\frac{1}{1-\alpha}}.$$

For $\alpha=1/3$, $s=0.24$, and $n+g+\delta=0.06$, $k^*=(4)^{3/2}=8$ and $y^*=8^{1/3}=2$. A higher $s$ raises $k^*$ and $y^*$, but output per effective worker still stops growing in the new steady state.
<!-- bilingual-en:end -->

## 收敛与增长核算
<!-- bilingual-en:start -->
*Convergence and growth accounting*
<!-- bilingual-en:end -->

相同结构参数的经济体有条件收敛，初始资本低者增长更快；现实参数差异破坏无条件收敛。增长核算把产出增长分为资本、劳动和 TFP 残差，残差含技术也含测量和利用率。
<!-- bilingual-en:start -->
Economies with the same structural parameters exhibit conditional convergence, so an economy with lower initial capital grows faster. Parameter differences in reality prevent unconditional convergence. Growth accounting separates output growth into capital, labor, and a TFP residual; the residual includes technology but also measurement and utilization.
<!-- bilingual-en:end -->

对 Cobb–Douglas $Y=AK^\alpha L^{1-\alpha}$ 取对数差分，得

$$g_Y=g_A+\alpha g_K+(1-\alpha)g_L.$$

因而 $g_A$ 是在生产函数、份额权重和完全测量成立下所推算的残差。它不是直接观测的“纯技术”，还可吸收资本利用率、投入质量和错配的变化。
<!-- bilingual-en:start -->
Log-differentiating Cobb–Douglas $Y=AK^\alpha L^{1-\alpha}$ gives

$$g_Y=g_A+\alpha g_K+(1-\alpha)g_L.$$

Thus $g_A$ is a residual inferred under the production-function form, factor-share weights, and accurate measurement. It is not directly observed “pure technology” and may absorb capital utilization, input quality, and misallocation.
<!-- bilingual-en:end -->

## 技术进步的外生位置
<!-- bilingual-en:start -->
*The exogenous place of technological progress*
<!-- bilingual-en:end -->

外生技术增长在 Solow 中决定长期人均增长，但模型不解释其来源。把技术当公共知识、研发结果或人力资本积累是内生增长的任务。
<!-- bilingual-en:start -->
Exogenous technological growth determines long-run growth per person in Solow, but the model does not explain where that technology comes from. Treating technology as public knowledge, an R&D outcome, or human-capital accumulation is the task of endogenous growth theory.
<!-- bilingual-en:end -->

## 失败诊断
<!-- bilingual-en:start -->
*Failure diagnosis*
<!-- bilingual-en:end -->

- 把储蓄率上升的过渡增长误写成人均增长率永久上升。
- 说“稳态没有增长”时不指明是每有效劳动变量，与人均变量混淆。
- 把条件收敛误写成任何穷国都会无条件追上任何富国。
- 把 Solow 残差直接叫作纯技术进步。
<!-- bilingual-en:start -->
- Mistaking transitional growth after a higher saving rate for a permanent increase in growth per person.
- Saying “there is no growth in steady state” without specifying that the statement concerns per-effective-worker variables.
- Misreading conditional convergence as every poor country unconditionally catching every rich country.
- Calling the Solow residual pure technological progress.
<!-- bilingual-en:end -->

## 最小自检
<!-- bilingual-en:start -->
*Minimum self-check*
<!-- bilingual-en:end -->

### Solow 模型中提高储蓄率长期改变增长率还是水平？
<!-- bilingual-en:start -->
*In the Solow model, does a higher saving rate change the long-run growth rate or the level?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 在外生技术增长率不变时，主要提高稳态人均产出水平并产生过渡增长，不永久提高稳态人均增长率。
<!-- bilingual-en:start -->
> [!answer]- Answer
> With unchanged exogenous technological growth, it mainly raises the steady-state level of output per person and creates transitional growth, not a permanent increase in the steady-state growth rate per person.
<!-- bilingual-en:end -->

### 用自己的话解释 break-even investment。
<!-- bilingual-en:start -->
*Explain break-even investment in your own words.*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 它是用于补偿折旧，并为新增人口和有效劳动配备资本，从而仅保持现有每有效劳动资本不变的投资。
<!-- bilingual-en:start -->
> [!answer]- Answer
> It is the investment needed to replace depreciation and equip additional population and effective labor, merely keeping capital per effective worker unchanged.
<!-- bilingual-en:end -->

### 增长核算的 TFP 残差能否直接叫纯技术进步？
<!-- bilingual-en:start -->
*Can the TFP residual in growth accounting be called pure technological progress?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 不能；还含测量误差、产能利用、资源配置和遗漏投入质量。
<!-- bilingual-en:start -->
> [!answer]- Answer
> No. It can also contain measurement error, capacity utilization, resource allocation, and omitted input quality.
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [[02_Economy/10_发展经济学/发展经济学拍屏ppt.pdf]]：支持课程范围、案例和课堂模型。
- Solow（1956）《[A Contribution to the Theory of Economic Growth](https://doi.org/10.2307/1884513)》：核对资本递减收益、要素替代和稳定调整。
- Solow（1957）《[Technical Change and the Aggregate Production Function](https://doi.org/10.2307/1926047)》：核对增长核算口径与残差解释。
- 已从 $sk^\alpha=(n+g+\delta)k$ 重做 Cobb–Douglas 稳态推导，并重算 $k^*=8, y^*=2$。
<!-- bilingual-en:start -->
- The course slide PDF supports course scope, examples, and classroom models.
- Solow (1956), “[A Contribution to the Theory of Economic Growth](https://doi.org/10.2307/1884513),” verifies diminishing returns to capital, factor substitution, and stable adjustment.
- Solow (1957), “[Technical Change and the Aggregate Production Function](https://doi.org/10.2307/1926047),” verifies growth-accounting conventions and residual interpretation.
- The Cobb–Douglas steady state was rederived from $sk^\alpha=(n+g+\delta)k$, and $k^*=8, y^*=2$ were recomputed.
<!-- bilingual-en:end -->
