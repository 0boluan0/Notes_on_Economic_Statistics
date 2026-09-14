---
aliases:
  - "Proxy SVAR 用外部工具从简约型创新中识别目标结构冲击"
  - External-instrument SVAR
  - Proxy SVAR
  - SVAR 外部工具
student_os: knowledge-atom
atom_id: TS-VAR-015
atom_set: vector-autoregression
atom_type: definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[结构VAR]]"
  - "[[简约型VAR创新]]"
related:
  - "[[外部工具识别条件]]"
  - "[[结构脉冲响应]]"
part_of:
  - "[[VAR、脉冲响应与 Granger 因果.canvas]]"
---

# Proxy SVAR 用外部工具从简约型创新中识别目标结构冲击
<!-- bilingual-en:start -->
*A Proxy SVAR uses an external instrument to identify a target structural shock inside the reduced-form innovations*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> Proxy SVAR（外部工具 SVAR）不靠变量排序直接命名冲击，而是用一个系统外变量 $z_t$ 为目标结构冲击提供方向信息。它通过 $z_t$ 与简约型创新 $u_t$ 的共变动，识别目标冲击的当期影响向量，再用 VAR 的 VMA 动态把这个方向向后传播。

<!-- bilingual-en:start -->
> [!summary] What it is
> A Proxy SVAR, or external-instrument SVAR, uses a variable $z_t$ from outside the system to provide directional information about a target structural shock. The covariance between $z_t$ and the reduced-form innovations $u_t$ identifies the shock's impact vector up to scale, which is then propagated through the VAR's VMA dynamics.
<!-- bilingual-en:end -->

若结构映射为
$$
u_t=B\varepsilon_t,
$$
并且 $z_t$ 只与目标冲击 $\varepsilon_{1t}$ 相关，则
$$
E(z_tu_t)=B\,E(z_t\varepsilon_t)
=b_1E(z_t\varepsilon_{1t}),
$$
其中 $b_1$ 是 $B$ 的第一列。所以 $E(z_tu_t)$ 的方向与目标冲击的当期影响向量 $b_1$ 平行。其比例尺度 $E(z_t\varepsilon_{1t})$ 仍未知，故需要另外规定“一个冲击单位”的经济定标。

<!-- bilingual-en:start -->
If $u_t=B\varepsilon_t$ and $z_t$ is correlated only with target shock $\varepsilon_{1t}$, then $E(z_tu_t)=b_1E(z_t\varepsilon_{1t})$, where $b_1$ is the first column of $B$. The covariance vector is therefore parallel to the target shock's impact vector. Its scale remains unknown and must be normalized into an economically interpretable shock unit.
<!-- bilingual-en:end -->

实证上，先估计简约型 VAR 并取得 $\hat u_t$，再用工具与这些创新的样本共变动估计影响方向，最后通过 $\Phi_h$ 得到各期限响应。Proxy SVAR 可以只识别一个目标冲击，不要求把所有其他创新也逐一命名。这是它与一个完全递归 Cholesky 系统的重要差别。

<!-- bilingual-en:start -->
In practice, estimate the reduced-form VAR, obtain $\hat u_t$, estimate the impact direction from its covariance with the instrument, and propagate that direction with $\Phi_h$. A Proxy SVAR may identify only one target shock; it need not assign economic names to every remaining innovation, unlike a fully recursive Cholesky system.
<!-- bilingual-en:end -->

这个方法的定义不等于它在某个应用中有效。工具是否真的只代理目标冲击、是否足够强，以及动态传播是否满足可恢复性或相应排除条件，应在独立的 [[外部工具识别条件]] 中审计。

<!-- bilingual-en:start -->
Defining the method is not the same as validating an application. Whether the instrument exclusively tracks the target shock, is sufficiently strong, and satisfies the recoverability or dynamic-exclusion conditions is audited separately in [[外部工具识别条件|the external-instrument validity atom]].
<!-- bilingual-en:end -->

> [!question]- 自检
> Proxy SVAR 是否必须把 VAR 中的每一个创新都识别并命名？
>
> **答案：** 不必须。一个有效外部工具可以只识别目标冲击的方向与响应；其他创新可以保持为未命名的正交补空间。

## 来源与核验

- [Stock & Watson (2018), *Identification and Estimation of Dynamic Causal Effects in Macroeconomics Using External Instruments*](https://www.nber.org/papers/w24216)：核对外部工具如何通过简约型创新识别目标冲击方向及动态响应。
- [Mertens & Ravn (2013), *The Dynamic Effects of Personal and Corporate Income Tax Changes in the United States*](https://doi.org/10.1257/aer.103.4.1212)：核对以叙事税收变化代理目标结构冲击的应用。
- [Kilian & Lütkepohl (2017), *Structural Vector Autoregressive Analysis*](https://doi.org/10.1017/9781108164818)：核对外部工具在 SVAR 识别谱系中的位置。
