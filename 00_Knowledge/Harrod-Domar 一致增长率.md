---
aliases:
  - "固定资本产出比与储蓄转投资给出计划一致增长率 s 除以 v 而不是预测"
  - Fixed capital output ratio and saving imply a consistency rate s over v
  - Harrod Domar s over v
  - 保证增长率 s 除以 v
student_os: knowledge-atom
atom_id: DEV-HD-002
atom_set: harrod-domar-growth
atom_type: derivation-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[投资双重效应]]"
part_of:
  - "[[Harrod—Domar 增长模型.canvas]]"
implies:
  - "[[Harrod 三种增长率]]"
related:
  - "[[Harrod 与 Domar 的机制差异]]"
---

# 固定资本产出比与储蓄转投资给出计划一致增长率 s 除以 v 而不是预测
<!-- bilingual-en:start -->
*A fixed capital–output ratio and saving-to-investment conversion imply the consistency rate $s/v$, not a forecast*
<!-- bilingual-en:end -->

> [!summary] 原子推导
> 在现代教科书的固定系数重写中，令 $s=S/Y$ 为计划储蓄率，$v=\Delta K/\Delta Y$ 为增量资本—产出比。若计划储蓄与企业所需投资相等、$S=I^d$，净投资形成资本、$I^d=\Delta K$，并且生产额外产出需要固定比例资本、$\Delta K=v\Delta Y$，则
> $$g_w=\frac{\Delta Y}{Y}=\frac{s}{v}.$$
> 这给出使计划储蓄、所需投资与产能扩张彼此一致的增长率，不保证实际经济会实现它。
> <!-- bilingual-en:start -->
> Under fixed coefficients, net capital formation, and equality between planned saving and required investment, $g_w=s/v$ is the growth rate consistent with firms' capital requirements. It is a conditional consistency relation, not an unconditional prediction.
> <!-- bilingual-en:end -->

教科书重写可用三步表示：
$$
S=sY,\qquad S=I^d=I=\Delta K,\qquad \Delta K=v\Delta Y.
$$
把三式连接起来，
$$
sY=v\Delta Y
\quad\Longrightarrow\quad
\frac{\Delta Y}{Y}=\frac{s}{v}.
$$

这里的 $S=I^d$ 是**计划一致条件**。闭合经济中实现值的储蓄与投资相等是一条事后核算恒等式；它本身不能证明企业原先想投的数量恰好等于家庭原先想储蓄的数量，也不能给出离开保证路径后的调整过程。

不同教材可能用相反记号。本课程在资本充分利用的支路写 $Y=VK$，把 $V=Y/K$ 称作产出—资本比；若劳动约束先绑定，则实际产出满足 $Y<VK$，实际 $Y/K$ 并不等于这个产能参数。只有当资本支路绑定、$V$ 恒定、关系通过原点且同一约束适用于增量时，才有
$$
v=\frac{\Delta K}{\Delta Y}=\frac{K}{Y}=\frac{1}{V},
$$
从而同一条件写成 $g_w=Vs$。看到 $s/v$ 或 $Vs$ 时，必须先确认水平比和增量比能否这样互换。

例如 $s=0.20$、$v=4$，则简化的一致增长率为 $5\%$。这个计算并没有证明计划会实现，也没有证明资本系数固定、产能会被利用、劳动与进口设备不构成约束。

> [!warning] 不是政策乘法器
> 从公式看，提高 $s$ 会提高条件性的 $g_w$；但若额外储蓄没有形成所需资本，或需求、利用率、技术和劳动力条件改变，实际增长不会按同一比例机械上升。若 $I$ 是总投资且折旧率为 $\delta$，资本积累式变成 $\Delta K=I-\delta K$；在固定 $K/Y=v$ 的平衡路径上，相应关系是 $g=s/v-\delta$，而不是继续无条件使用 $s/v$。

> [!question]- 自检
> 为什么 $s/v=5\%$ 不能直接读作“明年的实际 GDP 增长率是 5%”？
>
> **答案：** 它依赖计划储蓄等于所需投资、固定资本系数、净投资口径和产能利用等条件，只给出计划一致所需的增长率；实际需求与投资行为未被等式自动决定。

## 来源与核验

- Harrod（1939），[An Essay in Dynamic Theory](https://doi.org/10.2307/2225181)：核对保证增长率作为储蓄供给与投资需求相容条件的原始问题。
- Blume 与 Sargent（2015），[Harrod 1939](https://doi.org/10.1111/ecoj.12224)：核对 Harrod 原式 $S_t=sY_t$、$I_t=g(Y_{t+1}-Y_t)$ 与 $S_t=I_t$，以及把 $g$ 固定解释为资本—产出系数是后来的教科书读法。
- [[02_Economy/10_发展经济学/06_经济增长理论.md#2.2 模型内容|发展经济学课程 §2.2]]：核对本课程以 $V=Y/K$ 写成 $g_w=Vs$ 的相反记号。
