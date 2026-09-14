---
aliases:
  - 常储蓄 AK 的人均资本与产出按 sA−δ−n 增长
  - Constant-saving AK per-capita growth condition
student_os: knowledge-atom
atom_id: MACRO-ENDO-004
atom_type: proposition
status: source-checked
part_of:
  - "[[内生增长与创新.canvas]]"
---

# 常储蓄 AK 的人均资本与产出按 sA−δ−n 增长
<!-- bilingual-en:start -->
*In the constant-saving AK model, capital and output per person grow at sA−δ−n.*
<!-- bilingual-en:end -->

在连续时间 [[AK模型]] 中，设 $Y=AK$，总投资为 $I=sY$，其中 $A>0$、$0<s<1$、折旧率 $\delta\ge0$ 均固定；人口与劳动人数相同，$L(t)=L_0e^{nt}$，$n\ge0$。从正的初始资本出发，人均资本 $k=K/L$ 与人均产出 $y=Y/L=Ak$ 能持续正增长，当且仅当 $sA>\delta+n$。总量正增长所需的 $sA>\delta$ 是另一条件。
<!-- bilingual-en:start -->
Consider a continuous-time [[AK模型|AK model]] with $Y=AK$, investment $I=sY$, fixed $A>0$, $0<s<1$ and depreciation $\delta\ge0$. Population and labour coincide and satisfy $L(t)=L_0e^{nt}$, $n\ge0$. Starting from positive capital, per-capita capital $k=K/L$ and output $y=Ak$ grow persistently at a positive rate exactly when $sA>\delta+n$. Positive aggregate growth instead requires $sA>\delta$.
<!-- bilingual-en:end -->

先从总量资源约束得到净投资，再对人均定义求导：
<!-- bilingual-en:start -->
First obtain net investment from the aggregate resource constraint, then differentiate the per-capita definition:
<!-- bilingual-en:end -->

$$
\dot K=sAK-\delta K,
\qquad
\frac{\dot k}{k}
=\frac{\dot K}{K}-\frac{\dot L}{L}
=sA-\delta-n\equiv g_k.
$$

因此总资本和总产出按 $sA-\delta$ 增长；每人资本、产出以及消费 $c=(1-s)Ak$ 都按 $g_k$ 增长。$n$ 是分母增加造成的人口稀释，不是额外的资本折旧。这里没有劳动增进技术的有效劳动分母，也不应再减一个技术增长率，相关口径见 [[Solow 人均化边界]]。
<!-- bilingual-en:start -->
Aggregate capital and output grow at $sA-\delta$, while per-capita capital, output and consumption $c=(1-s)Ak$ grow at $g_k$. The term $n$ reflects a growing population denominator, not additional physical depreciation. No effective-labour denominator is used here, so no technological growth rate should be subtracted; see [[Solow 人均化边界|the normalisation boundary]].
<!-- bilingual-en:end -->

运动方程的解为
<!-- bilingual-en:start -->
The solution to the law of motion is:
<!-- bilingual-en:end -->

$$
k(t)=k(0)e^{g_kt},\qquad y(t)=Ak(0)e^{g_kt}.
$$

若 $g_k>0$，人均水平持续上升；若 $g_k=0$，每个正初值都给出一条人均常值路径；若 $g_k<0$，人均水平下降。总量增长并不排除最后一种情况，例如 $0<sA-\delta<n$ 时，产出增长赶不上人口增长。
<!-- bilingual-en:start -->
For $g_k>0$, per-capita levels rise indefinitely. For $g_k=0$, every positive initial value gives a constant per-capita path. For $g_k<0$, per-capita levels decline. Aggregate output can still rise in the last case: if $0<sA-\delta<n$, output grows more slowly than population.
<!-- bilingual-en:end -->

构造例子：以年为时间单位，取 $A=0.5$、$s=0.20$、$\delta=0.05$、$n=0.02$、$K(0)=100$、$L(0)=10$。初始总产出为 $50$，总投资为 $10$，消费为 $40$，折旧为 $5$，所以净投资为 $5$。总量增长率是每年 $5\%$，人均增长率是每年 $3\%$；两者均为连续复利率。
<!-- bilingual-en:start -->
For a constructed annual example, take $A=0.5$, $s=0.20$, $\delta=0.05$, $n=0.02$, $K(0)=100$ and $L(0)=10$. Initial aggregate output is $50$, investment $10$, consumption $40$ and depreciation $5$, leaving net investment of $5$. Aggregate growth is $5\%$ and per-capita growth $3\%$ per year, both continuously compounded.
<!-- bilingual-en:end -->

十年后的计算同时保留总量和人均分母；数值四舍五入至四位小数：
<!-- bilingual-en:start -->
The ten-year calculation keeps aggregate quantities and the per-capita denominator explicit. Values are rounded to four decimal places:
<!-- bilingual-en:end -->

| 变量 / Variable | $t=0$ | $t=10$ |
| --- | ---: | ---: |
| 总资本 / Aggregate capital $K$ | 100 | $100e^{0.05\times10}=164.8721$ |
| 人口 / Population $L$ | 10 | $10e^{0.02\times10}=12.2140$ |
| 人均资本 / Capital per person $k$ | 10 | 13.4986 |
| 人均产出 / Output per person $y$ | 5 | 6.7493 |
| 人均消费 / Consumption per person $c$ | 4 | 5.3994 |

## 来源与核验

- [Acemoglu，MIT 14.452，2016 Lectures 2–3](https://ocw.mit.edu/courses/14-452-economic-growth-fall-2016/2b68057aa4e74410d00ae89a0c49752f_MIT14_452F16_Lec2and3.pdf#page=57)，PDF第57–59页：支持固定生产率 AK、人口增长、人均率 $sA-\delta-n$ 与指数解。总量/人均推导、零增长和负增长情况及数例均由该运动方程独立核算。
<!-- bilingual-en:start -->
The lecture supplies the AK specification, population growth, per-capita rate and exponential solution. The aggregate–per-capita derivation, zero and negative cases, and numerical example are independently calculated from that law of motion.
<!-- bilingual-en:end -->
