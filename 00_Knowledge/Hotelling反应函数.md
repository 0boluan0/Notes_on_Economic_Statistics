---
aliases:
  - "在端点企业、均匀消费者、线性运输成本 t>0、全市场覆盖的单位 Hotelling 线性城市中，若相同边际成本 c≥0 且价格可在 [0,∞) 连续选择，则企业 i 在市场内点分割分支上的价格反应候选为 (t+c+p_j)/2，斜率 1/2 使该分支中的价格成为战略互补"
  - In a covered unit Hotelling linear city with endpoint firms, uniform consumers, linear transport cost t>0, common marginal cost c≥0, and a continuous price set [0,∞), firm i's candidate on the interior-split price-response branch is (t+c+p_j)/2 with slope 1/2, so prices are strategic complements on that branch
student_os: knowledge-atom
atom_id: GT-OLI-013
atom_set: oligopoly-competition
atom_type: best-response-derivation
status: source-checked
mastery_state: unassessed
requires:
  - "[[Hotelling需求系统]]"
  - "[[最佳反应]]"
related:
  - "[[战略替代与互补]]"
leads_to:
  - "[[Hotelling均衡]]"
part_of:
  - "[[寡头竞争.canvas|寡头竞争]]"
---

# 在端点企业、均匀消费者、线性运输成本 t>0、全市场覆盖的单位 Hotelling 线性城市中，若相同边际成本 c≥0 且价格可在 [0,∞) 连续选择，则企业 i 在市场内点分割分支上的价格反应候选为 (t+c+p_j)/2，斜率 1/2 使该分支中的价格成为战略互补
<!-- bilingual-en:start -->
*In a covered unit Hotelling linear city with endpoint firms, uniform consumers, linear transport cost $t>0$, common marginal cost $c\ge0$, and a continuous price set $[0,\infty)$, firm $i$'s candidate on the interior-split price-response branch is $(t+c+p_j)/2$ with slope $1/2$, so prices are strategic complements on that branch*
<!-- bilingual-en:end -->

> [!summary] 对手提价会放松本方面对的需求压力
> 空间差异使高价企业仍能保留附近消费者。对手价格上升时，本方在任意给定价格下获得更多需求，因此其内点利润最大化价格也上升；这正是价格战略互补的反应函数含义。
>
> <!-- bilingual-en:start -->
> Spatial differentiation lets a higher-price firm retain nearby consumers. A rival price increase raises the firm's demand at any given own price, so its interior profit-maximizing price also rises. This is the best-response meaning of strategic complementarity.
> <!-- bilingual-en:end -->

## 从需求函数优化

设两家相同边际成本为 $c\ge0$，价格可在 $[0,\infty)$ 连续选择。由 [[Hotelling需求系统]] 的内点需求

$$
q_i(p_i,p_j)=\frac12+\frac{p_j-p_i}{2t},
$$

企业 $i$ 的利润为

$$
\pi_i=(p_i-c)\left(\frac12+\frac{p_j-p_i}{2t}\right).
$$

一阶与二阶条件是

$$
\frac{\partial\pi_i}{\partial p_i}
=\frac{t+p_j+c-2p_i}{2t}=0,
\qquad
\frac{\partial^2\pi_i}{\partial p_i^2}=-\frac1t<0.
$$

所以内点分支的候选反应为

$$
BR_i^{\mathrm{int}}(p_j)=\frac{t+c+p_j}{2},
\qquad
\frac{dBR_i^{\mathrm{int}}}{dp_j}=\frac12>0.
$$

<!-- bilingual-en:start -->
Substituting the interior demand system into profit gives a strictly concave quadratic in own price. Its first-order condition yields the interior branch $BR_i^{\mathrm{int}}(p_j)=(t+c+p_j)/2$, and the positive slope $1/2$ classifies prices as strategic complements within that branch.
<!-- bilingual-en:end -->

## “内点”是命题的一部分

上标 $\mathrm{int}$ 明确表示这个公式只在无差异消费者仍位于 $(0,1)$ 的分支上成立。若候选价格使一家企业取得全部市场或完全失去市场，全局最佳反应必须把角点利润与内点候选一起比较，不能继续外推同一条直线。[[Hotelling均衡]]只在对称候选处完成这项全局角点核验。

<!-- bilingual-en:start -->
The formula is an interior result. If the candidate price gives one firm the whole market or no demand, the global best response must compare corner profits with the interior candidate. [[Hotelling均衡]] performs that global check at the symmetric price pair.
<!-- bilingual-en:end -->

> [!question]- 自检
> 为什么反应函数斜率 $1/2>0$ 足以说价格是战略互补，却不能直接推出均衡唯一或福利更高？
>
> **答案：** 正斜率只说明对手提价会提高本方的最优价格。均衡数量、唯一性和福利还取决于两条完整反应、角点以及消费者与成本结构。

## 来源与核验

- MIT 14.271, [*Static Competition and Models of Differentiation, Part 1*, Hotelling Competition](https://ocw.mit.edu/courses/14-271-industrial-organization-i-fall-2022/mit14_271_f22_lec5slides.pdf)：核对内点需求、利润、一阶条件、反应函数及战略互补。
- [[01_Math/03_game theory/04_案例（囚徒困境与纳什均衡）#5. Hotelling 模型|本地课程：Hotelling 模型]]：核对课程使用的端点、均匀消费者、线性运输成本与反应函数记号。
- [[Hotelling需求系统]]拥有需求分配；本卡只拥有给定该需求后的内点最优化，不把均衡价格重复存入此处。
