---
aliases:
  - "基础 Merton 模型只在单一债务到期日以公司资产价值低于债务面值判定违约并把股权视为欧式看涨期权，用无风险漂移得到的 N(-d₂) 是风险中性违约概率而非真实世界 PD"
  - "Merton structural default model"
  - "Merton 模型"
student_os: knowledge-atom
atom_id: RM-CR-008
atom_set: credit-risk-parameters-and-models
atom_type: structural-model
status: source-checked
mastery_state: unassessed
requires:
  - "[[无风险利率口径]]"
  - "[[违约概率口径]]"
related:
  - "[[违约强度模型]]"
  - "[[单期信用债定价]]"
  - "[[信用利差与违约概率]]"
part_of:
  - "[[信用风险参数与模型.canvas|信用风险参数与模型]]"
---

# 基础 Merton 模型只在单一债务到期日以公司资产价值低于债务面值判定违约并把股权视为欧式看涨期权，用无风险漂移得到的 N(-d₂) 是风险中性违约概率而非真实世界 PD
<!-- bilingual-en:start -->
*The basic Merton model declares default only at a single debt maturity when firm asset value is below face value and treats equity as a European call; N(-d₂) obtained with risk-free drift is a risk-neutral default probability, not a real-world PD*
<!-- bilingual-en:end -->

> [!summary] 资本结构把信用风险变成期权问题
> 公司当前资产价值为 $V_0$，到期 $T$ 只有一笔面值为 $F$ 的零息债务。到期时若 $V_T\ge F$，债权人获得 $F$，股东取得剩余；若 $V_T<F$，公司在该模型中违约，股东因有限责任放弃公司，债权人获得 $V_T$。所以
> $$
> E_T=(V_T-F)^+,
> $$
> 即股权是以公司资产为标的、执行价为 $F$、到期日为 $T$ 的欧式看涨期权。

在基础的无分红、常数无风险利率与资产波动率设定下，风险中性测度 $\mathbb Q$ 中
$$
\frac{dV_t}{V_t}=r\,dt+\sigma_V\,dW_t^{\mathbb Q},
$$
股权价值为
$$
E_0=V_0\Phi(d_1)-Fe^{-rT}\Phi(d_2),
$$
其中
$$
d_1=\frac{\ln(V_0/F)+(r+\tfrac12\sigma_V^2)T}{\sigma_V\sqrt T},
\qquad
d_2=d_1-\sigma_V\sqrt T.
$$
因为风险中性分布下的违约事件是 $V_T<F$，定价口径的违约概率为
$$
\mathbb Q(V_T<F)=\Phi(-d_2).
$$

## 真实世界 PD 必须换回真实漂移

若要预测实际频率，应在物理测度 $\mathbb P$ 下说明公司资产的真实漂移 $\mu$：
$$
\frac{dV_t}{V_t}=\mu\,dt+\sigma_V\,dW_t^{\mathbb P}.
$$
同一到期事件的物理概率为
$$
\mathbb P(V_T<F)=\Phi(-d_2^{\mathbb P}),
\qquad
d_2^{\mathbb P}=\frac{\ln(V_0/F)+(\mu-\tfrac12\sigma_V^2)T}{\sigma_V\sqrt T}.
$$
把 $r$ 放入公式得到的 $\Phi(-d_2)$ 服务于无套利定价；把 $\mu$ 放入公式才是在该资产过程假设下的真实世界预测。二者之间包含风险价格，不能只改一个标签。实际 KMV 类方法还会用经验违约数据把 distance-to-default 映射到违约频率，因此也不等于机械采用正态尾概率。

## 模型给出的机制与不能给出的结论

- **给出：** 杠杆 $F/V_0$、资产波动率 $\sigma_V$、期限和无风险利率怎样共同影响股权、债务和到期违约边界。
- **需要反推：** 公司总资产的市场价值 $V_0$ 与资产波动率 $\sigma_V$ 通常不可直接观察；实践中常结合股权市值和股权波动率求解，因此输入本身带有模型误差。
- **基础模型不含：** 到期日前触碰壁垒即违约、多个债务到期日、票息与复杂优先级、流动性溢价、跳跃、随机利率、战略违约和法律处置过程。
- **恢复含义有限：** 基础模型在违约时让债权人取得 $V_T$，不等于现实清算中给定的固定 recovery rate。

> [!question]- 最小自检
> 分析者用 $r$ 代入 $d_2$，算得 $\Phi(-d_2)=4\%$，随后说“该公司未来一年真实违约频率就是 4%”。错误在哪里？
>
> **答案：** 用 $r$ 得到的是该模型和定价假设下的风险中性概率，其中包含风险价格。真实世界 PD 需要物理测度下的资产漂移或经历史违约校准的映射；还要承认基础模型只在单一债务到期日判定违约。

## 边界

- “资产跌破债务就立即违约”不是基础 Merton 结论；基础模型只检查 $T$ 时点的 $V_T<F$。早违约需要 barrier/first-passage 等扩展。
- 股权等于欧式看涨期权依赖单一到期债务、有限责任和模型中的资产过程，不是任意现实资本结构的会计恒等式。
- $\Phi(-d_2)$ 的测度由 $d_2$ 使用的漂移决定；只写“违约概率”而不写 $\mathbb P$ 或 $\mathbb Q$ 会混淆预测与定价。
- 模型能形成结构化风险信号，不证明真实公司资产服从连续对数正态过程，也不能单凭拟合价格确认违约机制。

## 来源与核验

- Merton (1974), [*On the Pricing of Corporate Debt: The Risk Structure of Interest Rates*](https://doi.org/10.1111/j.1540-6261.1974.tb03058.x)：原始结构模型，核验单一贴现债务、到期偿付边界、股权期权性和公司债务定价。
- Federal Reserve Board, [*Distress in the Financial Sector and Economic Activity*](https://www.federalreserve.gov/PUBS/FEDS/2008/200843/revision/)：核验 Merton distance-to-default 的实践解释边界，包括风险中性假设和现实监管关闭点与模型违约点不同。
- Hong Kong Monetary Authority, [*Assessing Default Risk of Chinese Firms: A Merton-KMV Approach*](https://www.hkma.gov.hk/media/eng/publication-and-research/research/working-papers/pre2007/RM24-2005.pdf)：核验用股权和会计信息反推资产价值、资产波动率与物理漂移的实施口径。
- 作者逐项核验日：2026-08-30；独立模型复核通过。
