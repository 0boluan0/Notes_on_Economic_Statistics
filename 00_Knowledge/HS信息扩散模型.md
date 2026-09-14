---
aliases:
  - HS信息扩散模型是信息观察者与受限动量交易者互动形成迟缓调整和价格超调的模型
  - Hong–Stein information-diffusion model
student_os: knowledge-atom
atom_id: FI-BF-046
atom_type: definition
status: source-checked
related:
  - "[[有限注意]]"
  - "[[BSV信念模型]]"
  - "[[DHS信心反馈]]"
  - "[[动量识别边界]]"
part_of:
  - "[[行为金融与套利限制.canvas]]"
---

# HS信息扩散模型是信息观察者与受限动量交易者互动形成迟缓调整和价格超调的模型
<!-- bilingual-en:start -->
*The HS model generates slow adjustment and overshooting through newswatchers and constrained momentum traders*
<!-- bilingual-en:end -->

Hong–Stein 模型有两组交易者：信息观察者依据各自逐步得到的基本面信息交易，但不从价格中提取别人的信息；动量交易者看过去价格变化，却只能使用受限的简单策略。两组都不是拥有全部信息并充分处理它的交易者。
<!-- bilingual-en:start -->
Newswatchers trade on gradually received fundamentals without extracting others' information from prices. Momentum traders use past price changes through restricted simple strategies. Neither processes the complete information set.
<!-- bilingual-en:end -->

先只有信息观察者，好消息在人群中逐步扩散，价格因此调整较慢。加入动量交易者后，早期追涨加快了信息进入价格；但后来者无法充分分辨这次涨价来自新基本面还是前一轮追涨，继续买入便可能把价格推过头。迟缓调整和超调由同一段互动产生，不需要另加一次坏消息来“解释反转”。
<!-- bilingual-en:start -->
Gradual diffusion first slows adjustment. Early momentum orders accelerate incorporation, but later traders cannot fully separate fundamental news from prior trend-following orders. Continued buying can overshoot. The same interaction generates delay and overshooting without a separate adverse-news shock.
<!-- bilingual-en:end -->

最小辨析：若交易者能识别每次涨价的来源、同时利用完整历史，原模型限制就变了；不能继续无条件套用该结论。这是模型规定的信息处理限制，可以与[[有限注意]]对照，但不等于已经识别了分心机制。它也不同于 BSV 的错误盈利模型，以及 DHS 的精度偏差。三者可能产生相似收益图形，却不能互相替代。
<!-- bilingual-en:start -->
Allowing traders to identify each price move's source changes the assumptions. These imposed processing restrictions can be compared with [[有限注意|limited attention]], but do not identify distraction as their cause. They also differ from BSV's earnings misspecification and DHS's precision bias. Similar return patterns do not make the mechanisms interchangeable.
<!-- bilingual-en:end -->

## 来源与核验

- [Hong & Stein (1999), *A Unified Theory of Underreaction, Momentum Trading, and Overreaction in Asset Markets*，pp. 2144–2146、§II](https://stein.scholars.harvard.edu/sites/g/files/omnuum5951/files/stein/files/unifiedtheory.pdf)：核对两组信息限制、渐进扩散与早晚动量交易者之间的机制。文本为机制说明，不冒充完整均衡推导。

<!-- bilingual-en:start -->
The original paper supports the information restrictions and interaction mechanism. This card explains the mechanism rather than reproducing the full equilibrium derivation.
<!-- bilingual-en:end -->
