---
aliases:
  - 现有设计提高后续研发效率，使固定研究人力能够持续按比例增加知识存量
  - Knowledge spillovers in Romer research
student_os: knowledge-atom
atom_id: MACRO-ENDO-014
atom_type: mechanism
status: source-checked
part_of:
  - "[[内生增长与创新.canvas]]"
---

# 现有设计提高后续研发效率，使固定研究人力能够持续按比例增加知识存量
<!-- bilingual-en:start -->
*Existing designs raise subsequent research productivity, allowing a fixed research input to increase knowledge at a sustained proportional rate.*
<!-- bilingual-en:end -->

[[Romer品种扩张模型]]把过去的设计知识视为研究者能够共同使用的投入。新设计除了允许制造一种新机器，还提高后来研究者的研发能力；原设计者不能对这种后续研究收益收费。这是相对于设计投资决策的 [[外部性|正外部性]]，其关键是新设计改变了谁的生产条件，以及哪些收益没有进入原投资者的回报。
<!-- bilingual-en:start -->
The [[Romer品种扩张模型|Romer expanding-variety model]] treats past design knowledge as an input available to all researchers. A new design enables a new machine and raises the productivity of later research, without letting its creator charge for that later benefit. This is a [[外部性|positive externality]] relative to the original design investment: identify whose productive opportunities improve and which benefits the investor cannot capture.
<!-- bilingual-en:end -->

设 $A>0$ 为现有设计存量，$H_A\ge0$ 为投入研发的人力资本，$\delta>0$ 为研发效率。原文的研发方程是
<!-- bilingual-en:start -->
Let $A>0$ be the existing design stock, $H_A\ge0$ research human capital and $\delta>0$ research productivity. The original research equation is:
<!-- bilingual-en:end -->

$$\dot A=\delta H_A A,\qquad g_A=\frac{\dot A}{A}=\delta H_A.$$

每单位研究人力的设计产出是 $\delta A$：已有知识越多，同一研究人力越有效率。若 $H_A$ 保持正的常数，知识存量便以常数 $\delta H_A$ 增长；研发人力为什么留在研究部门，要由 [[Romer研发人力配置]] 进一步解释。这里的 $\delta$ 是研发效率，并非折旧率，总人力资本 $H$ 在原模型中固定。
<!-- bilingual-en:start -->
Each unit of research human capital produces designs at rate $\delta A$, so existing knowledge increases its productivity. Constant positive $H_A$ yields constant proportional knowledge growth. Why researchers remain in that sector is established by [[Romer研发人力配置|research allocation]]. Here $\delta$ measures research productivity, not depreciation, and the model's total human capital $H$ is fixed.
<!-- bilingual-en:end -->

例如取 $\delta=0.02$、$H_A=1$。$A=100$ 时每期新增设计流量为2；$A=200$ 时为4，两者的瞬时比例增长率都是2%。第二种状态的研究者没有多一倍个人教育，却能利用更丰富的既有设计。这是知识存量改变生产条件的教学例子；“2、4”是连续时间流量，不是规定每期必须发现整数个设计。
<!-- bilingual-en:start -->
With $\delta=0.02$ and $H_A=1$, a design stock of 100 generates a flow of two new designs, while a stock of 200 generates four. Both imply instantaneous proportional growth of 2%. Researchers in the second state need not have twice as much education: they use more existing knowledge. These are continuous-time flows in a constructed example, not integer discoveries required in each period.
<!-- bilingual-en:end -->

持续比例增长尤其依赖右侧对 $A$ 的**线性**。非竞争性允许共同使用知识，却不能单独证明现实研发生产率一定与 $A$ 成正比；改变这个指数会改变 [[研发规模效应边界|长期增长与规模效应]]。[[学习外部性]]提供“知识收益未被投入者取得”的一般对照，本卡特有的内容是这项外溢如何进入新设计的生产方程。
<!-- bilingual-en:start -->
Sustained proportional growth particularly depends on linearity in $A$. Nonrivalry permits shared knowledge use but does not establish that actual research productivity is proportional to the knowledge stock. Changing that exponent changes [[研发规模效应边界|growth and scale effects]]. [[学习外部性|Learning externalities]] offer a general comparison; this card specifies how the uncaptured benefit enters the production of new designs.
<!-- bilingual-en:end -->

## 来源与核验

- Romer（1990），[Endogenous Technological Change](https://web.stanford.edu/~klenow/Romer_1990.pdf#page=14)，印刷S83–S85（PDF第14–16页）：式(3)、研究者共享既有知识、对 $A$ 线性的关键作用、设计两种用途的排他边界；印刷S96（PDF第27页）：后续研发收益未进入设计的市场价格。
- 已从式(3)除以正的 $A$ 得出增长率，并检查数例。固定 $H_A$ 是这里解释增长机制的条件，不是对均衡研发配置的预设结论。
<!-- bilingual-en:start -->
Equation (3) and pages S83–S85 specify shared knowledge and linear research productivity. Page S96 identifies the omitted return to subsequent research. The growth-rate relation and numerical example follow directly; equilibrium research allocation is established separately.
<!-- bilingual-en:end -->
