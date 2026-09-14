---
aliases:
  - "Bradley–Terry 模型把同一提示下两个回答的选择概率写成二者标量分数之差的 logistic 函数"
  - "The Bradley–Terry model writes the choice probability between two responses to the same prompt as a logistic function of their scalar-score difference"
  - "Bradley–Terry 型奖励模型以两回答的分数差拟合被选择概率，因此学习的是给定偏好数据中的相对排序而非绝对真值"
  - "A Bradley–Terry reward model fits choice probability from the score difference between two responses, thereby learning relative ranking in the observed preference data rather than absolute truth"
student_os: knowledge-atom
atom_id: LLM-PT-016
atom_set: llm-post-training
atom_type: model
status: source-checked
mastery_state: unassessed
requires:
  - "[[成对偏好模型]]"
related:
  - "[[提示内奖励平移不变]]"
  - "[[偏好分数解释边界]]"
leads_to:
  - "[[显式奖励模型RLHF]]"
  - "[[DPO目标]]"
part_of:
  - "[[LLM 后训练.canvas|LLM 后训练]]"
---

# Bradley–Terry 模型把同一提示下两个回答的选择概率写成二者标量分数之差的 logistic 函数
<!-- bilingual-en:start -->
*The Bradley–Terry model writes the choice probability between two responses to the same prompt as a logistic function of their scalar-score difference*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> 给定提示 $x$ 和两个候选回答 $y_a,y_b$，模型先为每个回答给出标量分数 $r(x,y)$，再令
> $$
> P(y_a\succ y_b\mid x)
> =\sigma\!\left(r(x,y_a)-r(x,y_b)\right)
> =\frac{e^{r(x,y_a)}}{e^{r(x,y_a)}+e^{r(x,y_b)}}.
> $$
> 分数差为零时二者各有一半概率被选；差值越大，模型越确信 $y_a$ 会胜过 $y_b$。这里的 Bradley–Terry 是把分数接到选择概率上的统计模型，不是生成分数的神经网络架构。
>
> <!-- bilingual-en:start -->
> Given a prompt $x$ and responses $y_a,y_b$, the model assigns each response a scalar score $r(x,y)$ and sets $P(y_a\succ y_b\mid x)=\sigma(r(x,y_a)-r(x,y_b))$. Equal scores imply an even choice probability; a larger difference makes selection of $y_a$ more probable. Bradley–Terry is the statistical link from scores to choices, not the neural architecture that produces the scores.
> <!-- bilingual-en:end -->

## 一条比较怎样进入损失
<!-- bilingual-en:start -->
*How one comparison enters the loss*
<!-- bilingual-en:end -->

若数据记录 $y_w$ 胜过 $y_l$，这条记录的负对数似然是
$$
-\log \sigma\!\left(r(x,y_w)-r(x,y_l)\right).
$$
当模型把胜者排得比败者更高时，损失下降；若顺序相反，损失上升。多条比较共同约束一个可泛化的评分函数，因而模型可以对训练时没有成对出现过的新回答作出预测。

<!-- bilingual-en:start -->
For an observation in which $y_w$ beats $y_l$, the negative log-likelihood is $-\log\sigma(r(x,y_w)-r(x,y_l))$. The loss falls when the winner receives the higher score and rises when the order is reversed. Many comparisons jointly constrain a scoring function that can generalise to responses not paired during training.
<!-- bilingual-en:end -->

取对数几率后，模型的含义更直接：
$$
\log\frac{P(y_a\succ y_b\mid x)}{1-P(y_a\succ y_b\mid x)}
=r(x,y_a)-r(x,y_b).
$$
因此分数差是选择对数几率，而不是一个回答脱离比较对象后的成功概率。若差值为 $\log 3$，模型给出的选择概率是 $3/(1+3)=0.75$。

<!-- bilingual-en:start -->
Taking log odds gives $\log\frac{P}{1-P}=r(x,y_a)-r(x,y_b)$. The score difference is therefore a choice log-odds, not a context-free probability that one response is good. A difference of $\log 3$ corresponds to a choice probability of $0.75$.
<!-- bilingual-en:end -->

## 模型形式带来的边界
<!-- bilingual-en:start -->
*Boundaries imposed by the model form*
<!-- bilingual-en:end -->

同一提示下的选择只通过一个标量差值进入概率，所以标准形式不能直接表示平局、多维理由或随比较对象改变的评价准则。实际系统可以扩展平局、margin、列表排序或多维奖励；这些是别的模型设定，不能悄悄算作标准 Bradley–Terry 的结论。

此外，式子只依赖分数差。共同平移为何不可识别，以及这种平移在 KL 正则策略目标中为何不改变最优策略，单独见 [[提示内奖励平移不变]]；分数能否被解释为事实性、安全性或广泛效用，单独见 [[偏好分数解释边界]]。

<!-- bilingual-en:start -->
Choices enter the standard model through one scalar difference, so it does not directly represent ties, multidimensional reasons, or comparison-dependent criteria. Ties, margins, listwise ranking, and vector rewards require extensions. The likelihood also depends only on differences: [[提示内奖励平移不变|prompt-wise reward-shift invariance]] isolates the resulting non-identification, while [[偏好分数解释边界|the interpretation boundary]] asks what the fitted scores can validly mean.
<!-- bilingual-en:end -->

> [!question]- 自检
> 同一提示下，回答 A 与 B 的分数分别为 $2$ 和 $1$。模型给 A 胜过 B 的概率是多少？这个数能否直接解释为“A 有 73% 概率符合事实”？
>
> <!-- bilingual-en:start -->
> For one prompt, responses A and B receive scores $2$ and $1$. What probability does the model assign to A beating B? Can this be read as “A has a 73% probability of being factual”?
> <!-- bilingual-en:end -->
>
> **答案：** $\sigma(1)\approx0.731$。它是既定模型下 A 相对 B 被选择的概率，不是 A 单独成立的事实概率；后一种解释还需要独立的事实标注与验证。
>
> <!-- bilingual-en:start -->
> **Answer:** $\sigma(1)\approx0.731$. This is the modeled probability that A is chosen over B, not a standalone probability that A is factual. The latter interpretation requires separate factual labels and validation.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [Ouyang et al. (2022), *Training language models to follow instructions with human feedback*](https://arxiv.org/html/2203.02155#S3.SS3)：第 3.3 节给出按 chosen/rejected 奖励差训练标量奖励模型的公开实现。
- [Rafailov et al. (2023), *Direct Preference Optimization*](https://arxiv.org/html/2305.18290#S3)：第 3 节明确写出 Bradley–Terry 偏好模型，并从该形式推导直接偏好目标。

<!-- bilingual-en:start -->
- Ouyang et al. (2022), §3.3, provides a public scalar reward model trained from reward differences between chosen and rejected responses.
- Rafailov et al. (2023), §3, explicitly states the Bradley–Terry preference model used in the DPO derivation.
<!-- bilingual-en:end -->
