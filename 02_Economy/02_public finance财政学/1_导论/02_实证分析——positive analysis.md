# 基础信息(Positive Analysis)
<!-- bilingual-en:start -->
*Foundations of Positive Analysis*
<!-- bilingual-en:end -->

[[02_Economy/02_public finance财政学/01_福利评价与政府干预#1. 从政府在经济中做什么开始|本节连续讲解]] · [[福利经济学与政府干预.canvas|主题关系图]]

* 超越一切价值判断，只描述、解释实证研究对象的各种现象，研究其内在规律或者检验有关理论，并运用理论构造模型，分析并预测人们在一定条件下的行为趋势或者概率。
* 目标：解释这个世界。
* 回答的问题：是什么(what)、为什么(why)、怎么样(how)
* 起点是客观的，结论是描述性的，对结论的好或坏不做任何评价
<!-- bilingual-en:start -->
* It brackets value judgements and instead describes and explains observed phenomena, investigates their underlying regularities, tests theories, and uses theory to construct models that analyse or predict how people are likely to behave under specified conditions.
* Its aim is to explain the world.
* It asks what happens, why it happens, and how it happens.
* It begins from objective observations and reaches descriptive conclusions without judging whether those conclusions are good or bad.
<!-- bilingual-en:end -->

<!-- topic49-calibration:positive-scope:start -->
> [!note] 实证分析的范围
> [[实证分析]]的结论描述或解释后果，并不直接判定后果是否值得；这不表示研究的问题选择、测量和模型假设天然没有价值或方法争议。理论推导与经验检验都可以属于实证分析。
> <!-- bilingual-en:start -->
> [[实证分析|Positive analysis]] describes or explains consequences without itself judging their desirability. Question selection, measurement, and model assumptions can still involve value or methodological disputes. Both theoretical derivation and empirical testing can be positive analysis.
> <!-- bilingual-en:end -->
<!-- topic49-calibration:positive-scope:end -->

## 理论的作用
<!-- bilingual-en:start -->
*The Role of Theory*
<!-- bilingual-en:end -->

经济模型，为思考影响模型的因素提供框架
<!-- bilingual-en:start -->
An economic model provides a framework for identifying and reasoning about the factors that shape an outcome.
<!-- bilingual-en:end -->
# 案例分析
<!-- bilingual-en:start -->
*Case Study*
<!-- bilingual-en:end -->
## 案例概述
<!-- bilingual-en:start -->
*Scenario*
<!-- bilingual-en:end -->
>假设一个人每天的时间是一定的，那么他应该如何分配他的工作和休息时间以达到最爽的状态。（每小时时薪是10美刀）
> <!-- bilingual-en:start -->
>Suppose a person has a fixed amount of time each day. How should they divide it between work and leisure to achieve the greatest satisfaction if the hourly wage is ten dollars?
> <!-- bilingual-en:end -->

在这种情况下，应该会有一个比较均衡的分配以使得这个人既有钱花又有时间花钱。
此处不妨将休息看作是一个消耗10美刀又一小时的商品。
<!-- bilingual-en:start -->
The person will presumably choose a balance that leaves both money to spend and time in which to spend it.
Leisure can be treated as a good whose opportunity cost is one hour and the ten dollars that could have been earned during it.
<!-- bilingual-en:end -->

## 问题描述与分析
<!-- bilingual-en:start -->
*Question and Analysis*
<!-- bilingual-en:end -->

此时，向此人每小时的收入征收20%的税，他的时间分配将如何改变？
根据[[劳动供给效应|收入效应与替代效应]]，此人的时间分配并不能仅靠现在的信息得知。
<!-- bilingual-en:start -->
How would this person's allocation of time change if their hourly earnings were taxed at 20%?
Because the [[劳动供给效应|income and substitution effects]] work in opposing directions, the information given is not enough to determine the answer.
<!-- bilingual-en:end -->

>在这个问题中，对问题的分析仰赖于两个模型，这部分工作就是实证分析。
> <!-- bilingual-en:start -->
>The analysis relies on two theoretical effects. Using those models to determine what can be inferred is an example of positive analysis.
> <!-- bilingual-en:end -->

# 因果推断
<!-- bilingual-en:start -->
*Causal Inference*
<!-- bilingual-en:end -->

**实证分析的重要部分**
<!-- bilingual-en:start -->
**An important part of positive analysis.**
<!-- bilingual-en:end -->

X先于Y，X才有可能导致Y的发生
<!-- bilingual-en:start -->
For $X$ to cause $Y$, $X$ must occur before $Y$.
<!-- bilingual-en:end -->

**XY相关不代表二者有因果关系**，想证明二者有因果关系需要排除其他的所有因素的影响，这部分工作的多寡好坏直接影响实证分析的高度
<!-- bilingual-en:start -->
**Correlation between $X$ and $Y$ does not establish a causal relationship.** Establishing causality requires ruling out alternative explanations; the quality and thoroughness of that work determine the strength of the positive analysis.
<!-- bilingual-en:end -->

<!-- topic49-calibration:causal-design:start -->
> [!note] 因果主张需要识别依据
> 时间先后本身不证明因果；政策研究还须定义何时开始处理、是否有预期反应，以及可信的未处理反事实。“排除其他解释”也不等于在回归中机械控制所有变量。具体检查见[[因果研究设计检查]]；某些识别假设需要实质论证，不能只靠一次统计检验确认。
> <!-- bilingual-en:start -->
> Temporal order alone does not establish causation. A policy study must define treatment timing, consider anticipation, and justify the untreated counterfactual. Ruling out alternative explanations does not mean mechanically controlling for every variable. See the [[因果研究设计检查|causal-design check]]; some identifying assumptions require substantive arguments rather than a single statistical test.
> <!-- bilingual-en:end -->
<!-- topic49-calibration:causal-design:end -->


## [[双重差分法|DID]]
<!-- bilingual-en:start -->
*Difference-in-Differences (DID)*
<!-- bilingual-en:end -->
双重差分比较处理组与对照组各自的前后变化，再比较这两个变化之差。对照组不是随意“人为分组”，而是要有理由代表处理组若未受政策时本会经历的变化。
<!-- bilingual-en:start -->
Difference-in-differences compares the before–after change in the treated group with the corresponding change in a comparison group. The comparison group is not an arbitrary partition: it must credibly represent how the treated group would have changed without the policy.
<!-- bilingual-en:end -->

计量方法的连续阅读见[[02_Economy/01_Econometrics/14_双重差分法#双重差分法（DID）|双重差分法课程路径]]；假设、诊断和错位处理的全局关系见[[双重差分法（DID）.canvas|DID 主题 Canvas]]。
<!-- bilingual-en:start -->
For the continuous econometric treatment, see the [[02_Economy/01_Econometrics/14_双重差分法#双重差分法（DID）|DID course path]]; the [[双重差分法（DID）.canvas|DID topic Canvas]] shows assumptions, diagnostics, and staggered-treatment relationships.
<!-- bilingual-en:end -->

经济学中的随机实验常受伦理、制度或成本限制，因此研究者会利用政策分期、门槛或外生冲击形成的准实验变异。但“准实验”不等于只看历史数据：因果解释仍取决于处理时点、对照组和识别假设是否可信。
<!-- bilingual-en:start -->
Randomized experiments in economics are often constrained by ethics, institutions, or cost, so researchers use quasi-experimental variation generated by policy timing, thresholds, or external shocks. “Quasi-experimental” does not merely mean analysing historical data; causal interpretation still depends on credible treatment timing, comparison groups, and identifying assumptions.
<!-- bilingual-en:end -->
