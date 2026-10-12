<!-- visual-lab:generated:start -->
# 策略梯度如何改变行动概率

<!-- bilingual-en:start -->
*Exact policy gradients with recurrent decisions*
<!-- bilingual-en:end -->

把同一小环境的冒险概率写成 sigmoid 参数，直接求出回报和梯度，并核对折扣访问次数。

> [!question] 先预测，再操作
> 每次回到 S₀ 都能再行动，策略梯度为什么必须计入这些再次访问？

<!-- visual-lab:film:start -->
## 动画：分步观察

<!-- bilingual-en:start -->
*Animation: follow the mechanism*
<!-- bilingual-en:end -->

![[07_Programme/可视化实验室/media/rl-policy.mp4]]

约 43 秒，中英字幕。可暂停、点击进度条定位或全屏观看。

<!-- bilingual-en:start -->
About 43 seconds, with Chinese and English captions. Pause, seek on the timeline or watch full screen.
<!-- bilingual-en:end -->

- **0:00** 参数先改变选动作的概率
- **0:10** 梯度问：稍微增加 θ，回报会怎样？
- **0:16** 沿梯度更新，再重新评估策略
- **0:33** 更常选择哪个动作，取决于当前环境

<!-- bilingual-en:start -->
- **0:00** The parameter changes action probabilities.
- **0:10** The gradient asks how return changes with theta.
- **0:16** Update the parameter, then reevaluate the policy.
- **0:33** The beneficial update direction depends on the environment.
<!-- bilingual-en:end -->

## 亲手实验

<!-- bilingual-en:start -->
*Try it yourself*
<!-- bilingual-en:end -->

在下方展开“影片参数与延伸对照”，选择“载入本片的模型与数据”，先核对影片中的一个对象或运算，再改变一个条件。其他对照用于继续探索。

<!-- bilingual-en:start -->
Expand the film inputs below and load the model and data used in this film. Check one object or calculation, then change one condition. The other comparisons provide further exploration.
<!-- bilingual-en:end -->

<!-- visual-lab:film:end -->
```visual-lab
id: rl-policy
height: 980
```

## 观察路线

1. 先对照 Bellman 实验明确同一环境，再查看 θ=0 时 q=1/2 的随机策略。
2. 改变 θ 但不迭代，比较解析 J 曲线和初始梯度符号。
3. 启动不同步数的精确梯度上升，观察概率与回报如何一起变化。
4. 接近 q=0 或 1 时，辨认 sigmoid 饱和与策略不佳是否同时发生。

## 公式与适用条件

$$
q=\sigma(\theta),\quad J(\theta)=\frac{1+q(c+3\gamma p-1)}{1-\gamma q(1-p)},\quad \frac{dJ}{d\theta}=\frac{q(1-q)}{1-\gamma q(1-p)}\,[Q(S_0,\mathrm{risky})-1]
$$

- 同一状态使用同一个固定 q，环境和奖励与 Bellman 实验一致。
- 目标是从 S₀ 开始的期望折扣回报；访问权重为未归一化折扣次数。
- θ 连续、q=σ(θ)；有限 θ 不能给出精确 0 或 1 的确定性策略。

## 回到知识点

- [[00_Knowledge/梯度最陡方向|梯度最陡方向]]：对解析期望回报作梯度上升；局部方向含义与有限步数达到确定性最优策略需区分。
- [[00_Knowledge/链式法则|链式法则]]：从 q=σ(θ) 到 J(q) 的导数，检查 sigmoid 饱和以及折扣访问权重。

[[07_Programme/可视化实验室/00_实验库入口|回到实验库]] · [[07_Programme/可视化实验室/实验/rl-bellman|Bellman 备份如何传播未来奖励]] · [[07_Programme/可视化实验室/实验/logistic|概率、阈值与交叉熵]] · [[07_Programme/可视化实验室/实验/gradient|梯度下降何时收敛]] · [[07_Programme/可视化实验室/实验/backprop|反向传播把局部导数串起来]]

## 来源与核验

- [Sutton & Barto · Reinforcement Learning（作者教材草稿）](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf) — Ch.3 Bellman equations; §4.4 Value Iteration, Eq. (4.10)。支持：有限 MDP、折扣价值与 Bellman 最优备份。实验中的三状态 MDP 和解析式为独立构造。
- [Sutton, McAllester, Singh & Mansour · Policy Gradient Methods](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf) — §1 Theorem 1, start-state formulation; Appendix Eq. (7)。支持：指定起点的折扣状态访问权重与策略梯度；本页使用真实折扣访问次数，与直接微分解析回报互相核验。

实验使用明确给定的公式、合成数据或固定小型示例。代码、可复现检查和逐轮修订记录见 [[07_Programme/可视化实验室/01_使用与维护|使用与维护]]。
<!-- visual-lab:generated:end -->
