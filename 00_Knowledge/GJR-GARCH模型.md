---
aliases:
  - "GJR-GARCH 用冲击符号改变平方冲击对条件方差的作用"
  - GJR-GARCH model
  - Glosten-Jagannathan-Runkle GARCH
  - GJR 方差模型
student_os: knowledge-atom
atom_id: TS-VOL-021
atom_set: conditional-volatility
atom_type: definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[波动不对称与杠杆]]"
  - "[[GARCH(p,q)模型]]"
related:
  - "[[ARCH-GARCH正性条件]]"
  - "[[TARCH命名边界]]"
  - "[[EGARCH模型]]"
part_of:
  - "[[条件异方差：ARCH 与 GARCH.canvas]]"
---

# GJR-GARCH 用冲击符号改变平方冲击对条件方差的作用
<!-- bilingual-en:start -->
*GJR-GARCH lets the sign of a shock change its squared effect on conditional variance*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> 一种 GJR-GARCH(1,1) 写法是
> $$h_t=\omega+\bigl(\alpha+\gamma I\{\varepsilon_{t-1}<0\}\bigr)\varepsilon_{t-1}^2+\beta h_{t-1}.$$
> 大小相同的正、负创新分别以 $\alpha$ 与 $\alpha+\gamma$ 进入下一期条件方差。在这个指标和符号约定下，$\gamma>0$ 表示负创新带来更大的预测方差更新。

<!-- bilingual-en:start -->
> [!summary] What it is
> A common GJR-GARCH(1,1) specification is $h_t=\omega+[\alpha+\gamma I\{\varepsilon_{t-1}<0\}]\varepsilon_{t-1}^2+\beta h_{t-1}$. Equal-sized positive and negative innovations enter with coefficients $\alpha$ and $\alpha+\gamma$. Under this indicator and sign convention, $\gamma>0$ means a larger variance response to negative news.
<!-- bilingual-en:end -->

对所有历史保证正性的常用充分条件是
$$\omega>0,\qquad \beta\ge0,\qquad \alpha\ge0,\qquad \alpha+\gamma\ge0.$$
所以 $\gamma$ 本身不必非负。若 $z_t=\varepsilon_t/\sqrt{h_t}$ 对称且已标准化，有限二阶矩的常用条件为
$$\alpha+\beta+\frac{\gamma}{2}<1.$$
若冲击分布不对称，$1/2$ 要换成 $E[z_t^2I\{z_t<0\}]$；严格平稳则仍要检查相应随机系数的 log-moment 条件。

<!-- bilingual-en:start -->
Standard sufficient positivity restrictions are $\omega>0$, $\beta\ge0$, $\alpha\ge0$, and $\alpha+\gamma\ge0$, so $\gamma$ itself need not be nonnegative. With symmetric standardized shocks, the familiar finite-second-moment condition is $\alpha+\beta+\gamma/2<1$; under asymmetry, $1/2$ is replaced by $E[z_t^2I\{z_t<0\}]$. Strict stationarity remains a separate log-moment question.
<!-- bilingual-en:end -->

这张卡定义的是具体的**方差递推**。有些教材或软件也把它叫 TARCH/TGARCH，但该标签还可能指递推条件标准差的 Zakoïan 模型；跨来源读取时应先看 [[TARCH命名边界|公式的命名边界]]，再迁移参数限制与系数解释。

<!-- bilingual-en:start -->
This atom defines a variance recursion. Some texts and packages call it TARCH or TGARCH, but the same label may instead denote Zakoïan's conditional-standard-deviation recursion. When moving across sources, inspect the equation before transferring parameter restrictions or coefficient interpretations.
<!-- bilingual-en:end -->

> [!example] 同幅冲击的比较
> 若 $\alpha=0.08$、$\gamma=0.12$，则 $\varepsilon_{t-1}=2$ 给方差方程增加 $0.08\times4=0.32$，而 $\varepsilon_{t-1}=-2$ 增加 $(0.08+0.12)\times4=0.80$。比较必须固定冲击大小，才能把差异归到符号项。

> [!question]- 自检
> 若 $\alpha=0.08,\gamma=-0.03$，模型一定违反正性吗？
>
> **答案：** 不一定。正冲击系数为 $0.08$，负冲击系数为 $0.05$，二者仍非负；还要检查 $\omega$、$\beta$ 以及平稳和矩条件。

## 来源与核验

- [Glosten, Jagannathan & Runkle (1993)](https://doi.org/10.1111/j.1540-6261.1993.tb05128.x)：核对符号相关平方项、系数解释及有限矩限制。
- [Engle & Ng (1993), *Measuring and Testing the Impact of News on Volatility*](https://doi.org/10.1111/j.1540-6261.1993.tb05127.x)：核对 GJR news-impact curve 与不对称响应比较。
- [[01_Math/06_时间序列分析/lecture.pdf#page=176|课程讲义 p. 176]]：核对课程所用的指标函数方差递推；课程称其为 TARCH，本卡按公式归入 GJR-GARCH。
