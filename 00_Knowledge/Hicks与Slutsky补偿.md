---
aliases:
  - "Hicks补偿保持原效用而Slutsky补偿只保证原消费束在新价格下仍买得起"
  - Hicks and Slutsky compensation
  - Utility compensation versus purchasing-power compensation
student_os: knowledge-atom
atom_id: MICRO-CONS-004
atom_set: income-substitution
atom_type: distinction
status: source-checked
mastery_state: unassessed
requires:
  - "[[普通与补偿需求]]"
related:
  - "[[价格效应分解]]"
  - "[[Slutsky方程]]"
part_of:
  - "[[收入效应与替代效应.canvas]]"
---

# Hicks补偿保持原效用而Slutsky补偿只保证原消费束在新价格下仍买得起
<!-- bilingual-en:start -->
*Hicks compensation preserves the original utility, whereas Slutsky compensation only keeps the original bundle affordable at the new prices*
<!-- bilingual-en:end -->

> [!summary] 有限价格变化的两个不同中间点
> 价格从 $p^0$ 变为 $p^1$，原选择为 $x^0=x(p^0,m)$、原效用为 $u^0$。Hicks 补偿把新收入设为 $m^H=e(p^1,u^0)$；Slutsky 补偿把新收入设为 $m^S=p^1\cdot x^0$。前者刚好维持原效用，后者刚好让原组合仍可购买。有限变化时，两种补偿通常给出不同的中间选择。
> <!-- bilingual-en:start -->
> After a finite price change, Hicks compensation sets income to the minimum expenditure needed to attain the original utility. Slutsky compensation sets income just high enough to afford the original bundle. These generally generate different compensated bundles.
> <!-- bilingual-en:end -->

若原预算绑定，$m=p^0\cdot x^0$。两种收入调整分别是
$$
\Delta m^H=e(p^1,u^0)-m,
\qquad
\Delta m^S=(p^1-p^0)\cdot x^0,
$$
因而 $m+\Delta m^S=p^1\cdot x^0$。对应的中间选择为
$$
x^H=h(p^1,u^0)=x(p^1,e(p^1,u^0)),
$$
以及
$$
x^S=x(p^1,p^1\cdot x^0).
$$

原组合 $x^0$ 在新价格下本身就是一个能够达到 $u^0$ 的可行方案，所以支出最小化必有
$$
e(p^1,u^0)\le p^1\cdot x^0.
$$
因此在上述定义下，Slutsky 补偿必不小于 Hicks 补偿。消费者拿到 Slutsky 补偿后不必继续购买原组合；重新优化可能达到严格高于 $u^0$ 的效用。把“原组合买得起”写成“效用保持不变”，正是混淆两种补偿的关键错误。

以 $u(x,y)=\sqrt{xy}$、$m=100$、$p_y=1$、$p_x:10\to20$ 为例。原组合为 $(5,50)$，$u^0=\sqrt{250}$。Hicks 新收入与 $x$ 的中间需求是
$$
m^H=2u^0\sqrt{20}\approx141.421,
\qquad
x^H=u^0\sqrt{\frac1{20}}\approx3.536.
$$
Slutsky 新收入则是
$$
m^S=20\times5+1\times50=150,
$$
重新优化得到 $x^S=150/(2\times20)=3.75$。两个中间点显然不同。

在需求足够光滑且选择唯一时，价格变化缩小时，两种补偿的差是高阶小量，在初始点给出相同的一阶补偿需求导数。这解释了为什么微分形式的 Slutsky 方程可以用 Hicksian demand 表示；它不意味着有限变化下两张补偿预算线完全相同。

> [!question]- 自检
> 为什么 Slutsky 补偿后，消费者的效用可能高于价格变化前？
>
> **答案：** 因为补偿只要求旧组合在新价格下仍可购买。旧组合未必是新价格下达到旧效用的最低成本方案；消费者可以重新配置支出，选到一个更受偏好的组合。

## 来源与核验

- [Princeton ECO 305, Lecture 6: Applications of the Expenditure Function](https://www.princeton.edu/~dixitak/Teaching/MicroHighCalculus/Notes%26Slides/Lec06.pdf)，pp. 1–3：核对 Hicks 补偿、Slutsky 补偿、两者金额及有限价格变化下中间点的区别。
- [Boston University EC 701, Microeconomic Theory slides](https://sites.bu.edu/manove/files/2013/05/MicroSlides3IndUtilF05x2.pdf)，slides 201–204（PDF pp. 11–12）：核对 $x^S(p',x)=x(p',p'x)$、$h(p',u)=x(p',e(p',u))$，以及两种补偿在初始点具有相同一阶导数而有限变化不同。
