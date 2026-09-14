---
aliases:
  - "按每单位有效劳动计量时储蓄增加资本而折旧人口与技术进步共同稀释资本"
  - "Saving raises capital per effective worker while depreciation population and technology dilute it"
  - "Solow capital accumulation equation"
student_os: knowledge-atom
atom_id: MACRO-SOLOW-002
atom_set: solow-growth
atom_type: law-of-motion
status: source-checked
mastery_state: unassessed
requires:
  - "[[Solow 人均化边界]]"
part_of:
  - "[[Solow 增长模型、稳态与收敛.canvas]]"
implies:
  - "[[Solow 稳态条件]]"
related:
  - "[[储蓄率的水平效应]]"
---

# 按每单位有效劳动计量时储蓄增加资本而折旧人口与技术进步共同稀释资本
<!-- bilingual-en:start -->
*Saving raises capital per effective worker while depreciation, population, and technology dilute it*
<!-- bilingual-en:end -->

> [!summary] 原子方程
> 在闭合的基本 Solow 模型中，$\dot K=sY-\delta K$。若 $\dot L/L=n$、劳动增进技术满足 $\dot A/A=g$，并定义 $k=K/(AL)$、$f(k)=Y/(AL)$，则
> $$
> \dot k=s f(k)-(\delta+n+g)k.
> $$
> 第一项是每单位有效劳动的实际投资；第二项是仅为抵消折旧、给新增劳动配资本并跟上有效劳动增长所需的 break-even investment。
> <!-- bilingual-en:start -->
> With labor-augmenting technology, capital per effective worker obeys $\dot k=sf(k)-(\delta+n+g)k$. Saving finances new capital; depreciation, population growth, and growth of effective labor are three distinct reasons that investment is needed merely to keep $k$ unchanged.
> <!-- bilingual-en:end -->

从定义出发，
$$
\frac{\dot k}{k}=\frac{\dot K}{K}-\frac{\dot A}{A}-\frac{\dot L}{L}.
$$
再代入 $\dot K=sY-\delta K$ 与 $Y/(AL)=f(k)$，便得到核心方程。若模型没有技术进步，分母改为 $L$、令 $g=0$，方程退化为课程使用的
$$
\dot k=s f(k)-(n+\delta)k,\qquad k=K/L.
$$

因此，写下 $(n+g+\delta)k$ 时必须同时说明 $k$ 是每单位有效劳动资本。若 $k$ 仍定义成 $K/L$，却把 $gk$ 也当作稀释项，就混合了两套口径。

> [!question]- 自检
> 为什么技术进步会出现在“维持 $k$ 不变”的投资中？
>
> **答案：** 因为这里的 $k=K/(AL)$。即使劳动人数不变，$A$ 上升也使有效劳动 $AL$ 增加；资本必须同步增加，才能维持每单位有效劳动资本不变。

## 来源与核验

- [[02_Economy/10_发展经济学/发展经济学拍屏ppt.pdf#page=71|发展经济学课程 PDF pp. 71–72]]：核对无技术版本 $\dot k=sf(k)-(n+\delta)k$ 及 break-even investment 图。
- MIT 14.452，[The Solow Growth Model, Lectures 2–3](https://ocw.mit.edu/courses/14-452-economic-growth-fall-2016/2b68057aa4e74410d00ae89a0c49752f_MIT14_452F16_Lec2and3.pdf)，pp. 49–52、75–81：核对总量资本积累、人口稀释、有效劳动口径及 $\delta+n+g$ 三项。
- 已从 $k=K/(AL)$ 的对数导数逐项重做推导，没有把 $g$ 加入每名劳动者口径。
