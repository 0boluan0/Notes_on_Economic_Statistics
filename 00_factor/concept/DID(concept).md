---
aliases:
tags:
  - concept
---

DID是用来评估一个 政策/事件/处理 对结果的因果影响.在现实生活中不可能有一个政策对同一个对象既实施又没实施.所以使用这个方法==用一个没被政策影响的对照组扣除共同的时间变化,从而得到一个近似的净效应==

# 数学符号

- 有两组个体：处理组（treated）与对照组（control）。用组指示变量 $D_i\in{0,1}$ 表示个体 $i$ 是否属于处理组。
- 有两个时期：政策前（pre）与政策后（post）。用时间指示变量 $Post_t\in{0,1}$ 表示时期 $t$ 是否为政策后。
- 结果变量：$Y_{it}$（例如收入、产出、违约率等）。

# 模型内容




























-----

## 1. 研究对象与符号设定 What we are trying to estimate

**中文**  
我们观测到很多个体/地区/企业$i$在多个时期$t$的结果变量$Y_{it}$（比如工资、利润、排放）。某个政策/事件在某个时间点后对一部分单位生效。我们用处理指示变量$D_{it}\in{0,1}$表示单位$i$在时期$t$是否“受到处理”（政策是否生效于它）。

在$DiD$最经典的“两组两期”（two groups, two periods）设定里：

- 组别指示$G_i\in{0,1}$：$G_i=1$表示处理组（treated group），$G_i=0$表示对照组（control group）。
- 时间指示$Post_t\in{0,1}$：$Post_t=1$表示政策后，$Post_t=0$表示政策前。
- 典型情况下处理发生在“处理组×政策后”这一格，因此  
    $$  
    D_{it}=G_i\cdot Post_t.  
    $$
    

**English**  
We observe an outcome $Y_{it}$ for unit $i$ over time $t$. A policy/event turns on after a certain time for some units. Let $D_{it}\in{0,1}$ indicate whether unit $i$ is treated at time $t$.

In the canonical “two groups, two periods” DiD setup:

- Group indicator $G_i\in{0,1}$: $G_i=1$ for treated group, $G_i=0$ for control group.
- Time indicator $Post_t\in{0,1}$: $Post_t=1$ post-policy, $Post_t=0$ pre-policy.
- Typically only treated units are treated after the policy, so  
    $$  
    D_{it}=G_i\cdot Post_t.  
    $$

---

## 2. 经济学严谨模型：潜在结果 Potential outcomes model (econometric causality)

**中文**  
对每个单位$i$、时期$t$，定义两个潜在结果：

- $Y_{it}(1)$：如果$i$在$t$期受到处理（政策生效），结果是多少；
- $Y_{it}(0)$：如果$i$在$t$期不受处理，结果是多少。
    

真实世界只会发生一种状态，所以观测到的结果是“选择性显现”的：  
$$  
Y_{it}=D_{it}Y_{it}(1)+(1-D_{it})Y_{it}(0).  
$$

我们关心的因果效应通常是政策后的**处理组平均处理效应**（常见目标是$ATT$）：  
$$  
ATT:=E!\left[Y_{it}(1)-Y_{it}(0)\mid G_i=1,\ Post_t=1\right].  
$$

**English**  
For each unit $i$ at time $t$, define two potential outcomes:

- $Y_{it}(1)$ if treated,
    
- $Y_{it}(0)$ if not treated.
    

Only one is observed, so the observed outcome is  
$$  
Y_{it}=D_{it}Y_{it}(1)+(1-D_{it})Y_{it}(0).  
$$

A common target estimand is the post-policy **average treatment effect on the treated**:  
$$  
ATT:=E!\left[Y_{it}(1)-Y_{it}(0)\mid G_i=1,\ Post_t=1\right].  
$$

---

## 3. $DiD$估计量：差上加差 The DiD estimator (difference in differences)

**中文**  
在“两组两期”里，把每个组-期的平均结果记为：  
$$  
\mu_{g,p}:=E!\left[Y_{it}\mid G_i=g,\ Post_t=p\right],\quad g\in{0,1},\ p\in{0,1}.  
$$

那么$DiD$估计量（总体版本）定义为：  
$$  
DiD:=\left(\mu_{1,1}-\mu_{1,0}\right)-\left(\mu_{0,1}-\mu_{0,0}\right).  
$$

样本里你看到的是把$\mu_{g,p}$换成样本均值$\bar Y_{g,p}$得到：  
$$  
\widehat{DiD}:=\left(\bar Y_{1,1}-\bar Y_{1,0}\right)-\left(\bar Y_{0,1}-\bar Y_{0,0}\right).  
$$

**English**  
Let the group-by-period means be  
$$  
\mu_{g,p}:=E!\left[Y_{it}\mid G_i=g,\ Post_t=p\right].  
$$

Then the (population) DiD estimand is  
$$  
DiD:=\left(\mu_{1,1}-\mu_{1,0}\right)-\left(\mu_{0,1}-\mu_{0,0}\right),  
$$  
and the sample estimator replaces $\mu_{g,p}$ by sample means $\bar Y_{g,p}$:  
$$  
\widehat{DiD}:=\left(\bar Y_{1,1}-\bar Y_{1,0}\right)-\left(\bar Y_{0,1}-\bar Y_{0,0}\right).  
$$

---

## 4. 识别关键：平行趋势 + 无预期等假设 Identification assumptions (why DiD works)

下面是让$DiD$识别$ATT$的“最核心假设集合”。  
Here is the core set of assumptions under which DiD identifies $ATT$.

### 4.1 平行趋势 Parallel trends (in potential outcomes)

 **中文（潜在结果表述，最标准）**  
平行趋势说：在**没有处理**的世界里，处理组与对照组从前到后的平均变化相同：  
$$  
E!\left[Y_{it}(0)\mid G_i=1,\ Post_t=1\right]-E!\left[Y_{it}(0)\mid G_i=1,\ Post_t=0\right]

E!\left[Y_{it}(0)\mid G_i=0,\ Post_t=1\right]-E!\left[Y_{it}(0)\mid G_i=0,\ Post_t=0\right].  
$$

 **English**  
Parallel trends in potential outcomes requires that, in the no-treatment world, treated and control groups would have the same average change:  
$$  
E!\left[Y_{it}(0)\mid G_i=1,\ Post_t=1\right]-E!\left[Y_{it}(0)\mid G_i=1,\ Post_t=0\right]

E!\left[Y_{it}(0)\mid G_i=0,\ Post_t=1\right]-E!\left[Y_{it}(0)\mid G_i=0,\ Post_t=0\right].  
$$

### 4.2 无预期（处理前不受影响）No anticipation

**中文**  
处理组在政策前没有被“提前影响”，即政策前处理组的潜在结果不因未来处理而改变：  
$$  
E!\left[Y_{it}(1)\mid G_i=1,\ Post_t=0\right]=E!\left[Y_{it}(0)\mid G_i=1,\ Post_t=0\right].  
$$

**English**  
No anticipation means outcomes in the pre period are unaffected by future treatment:  
$$  
E!\left[Y_{it}(1)\mid G_i=1,\ Post_t=0\right]=E!\left[Y_{it}(0)\mid G_i=1,\ Post_t=0\right].  
$$

### 4.3 无干扰/无溢出（SUTVA的一部分）No interference / no spillovers

**中文**  
对照组不被处理组的政策间接影响（不然对照组不再是“反事实”参照）。  
**English**  
Control units are not indirectly affected by treated units (otherwise controls are not valid counterfactuals).

---

## 5. 严谨结论：在这些假设下，$DiD=ATT$ Formal identification result

**中文（给你一个“能写进作业”的推导骨架）**  
在“两组两期、只在处理组政策后处理”的设定下，对照组与处理组在政策后观测到的均值分别为：

- 因为对照组始终未处理，所以  
    $$  
    E!\left[Y_{it}\mid G_i=0,\ Post_t=p\right]=E!\left[Y_{it}(0)\mid G_i=0,\ Post_t=p\right],\quad p\in{0,1}.  
    $$
    
- 处理组在政策前未处理，所以  
    $$  
    E!\left[Y_{it}\mid G_i=1,\ Post_t=0\right]=E!\left[Y_{it}(0)\mid G_i=1,\ Post_t=0\right].  
    $$
    
- 处理组在政策后被处理，所以  
    $$  
    E!\left[Y_{it}\mid G_i=1,\ Post_t=1\right]=E!\left[Y_{it}(1)\mid G_i=1,\ Post_t=1\right].  
    $$
    

把这些代入$DiD$定义：  
$$  
DiD=\Big(E[Y_{it}(1)\mid G_i=1,Post_t=1]-E[Y_{it}(0)\mid G_i=1,Post_t=0]\Big)  
-\Big(E[Y_{it}(0)\mid G_i=0,Post_t=1]-E[Y_{it}(0)\mid G_i=0,Post_t=0]\Big).  
$$

再用平行趋势把“处理组无处理反事实的变化”替换为“对照组的变化”，即可得到：  
$$  
DiD=E!\left[Y_{it}(1)-Y_{it}(0)\mid G_i=1,\ Post_t=1\right]=ATT.  
$$

**English (same skeleton)**  
Under the canonical setup (only treated group is treated in the post period), we have:  
$$  
E[Y_{it}\mid G_i=0,Post_t=p]=E[Y_{it}(0)\mid G_i=0,Post_t=p],  
$$  
$$  
E[Y_{it}\mid G_i=1,Post_t=0]=E[Y_{it}(0)\mid G_i=1,Post_t=0],  
$$  
$$  
E[Y_{it}\mid G_i=1,Post_t=1]=E[Y_{it}(1)\mid G_i=1,Post_t=1].  
$$

Plugging into the DiD expression and applying parallel trends yields:  
$$  
DiD=E!\left[Y_{it}(1)-Y_{it}(0)\mid G_i=1,\ Post_t=1\right]=ATT.  
$$

---

## 6. 统计学模型：什么是“$DiD$回归” The DiD regression model

**中文**  
在两组两期中，最标准的$DiD$回归是一个“完全饱和的均值模型”（cell-means model的重参数化）：  
$$  
Y_{it}=\alpha+\gamma G_i+\delta Post_t+\beta(G_i\cdot Post_t)+u_{it},  
$$  
并要求  
$$  
E[u_{it}\mid G_i,Post_t]=0.  
$$

这里$\beta$就是政策效应（在识别条件下等于$ATT$）。核心原因是：交互项$G_i\cdot Post_t$只在“处理组&政策后”的那一格等于$1$，因此$\beta$刻画了那一格相对其它格子的净增量。

**English**  
The canonical DiD regression (two groups, two periods) is a re-parameterized saturated means model:  
$$  
Y_{it}=\alpha+\gamma G_i+\delta Post_t+\beta(G_i\cdot Post_t)+u_{it},  
$$  
with  
$$  
E[u_{it}\mid G_i,Post_t]=0.  
$$  
Here $\beta$ is the policy effect (and equals $ATT$ under the identification assumptions).

---

## 7. 严谨对应：为什么回归系数$\beta$等于“差上加差” Why the OLS $\beta$ equals DiD

**中文**  
把回归模型对四个格子取条件期望（利用$$E[u_{it}\mid G_i,Post_t]=0$$），得到：  
$$  
E[Y_{it}\mid G_i=0,Post_t=0]=\alpha,  
$$  
$$  
E[Y_{it}\mid G_i=1,Post_t=0]=\alpha+\gamma,  
$$  
$$  
E[Y_{it}\mid G_i=0,Post_t=1]=\alpha+\delta,  
$$  
$$  
E[Y_{it}\mid G_i=1,Post_t=1]=\alpha+\gamma+\delta+\beta.  
$$

因此：  
$$  
\Big(E[Y\mid 1,1]-E[Y\mid 1,0]\Big)-\Big(E[Y\mid 0,1]-E[Y\mid 0,0]\Big)=\beta.  
$$

这说明：在两组两期、并且用这一套回归时，$\hat\beta$在样本中对应的就是$\widehat{DiD}$（用样本均值替换期望）。

**English**  
Taking conditional expectations across the four cells yields:  
$$  
E[Y\mid 0,0]=\alpha,\quad E[Y\mid 1,0]=\alpha+\gamma,  
$$  
$$  
E[Y\mid 0,1]=\alpha+\delta,\quad E[Y\mid 1,1]=\alpha+\gamma+\delta+\beta.  
$$  
Hence the difference-in-differences in conditional means equals $\beta$.

---

## 8. 一个你必须能复述的“最小例子” Minimal numeric example you must be able to reproduce

**中文**  
设四格均值为：

- $\bar Y_{1,0}=10$（处理组、前）
    
- $\bar Y_{1,1}=12$（处理组、后）
    
- $\bar Y_{0,0}=11$（对照组、前）
    
- $\bar Y_{0,1}=12$（对照组、后）
    

则  
$$  
\widehat{DiD}=(12-10)-(12-11)=1.  
$$  
在$DiD$回归里，$\hat\beta$就等于这个$1$。

**English**  
With cell means $\bar Y_{1,0}=10,\bar Y_{1,1}=12,\bar Y_{0,0}=11,\bar Y_{0,1}=12$,  
$$  
\widehat{DiD}=(12-10)-(12-11)=1,  
$$  
and the DiD regression coefficient $\hat\beta$ equals $1$ in this setup.

---

## 9. 你学到这里算“过关”的检查点 Quick mastery checklist

**中文**  
你现在需要能做到三件事：

1. 说清$Y_{it}(1)$、$Y_{it}(0)$、$D_{it}$与观测$Y_{it}$的关系；
    
2. 写出平行趋势假设（用$Y_{it}(0)$写）；
    
3. 写出$DiD$回归并说明为什么$\beta$等于差上加差。
    

**English**  
You should be able to:

1. relate $Y_{it}(1)$, $Y_{it}(0)$, $D_{it}$ to observed $Y_{it}$;
    
2. state parallel trends using $Y_{it}(0)$;
    
3. write the DiD regression and explain why $\beta$ equals the DiD contrast.
    

---

如果你同意，我们下一讲就紧接着做：**为什么$DiD$回归里不能用普通标准误，必须考虑“按簇聚类”**（这正是Hansen论文的入口）。  
If you’re good with this, next we’ll move directly to: **why ordinary standard errors fail in DiD and why clustering matters**, which is exactly the gateway to Hansen’s paper.

你想用什么贯穿例子？（工资/房价/企业利润/碳排放都行）  
Which running example do you prefer (wages, housing prices, firm profits, emissions)?

