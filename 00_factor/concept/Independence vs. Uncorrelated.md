---
aliases:
- 独立性与不相关
- Independence vs. Uncorrelated
tags:
  - concept
  - probability
---
# Independence vs. Uncorrelated

**独立性（independence）**是概率分布层面的概念：知道变量 $V_1$ 的取值不会改变 $V_2$ 的条件分布。对离散变量，等价地，所有取值满足
$$
\Pr(V_1=v_1,V_2=v_2)=\Pr(V_1=v_1)\Pr(V_2=v_2).
$$

**零相关（zero correlation）**在二阶矩存在且方差非零时指 $\rho(V_1,V_2)=0$，等价于
$$
E[V_1V_2]=E[V_1]E[V_2].
$$
它只排除线性相关，不能排除非线性依赖。

**独立 $\not\equiv$ 零相关**的经典例子：  
设 $V_1$ 以相同概率取 $-1,0,1$，并令 $V_2=V_1^2$。于是
- $\Pr(V_1=-1,V_2=1)=\Pr(V_1=1,V_2=1)=1/3$；
- $\Pr(V_1=0,V_2=0)=1/3$。
在该情形下，$V_2$ 由 $V_1$ 决定，所以两者不独立；但
$$
E(V_1) = \frac{-1 + 0 + 1}{3} = 0,\qquad
E(V_2) = \frac{1 + 1 + 0}{3} = \frac{2}{3}\,,
$$ 
$$
E(V_1 V_2) = (-1)\cdot1\cdot\frac{1}{3} + (0)\cdot0\cdot\frac{1}{3} + (1)\cdot1\cdot\frac{1}{3} = 0 \,.
$$ 
因此 $\operatorname{Cov}(V_1,V_2)=0$，即两者零相关但不独立。


## 最小例子

把 **Independence vs. Uncorrelated** 放在最简单的可计算情形中：先给定定义所需的最小输入，再按定义计算输出；若关键关系不成立，就不能把该对象归入本概念。这个检查也能帮助区分相近概念。
## 概念边界

本卡片只回答“它是什么”，集中在定义、核心关系与最小例子；具体估计步骤、证明和诊断流程应分别放在 procedure、proof 或 system 卡片中。
## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM ""
WHERE (
  contains(file.path, "01_Math/") OR
  contains(file.path, "02_Economy/") OR
  contains(file.path, "03_Computer_Science/")
) AND contains(file.outlinks, this.file.link)
SORT file.mtime DESC
```
