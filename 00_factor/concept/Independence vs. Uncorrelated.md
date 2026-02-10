---
aliases:
- 独立性与不相关
- Independence vs. Uncorrelated
- Independence
tags:
- concept
---
**独立性（Independence）**是概率分布层面的概念：$如果知道变量 V_1 的取值不会影响另一个变量 V_2 的分布，则称 V_1 和 V_2 相互独立。数学上，独立性意味着联合密度函数可以分解为边际密度之积：f_{V_1,V_2}(v_1,v_2) = f_{V_1}(v_1)\,f_{V_2}(v_2)。$

$**零相关（Zero Correlation）**仅指相关系数 \rho=0，即 E(V_1 V_2) = E(V_1)E(V_2)。零相关并不必然意味着独立——零相关只反映线性关系的消失，但两变量仍可能存在非线性依赖关系。$

**独立 $\not\equiv$ 零相关**的经典例子：  
设 $V_1$ 取值 -1、0、+1，三者概率相等各为$\frac{1}{3}$。定义 $V_2$ 满足：$当 V_1 = \pm 1 时，V_2 = 1；当 V_1 = 0 时，V_2 = 0。由此构造的 (V_1, V_2) 满足：$
- $P(V_1 = -1, V_2 = 1) = 1/3,\; P(V_1 = 1, V_2 = 1) = 1/3,\; P(V_1 = 0, V_2 = 0) = 1/3。$
在该情形下，$V_2$ 显然依赖于 $V_1$（因为 $V_2$ 的取值由 $V_1$ 决定），两者不是独立的。但计算相关系数：
$$
E(V_1) = \frac{-1 + 0 + 1}{3} = 0,\qquad
E(V_2) = \frac{1 + 1 + 0}{3} = \frac{2}{3}\,,
$$ 
$$
E(V_1 V_2) = (-1)\cdot1\cdot\frac{1}{3} + (0)\cdot0\cdot\frac{1}{3} + (1)\cdot1\cdot\frac{1}{3} = 0 \,.
$$ 
$因此 Cov(V_1,V_2) = E(V_1V_2) - E(V_1)E(V_2) = 0 - 0 \cdot \frac{2}{3} = 0，导致 \rho_{1,2} = 0。也就是说 V_1 和 V_2 **零相关但不独立**。这个例子说明了零相关并不保证独立性。$

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
