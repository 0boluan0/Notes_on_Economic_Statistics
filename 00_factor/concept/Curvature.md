---
aliases:
- 曲率
- Curvature
tags:
- concept
---
## 一、公式说明

- $C$ 是凸性（曲率），$D$ 是久期，$B$ 是债券价格，$y$ 是到期收益率
- 价格变动近似公式：
    $$
\frac{\Delta B}{B} = -D \Delta y + \frac{1}{2}C (\Delta y)^2
    $$
- 意思是：价格对利率变动的敏感度，除了久期线性部分，还有二阶的凸性修正项。

## **二、数值例子**

- 债券面值 $1000$，现价 $B=1000$
- 久期 $D=5$
- 凸性 $C=30$
- $市场利率上升 1\%，即 \Delta y = 0.01$

### **1. 只用久期近似**

$$
\frac{\Delta B}{B} \approx -D \Delta y = -5 \times 0.01 = -0.05
$$

$也就是说，债券价格大约**下跌5%**，变为 1000\times(1-0.05)=950 元。$

### **2. 用久期+凸性修正**

$$

\frac{\Delta B}{B} \approx -D \Delta y + \frac{1}{2}C (\Delta y)^2

$$

代入：
- $D=5$
- $C=30$
- $\Delta y = 0.01$
$$

\frac{\Delta B}{B} \approx -5 \times 0.01 + \frac{1}{2} \times 30 \times (0.01)^2 \

= -0.05 + 0.5 \times 30 \times 0.0001 \

= -0.05 + 0.0015 = -0.0485

$$

$所以，**价格大约下跌4.85%**，变为 1000\times(1-0.0485)=951.5 元。$

### **3. 对比和理解**

- 只用久期近似，跌5%，价格950元
- 加上凸性修正，跌4.85%，价格951.5元
- 差值是 $1.5$ 元，就是凸性在价格上的“保护作用”
    （同久期情况下，凸性越高的债券，下跌幅度越小）

### **4. 结论**

- **久期反映一阶敏感性，价格对利率小变动的线性反应**
- **凸性反映二阶敏感性，修正较大变动时久期低估的跌幅**
- 市场利率波动较大时，有凸性的债券价格表现更“抗跌”，投资者更喜欢高凸性
    
## **一句话记忆**

> **凸性让你的债券在利率大变动时少跌一点，久期只是最基本的线性近似。**

## 相关链接

- 一阶风险：[[duration|久期]], [[Macaulay Duration|马考利久期]], [[Modified Duration|修正久期]]
- 完整利率风险框架：久期（一阶）+ 曲率（二阶）

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
