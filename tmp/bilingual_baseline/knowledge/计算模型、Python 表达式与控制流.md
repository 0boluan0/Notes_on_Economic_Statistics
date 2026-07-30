---
aliases:
  - "Python Fundamentals and Control Flow"
  - "Python Basics"
  - "Python基础"
status: source-checked
---

# 计算模型、Python 表达式与控制流

> [!summary] 快速恢复
> **它解决什么：** 把问题写成 Python 对象、表达式和明确控制流，理解程序是状态随语句执行而变化的过程。
> **具体锚点：** `input()` 返回字符串；若要数值比较必须显式转换，否则 `"10" < "2"` 按字符顺序为真。
> **核心难点：** 对象有类型和值，名称只是绑定；分支只执行一条路径，循环需保证状态推进和终止。
> **为什么重要：** 后续函数、数据结构、测试和算法都假设能准确模拟基本求值。
> **继续：** 用小例子手工 trace 表达式、分支和循环，再写程序。

> [!source] 本节依据
> - MIT 6.100L 本地官方 slides、transcripts、finger exercises 与 problem sets：支持每个概念、示例和课程范围。
> - [[03_Computer_Science/03_MIT 6.100L/Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf]]：交叉核验 Python、算法、复杂度、OOP 与模拟。

## 计算与程序

程序把输入经算法映射为输出。声明式知识说结果性质，命令式知识给步骤。Python 解释器执行语句并维护环境；同一算法可用不同语言表达，语言不是算法本身。

## 对象、类型与表达式

数字、布尔、字符串等都是对象；操作符按类型定义。`type` 决定允许操作，转换创建相应值。`==` 比较值，`is` 比较对象身份，后者通常只用于 `None` 等单例。

## 名称与赋值

赋值让名称指向对象，不是把值永久装进盒子。右侧先求值再绑定左侧；多重赋值可安全交换。不可变对象“变化”实际是名称改绑到新对象。

## 字符串与输入输出

字符串是不可变序列，索引从 0，切片半开区间。`input` 返回字符串，格式化输出应显式控制类型和精度。用户输入是信任边界，要验证并处理转换失败。

## 分支

`if/elif/else` 按顺序选择第一条真分支。布尔表达式短路，可用于安全 guard；过度依赖 truthiness 会模糊 0、空容器和 `None` 的差别。

## 迭代

`while` 适合条件驱动，`for` 遍历 iterable。循环要有不变量和进度量，注意 off-by-one、边界和修改正在遍历的容器。`range` 也是半开区间。

## 分解问题

先写示例和输入输出，再把大步骤拆成可测试小块。能用现有内置/标准库表达时不手写复杂循环。

## 最小自检

### Python 赋值语句改变的是对象还是名称绑定？

> [!answer]- 答案
> 通常先求右侧对象，再让名称绑定到它；不可变对象本身不被修改。
### `is` 和 `==` 何时不同？

> [!answer]- 答案
> `==` 比较值是否相等，`is` 比较是否同一对象；普通数值/字符串应使用 `==`。
### 一个 `while` 循环如何证明会终止？

> [!answer]- 答案
> 找一个每轮严格朝边界推进、且有下界/上界的度量，并确保所有分支都更新它。

## 来源与核验

- MIT 6.100L 本地官方 slides、transcripts、finger exercises 与 problem sets：支持每个概念、示例和课程范围。
- [[03_Computer_Science/03_MIT 6.100L/Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf]]：交叉核验 Python、算法、复杂度、OOP 与模拟。
