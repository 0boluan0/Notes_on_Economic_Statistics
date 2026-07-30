---
aliases:
  - "Python Sequences, Mutability and Aliasing"
  - "Aliasing and Cloning"
  - "Python序列"
status: source-checked
---

# Python 序列、可变性与别名

> [!summary] 快速恢复
> **它解决什么：** 准确判断 tuple、list、string 和 dictionary 中哪些对象可变、哪些名称共享同一对象，以及修改会传播到哪里。
> **具体锚点：** `b = a` 不复制列表；`b.append(...)` 后 a 也看到变化，因为两个名称指向同一对象。
> **核心难点：** 容器不可变不等于其中对象不可变；浅拷贝只复制外层，嵌套元素仍可共享。
> **为什么重要：** 大量“莫名被修改”的 bug、默认参数陷阱和函数副作用都来自别名。
> **继续：** 画对象图而非把变量想成盒子；修改前决定共享、浅拷贝还是深拷贝。

> [!source] 本节依据
> - MIT 6.100L 本地官方 slides、transcripts、finger exercises 与 problem sets：支持每个概念、示例和课程范围。
> - [[03_Computer_Science/03_MIT 6.100L/Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf]]：交叉核验 Python、算法、复杂度、OOP 与模拟。

## 序列共同接口

string、tuple、list 支持长度、索引、切片、迭代和成员测试，但元素/对象可变性不同。切片通常创建新外层序列；负索引和半开区间使边界组合方便。

## 可变与不可变

字符串、tuple、数字不可原地改，操作返回新对象；list/dict/set 可原地修改。tuple 不可变只指其元素引用槽不能换，若元素是 list，内部 list 仍可变。

## aliasing

`b=a` 创建另一个名称绑定同一对象。身份用 `is` 判断，值相等用 `==`。函数收到 list 后原地修改会被调用者观察；是否允许应写在 contract。

## cloning

`a[:]`、`list(a)` 或 `copy.copy` 生成浅拷贝，嵌套对象仍共享。`copy.deepcopy` 递归复制但成本高、对外部资源/自定义对象未必合适。最小需要的复制层级最好。

## list comprehension

列表推导表达“遍历—过滤—映射”，适合无副作用构造。复杂嵌套或副作用用显式循环更清楚。生成器表达式惰性产生值，节省内存但只能按迭代消费。

## dictionary 与哈希

dict 把可哈希 key 映射到 value，平均查找近 O(1)。可哈希对象需哈希稳定且相等对象哈希相同，所以 list 不能作 key；tuple 仅在全部元素可哈希时可作 key。

## 安全迭代

遍历容器时结构修改可能跳过元素或报错；构造新容器、遍历副本或分两阶段修改。

## 最小自检

### `b=a` 和 `b=a[:]` 对列表有何差别？

> [!answer]- 答案
> 前者共享同一列表；后者复制外层列表，但嵌套元素仍可能共享。
### tuple 为什么可能间接变化？

> [!answer]- 答案
> tuple 的引用槽固定，但若槽指向可变对象，该对象内部可修改。
### 为什么 list 不能作 dict key？

> [!answer]- 答案
> list 可变，内容变化会破坏哈希值稳定性和哈希表查找不变量。

## 来源与核验

- MIT 6.100L 本地官方 slides、transcripts、finger exercises 与 problem sets：支持每个概念、示例和课程范围。
- [[03_Computer_Science/03_MIT 6.100L/Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf]]：交叉核验 Python、算法、复杂度、OOP 与模拟。
