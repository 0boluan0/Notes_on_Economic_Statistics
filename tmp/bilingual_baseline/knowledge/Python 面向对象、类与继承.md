---
aliases:
  - "Python OOP, Classes and Inheritance"
  - "Object-Oriented Programming"
  - "Python面向对象"
status: source-checked
---

# Python 面向对象、类与继承

> [!summary] 快速恢复
> **它解决什么：** 把状态与操作组合成有清晰不变量的对象，并在真正的“is-a”关系下复用和替换行为。
> **具体锚点：** `Account` 对象同时保存余额和存取规则；直接让任何代码随意改余额会破坏不变量。
> **核心难点：** 类属性与实例属性不同；继承不是代码去重工具，错误层级会产生脆弱耦合。
> **为什么重要：** OOP 的价值是管理状态和接口，不是把每个函数都塞进类。
> **继续：** 先设计对象责任和不变量，再选组合或继承；默认优先组合。

> [!source] 本节依据
> - MIT 6.100L 本地官方 slides、transcripts、finger exercises 与 problem sets：支持每个概念、示例和课程范围。
> - [[03_Computer_Science/03_MIT 6.100L/Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf]]：交叉核验 Python、算法、复杂度、OOP 与模拟。

## 类、实例与方法

类定义创建对象的模板/命名空间，实例有独立属性。实例方法首参数 `self` 接收调用对象；`__init__` 初始化而非构造对象本身。属性查找先实例再类及 MRO。

## 类属性与实例属性

类属性共享，实例赋同名属性会遮蔽而非修改其他实例。可变类属性若用作每实例容器会意外共享；应在 `__init__` 创建实例状态。

## 封装与不变量

Python 以约定和 property 管理访问，不提供绝对私有。方法应维护余额非负、坐标有效等不变量。公开 API 越小，状态空间越易推理。

## 特殊方法

`__repr__`、`__str__`、`__eq__`、迭代等让对象接入 Python data model。相等语义与哈希必须一致；实现 `__eq__` 的可变对象通常不应可哈希。

## 继承与多态

子类应可在需要父类处替换而不破坏契约（Liskov intuition）。override 可调用 `super()` 复用协作实现；多重继承需理解 MRO。只为复用几行代码而继承常破坏语义。

## 组合

“has-a”关系用成员对象委托，耦合更小、运行时可替换。策略、适配器等许多模式本质是简单组合；不用为单一实现先造抽象层。

## dataclass 与简化

主要是数据载体时 `dataclass` 自动生成样板；行为简单时字典/tuple/函数可能已经足够。类只有在状态、身份或不变量值得集中管理时存在。

## 最小自检

### 可变类属性为什么常造成 bug？

> [!answer]- 答案
> 所有未遮蔽它的实例共享同一对象，一个实例修改会被其他实例看到。
### 什么时候组合优于继承？

> [!answer]- 答案
> 关系是 has-a、只想复用行为、需要运行时替换，或子对象不满足父类完整契约时。
### 子类可替换父类意味着什么？

> [!answer]- 答案
> 调用者只依赖父类契约时，换成子类仍保持前置/后置条件和可观察行为。

## 来源与核验

- MIT 6.100L 本地官方 slides、transcripts、finger exercises 与 problem sets：支持每个概念、示例和课程范围。
- [[03_Computer_Science/03_MIT 6.100L/Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf]]：交叉核验 Python、算法、复杂度、OOP 与模拟。
