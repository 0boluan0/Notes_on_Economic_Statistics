---
aliases:
  - "Python OOP, Classes and Inheritance"
  - "Python Object-Oriented Programming"
  - "Python面向对象"
status: source-checked
---

# Python 面向对象、类与继承
<!-- bilingual-en:start -->
*Object-Oriented Programming, Classes, and Inheritance in Python*
<!-- bilingual-en:end -->

> [!summary] 快速恢复
> **它解决什么：** 把状态与操作组合成有清晰不变量的对象，并在真正的“is-a”关系下复用和替换行为。
> **具体锚点：** `Account` 对象同时保存余额和存取规则；直接让任何代码随意改余额会破坏不变量。
> **核心难点：** 类属性与实例属性不同；继承不是代码去重工具，错误层级会产生脆弱耦合。
> **为什么重要：** OOP 的价值是管理状态和接口，不是把每个函数都塞进类。
> **继续：** 先设计对象责任和不变量，再选组合或继承；默认优先组合。
> <!-- bilingual-en:start -->
> **Problem addressed:** Combine state and operations in objects with explicit invariants, and reuse or substitute behavior only under a genuine “is-a” relationship.
> **Concrete anchor:** An `Account` object stores both a balance and the rules for deposits and withdrawals; allowing arbitrary code to modify the balance would break its invariant.
> **Central difficulty:** Class attributes differ from instance attributes, and inheritance is not merely a tool for deduplicating code; a false hierarchy creates fragile coupling.
> **Why it matters:** The value of object-oriented programming is managing state and interfaces, not placing every function inside a class.
> **Continue with:** Design object responsibilities and invariants first, then choose composition or inheritance, preferring composition by default.
> <!-- bilingual-en:end -->

> [!source] 本节依据
> - MIT 6.100L 本地官方 slides、transcripts、finger exercises 与 problem sets：支持每个概念、示例和课程范围。
> - [[03_Computer_Science/03_MIT 6.100L/Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf]]：交叉核验 Python、算法、复杂度、OOP 与模拟。
> <!-- bilingual-en:start -->
> - Local official MIT 6.100L slides, transcripts, finger exercises, and problem sets support the concepts, examples, and course scope.
> - [[03_Computer_Science/03_MIT 6.100L/Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Introduction to Computation and Programming Using Python]] cross-checks Python, algorithms, complexity, object-oriented programming, and simulation.
> <!-- bilingual-en:end -->

## 类、实例与方法
<!-- bilingual-en:start -->
*Classes, Instances, and Methods*
<!-- bilingual-en:end -->

类定义创建对象的模板/命名空间，实例有独立属性。实例方法首参数 `self` 接收调用对象；`__init__` 初始化而非构造对象本身。属性查找先实例再类及 MRO。
<!-- bilingual-en:start -->
A class definition creates a template and namespace, while each instance can hold its own attributes. The first parameter of an instance method, conventionally `self`, receives the calling object; `__init__` initializes an already created instance. Attribute lookup searches the instance, then the class and its method resolution order.
<!-- bilingual-en:end -->

## 类属性与实例属性
<!-- bilingual-en:start -->
*Class and Instance Attributes*
<!-- bilingual-en:end -->

类属性共享，实例赋同名属性会遮蔽而非修改其他实例。可变类属性若用作每实例容器会意外共享；应在 `__init__` 创建实例状态。
<!-- bilingual-en:start -->
A class attribute is shared. Assigning an attribute of the same name on an instance shadows the class attribute rather than changing other instances. A mutable class attribute accidentally shares per-instance state, so such containers should be created in `__init__`.
<!-- bilingual-en:end -->

## 封装与不变量
<!-- bilingual-en:start -->
*Encapsulation and Invariants*
<!-- bilingual-en:end -->

Python 以约定和 property 管理访问，不提供绝对私有。方法应维护余额非负、坐标有效等不变量。公开 API 越小，状态空间越易推理。
<!-- bilingual-en:start -->
Python manages access through conventions and properties rather than absolute privacy. Methods should preserve invariants such as a nonnegative balance or valid coordinates. A smaller public API makes the state space easier to reason about.
<!-- bilingual-en:end -->

不变量要在每个能改变状态的入口维护，而不是只在构造时检查。若属性必须联动更新，把规则放在一个方法或 property 中，避免调用者各自复制规则。
<!-- bilingual-en:start -->
An invariant must be maintained at every entry point that can change state, not only during initialization. If attributes must change together, centralize the rule in one method or property instead of asking every caller to reproduce it.
<!-- bilingual-en:end -->

## 特殊方法
<!-- bilingual-en:start -->
*Special Methods*
<!-- bilingual-en:end -->

`__repr__`、`__str__`、`__eq__`、迭代等让对象接入 Python data model。相等语义与哈希必须一致；实现 `__eq__` 的可变对象通常不应可哈希。
<!-- bilingual-en:start -->
Special methods such as `__repr__`, `__str__`, `__eq__`, and iteration protocols integrate an object with the Python data model. Equality and hashing must agree; a mutable object that defines value equality should usually not be hashable.
<!-- bilingual-en:end -->

## 继承与多态
<!-- bilingual-en:start -->
*Inheritance and Polymorphism*
<!-- bilingual-en:end -->

子类应可在需要父类处替换而不破坏契约（Liskov intuition）。override 可调用 `super()` 复用协作实现；多重继承需理解 MRO。只为复用几行代码而继承常破坏语义。
<!-- bilingual-en:start -->
A subclass should be substitutable wherever the parent contract is expected, following the Liskov intuition. An override can call `super()` to cooperate with parent implementations, while multiple inheritance requires understanding the MRO. Inheriting merely to reuse a few lines often breaks the semantic relationship.
<!-- bilingual-en:end -->

## 组合
<!-- bilingual-en:start -->
*Composition*
<!-- bilingual-en:end -->

“has-a”关系用成员对象委托，耦合更小、运行时可替换。策略、适配器等许多模式本质是简单组合；不用为单一实现先造抽象层。
<!-- bilingual-en:start -->
A “has-a” relationship delegates to a member object, reducing coupling and allowing runtime substitution. Many strategy and adapter patterns are simple composition; do not invent an abstraction layer for a single implementation.
<!-- bilingual-en:end -->

## Worked example：用不变量保护账户
<!-- bilingual-en:start -->
*Worked Example: Protect an Account Invariant*
<!-- bilingual-en:end -->

提款方法负责验证金额和余额，因此任何成功调用都保持 `balance >= 0`。若调用者直接写 `account.balance -= amount`，对象无法保证自己的契约。
<!-- bilingual-en:start -->
The withdrawal method validates both the amount and the available balance, so every successful call preserves `balance >= 0`. If callers directly execute `account.balance -= amount`, the object cannot enforce its contract.
<!-- bilingual-en:end -->

```python
class Account:
    def __init__(self, opening_balance=0):
        if opening_balance < 0:
            raise ValueError("opening balance must be nonnegative")
        self._balance = opening_balance

    @property
    def balance(self):
        return self._balance

    def withdraw(self, amount):
        if amount <= 0:
            raise ValueError("amount must be positive")
        if amount > self._balance:
            raise ValueError("insufficient funds")
        self._balance -= amount
```

## dataclass 与简化
<!-- bilingual-en:start -->
*Dataclasses and Simpler Alternatives*
<!-- bilingual-en:end -->

主要是数据载体时 `dataclass` 自动生成样板；行为简单时字典/tuple/函数可能已经足够。类只有在状态、身份或不变量值得集中管理时存在。
<!-- bilingual-en:start -->
When an object is mainly a data carrier, `dataclass` can generate boilerplate. For simple behavior, a dictionary, tuple, or function may already be sufficient. A class earns its existence when state, identity, or invariants need centralized management.
<!-- bilingual-en:end -->

## 失败诊断
<!-- bilingual-en:start -->
*Failure Diagnosis*
<!-- bilingual-en:end -->

- 多个实例共享列表：检查容器是否误写成类属性，或默认参数是否复用了同一对象。
  <!-- bilingual-en:start -->
  Instances share a list: check whether the container was defined as a class attribute or reused as a mutable default.
  <!-- bilingual-en:end -->
- override 后父类行为失效：逐项比较父类契约，检查子类是否加强前置条件或削弱后置条件。
  <!-- bilingual-en:start -->
  A subclass breaks parent behavior: compare the parent contract item by item and check whether the subclass strengthened preconditions or weakened postconditions.
  <!-- bilingual-en:end -->
- 层级不断加深：重新问关系是否真是 is-a；若只是借用能力，改为组合和委托。
  <!-- bilingual-en:start -->
  The hierarchy keeps deepening: ask again whether the relationship is genuinely “is-a”; if it merely borrows capability, use composition and delegation.
  <!-- bilingual-en:end -->

## 最小自检
<!-- bilingual-en:start -->
*Minimum Self-Check*
<!-- bilingual-en:end -->

### 可变类属性为什么常造成 bug？
<!-- bilingual-en:start -->
*Why do mutable class attributes often cause bugs?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 所有未遮蔽它的实例共享同一对象，一个实例修改会被其他实例看到。
> <!-- bilingual-en:start -->
> Every instance that has not shadowed the attribute shares the same object, so one instance's mutation is visible to the others.
> <!-- bilingual-en:end -->

### 什么时候组合优于继承？
<!-- bilingual-en:start -->
*When is composition preferable to inheritance?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 关系是 has-a、只想复用行为、需要运行时替换，或子对象不满足父类完整契约时。
> <!-- bilingual-en:start -->
> When the relationship is “has-a,” only behavior reuse is needed, runtime substitution matters, or the child does not satisfy the parent's complete contract.
> <!-- bilingual-en:end -->

### 子类可替换父类意味着什么？
<!-- bilingual-en:start -->
*What does it mean for a subclass to be substitutable for its parent?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 调用者只依赖父类契约时，换成子类仍保持前置/后置条件和可观察行为。
> <!-- bilingual-en:start -->
> A caller relying only on the parent contract should observe valid preconditions, postconditions, and behavior after the subclass is substituted.
> <!-- bilingual-en:end -->

### 为什么把余额设成只读 property 仍不能自动保证账户正确？
<!-- bilingual-en:start -->
*Why does a read-only balance property not automatically make an account correct?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 所有内部修改方法仍必须维护不变量，而且可变成员对象也可能从其他路径泄露；property 只是一个入口控制工具。
> <!-- bilingual-en:start -->
> Every internal mutator must still preserve the invariant, and mutable member objects can leak through other paths; a property controls one entry point only.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and Verification*
<!-- bilingual-en:end -->

- MIT 6.100L 本地官方 slides、transcripts、finger exercises 与 problem sets：支持每个概念、示例和课程范围。
  <!-- bilingual-en:start -->
  Local official MIT 6.100L slides, transcripts, finger exercises, and problem sets support the concepts, examples, and course scope.
  <!-- bilingual-en:end -->
- [[03_Computer_Science/03_MIT 6.100L/Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf]]：交叉核验 Python、算法、复杂度、OOP 与模拟。
  <!-- bilingual-en:start -->
  [[03_Computer_Science/03_MIT 6.100L/Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Introduction to Computation and Programming Using Python]] cross-checks Python, algorithms, complexity, object-oriented programming, and simulation.
  <!-- bilingual-en:end -->
- [The Python Tutorial: Classes](https://docs.python.org/3/tutorial/classes.html)：复核实例、类属性、继承、作用域与迭代器的语言规则。
  <!-- bilingual-en:start -->
  [The Python Tutorial: Classes](https://docs.python.org/3/tutorial/classes.html) verifies language rules for instances, class attributes, inheritance, scope, and iterators.
  <!-- bilingual-en:end -->
