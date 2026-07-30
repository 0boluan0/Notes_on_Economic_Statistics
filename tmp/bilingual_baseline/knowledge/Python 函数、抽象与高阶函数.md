---
aliases:
  - "Python Functions and Abstraction"
  - "Higher-Order Functions"
  - "Python函数"
status: source-checked
---

# Python 函数、抽象与高阶函数

> [!summary] 快速恢复
> **它解决什么：** 用函数建立可测试接口、控制作用域，并把行为本身作为数据组合。
> **具体锚点：** `map`/自定义高阶函数可接收一个函数并把同一遍历框架用于不同转换，固定控制结构而替换行为。
> **核心难点：** 函数定义时不执行函数体；参数绑定、默认值求值和局部作用域决定调用行为。
> **为什么重要：** 良好抽象减少重复推理，而不是只减少行数。
> **继续：** 先写 contract 和纯函数，再使用 first-class function、closure 与高阶组合。

> [!source] 本节依据
> - MIT 6.100L 本地官方 slides、transcripts、finger exercises 与 problem sets：支持每个概念、示例和课程范围。
> - [[03_Computer_Science/03_MIT 6.100L/Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf]]：交叉核验 Python、算法、复杂度、OOP 与模拟。

## 分解与抽象

函数应有单一清楚职责、输入前提、返回和副作用。调用者只依赖 contract；实现细节可替换。过短包装若没有稳定概念只增加跳转，抽象边界应围绕变化原因。

## 参数、返回与作用域

调用时 positional/keyword 参数绑定，局部名称在函数 frame。`return` 立即结束并给值，未显式返回得到 `None`。不要用打印替代返回值；打印是副作用，返回才可组合。

## 默认参数陷阱

默认表达式在 `def` 执行时求值一次；可变默认列表会跨调用共享。使用 `None` 哨兵并在函数内新建容器。类型提示和文档不自动执行验证。

## 函数作为对象

函数可绑定名称、放进容器、传参和返回。不要加括号时传函数对象，加括号是调用结果。高阶函数把“遍历/重试/计时”等框架与具体工作分离。

## closure

内层函数捕获定义环境，可构造参数化函数和保持私有状态。捕获可变变量时注意 late binding；循环生成 closure 可用默认参数或工厂函数固定当前值。

## lambda 与可读性

lambda 只含表达式，适合短小无歧义的回调；复杂逻辑用命名函数以便测试、文档和 traceback。

## 规格与测试

测试接口的正常、边界和错误行为，避免只测试实现细节。纯函数最易复现；副作用应隔离。

## 最小自检

### 为什么可变默认参数会跨调用共享？

> [!answer]- 答案
> 默认对象在函数定义时创建一次，后续调用未传该参数时复用同一对象。
### 传 `f` 与传 `f()` 有什么区别？

> [!answer]- 答案
> 前者传函数对象供以后调用，后者立即调用并传它的返回值。
### 函数抽象的价值为什么不只是减少代码行？

> [!answer]- 答案
> 它建立稳定 contract，让调用者无需反复理解实现，并把变化隔离在一个位置。

## 来源与核验

- MIT 6.100L 本地官方 slides、transcripts、finger exercises 与 problem sets：支持每个概念、示例和课程范围。
- [[03_Computer_Science/03_MIT 6.100L/Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf]]：交叉核验 Python、算法、复杂度、OOP 与模拟。
