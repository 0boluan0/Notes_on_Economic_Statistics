---
aliases:
  - "Python Functions, Scope, Closures, and Higher-Order Functions"
  - "Python函数"
status: source-checked
---

# Python 函数、作用域、闭包与高阶函数
<!-- bilingual-en:start -->
*Python Functions, Scope, Closures, and Higher-Order Functions*
<!-- bilingual-en:end -->

> [!summary] 快速恢复
> **它解决什么：** 用函数建立可测试接口、控制作用域，并把行为本身作为数据组合。
> **具体锚点：** `map`/自定义高阶函数可接收一个函数并把同一遍历框架用于不同转换，固定控制结构而替换行为。
> **核心难点：** 函数定义时不执行函数体；参数绑定、默认值求值和局部作用域决定调用行为。
> **为什么重要：** 良好抽象减少重复推理，而不是只减少行数。
> **继续：** 先写 contract 和纯函数，再使用 first-class function、closure 与高阶组合。
> <!-- bilingual-en:start -->
> **Problem addressed:** Build testable interfaces with functions, control scope, and compose behavior as data.
> **Concrete anchor:** `map` or a custom higher-order function accepts a function, retaining one traversal structure while substituting different behavior.
> **Central difficulty:** Defining a function does not execute its body; parameter binding, default-value evaluation, and lexical scope determine each call.
> **Why it matters:** A good abstraction reduces repeated reasoning, not merely lines of code.
> **Continue with:** Write a contract and a pure function first, then introduce first-class functions, closures, and higher-order composition.
> <!-- bilingual-en:end -->

> [!source] 本节依据
> - MIT 6.100L 本地官方 slides、transcripts、finger exercises 与 problem sets：支持每个概念、示例和课程范围。
> - [[03_Computer_Science/03_MIT 6.100L/Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf]]：交叉核验 Python、算法、复杂度、OOP 与模拟。
> <!-- bilingual-en:start -->
> - Local official MIT 6.100L slides, transcripts, finger exercises, and problem sets support the concepts, examples, and course scope.
> - [[03_Computer_Science/03_MIT 6.100L/Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Introduction to Computation and Programming Using Python]] cross-checks Python, algorithms, complexity, object-oriented programming, and simulation.
> <!-- bilingual-en:end -->

## 函数与递归的统一视角
<!-- bilingual-en:start -->
*A Unified View of Functions and Recursion*
<!-- bilingual-en:end -->

> [!summary] 快速恢复
> **它解决什么：** 用函数把“做什么”与“怎样做”分开，并用环境模型解释名称绑定、高阶函数和递归调用。
> **具体锚点：** `make_adder(3)` 返回的函数以后仍记得 `3`，因为闭包捕获定义时环境，不是把数字藏进函数名字。
> **核心难点：** 表达式求值发生在具体 frame；递归不是函数复制自己，而是每次调用建立新 frame 并等待子调用返回。
> **为什么重要：** 这是 CS61A 后续数据抽象、解释器和声明式编程的共同基础。
> **继续：** 先用环境图追踪调用，再写高阶函数与递归；不能画清 frame 就不要只靠运行猜。
> <!-- bilingual-en:start -->
> **Problem addressed:** Separate what a computation does from how it does it, and use the environment model to explain name binding, higher-order functions, and recursive calls.
> **Concrete anchor:** A function returned by `make_adder(3)` still remembers `3` because a closure captures its defining environment; the number is not hidden in the function's name.
> **Central difficulty:** Expressions are evaluated in concrete frames. Recursion is not a function copying itself; every call creates a fresh frame and waits for a subcall to return.
> **Why it matters:** This is the common foundation for later CS61A work on data abstraction, interpreters, and declarative programming.
> **Continue with:** Trace calls with environment diagrams before writing higher-order and recursive functions; if the frames cannot be drawn, do not rely on running the code and guessing.
> <!-- bilingual-en:end -->

> [!source] 本节依据
> - [Composing Programs](https://www.composingprograms.com/) 与本地 CS61A 教材：核验函数抽象、环境模型和递归。
> <!-- bilingual-en:start -->
> - [Composing Programs](https://www.composingprograms.com/) and the local CS61A text verify functional abstraction, the environment model, and recursion.
> <!-- bilingual-en:end -->

## 分解与抽象
<!-- bilingual-en:start -->
*Decomposition and Abstraction*
<!-- bilingual-en:end -->

函数应有单一清楚职责、输入前提、返回和副作用。调用者只依赖 contract；实现细节可替换。过短包装若没有稳定概念只增加跳转，抽象边界应围绕变化原因。
<!-- bilingual-en:start -->
A function should have one clear responsibility, explicit input preconditions, a return contract, and known side effects. Callers depend on the contract, so implementation details can change. A tiny wrapper without a stable concept merely adds navigation; an abstraction boundary should follow a reason for change.
<!-- bilingual-en:end -->

函数用名称、参数、文档和返回契约隐藏实现。调用者依赖接口而非内部步骤；局部重构不应改变可观察行为。把重复代码抽出只有在形成清晰概念时有价值，而不是为了制造一行包装。
<!-- bilingual-en:start -->
A function hides implementation behind a name, parameters, documentation, and a return contract. Callers depend on the interface rather than internal steps, so a local refactor should not alter observable behavior. Extracting repeated code is useful only when it creates a clear concept, not when it merely creates a one-line wrapper.
<!-- bilingual-en:end -->

## 参数、返回与默认值
<!-- bilingual-en:start -->
*Parameters, Return Values, and Defaults*
<!-- bilingual-en:end -->

调用时 positional/keyword 参数绑定，局部名称在函数 frame。`return` 立即结束并给值，未显式返回得到 `None`。不要用打印替代返回值；打印是副作用，返回才可组合。
<!-- bilingual-en:start -->
At call time, positional and keyword arguments bind parameters inside a function frame. `return` ends execution immediately and supplies a value; falling off the end returns `None`. Printing is a side effect and cannot replace a return value, which is what makes results composable.
<!-- bilingual-en:end -->

默认表达式在 `def` 执行时求值一次；可变默认列表会跨调用共享。使用 `None` 哨兵并在函数内新建容器。类型提示和文档不自动执行验证。
<!-- bilingual-en:start -->
A default expression is evaluated once when the `def` statement runs, so a mutable default list is shared across calls. Use a `None` sentinel and create the container inside the function. Type hints and documentation do not perform runtime validation automatically.
<!-- bilingual-en:end -->

## 表达式、调用与环境模型
<!-- bilingual-en:start -->
*Expressions, Calls, and the Environment Model*
<!-- bilingual-en:end -->

求值先确定 operator 和 operands，再把实参绑定到新调用 frame 的形参，执行函数体并返回值。名称查找沿当前 frame 的 parent environment 向外。求值顺序和副作用会影响结果，纯表达式更易推理。
<!-- bilingual-en:start -->
Evaluation determines the operator and operands, binds arguments to parameters in a new call frame, executes the body, and returns a value. Name lookup proceeds from the current frame through its parent environment. Evaluation order matters when side effects are present, which is why pure expressions are easier to reason about.
<!-- bilingual-en:end -->

赋值把名称绑定到对象，调用创建 frame，嵌套函数的 parent 指向定义环境。`nonlocal`/`global` 改变绑定位置。环境图能区分同名变量、对象共享和闭包状态。
<!-- bilingual-en:start -->
Assignment binds a name to an object, a call creates a frame, and a nested function's parent points to its defining environment. `nonlocal` and `global` change where a binding is updated. An environment diagram distinguishes shadowed names, shared objects, and closure state.
<!-- bilingual-en:end -->

## 函数作为对象与高阶函数
<!-- bilingual-en:start -->
*Functions as Objects and Higher-Order Functions*
<!-- bilingual-en:end -->

函数可绑定名称、放进容器、传参和返回。不要加括号时传函数对象，加括号是调用结果。高阶函数把“遍历/重试/计时”等框架与具体工作分离。
<!-- bilingual-en:start -->
A function can be bound to a name, stored in a container, passed as an argument, or returned. Passing `f` passes the function object, whereas `f()` evaluates the call and passes its result. Higher-order functions separate a stable framework such as traversal, retrying, or timing from the work supplied to it.
<!-- bilingual-en:end -->

函数可作为参数和返回值，允许把变化的行为与固定控制结构分离。closure 保存函数代码和定义环境；返回后捕获变量仍可访问。lambda 适合短小表达式，不应遮蔽复杂逻辑。
<!-- bilingual-en:start -->
Because functions can be arguments and return values, changing behavior can be separated from fixed control structure. A closure stores both function code and its defining environment, so captured variables remain accessible after the outer call returns. A lambda suits a short expression but should not hide complex logic.
<!-- bilingual-en:end -->

## closure
<!-- bilingual-en:start -->
*Closures*
<!-- bilingual-en:end -->

内层函数捕获定义环境，可构造参数化函数和保持私有状态。捕获可变变量时注意 late binding；循环生成 closure 可用默认参数或工厂函数固定当前值。
<!-- bilingual-en:start -->
An inner function captures its defining environment, enabling parameterized functions and private state. When closures capture a variable that later changes, beware of late binding; a factory function or a default argument can freeze the current value when generating closures in a loop.
<!-- bilingual-en:end -->

`nonlocal` 允许闭包重绑定外层局部名称，但它会引入隐式状态。若状态变化本身是主题，可以使用；若只为绕过返回值，应优先返回新值，让数据流可见。
<!-- bilingual-en:start -->
`nonlocal` lets a closure rebind a name in an enclosing local scope, but it introduces implicit state. Use it when state evolution is the point; if it merely avoids returning a value, prefer returning the new value so that data flow remains visible.
<!-- bilingual-en:end -->

## Worked example：生成一组倍率函数
<!-- bilingual-en:start -->
*Worked Example: Generate a Family of Scaling Functions*
<!-- bilingual-en:end -->

下面的工厂函数把倍率保存在定义环境中。调用 `make_scaler(3)` 后，外层 frame 虽已返回，返回的函数仍能沿 parent environment 找到 `factor=3`。
<!-- bilingual-en:start -->
The factory below stores a multiplier in its defining environment. After `make_scaler(3)` returns, the returned function can still find `factor=3` through its parent environment.
<!-- bilingual-en:end -->

```python
def make_scaler(factor):
    """Return a function that multiplies its input by factor."""
    def scale(value):
        return factor * value

    return scale


triple = make_scaler(3)
print(triple(10))
```

若在循环中直接写 `lambda x: i * x`，所有函数可能在调用时读取同一个最终 `i`。用 `lambda x, factor=i: factor * x` 或调用工厂函数，才会为每次迭代固定值。
<!-- bilingual-en:start -->
If a loop directly creates `lambda x: i * x`, every function may read the same final value of `i` when called. Use `lambda x, factor=i: factor * x` or call a factory function to capture a distinct value for each iteration.
<!-- bilingual-en:end -->

## lambda 与可读性
<!-- bilingual-en:start -->
*Lambda and Readability*
<!-- bilingual-en:end -->

lambda 只含表达式，适合短小无歧义的回调；复杂逻辑用命名函数以便测试、文档和 traceback。
<!-- bilingual-en:start -->
A lambda contains one expression and suits a short, unambiguous callback. Use a named function for complex logic so that it can be tested, documented, and identified clearly in a traceback.
<!-- bilingual-en:end -->

## 数据抽象入口
<!-- bilingual-en:start -->
*Entry Point to Data Abstraction*
<!-- bilingual-en:end -->

constructor 和 selector 建立抽象屏障，表示可从 tuple 改为函数或类而不改上层代码。使用者不应绕过 selector 依赖内部表示。
<!-- bilingual-en:start -->
Constructors and selectors create an abstraction barrier: a representation can change from a tuple to a function or class without changing client code. Clients should not bypass selectors and depend on the internal representation.
<!-- bilingual-en:end -->

## 规格、测试与失败诊断
<!-- bilingual-en:start -->
*Specifications, Testing, and Failure Diagnosis*
<!-- bilingual-en:end -->

测试接口的正常、边界和错误行为，避免只测试实现细节。纯函数最易复现；副作用应隔离。
<!-- bilingual-en:start -->
Test normal, boundary, and error behavior at the interface instead of testing only implementation details. Pure functions are easiest to reproduce; isolate side effects.
<!-- bilingual-en:end -->

- 返回 `None`：检查是否漏写 `return`，或把 `print` 误当成返回。
  <!-- bilingual-en:start -->
  Unexpected `None`: check for a missing `return` or a `print` statement being mistaken for a return value.
  <!-- bilingual-en:end -->
- 名称取值意外：画出当前 frame 与 parent environment，检查局部遮蔽、`nonlocal` 和 late binding。
  <!-- bilingual-en:start -->
  An unexpected name value: draw the current frame and its parent environment, then inspect local shadowing, `nonlocal`, and late binding.
  <!-- bilingual-en:end -->
- 多次调用互相污染：检查可变默认参数、全局状态或被闭包共享的可变对象。
  <!-- bilingual-en:start -->
  Calls contaminate one another: inspect mutable defaults, global state, and mutable objects shared by a closure.
  <!-- bilingual-en:end -->

## 最小自检
<!-- bilingual-en:start -->
*Minimum Self-Check*
<!-- bilingual-en:end -->

### 为什么可变默认参数会跨调用共享？
<!-- bilingual-en:start -->
*Why is a mutable default argument shared across calls?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 默认对象在函数定义时创建一次，后续调用未传该参数时复用同一对象。
<!-- bilingual-en:start -->
> [!answer]- Answer
> The default object is created once when the function is defined and reused whenever a later call omits that argument.
<!-- bilingual-en:end -->

### 传 `f` 与传 `f()` 有什么区别？
<!-- bilingual-en:start -->
*What is the difference between passing `f` and passing `f()`?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 前者传函数对象供以后调用，后者立即调用并传它的返回值。
<!-- bilingual-en:start -->
> [!answer]- Answer
> The former passes the function object for later use; the latter calls it immediately and passes its return value.
<!-- bilingual-en:end -->

### 函数抽象的价值为什么不只是减少代码行？
<!-- bilingual-en:start -->
*Why is the value of functional abstraction not merely fewer lines of code?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 它建立稳定 contract，让调用者无需反复理解实现，并把变化隔离在一个位置。
<!-- bilingual-en:start -->
> [!answer]- Answer
> It establishes a stable contract, lets callers avoid repeatedly understanding the implementation, and isolates change in one place.
<!-- bilingual-en:end -->

### 闭包为什么在外层函数返回后仍记得变量？
<!-- bilingual-en:start -->
*Why does a closure remember variables after the outer function returns?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 函数对象携带定义时环境的引用，调用时名称查找仍能到该环境。
<!-- bilingual-en:start -->
> [!answer]- Answer
> The function object retains a reference to its defining environment, so name lookup can still reach that environment when the function is called.
<!-- bilingual-en:end -->

### 环境模型中一次函数调用创建什么？
<!-- bilingual-en:start -->
*What does one function call create in the environment model?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 创建新 frame，把形参绑定到实参，并以函数定义环境为 parent。
<!-- bilingual-en:start -->
> [!answer]- Answer
> It creates a new frame, binds parameters to arguments, and uses the function's defining environment as the parent.
<!-- bilingual-en:end -->

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
- [Composing Programs](https://www.composingprograms.com/) 与本地 CS61A 教材：核验函数抽象、环境模型和递归。
  <!-- bilingual-en:start -->
[Composing Programs](https://www.composingprograms.com/) and the local CS61A text verify functional abstraction and the environment model; recursion is consolidated in [[递归与递归数据|Recursion and Recursive Data]].
  <!-- bilingual-en:end -->
- [The Python Tutorial: Defining Functions](https://docs.python.org/3/tutorial/controlflow.html#defining-functions)：复核默认参数、关键字参数、lambda 与文档字符串语义。
  <!-- bilingual-en:start -->
  [The Python Tutorial: Defining Functions](https://docs.python.org/3/tutorial/controlflow.html#defining-functions) verifies default arguments, keyword arguments, lambda expressions, and documentation strings.
  <!-- bilingual-en:end -->
