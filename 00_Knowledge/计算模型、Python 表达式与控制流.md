---
aliases:
  - "Python Fundamentals and Control Flow"
  - "Python Basics"
  - "Python基础"
status: source-checked
---

# 计算模型、Python 表达式与控制流
<!-- bilingual-en:start -->
*Computational Models, Python Expressions, and Control Flow*
<!-- bilingual-en:end -->

> [!summary] 快速恢复
> **它解决什么：** 把问题写成 Python 对象、表达式和明确控制流，理解程序是状态随语句执行而变化的过程。
> **具体锚点：** `input()` 返回字符串；若要数值比较必须显式转换，否则 `"10" < "2"` 按字符顺序为真。
> **核心难点：** 对象有类型和值，名称只是绑定；分支只执行一条路径，循环需保证状态推进和终止。
> **为什么重要：** 后续函数、数据结构、测试和算法都假设能准确模拟基本求值。
> **继续：** 用小例子手工 trace 表达式、分支和循环，再写程序。
> <!-- bilingual-en:start -->
> **Problem addressed:** Express a problem using Python objects, expressions, and explicit control flow, while understanding a program as state evolving as statements execute.
> **Concrete anchor:** `input()` returns a string; numeric comparison requires an explicit conversion, otherwise `"10" < "2"` is true under lexicographic ordering.
> **Central difficulty:** Objects have types and values, while names are bindings; a branch selects one path, and a loop must advance state toward termination.
> **Why it matters:** Functions, data structures, testing, and algorithms all assume that basic evaluation can be simulated accurately.
> **Continue with:** Trace small expressions, branches, and loops by hand before writing a complete program.
> <!-- bilingual-en:end -->

> [!source] 本节依据
> - MIT 6.100L 本地官方 slides、transcripts、finger exercises 与 problem sets：支持每个概念、示例和课程范围。
> - [[03_Computer_Science/03_MIT 6.100L/Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf]]：交叉核验 Python、算法、复杂度、OOP 与模拟。
> <!-- bilingual-en:start -->
> - Local official MIT 6.100L slides, transcripts, finger exercises, and problem sets support the concepts, examples, and course scope.
> - [[03_Computer_Science/03_MIT 6.100L/Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Introduction to Computation and Programming Using Python]] cross-checks Python, algorithms, complexity, object-oriented programming, and simulation.
> <!-- bilingual-en:end -->

## 计算与程序
<!-- bilingual-en:start -->
*Computation and Programs*
<!-- bilingual-en:end -->

程序把输入经算法映射为输出。声明式知识说结果性质，命令式知识给步骤。Python 解释器执行语句并维护环境；同一算法可用不同语言表达，语言不是算法本身。
<!-- bilingual-en:start -->
A program maps inputs to outputs through an algorithm. Declarative knowledge states properties of the result, whereas imperative knowledge supplies the procedure. The Python interpreter executes statements and maintains an environment; the same algorithm can be expressed in different languages, so a language is not the algorithm itself.
<!-- bilingual-en:end -->

执行程序时应同时追踪三件事：当前环境中的名称绑定、正在求值的表达式，以及下一条由控制流选中的语句。只盯着最终输出，会漏掉中间状态为何改变。
<!-- bilingual-en:start -->
When executing a program, track three things together: name bindings in the current environment, the expression being evaluated, and the next statement chosen by control flow. Looking only at the final output hides why intermediate state changed.
<!-- bilingual-en:end -->

## 对象、类型与表达式
<!-- bilingual-en:start -->
*Objects, Types, and Expressions*
<!-- bilingual-en:end -->

数字、布尔、字符串等都是对象；操作符按类型定义。`type` 决定允许操作，转换创建相应值。`==` 比较值，`is` 比较对象身份，后者通常只用于 `None` 等单例。
<!-- bilingual-en:start -->
Numbers, booleans, and strings are objects; operators are defined by type. A type determines which operations are valid, and conversion creates a value of the requested type. `==` compares values, while `is` compares object identity and is normally reserved for singletons such as `None`.
<!-- bilingual-en:end -->

表达式先按优先级和结合性建立求值结构，再对子表达式求值。不要凭视觉猜复杂表达式；加括号或拆成带名称的中间结果，既能澄清意图，也便于检查类型。
<!-- bilingual-en:start -->
An expression first acquires an evaluation structure from precedence and associativity, then its subexpressions are evaluated. Do not guess a dense expression by sight; parentheses or named intermediate results clarify intent and make types inspectable.
<!-- bilingual-en:end -->

## 名称与赋值
<!-- bilingual-en:start -->
*Names and Assignment*
<!-- bilingual-en:end -->

赋值让名称指向对象，不是把值永久装进盒子。右侧先求值再绑定左侧；多重赋值可安全交换。不可变对象“变化”实际是名称改绑到新对象。
<!-- bilingual-en:start -->
Assignment makes a name refer to an object; it does not permanently place a value inside a box. The right-hand side is evaluated before the left-hand name is bound, so multiple assignment can swap values safely. When an immutable object appears to “change,” the name has actually been rebound to a new object.
<!-- bilingual-en:end -->

## 字符串与输入输出
<!-- bilingual-en:start -->
*Strings and Input/Output*
<!-- bilingual-en:end -->

字符串是不可变序列，索引从 0，切片半开区间。`input` 返回字符串，格式化输出应显式控制类型和精度。用户输入是信任边界，要验证并处理转换失败。
<!-- bilingual-en:start -->
A string is an immutable sequence indexed from zero, and slices use half-open intervals. `input` returns a string, while formatted output should control type and precision explicitly. User input is a trust boundary: validate it and handle conversion failures.
<!-- bilingual-en:end -->

## 分支
<!-- bilingual-en:start -->
*Branching*
<!-- bilingual-en:end -->

`if/elif/else` 按顺序选择第一条真分支。布尔表达式短路，可用于安全 guard；过度依赖 truthiness 会模糊 0、空容器和 `None` 的差别。
<!-- bilingual-en:start -->
`if/elif/else` selects the first true branch in order. Boolean expressions short-circuit and can implement safe guards; excessive reliance on truthiness can blur the important distinction among zero, an empty container, and `None`.
<!-- bilingual-en:end -->

分支应覆盖业务上可能出现的状态，而不只是让代码“总能走到某条路”。当 `else` 同时代表多种异常情况时，先显式验证输入，再让每个分支表达一个清楚条件。
<!-- bilingual-en:start -->
Branches should cover states that can occur in the problem, not merely ensure that execution always takes some path. If one `else` represents several exceptional conditions, validate the input first and let each branch express one clear condition.
<!-- bilingual-en:end -->

## 迭代
<!-- bilingual-en:start -->
*Iteration*
<!-- bilingual-en:end -->

`while` 适合条件驱动，`for` 遍历 iterable。循环要有不变量和进度量，注意 off-by-one、边界和修改正在遍历的容器。`range` 也是半开区间。
<!-- bilingual-en:start -->
`while` suits condition-driven repetition, whereas `for` traverses an iterable. A loop needs an invariant and a progress measure; watch for off-by-one errors, boundary mistakes, and mutation of the container being traversed. `range` also uses a half-open interval.
<!-- bilingual-en:end -->

循环正确性可以分三步检查：进入循环前不变量成立；执行一轮后仍成立；退出条件与不变量共同推出目标。终止性还要求某个有界度量每轮严格推进。
<!-- bilingual-en:start -->
Check loop correctness in three steps: the invariant holds before entry, remains true after one iteration, and together with the exit condition implies the goal. Termination additionally requires a bounded measure that advances strictly on every iteration.
<!-- bilingual-en:end -->

## Worked example：读取并分类一个分数
<!-- bilingual-en:start -->
*Worked Example: Read and Classify a Score*
<!-- bilingual-en:end -->

目标是读取 `0` 到 `100` 的整数并输出等级。关键不是把 `if` 写出来，而是先把字符串转换失败、越界和正常分支分开。
<!-- bilingual-en:start -->
The goal is to read an integer from `0` to `100` and print a grade. The important step is not merely writing an `if`, but separating conversion failure, out-of-range input, and valid branches.
<!-- bilingual-en:end -->

```python
raw = input("Score: ")

try:
    score = int(raw)
except ValueError:
    print("Please enter an integer.")
else:
    if not 0 <= score <= 100:
        print("Score must be between 0 and 100.")
    elif score >= 70:
        print("Distinction")
    elif score >= 40:
        print("Pass")
    else:
        print("Fail")
```

手工 trace `raw="70"` 时，转换得到整数 `70`，越界 guard 为假，第一条等级条件为真，后续分支不再求值。对 `raw="seven"`，控制流在转换处转入异常处理，等级分支不会运行。
<!-- bilingual-en:start -->
Tracing `raw="70"` yields the integer `70`; the range guard is false, the first grading condition is true, and later branches are not evaluated. With `raw="seven"`, control transfers to exception handling during conversion, so no grading branch runs.
<!-- bilingual-en:end -->

## 分解问题
<!-- bilingual-en:start -->
*Problem Decomposition*
<!-- bilingual-en:end -->

先写示例和输入输出，再把大步骤拆成可测试小块。能用现有内置/标准库表达时不手写复杂循环。
<!-- bilingual-en:start -->
Write examples and specify inputs and outputs first, then divide the work into testable steps. Use built-ins or the standard library when they already express the operation instead of hand-writing a complicated loop.
<!-- bilingual-en:end -->

## 失败诊断
<!-- bilingual-en:start -->
*Failure Diagnosis*
<!-- bilingual-en:end -->

- 输出类型不对：在第一次产生该值的位置检查 `type`，不要等到最后一行才猜。
  <!-- bilingual-en:start -->
  Wrong output type: inspect `type` where the value is first created instead of guessing at the final line.
  <!-- bilingual-en:end -->
- 分支没有进入：写出每个条件的实际布尔值，并检查是否被更早的宽泛条件截获。
  <!-- bilingual-en:start -->
  A branch is never entered: record the actual boolean value of each condition and check whether an earlier, broader condition captured the case.
  <!-- bilingual-en:end -->
- 循环不结束：找出预期单调变化的进度量，检查所有路径是否更新它。
  <!-- bilingual-en:start -->
  A loop does not terminate: identify the intended monotone progress measure and verify that every path updates it.
  <!-- bilingual-en:end -->
- 边界少一个：把空输入、一个元素、第一项、最后一项以及恰好位于阈值的值列成表逐一 trace。
  <!-- bilingual-en:start -->
  A boundary is off by one: tabulate and trace empty input, one element, the first and last positions, and values exactly at each threshold.
  <!-- bilingual-en:end -->

## 最小自检
<!-- bilingual-en:start -->
*Minimum Self-Check*
<!-- bilingual-en:end -->

### Python 赋值语句改变的是对象还是名称绑定？
<!-- bilingual-en:start -->
*Does a Python assignment change an object or a name binding?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 通常先求右侧对象，再让名称绑定到它；不可变对象本身不被修改。
<!-- bilingual-en:start -->
> [!answer]- Answer
> Normally the right-hand object is evaluated first and the name is then bound to it; an immutable object itself is not modified.
<!-- bilingual-en:end -->

### `is` 和 `==` 何时不同？
<!-- bilingual-en:start -->
*When do `is` and `==` differ?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> `==` 比较值是否相等，`is` 比较是否同一对象；普通数值/字符串应使用 `==`。
<!-- bilingual-en:start -->
> [!answer]- Answer
> `==` asks whether values are equal, while `is` asks whether two references designate the same object; ordinary numbers and strings should be compared with `==`.
<!-- bilingual-en:end -->

### 一个 `while` 循环如何证明会终止？
<!-- bilingual-en:start -->
*How can termination of a `while` loop be justified?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 找一个每轮严格朝边界推进、且有下界/上界的度量，并确保所有分支都更新它。
<!-- bilingual-en:start -->
> [!answer]- Answer
> Identify a measure with a lower or upper bound that moves strictly toward that bound on every iteration, and verify that every branch updates it.
<!-- bilingual-en:end -->

### 为什么 `input()` 后立即做边界验证比在每个分支里补检查更可靠？
<!-- bilingual-en:start -->
*Why is validating immediately after `input()` more reliable than adding checks separately in every branch?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 它在共同信任边界一次排除非法状态，使后续每条路径都能依赖同一前提，避免某个分支漏检。
<!-- bilingual-en:start -->
> [!answer]- Answer
> It rejects invalid states once at the shared trust boundary, allowing every later path to rely on the same precondition and preventing a branch from omitting validation.
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
- [The Python Tutorial](https://docs.python.org/3/tutorial/): 核验表达式、控制流、字符串、输入输出与异常处理的语言语义。
  <!-- bilingual-en:start -->
  [The Python Tutorial](https://docs.python.org/3/tutorial/) verifies the language semantics of expressions, control flow, strings, input/output, and exception handling.
  <!-- bilingual-en:end -->
