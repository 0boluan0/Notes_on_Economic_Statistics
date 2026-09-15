---
aliases:
  - MIT 6.100L Lecture 12
  - 6.100L L12
  - List Comprehension, Functions as Objects, Testing, and Debugging
tags:
  - computer-science
  - python
  - mit-6.100l
  - lecture-note
科目: Computer Science
course: MIT 6.100L Introduction to CS and Programming Using Python
lecture: 12
---

# Lecture 12: List Comprehension, Functions as Objects, Testing, and Debugging

> [!tip] Hint
> - 这节课前半段先收尾 lists 和 functions，后半段才转入 testing/debugging，所以正文顺序不能写反。
> - list comprehension 的动机是：某类“建新列表”的 for-loop 模式太常见，Python 给了更紧凑写法。
> - comprehension 里既有“对每个元素做什么”，也可以有“满足什么条件才保留”。
> - keyword argument 和 default parameter 是从 `bisection_root_new` 这种具体函数一点点引出来的。
> - `epsilon=0.01` 这类默认值不是语法糖而已，它在定义“调用者省略参数时的默认行为”。
> - `make_prod(a)` 返回函数，是为了再推一步“函数对象可以被制造和返回”。
> - 课堂后半段把 testing 和 debugging 分开讲：一个是设计检查方法，一个是定位并修 bug。
> - black-box testing 看 specification，glass-box testing 看实现路径；两者都要，但关心点不同。
> - buggy palindrome 的调试过程是本讲最值得模仿的部分：先复现，再加打印，再缩小问题，再修一个 bug，再发现第二个 bug。
> - 听完这节课，你应该能复述一个系统的 debugging recipe，而不是“出错了就随便改两行试试”。
> <!-- bilingual-en:start -->
> - The first half of the lecture finishes the treatment of lists and functions; only the second half turns to testing and debugging, so that order matters.
> - List comprehensions arise because the same kind of loop for constructing a new list appears so often that Python provides a compact notation for it.
> - A comprehension specifies both what to produce from each element and, optionally, which elements to retain.
> - Keyword arguments and default parameters are introduced incrementally through a concrete function such as `bisection_root_new`.
> - A default such as `epsilon=0.01` does more than shorten syntax: it defines what the function does when the caller omits that argument.
> - `make_prod(a)` returns a function, extending the idea that function objects can themselves be created and returned.
> - The second half distinguishes testing, which designs checks, from debugging, which locates and repairs faults.
> - Black-box tests follow the specification; glass-box tests inspect execution paths. Both are necessary, but they answer different questions.
> - The debugging of the faulty palindrome function is the model worth imitating: reproduce the failure, add diagnostic output, narrow the cause, repair one bug, and then uncover the next.
> - By the end, you should be able to give a systematic debugging recipe instead of changing random lines whenever something fails.
> <!-- bilingual-en:end -->

> [!links] 原子化入口
> 本讲的函数部分先用 [[默认参数求值]] 固定定义时点，再由 [[一等函数对象]]、[[词法作用域]] 和 [[闭包环境保留]] 解释 `make_prod` 为什么能返回仍可访问 `a` 的函数。后半段继续复用 [[03_Computer_Science/03_MIT 6.100L/12_测试与失败处理|测试与失败处理连续路径]]。
>
> *Atomic path: [[默认参数求值|default evaluation time]] → [[一等函数对象|first-class functions]] → [[词法作用域|lexical scope]] → [[闭包环境保留|closure environment retention]], followed by [[03_Computer_Science/03_MIT 6.100L/12_测试与失败处理|the testing and failure-handling reading path]].*
>
> [[测试、调试、异常与断言.canvas|测试、调试、异常与断言 · 关系图]]

## Lecture flow

### 1. 开场先说明：这讲是在收尾 lists 和 functions
<!-- bilingual-en:start -->
*1. Opening Context: Completing the Treatment of Lists and Functions*
<!-- bilingual-en:end -->
老师一开始就说这会是一节相对 “chill” 的课。  
但这不代表内容散，而是因为它在做两件事：

- 把前面 lists / functions 的一些常见模式整理起来
- 再把课堂视角抬到 testing 和 debugging

所以 Lecture 12 的推进顺序很重要：

1. list comprehension
2. keyword / default parameters
3. function returning function
4. testing strategies
5. debugging method
<!-- bilingual-en:start -->
The instructor calls this a relatively “chill” lecture, not because it lacks structure, but because it consolidates familiar list and function patterns before widening the view to testing and debugging. Its sequence is important: list comprehensions; keyword and default parameters; functions that return functions; testing strategies; and finally a debugging method.
<!-- bilingual-en:end -->

### 2. list comprehension：把“建新列表”的套路压成一行
<!-- bilingual-en:start -->
*2. List Comprehensions: Compressing a Common List-Building Pattern into One Line*
<!-- bilingual-en:end -->
老师先回顾一个大家已经很熟的模式：
<!-- bilingual-en:start -->
The instructor first revisits a familiar pattern:
<!-- bilingual-en:end -->

```python
new_list = []
for e in old_list:
    new_list.append(expr(e))
return new_list
```

这类代码实在太常见了，所以 Python 提供了更紧凑的写法：
<!-- bilingual-en:start -->
Because this pattern occurs so often, Python provides a more compact form:
<!-- bilingual-en:end -->

```python
[expr(e) for e in old_list]
```

如果还要先筛选，再加工，则是：
<!-- bilingual-en:start -->
When elements must first be filtered and then transformed, the form becomes:
<!-- bilingual-en:end -->

```python
[expr(e) for e in old_list if test(e)]
```

课堂这里不是追求“写短”，而是让你看出 comprehension 对应的原始结构是什么。
<!-- bilingual-en:start -->
The aim is not brevity for its own sake, but recognizing the expanded loop represented by the comprehension.
<!-- bilingual-en:end -->

> [!note]
> comprehension 永远都可以还原成“循环 + 条件 + append”的展开写法。
> <!-- bilingual-en:start -->
> Any comprehension can be expanded into a loop, an optional condition, and an `append` operation.
> <!-- bilingual-en:end -->

### 3. comprehension 的两个位置要分清
<!-- bilingual-en:start -->
*3. Keeping the Roles Within a Comprehension Distinct*
<!-- bilingual-en:end -->
老师随后反复用例子拆 comprehension 的结构。
<!-- bilingual-en:start -->
The instructor then repeatedly decomposes comprehensions into their parts.
<!-- bilingual-en:end -->

以：

```python
[e**2 for e in range(8) if e % 2 == 0]
```

为例：

- `e**2`：新列表中每个元素长什么样
- `for e in range(8)`：从哪里依次取元素
- `if e % 2 == 0`：哪些元素能留下
<!-- bilingual-en:start -->
- `e**2` determines the value placed in the new list.
- `for e in range(8)` determines where the input elements come from.
- `if e % 2 == 0` determines which input elements are retained.
<!-- bilingual-en:end -->

如果这里顺序看错，就会写出语法对但语义乱的表达式。  
这也是为什么老师花了不少时间让大家把展开版和压缩版互相翻译。
<!-- bilingual-en:start -->
Confusing these roles can produce an expression that is syntactically valid but semantically wrong. That is why the lecture spends time translating in both directions between expanded loops and compact comprehensions.
<!-- bilingual-en:end -->

### 4. list comprehension 不是新能力，而是旧能力的浓缩
<!-- bilingual-en:start -->
*4. A List Comprehension Condenses Existing Capability Rather Than Adding New Capability*
<!-- bilingual-en:end -->
老师还专门写了一个等价函数：
<!-- bilingual-en:start -->
The instructor makes the equivalence explicit with a function:
<!-- bilingual-en:end -->

```python
def f(expr, old_list, test=lambda x: True):
    new_list = []
    for e in old_list:
        if test(e):
            new_list.append(expr(e))
    return new_list
```

这一步很有教学意义，因为它让你看到：

- comprehension 没有创造新计算能力
- 它只是把旧模式浓缩成更紧的语法
<!-- bilingual-en:start -->
- A comprehension does not add a new computational ability.
- It packages an existing pattern in tighter syntax.
<!-- bilingual-en:end -->

所以如果你看 comprehension 看不懂，最好的办法不是死盯一行，而是把它先展开回普通循环。
<!-- bilingual-en:start -->
If a comprehension is difficult to read, expand it into an ordinary loop instead of staring at the compact expression.
<!-- bilingual-en:end -->

### 5. 接着讲参数接口：keyword arguments 和 default parameters
<!-- bilingual-en:start -->
*5. Moving to Function Interfaces: Keyword Arguments and Default Parameters*
<!-- bilingual-en:end -->
把 list comprehension 讲完后，课堂切回函数接口设计。

老师拿前面已经学过的平方根函数继续改造：

- 先是 `bisection_root(x)`
- 再变成 `bisection_root_new(x, epsilon)`
- 最后变成 `bisection_root_new(x, epsilon=0.01)`
<!-- bilingual-en:start -->
After list comprehensions, the lecture returns to interface design by progressively revising the previously studied square-root function: from `bisection_root(x)`, to `bisection_root_new(x, epsilon)`, and finally to `bisection_root_new(x, epsilon=0.01)`.
<!-- bilingual-en:end -->

这组推进顺序很重要，因为它说明 default parameter 不是凭空出现的：

- 先发现“某个参数经常取同一个值”
- 再考虑把它设为默认值
<!-- bilingual-en:start -->
This progression shows where a default parameter comes from: first notice that an argument repeatedly takes the same value, and only then consider making that value the default.
<!-- bilingual-en:end -->

### 6. default parameter 的意义：调用更轻，但语义要更清楚
<!-- bilingual-en:start -->
*6. Default Parameters: Lighter Calls, Clearer Semantics*
<!-- bilingual-en:end -->
课堂里老师反复围绕 `epsilon` 讨论：

- 如果调用者不传 `epsilon`，默认该是多少
- 为什么 `0.01` 可以是一个合理默认值
- 如果想覆盖默认行为，该怎么传参
<!-- bilingual-en:start -->
The discussion of `epsilon` asks what value should be used when it is omitted, why `0.01` might be a reasonable default, and how a caller can override that choice.
<!-- bilingual-en:end -->

所以默认参数的真正意义是：

- 给出一种常见、可接受的默认行为
- 让简单调用更短
- 但仍保留调用者显式指定的能力
<!-- bilingual-en:start -->
A good default supplies a common, acceptable behavior, shortens the simplest call, and still permits the caller to specify a different value explicitly.
<!-- bilingual-en:end -->

例如：

```python
bisection_root_new(123)
bisection_root_new(123, 0.5)
bisection_root_new(123, epsilon=0.00001)
bisection_root_new(epsilon=0.001, x=123)
```

这里也自然引出了 keyword arguments：

- 用参数名写调用
- 可读性更好
- 顺序也更灵活
<!-- bilingual-en:start -->
The examples naturally introduce keyword arguments: naming parameters at the call site can improve readability and makes argument order more flexible.
<!-- bilingual-en:end -->

### 7. 一个函数还能返回另一个函数：`make_prod`
<!-- bilingual-en:start -->
*7. A Function Can Return Another Function: `make_prod`*
<!-- bilingual-en:end -->
讲完 default parameter，老师又推了一步 functions-as-objects。

例子是：
<!-- bilingual-en:start -->
After default parameters, the instructor develops the functions-as-objects idea with this example:
<!-- bilingual-en:end -->

```python
def make_prod(a):
    def g(b):
        return a * b
    return g
```

这段代码的重要性在于：

- 函数不仅能作为参数传入
- 还可以作为返回值送出去
<!-- bilingual-en:start -->
The code matters because a function can not only be passed as an argument, but also returned as a result.
<!-- bilingual-en:end -->

于是：

```python
doubler = make_prod(2)
val = doubler(3)
```

就读成：

- 先生成一个“乘以 2”的函数
- 再在别处调用它
<!-- bilingual-en:start -->
The call first creates a function that multiplies by two and then invokes that generated function elsewhere.
<!-- bilingual-en:end -->

这一部分其实在为更高级的抽象方式做铺垫，但课堂里不会把术语讲得太重，重点还是让你接受“函数对象可以被构造和返回”。
<!-- bilingual-en:start -->
This prepares the ground for more advanced abstraction without overloading the lecture with terminology. The immediate goal is simply to accept that function objects can be constructed and returned.
<!-- bilingual-en:end -->

### 8. 课堂后半段突然抬高视角：不再只问“能不能跑”，而是“怎么验证”
<!-- bilingual-en:start -->
*8. Raising the Level of Inquiry: From “Does It Run?” to “How Do We Verify It?”*
<!-- bilingual-en:end -->
到中后段，老师明确转入 testing and debugging。

这时课程气质明显变了：  
前半段在教写法，后半段在教工程习惯。
<!-- bilingual-en:start -->
In the latter half, the lecture turns explicitly to testing and debugging. The first half teaches language patterns; the second teaches engineering habits.
<!-- bilingual-en:end -->

老师先把几个词分开：

- validation / [[软件测试|testing]]：程序是否按预期工作
- [[调试|debugging]]：程序不对时，如何系统地定位原因
<!-- bilingual-en:start -->
- Validation and [[软件测试|testing]] ask whether a program behaves as intended.
- [[调试|Debugging]] asks how to locate the cause systematically when it does not.
<!-- bilingual-en:end -->

这两个动作往往在现实里交织出现，但课堂故意拆开讲，是为了避免你把“乱试输入”和“修 bug”混为一谈。
<!-- bilingual-en:start -->
Although these activities intertwine in practice, the lecture separates them so that ad hoc input trials are not confused with the disciplined work of finding and repairing a bug.
<!-- bilingual-en:end -->

测试前先从[[函数契约]]写下输入、预期返回值及需要观察的副作用或异常，再运行并比较实际行为。课堂 slides pp. 33–35 强调先有规格和预期结果；“没有报错”本身不是结果正确的判据。
<!-- bilingual-en:start -->
Before execution, use the [[函数契约|function contract]] to record the input, expected return value, and any observable effects or exceptions; then compare the actual behavior. Slides pp. 33–35 require a specification and expected results. Merely completing without an error is not a correctness criterion.
<!-- bilingual-en:end -->

这里区分[[测试用例]]和[[测试判据]]：一次具体检查需要输入与执行条件，而判断结果是否可接受还需要独立于实际输出的预期规则。
<!-- bilingual-en:start -->
Distinguish a [[测试用例|test case]] from its [[测试判据|oracle]]: a concrete check needs inputs and execution conditions, while judging acceptability requires an expectation independent of the observed output.
<!-- bilingual-en:end -->

### 9. 测试策略：unit / regression / integration
<!-- bilingual-en:start -->
*9. Testing Strategies: Unit, Regression, and Integration Tests*
<!-- bilingual-en:end -->
老师先给出几种更工程化的 testing 视角：

- [[单元测试|unit testing]]：单个函数或模块分别测
- [[回归测试|regression testing]]：修改以后，确认原来通过的东西没被改坏
- [[集成测试|integration testing]]：多个部件组合后一起测
<!-- bilingual-en:start -->
- [[单元测试|Unit testing]] checks a function or module in isolation.
- [[回归测试|Regression testing]] checks that a change has not broken behavior that previously worked.
- [[集成测试|Integration testing]] checks several components after they have been combined.
<!-- bilingual-en:end -->

这部分的课堂重点是：  
测试不是“随便挑几个输入跑一下”，而是要知道自己在检查哪一层行为、出于什么目的。单元与集成描述组合范围；回归描述修改后重新检查已有行为的目的，可以在不同范围执行，并不是夹在两者之间的第三个层级。
<!-- bilingual-en:start -->
Testing is not a matter of running a few arbitrary inputs: identify both the scope and the purpose. Unit and integration testing describe the scope of composition; regression testing describes checking existing behavior after a change and can occur at either scope. It is not a third level between them.
<!-- bilingual-en:end -->

### 10. black-box testing：按 specification 设计输入
<!-- bilingual-en:start -->
*10. Black-Box Testing: Designing Inputs from the Specification*
<!-- bilingual-en:end -->
老师随后讲 **[[黑盒测试|black-box testing]]**。

黑箱测试的核心是：

- 把函数当成黑箱
- 只看 specification
- 不依赖内部实现细节来设计测试
<!-- bilingual-en:start -->
Black-box testing treats the function as opaque, derives tests from its specification, and does not rely on internal implementation details.
<!-- bilingual-en:end -->

所以你会优先想：

- 正常输入
- 边界输入
- 特殊输入
- 是否有空输入、重复值、极端值
<!-- bilingual-en:start -->
This perspective prompts tests of ordinary cases, boundaries, special cases, empty inputs, duplicate values, and extremes.
<!-- bilingual-en:end -->

如果函数实现被完全重写，但 specification 不变，那么黑箱测试仍然应该有效。
<!-- bilingual-en:start -->
If the implementation is rewritten while the specification remains unchanged, the same black-box tests should still be valid.
<!-- bilingual-en:end -->

课堂的具体规格是 `sqrt(x, eps)`：假设 `x, eps` 是浮点数、`x >= 0`、`eps > 0`，返回 `res` 满足 `x-eps <= res*res <= x+eps`。按[[分区与边界选例]]先选 `x=0`、完全平方数、`0<x<1`、无理平方根，再组合很大/很小的 `x` 和 `eps`（slides pp. 38–39）。这里应检验平方后的误差，不要擅自把规格换成 `abs(res-sqrt(x)) <= eps`。契约没有规定非法输入的结果时，不能把自己猜测的异常行为当作合格标准；若接口承诺拒绝非法输入，再测试该失败路径。
<!-- bilingual-en:start -->
The classroom specification for `sqrt(x, eps)` assumes floating-point arguments with `x >= 0` and `eps > 0`, and promises `x-eps <= res*res <= x+eps`. Using [[分区与边界选例|partition and boundary cases]], slides pp. 38–39 test zero, perfect squares, values between zero and one, irrational square roots, and extreme combinations of `x` and `eps`. Check the squared-result error stated by the contract, not a substituted bound on `abs(res-sqrt(x))`. An unspecified response to invalid input is not a test requirement; test rejection when the interface actually promises it.
<!-- bilingual-en:end -->

> [!note] 随机测试的结论边界
> Slides p. 37 用“更多测试提高程序正确的概率”作直觉说明。严格地说，在给定采样方式与判据下，更多有信息的测试可能提高发现某类错误的机会；不能不加模型地据此计算“程序正确概率”，也不能由有限抽样推出所有输入都正确。极端浮点案例还可能暴露规格过强，而不只是实现写错；本地 Guttag §6.1.1（印刷 p. 72）明确保留这种可能。
> <!-- bilingual-en:start -->
> Slide p. 37 informally says that more tests increase the probability of correctness. More informative tests can improve the chance of revealing a class of faults under a specified sampling scheme and criterion, but do not establish a model-free probability of correctness or prove correctness for every input. Extreme floating-point cases can also expose an unrealistic specification, as Guttag §6.1.1, printed p. 72, explicitly notes.
> <!-- bilingual-en:end -->

### 11. glass-box testing：看实现路径是否覆盖到
<!-- bilingual-en:start -->
*11. Glass-Box Testing: Checking Coverage of Implementation Paths*
<!-- bilingual-en:end -->
接着老师再讲 **[[白盒测试|glass-box testing]]**。

玻璃盒测试的视角恰恰相反：

- 你知道函数内部结构
- 于是你关心哪些分支、循环路径、特殊路径有没有真的走到
<!-- bilingual-en:start -->
Glass-box testing takes the opposite perspective: knowing the internal structure lets you ask whether particular branches, loop paths, and special cases are actually exercised.
<!-- bilingual-en:end -->

比如某个函数对正数和负数走不同分支，那么 glass-box testing 会逼你主动构造两类输入。
<!-- bilingual-en:start -->
If positive and negative inputs follow different branches, for example, glass-box testing requires examples from both classes.
<!-- bilingual-en:end -->

所以这两种测试不是互相替代，而是互补：

- black-box 逼你尊重 specification
- glass-box 逼你覆盖实现细节
<!-- bilingual-en:start -->
The two strategies complement rather than replace each other: black-box tests enforce the specification, while glass-box tests cover the implementation's structure.
<!-- bilingual-en:end -->

[[代码覆盖率]]描述指定结构被执行的情况，但[[覆盖率不能证明正确|路径覆盖仍不保证正确]]。课堂 slides pp. 40–41 的反例是：
<!-- bilingual-en:start -->
[[代码覆盖率|Code coverage]] describes execution of specified structures, but [[覆盖率不能证明正确|path coverage still does not guarantee correctness]]. Slides pp. 40–41 give this counterexample:
<!-- bilingual-en:end -->

```python
def abs(x):
    """Assumes x is an int; returns its absolute value."""
    if x < -1:
        return -x
    else:
        return x
```

测试 `2` 与 `-2` 已执行两个分支，却都通过；边界 `-1` 才揭示错误返回 `-1`。一般程序还可能因循环次数或递归深度没有上界而无法穷尽所有路径。实用选例包括循环执行 0 次、1 次、多次，各分支与不同退出原因；走过代码以后，仍须核对预期结果。
<!-- bilingual-en:start -->
The inputs `2` and `-2` exercise both branches and pass, but the boundary input `-1` reveals the incorrect result `-1`. Unbounded loop counts or recursion depths may make exhaustive path coverage impossible. Useful cases exercise loops zero times, once, and repeatedly, as well as different branches and exit causes; executing a path still requires checking its expected result.
<!-- bilingual-en:end -->

### 12. debugging recipe：不要乱改，先收集证据
<!-- bilingual-en:start -->
*12. A Debugging Recipe: Gather Evidence Before Changing Code*
<!-- bilingual-en:end -->
讲完 testing 之后，老师转向 debugging，并给出非常明确的方法论。
<!-- bilingual-en:start -->
After testing, the instructor presents a concrete debugging method.
<!-- bilingual-en:end -->

核心思想是：

1. 先复现 bug
2. [[最小失败复现|选最小测试案例]]
3. 在关键位置加打印
4. 观察期望和实际在哪一步开始分叉
5. 一次只修一个问题
6. 修完再重新测试
<!-- bilingual-en:start -->

&nbsp;
**1.** Reproduce the bug.<br>
**2.** [[最小失败复现|Choose the smallest useful failing test case]].<br>
**3.** Add diagnostic output at key points.<br>
**4.** Find the first step at which expected and actual behavior diverge.<br>
**5.** Repair one problem at a time.<br>
**6.** Test again after each repair.<br>
<!-- bilingual-en:end -->

这套 recipe 是整节课最值得模仿的部分，因为它把 debugging 从“靠感觉”变成了一个过程。
<!-- bilingual-en:start -->
This recipe is the lecture's most transferable lesson because it turns debugging from intuition-driven tinkering into an evidence-based process.
<!-- bilingual-en:end -->

[[调试假设检验]]要求每次实验前先说明假设和能推翻它的观察。例如在一个有明确预期状态的中间检查点打印值，判断本次失败是否已经出现，再缩小候选区段（slides pp. 45–46）。这不是只按代码行数机械对半，也不保证“一处值看起来正常，就证明此前没有任何 bug”；检查点必须足以区分当前假设，多个错误也可能同时存在。
<!-- bilingual-en:start -->
[[调试假设检验|Hypothesis-driven debugging]] states the hypothesis and an observation that would refute it before each experiment. At an intermediate checkpoint with a known expected state, inspect whether the observed failure is already present and narrow the candidate region (slides pp. 45–46). This is not mechanical halving by line count: one apparently normal value does not prove that all preceding code is bug-free. A checkpoint must distinguish the current hypotheses, and multiple faults may coexist.
<!-- bilingual-en:end -->

### 13. buggy palindrome：课堂现场示范如何 debug
<!-- bilingual-en:start -->
*13. The Buggy Palindrome Function: A Live Debugging Demonstration*
<!-- bilingual-en:end -->
老师把这套 recipe 用在了一个有 bug 的 palindrome 函数上。

最初版本大致是：
<!-- bilingual-en:start -->
The instructor applies the recipe to a faulty palindrome function whose initial version is approximately:
<!-- bilingual-en:end -->

```python
def is_pal(x):
    temp = x
    temp.reverse
    if temp == x:
        return True
    else:
        return False
```

然后按步骤调：

1. 先跑测试用例
2. 发现结果不对
3. 在中间打印 `temp` 和 `x`
4. 发现 `reverse` 甚至没真正调用，因为少了括号
5. 修成 `temp.reverse()`
6. 又发现 `temp = x` 带来 aliasing，反转的是同一个列表
7. 最后改成 `temp = x[:]`
<!-- bilingual-en:start -->
The investigation proceeds step by step: run a test; observe the wrong result; print `temp` and `x`; discover that `reverse` was referenced but never called because parentheses were missing; change it to `temp.reverse()`; discover that `temp = x` creates an alias and therefore reverses the original list too; and finally clone the list with `temp = x[:]`.
<!-- bilingual-en:end -->

这个示范非常好，因为它连续暴露了两种不同 bug：

- 调用层面错误：`temp.reverse` 是合法的属性访问，只取得方法对象，没有调用它；不是 `SyntaxError`
- aliasing / mutability 层面错误：`temp = x` 形成[[对象别名]]，`temp = x[:]` 才按[[复制层级边界]]建立独立的外层列表
<!-- bilingual-en:start -->
The demonstration exposes two faults: `temp.reverse` is valid attribute access that retrieves a method without calling it, not a `SyntaxError`; then `temp = x` creates [[对象别名|an alias]], whereas `temp = x[:]` creates a separate outer list according to the [[复制层级边界|copying boundary]].
<!-- bilingual-en:end -->

原代码用 `list('ab')` 作最小非回文失败例，修复后还重跑原先通过的 `list('abcba')`；这一步就是[[回归测试]]。修复后的核心可以保留为：
<!-- bilingual-en:start -->
The source uses `list('ab')` as a minimal failing non-palindrome and reruns the previously passing `list('abcba')` after the fixes: this is [[回归测试|regression testing]]. The repaired core is:
<!-- bilingual-en:end -->

```python
def is_pal(x):
    temp = x[:]
    temp.reverse()
    return temp == x
```

> [!example]
> 真正好的 debugging 不是一步到位，而是先修掉第一个确认的问题，再看是否还有第二层问题。
> <!-- bilingual-en:start -->
> Good debugging need not solve everything in one leap. Repair the first confirmed fault, then test whether another layer of failure remains.
> <!-- bilingual-en:end -->

### 14. Wordle 文件：让你把 debugging 方法迁移出去
<!-- bilingual-en:start -->
*14. The Wordle File: Transferring the Debugging Method to New Code*
<!-- bilingual-en:end -->
课程最后提到 `lec12_wordle.py` 是有 bug 的，鼓励你自己去修。

这不是附带作业，而是在告诉你：

- 课堂示范的 debugging 方法不是只对 palindrome 有效
- 只要代码稍微变大，系统方法就更重要
<!-- bilingual-en:start -->
The lecture ends by inviting students to repair the bugs in `lec12_wordle.py`. This is not a throwaway exercise: the method demonstrated on the palindrome function should transfer to other programs, and disciplined debugging becomes more—not less—important as code grows.
<!-- bilingual-en:end -->

Lecture 12 到这里实际上完成了一个关键转折：  
课程开始正式要求你把“写代码”升级成“写、测、修代码”。
<!-- bilingual-en:start -->
Lecture 12 thus marks a transition from merely writing code to writing, testing, and repairing it.
<!-- bilingual-en:end -->

> [!warning] Wordle 原文件的核验边界
> 原文件同时保留 buggy 与 `FIXES TO THE BUGGY CODE` 两段，后者也不能当作完整正确答案：`get_word_list(words_str)` 仍读取全局 `words`，且未按其 docstring 转小写；`struck` 对 `strike` 按所述逐位置规则应得 `'STR  k'`，不是注释中的 `'ST   k'`；修订版重复字母处理仍与该文字规则不一致，输入合法性检查也仍被注释。保留文件作为调试练习，判断行为时先写清当前函数的合同，不把“FIXES”标签当作验证结果。
> <!-- bilingual-en:start -->
> The file preserves both a buggy section and a `FIXES TO THE BUGGY CODE` section, but the latter is not a fully verified solution. `get_word_list(words_str)` still reads global `words` and does not lowercase words as its docstring promises. The stated position-by-position rule gives `'STR  k'` for `struck` against `strike`, not the documented `'ST   k'`. Duplicate-letter behavior still conflicts with that rule, and input validation remains commented out. Retain the file as debugging practice and judge it against an explicit contract, not its “FIXES” label.
> <!-- bilingual-en:end -->

## Exercise log

> [!example] Finger exercise 12
> 官方练习是 `count_sqrts(nums_list)`：
> - `nums_list` 只含正数且无重复
> - 统计列表中有多少元素，其平方也在同一个列表里
> <!-- bilingual-en:start -->
> The official exercise is `count_sqrts(nums_list)`:
> - `nums_list` contains distinct positive numbers only.
> - Count how many elements have their square in the same list.
> <!-- bilingual-en:end -->

官方解法非常短：
<!-- bilingual-en:start -->
The official solution is very short:
<!-- bilingual-en:end -->

```python
cnt = 0
for i in nums_list:
    if i * i in nums_list:
        cnt += 1
return cnt
```

官方题干从“列表中哪些值是另一个元素的平方”描述目标，而代码从根 `i` 检查 `i*i`。在正数、无重复的前提下，两种计数一一对应；`1` 的平方还是自身，因此也计数。PDF p. 1 的题干末尾被截断，[[MIT 6.100L-OCW-offline-site/pages/lecture-12-list-comprehension-functions-as-objects-testing-debugging/index.html|官方离线页面题干]]保留完整的 `including itself`。上方代码是函数体片段，缩进按其逻辑整理，不照搬 PDF 解答框的排版错位。
<!-- bilingual-en:start -->
The official specification counts values that are squares of elements in the list, whereas the code tests each root `i` for membership of `i*i`. With distinct positive inputs, these counts correspond one-to-one; `1` counts because it is its own square. The end of the statement is clipped on PDF p. 1, while the [[MIT 6.100L-OCW-offline-site/pages/lecture-12-list-comprehension-functions-as-objects-testing-debugging/index.html|official offline page]] retains “including itself.” The code above is a function-body excerpt with logical indentation restored from the PDF's misaligned solution layout.
<!-- bilingual-en:end -->

这题虽然不复杂，但很适合放在本讲之后，因为它逼你把几种“列表处理模式”压缩到一个短函数里：

- 遍历列表
- membership test
- 计数器模式
<!-- bilingual-en:start -->
Although simple, the exercise is well placed here because it compresses three list-processing patterns into a short function: traversal, a membership test, and a counter.
<!-- bilingual-en:end -->

如果你想进一步练本讲前半段内容，可以把它改写成 list comprehension 风格去理解。
<!-- bilingual-en:start -->
To extend the first half of the lecture, you can reinterpret or rewrite it in a list-comprehension style.
<!-- bilingual-en:end -->

## Links to follow-up practice
- Slides: [[MIT 6.100L-slides/mit6_100l_lec12.pdf|Lecture 12 slides]]；[[MIT 6.100L-slides/mit6_100l_lec12.pdf#page=33|pp. 33–46: testing, black/glass-box cases, and debugging]]
- Lecture code: [[MIT 6.100L-OCW-offline-site/static_resources/mit6_100l_f22_lec12_code.zip|Lecture 12 code (zip)]]
- Finger exercise: [[MIT 6.100L-finger-exercises/mit6_100l_ex12_sol.pdf|Lecture 12 finger exercise solution]]；[[MIT 6.100L-finger-exercises/mit6_100l_ex12_sol.pdf#page=1|p. 1: count_sqrts specification and solution]]
- Transcript: [[MIT 6.100L-transcripts/mit6_100l_lec12_transcript.pdf|Lecture 12 transcript]]
- Recitation 6: [[MIT 6.100L-OCW-offline-site/static_resources/mit6_100l_f22_rec06.zip|Recitation 06 materials]]
- PS 3 out: [[MIT 6.100L-problem-sets/mit6_100l_ps3.pdf|PS3 statement]], [[MIT 6.100L-OCW-offline-site/static_resources/mit6_100l_f22_ps3_code.zip|PS3 starter code]]
- PS 2 due: [[MIT 6.100L-problem-sets/mit6_100l_ps2.pdf|PS2 statement]], [[MIT 6.100L-OCW-offline-site/static_resources/mit6_100l_f22_ps2_code.zip|PS2 starter code]]
- Textbook: [[Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Guttag textbook]]；本地 2013 Revised and Expanded 版的对应阅读是 §4.1.2、§5.2.2、§5.3，以及 [[Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf#page=87|Ch 6 Testing and Debugging，印刷 pp. 70–83 / PDF pp. 87–100]]。
  <!-- bilingual-en:start -->
  In the local 2013 revised and expanded edition, the relevant sections are §4.1.2, §5.2.2, §5.3, and Chapter 6, printed pp. 70–83 / PDF pp. 87–100.
  <!-- bilingual-en:end -->

## Review checklist
- [ ] 我能把一条 list comprehension 展开成普通循环，也能反过来压缩回去。
- [ ] 我能解释 comprehension 中 expression、iteration、filter 各在什么位置。
- [ ] 我能说明 default parameter 为什么是接口设计问题，而不只是语法方便。
- [ ] 我能区分位置参数、keyword argument、default value 的角色。
- [ ] 我能解释 `make_prod` 为什么说明函数也可以作为返回值。
- [ ] 我能区分 testing 和 debugging。
- [ ] 我能解释 black-box testing 和 glass-box testing 的不同关注点。
- [ ] 我能复述老师示范的 debugging recipe。
- [ ] 我能说出 buggy palindrome 里两个不同层面的 bug 分别是什么。
- [ ] 我能按课堂顺序复述：list comprehension -> function parameters -> testing -> debugging。
<!-- bilingual-en:start -->
- [ ] I can expand a list comprehension into an ordinary loop and compress the loop back into a comprehension.
- [ ] I can identify the expression, iteration, and filter positions in a comprehension.
- [ ] I can explain why a default parameter is an interface-design decision rather than mere syntactic convenience.
- [ ] I can distinguish positional arguments, keyword arguments, and default values.
- [ ] I can explain how `make_prod` demonstrates that a function may be returned as a value.
- [ ] I can distinguish testing from debugging.
- [ ] I can explain the different concerns of black-box and glass-box testing.
- [ ] I can reconstruct the instructor's debugging recipe.
- [ ] I can identify the two different faults in the buggy palindrome function.
- [ ] I can reconstruct the lecture sequence: list comprehensions -> function parameters -> testing -> debugging.
<!-- bilingual-en:end -->

> [!warning] Common mistakes
> - 把 list comprehension 当成“新功能”，却不知道它对应哪段普通循环。
> - default parameter 写了以后，没想清默认行为是否合理。
> - 以为测试就是随便挑几个输入跑一下。
> - debugging 时到处乱改，不先缩小问题范围。
> <!-- bilingual-en:start -->
> - Treating a list comprehension as a new capability without knowing which ordinary loop it represents.
> - Adding a default parameter without deciding whether the default behavior is actually sensible.
> - Mistaking testing for running a few arbitrary inputs.
> - Changing code in many places while debugging instead of first narrowing the source of the failure.
> <!-- bilingual-en:end -->
