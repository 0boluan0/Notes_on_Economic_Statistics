---
aliases:
  - MIT 6.100L Lecture 13
  - 6.100L L13
  - Exceptions and Assertions
tags:
  - computer-science
  - python
  - mit-6.100l
  - lecture-note
科目: Computer Science
course: MIT 6.100L Introduction to CS and Programming Using Python
lecture: 13
---

# Lecture 13: Exceptions and Assertions

[[03_Computer_Science/03_MIT 6.100L/12_测试与失败处理|测试与失败处理 · 连续阅读]] · [[测试、调试、异常与断言.canvas|测试、调试、异常与断言 · 关系图]]

> [!tip] Hint
> - 这节课开场先从 “scary red errors” 讲起，目标不是怕错误，而是学会把错误纳入程序逻辑。
> - 异常是改变正常控制流的带类型信号；本讲先从 unexpected condition 引入，但异常不一定是程序员没预料到的错误。
> - `try`/`except` 的课堂语义很明确：先试正常路径，出错了再走备用处理逻辑。
> - `except ValueError` 和裸 `except` 的区别，是“你到底想捕获什么问题”。
> - 这节课反复强调：不是所有错误都该被吞掉，有时应该返回默认值，有时应该重新 raise。
> - `raise ValueError(...)` 的意义是把“这个输入不符合我函数约定”明确表达出来。
> - assertions 使用异常机制，更像是给程序员自己的内在假设加护栏。
> - 断言启用时，`assert condition` 的条件为假会抛 `AssertionError`；它也可被捕获，且 `-O` 优化可以删除断言。
> - exceptions 是处理运行时异常情况，assertions 更偏向开发时抓逻辑违例。
> - 听完这节课，你应该能判断某个问题该 try/except、该 raise、还是该 assert。
> <!-- bilingual-en:start -->
> - The lecture opens with “scary red errors.” Its goal is not to make errors frightening, but to bring them into the program's logic.
> - An exception is a typed signal that changes normal control flow. The lecture introduces unexpected conditions, but an exception need not be an unanticipated programming error.
> - The classroom meaning of `try`/`except` is straightforward: attempt the normal path first, then use a fallback path if an error occurs.
> - The difference between `except ValueError` and a bare `except` is the difference between naming the failure you intend to handle and indiscriminately catching everything.
> - The lecture repeatedly stresses that not every error should be suppressed. Sometimes a default value is appropriate; at other times the exception should be raised again.
> - `raise ValueError(...)` explicitly states that an input violates the function's contract.
> - Assertions use the exception mechanism chiefly to guard assumptions made by the programmer.
> - With assertions enabled, a false condition raises `AssertionError`; that exception can also be caught, and `-O` optimization can remove assertions.
> - Exceptions handle exceptional runtime situations, whereas assertions are mainly a development-time check for violations of program logic.
> - By the end, you should be able to decide whether a situation calls for `try`/`except`, an explicit `raise`, or an `assert`.
> <!-- bilingual-en:end -->

## Lecture flow

### 1. 从 debugging ducks 进入：错误不是只拿来害怕的
<!-- bilingual-en:start -->
*1. Starting with Debugging Ducks: Errors Are More Than Something to Fear*
<!-- bilingual-en:end -->
Lecture 13 的开场延续了上一讲的 debugging 氛围。  
老师先拿 “debugging ducks” 活跃气氛，然后立刻转到今天主题：

- exceptions
- assertions

她把这些东西描述成我们经常看到的 “scary red errors”。  
这句话的重点不是渲染恐惧，而是让你接受：

- 错误信息不是程序世界的意外噪音
- 它们本身就是语言机制的一部分
<!-- bilingual-en:start -->
Lecture 13 continues the previous lecture's emphasis on debugging. After a lighthearted reference to “debugging ducks,” the instructor turns immediately to exceptions and assertions—the “scary red errors” students already recognize. The point is that error messages are not accidental noise around a program; they are part of the language's machinery.
<!-- bilingual-en:end -->

### 2. 什么是 exception：程序遇到了原本没预料到的情况
<!-- bilingual-en:start -->
*2. What Is an Exception? A Program Encounters an Unanticipated Condition*
<!-- bilingual-en:end -->
老师先从概念上解释 exception。

程序多数时候按正常路径运行；  
但一旦碰到某个意外条件，比如：

- index 越界
- 类型不匹配
- 名字不存在
- 除以 0

Python 就会抛出一个 exception。
<!-- bilingual-en:start -->
The instructor first explains the concept. A program normally follows its intended path, but Python raises an exception when execution encounters a condition such as an out-of-range index, a type mismatch, an undefined name, or division by zero.
<!-- bilingual-en:end -->

所以 exception 可以理解成：

> [!note]
> 程序执行偏离了“预期正常路径”时发出的信号。
> <!-- bilingual-en:start -->
> A signal emitted when execution departs from the expected normal path.
> <!-- bilingual-en:end -->

课堂这里也专门说，很多你已经见过的错误其实都是不同类型的 exception。
<!-- bilingual-en:start -->
The lecture also points out that many familiar errors are simply different types of exception.
<!-- bilingual-en:end -->

这里的 unexpected 是课堂的入门情境，不是[[Python异常]]的完整定义：函数也可以有意抛出异常来表达不能满足合同的情况。异常对象携带类型与信息，改变当前正常控制流；是否结束整个程序，要看外层是否有匹配处理器。`SyntaxError` 通常在解析代码时报告，而 `IndexError`、`TypeError` 等示例发生在相应操作执行时，不能统称为执行前就能查出的静态错误。
<!-- bilingual-en:start -->
“Unexpected” introduces the classroom examples, rather than defining every [[Python异常|Python exception]]. A function may deliberately raise an exception when it cannot fulfill its contract. The exception carries a type and information and changes normal control flow; a matching outer handler may prevent program termination. `SyntaxError` is normally reported while parsing code, whereas examples such as `IndexError` and `TypeError` arise when the relevant operation executes, not necessarily in a static check before execution.
<!-- bilingual-en:end -->

### 3. 以前程序一报错就崩，现在开始学会“接住”异常
<!-- bilingual-en:start -->
*3. Moving from Crashing on Every Error to Catching Exceptions*
<!-- bilingual-en:end -->
到这一步老师指出，之前我们的程序遇到异常时通常只有一个结果：

- 直接 crash
- 回去 debug

但 Python 其实允许你写代码去处理这些情况。  
这就是 `try` / `except` 的意义。

它们让程序可以说：

- 先按正常逻辑试试看
- 如果某类异常真的发生了
- 那就执行另一套处理代码
<!-- bilingual-en:start -->
Until now, an exception usually meant that the program crashed and you returned to debugging. Python can instead handle such situations explicitly with `try` and `except`: attempt the normal computation, and if a designated exception occurs, execute an alternative response.
<!-- bilingual-en:end -->

### 4. `try` / `except`：把正常路径和异常路径显式分开
<!-- bilingual-en:start -->
*4. `try` / `except`: Separating the Normal and Exceptional Paths Explicitly*
<!-- bilingual-en:end -->
老师先讲最基本框架：
<!-- bilingual-en:start -->
The instructor begins with the basic structure:
<!-- bilingual-en:end -->

```python
try:
    # potentially problematic code
except:
    # code to run if an exception occurs
```

理解这段结构时要抓住两条：

- `try` 里面放的是你希望正常执行的代码
- `except` 里面放的是“如果 try 失败，该怎么办”
<!-- bilingual-en:start -->
- The `try` block contains the code you expect to run normally.
- The `except` block states what to do if the attempted operation fails.
<!-- bilingual-en:end -->

如果 try 里没有异常，except 就不会运行。  
如果 try 里抛了异常，该次 try 中尚未执行的语句被跳过，Python 按书写顺序寻找第一个类型匹配的 except；没有匹配时，[[异常传播|异常沿外层代码与调用栈继续传播]]。处理器正常结束后才继续执行整个 try 语句后面的代码，不会返回失败语句重试。
<!-- bilingual-en:start -->
If the `try` block raises no exception, handlers are skipped. Otherwise, its remaining statements are skipped and Python searches handlers in order for the first matching exception type. If none matches, [[异常传播|the exception propagates through enclosing code and the call stack]]. A handler that completes normally continues after the entire `try` statement; it does not retry the failed statement.
<!-- bilingual-en:end -->

### 5. 具体例子：字符串求数字和
<!-- bilingual-en:start -->
*5. Worked Example: Summing Digits in a String*
<!-- bilingual-en:end -->
老师用 `sum_digits` 一类例子演示异常处理。

原始任务是：

- 遍历字符串
- 把其中数字字符转换成 int
- 求和
<!-- bilingual-en:start -->
The instructor demonstrates exception handling with a function such as `sum_digits`: traverse a string, convert its digit characters to integers, and add them.
<!-- bilingual-en:end -->

问题在于，如果字符串里混入非数字字符，`int(...)` 可能抛 `ValueError`。
<!-- bilingual-en:start -->
The difficulty is that `int(...)` may raise `ValueError` when the string contains a nonnumeric character.
<!-- bilingual-en:end -->

于是你就有几种选择：

- 忽略这个字符
- 给出默认处理
- 直接把异常继续抛给调用者
<!-- bilingual-en:start -->
- Ignore the offending character.
- Supply a default response.
- Let the exception propagate to the caller.
<!-- bilingual-en:end -->

课程这里要你看到的不是某一个“正确唯一答案”，而是：

- exception handling 让你能明确决定程序该怎么处理坏输入
<!-- bilingual-en:start -->
There is no single universally correct response. Exception handling makes the policy for invalid input an explicit program decision.
<!-- bilingual-en:end -->

对应原代码的四个版本如下。前两个示范“收集数字字符”，后两个转为“有不能转换的字符就拒绝”；它们不是同一失败合同的等价实现。以下整理其核心步骤，把源文件裸捕获收窄为此处预期的 `ValueError`，并让返回规则明确可见。
<!-- bilingual-en:start -->
The source develops four versions. The first two collect digit characters; the latter two reject a character that cannot be converted. They do not implement the same failure contract. The reconstructed cores below narrow the source's bare handlers to the expected `ValueError` and keep each return rule explicit.
<!-- bilingual-en:end -->

```python
def sum_digits(s):
    total = 0
    for char in s:
        if char in '0123456789':
            total += int(char)
    return total

def sum_digits_except(s):
    total = 0
    for char in s:
        try:
            total += int(char)
        except ValueError:
            print("couldn't convert character", char)
    return total

def sum_digits_raise(s):
    total = 0
    for char in s:
        try:
            total += int(char)
        except ValueError:
            raise ValueError("string contained a character")
    return total

def sum_digits_assert(s):
    assert len(s) != 0, "s is empty"
    total = 0
    for char in s:
        try:
            total += int(char)
        except ValueError:
            raise ValueError("string contained a character")
    return total
```

对课堂的 ASCII 示例 `"123abc"`，前两版返回 `6`（第二版还打印警告），后两版抛 `ValueError`；对 `""`，前三版返回 `0`，第四版在断言启用时抛 `AssertionError`。前三版 docstring 的“非空”是假设，并没有自动变成执行检查。第一版限定 ASCII 字符，`int(char)` 的可转换范围不应由这个例子推成完全相同。Slides p. 14 的代码画面缺少末尾 `return total`，原 `.py` 的 `sum_digits_assert` 保留了它；以上也保留该返回。
<!-- bilingual-en:start -->
For the classroom ASCII input `"123abc"`, the first two return `6` (with warnings from the second); the latter two raise `ValueError`. For `""`, the first three return `0`, while the fourth raises `AssertionError` when assertions are enabled. A nonempty-input assumption in a docstring does not itself execute a check. The first version explicitly selects ASCII digits, so this example does not establish identical character domains for that check and `int(char)`. Slide p. 14 omits the final `return total`, which is present in the original `.py` and retained above.
<!-- bilingual-en:end -->

### 6. 指定异常类型：不是所有错误都该被同一种方法处理
<!-- bilingual-en:start -->
*6. Naming Exception Types: Different Failures Need Different Responses*
<!-- bilingual-en:end -->
老师接下来强调，except 不一定要写得很宽泛。
<!-- bilingual-en:start -->
The instructor next emphasizes that an `except` clause need not—and usually should not—be overly broad.
<!-- bilingual-en:end -->

你可以写：

```python
except ValueError:
    ...
except ZeroDivisionError:
    ...
except Exception as err:
    ...
```

课堂在这里的重要观念是：

- 不同异常代表不同失败原因
- 处理方式也可能不同
<!-- bilingual-en:start -->
- Different exception types represent different causes of failure.
- Those failures may require different handling.
<!-- bilingual-en:end -->

所以如果你知道自己要防的就是 `ValueError`，那就应该写得更具体，而不是用一个笼统的裸 `except` 把所有问题吞掉。
<!-- bilingual-en:start -->
If `ValueError` is the failure you intend to handle, catch it specifically instead of using a bare `except` that suppresses unrelated problems as well.
<!-- bilingual-en:end -->

> [!warning]
> 裸 `except` 常常太宽，会把你原本想看见的 bug 也吞掉。
> <!-- bilingual-en:start -->
> A bare `except` is often too broad: it can hide the very bug you need to see.
> <!-- bilingual-en:end -->

[[异常类型匹配]]不仅包括同一个异常类，也包括它的子类，因此通常先写具体处理器。[[异常捕获边界|`except Exception` 不等于裸 `except`]]：后者还会截住 `KeyboardInterrupt`、`SystemExit` 等不属于 `Exception` 的异常。若只想记录一个未预期故障后交给上层，处理器末尾用不带参数的 `raise` 重新抛出，不能仅打印后假装计算成功。
<!-- bilingual-en:start -->
[[异常类型匹配|Exception matching]] includes subclasses, so specific handlers usually precede broader ones. [[异常捕获边界|`except Exception` is not a bare `except`]]: the latter also catches exceptions such as `KeyboardInterrupt` and `SystemExit` that are outside `Exception`. To record an unexpected fault and leave it for an outer layer, end the handler with a bare `raise` instead of merely printing and pretending the computation succeeded.
<!-- bilingual-en:end -->

Slides p. 8 还简要介绍两个子句：[[try的else分支|`else`]] 只在 try 正常走到末尾、没有异常或 `return`/`break`/`continue` 提前离开时运行，其中异常不由同组前面的 except 捕获；[[finally清理|`finally`]] 在离开 try 语句前做清理，即使存在待传播异常或待执行的返回。不要在 finally 中用新的返回掩盖原异常；“总会清理”也不是对进程被外部强制终止的保证。
<!-- bilingual-en:start -->
Slide p. 8 briefly introduces two further clauses. [[try的else分支|`else`]] runs only when the try suite reaches its end normally, without an exception or an early `return`, `break`, or `continue`; its exceptions are not caught by the preceding handlers of that same statement. [[finally清理|`finally`]] performs cleanup before leaving the try statement, including when an exception or return is pending. A new return in finally can hide an exception, and cleanup is not guaranteed if the process is forcibly terminated externally.
<!-- bilingual-en:end -->

> [!note] 原例输出不是除零的数学定义
> Slides p. 7 和 `divide_nums3` 在捕获除零后打印 `a/b = infinity`；这只是原示例选定的提示文本，不是 Python 算出的商，也不能把一般 `a/0`（尤其 `0/0`）定义为无穷大。稳妥的提示是本次除法没有有效结果。
> <!-- bilingual-en:start -->
> Slide p. 7 and `divide_nums3` print `a/b = infinity` after catching division by zero. This is the example's chosen message, not a quotient computed by Python or a definition of general `a/0`, especially `0/0`. The division has no valid result here.
> <!-- bilingual-en:end -->

### 7. `pairwise_div`：按合同拒绝含零分母
<!-- bilingual-en:start -->
*7. `pairwise_div`: Rejecting a Zero Denominator as the Contract Requires*
<!-- bilingual-en:end -->
课堂题（slides p. 11）给定两个非空、等长的数值列表，返回对应位置相除组成的新列表；如果 `Ldenom` 含 `0`，必须抛 `ValueError`。这题没有规定跳过坏元素或给默认值，不能把另一种容错政策当作它的答案。
<!-- bilingual-en:start -->
The classroom exercise (slide p. 11) takes two nonempty, equally long numeric lists and returns a new list of pairwise quotients. If `Ldenom` contains `0`, it must raise `ValueError`. Skipping an element or inserting a default would implement a different contract.
<!-- bilingual-en:end -->

```python
def pairwise_div(Lnum, Ldenom):
    if 0 in Ldenom:
        raise ValueError("denominator cannot be 0")
    return [Lnum[i] / Ldenom[i] for i in range(len(Lnum))]
```

这里仍假设非空、等长且元素是数值；第 11 节再对应 slides p. 15 加入开发期断言。也可在除法处捕获 `ZeroDivisionError` 后抛 `ValueError`，但一旦 raise，当前函数的正常循环就被中断，由外层决定是否处理。课堂 transcript pp. 7–8 明确展示了这两种写法。
<!-- bilingual-en:start -->
This version still assumes nonempty, equally long numeric lists; section 11 adds the developmental assertions from slide p. 15. Another implementation catches `ZeroDivisionError` at the division and raises `ValueError`. Raising interrupts the function's normal loop, leaving handling to an outer layer. Transcript pp. 7–8 explicitly demonstrates both approaches.
<!-- bilingual-en:end -->

> [!warning] 原代码的参数名误用
> `.py` 中 `ANSWERS TO YOU TRY IT` 版本检查的是 `if 0 in L2:`，误用了全局示例变量，而非参数 `Ldenom`。全局 `L2=[0]`、实际分母 `[2]` 会误拒绝；全局 `L2=[2]`、实际分母 `[0]` 又会漏检并抛出 `ZeroDivisionError`。上方课堂整理改为检查 `Ldenom`，原文件保持不动。文件后面还有同名 AT HOME 版本，会覆盖前面的定义；它进一步约定不能执行除法时抛 `ValueError`，但原裸捕获仍有过宽边界。
> <!-- bilingual-en:start -->
> The `.py` version under `ANSWERS TO YOU TRY IT` checks global `L2` rather than parameter `Ldenom`. Global `L2=[0]` wrongly rejects the actual denominator `[2]`, while global `L2=[2]` misses an actual denominator `[0]` and leaks `ZeroDivisionError`. The classroom reconstruction above checks the parameter; the original file is unchanged. Later AT HOME definitions overwrite the same name and broaden the contract to divisions that cannot be performed, but their bare handler remains overly broad.
> <!-- bilingual-en:end -->

### 8. 什么时候应该 `raise` 自己的异常
<!-- bilingual-en:start -->
*8. When to `raise` an Exception Yourself*
<!-- bilingual-en:end -->
讲到这里，老师把方向再推进一步：

- 你不只会“接住” Python 自带异常
- 你还可以[[主动抛出异常|主动 `raise` 自己的异常]]
<!-- bilingual-en:start -->
The lecture now moves from catching built-in exceptions to [[主动抛出异常|raising an exception deliberately]].
<!-- bilingual-en:end -->

例如你写函数时可能想表达：

- 输入为空不合法
- 分母列表里不允许出现 0
- 某个参数类型或范围违反函数前提
<!-- bilingual-en:start -->
A function may need to reject an empty input, forbid zero in a denominator list, or enforce a parameter's required type or range.
<!-- bilingual-en:end -->

这时你可以显式写：

```python
raise ValueError("denominator cannot be 0")
```

这样做的意义是把函数的前提条件说得更清楚，而不是等代码在深处莫名崩掉。
<!-- bilingual-en:start -->
Raising the exception at the boundary makes the function's precondition explicit instead of allowing execution to fail mysteriously deeper in the implementation.
<!-- bilingual-en:end -->

### 9. `raise` 的课堂语义：把“这是坏输入”写进接口里
<!-- bilingual-en:start -->
*9. The Meaning of `raise`: Encoding “Invalid Input” in the Interface*
<!-- bilingual-en:end -->
这一段的关键不是背 `raise` 的语法，而是理解它的设计角色。
<!-- bilingual-en:start -->
The important point is not memorizing the syntax of `raise`, but understanding its role in interface design.
<!-- bilingual-en:end -->

如果某个输入超出了函数原有的正常处理范围，课堂比较两种处理方向：

- 给默认值继续跑，但必须说明这种结果的含义和适用条件
- 明确拒绝它，并 raise 一个异常
<!-- bilingual-en:start -->
The lecture contrasts two responses when an input is outside the function's original normal case: continue with a default whose meaning and conditions are explicitly defined, or reject the input by raising an exception.
<!-- bilingual-en:end -->

当错误真的代表“调用者违反了接口约定”时，后者往往更合适。  
这让函数接口边界更清楚，也让 bug 更早暴露出来。
<!-- bilingual-en:start -->
Explicit rejection is often preferable when the caller has violated the contract. It sharpens the function boundary and exposes bugs earlier.
<!-- bilingual-en:end -->

选择[[异常处理策略]]时，默认值不一定错误，静默把失败伪装成有效结果才是问题。例如“没有成绩记为 0 分”是评分政策；“没有成绩所以均值在数学上就是 0”则不是同一陈述。失败返回值、警告、异常及副作用都应由[[函数契约]]说明。
<!-- bilingual-en:start -->
When choosing an [[异常处理策略|exception-handling policy]], a default is not inherently wrong; disguising a failure as a valid result is the problem. “No grades receive a score of zero” is a grading policy, not the mathematical statement that an empty collection has mean zero. Failure values, warnings, exceptions, and side effects belong in the [[函数契约|function contract]].
<!-- bilingual-en:end -->

### 10. assertions：给程序自己的假设加护栏
<!-- bilingual-en:start -->
*10. Assertions: Guardrails for the Program's Own Assumptions*
<!-- bilingual-en:end -->
讲完 exceptions 之后，老师再引入 assertions。

[[Python断言|assertion]] 也是一种异常机制，但角度不一样。
它的核心语句是：
<!-- bilingual-en:start -->
After exceptions, the instructor introduces [[Python断言|assertions]]. They also use the exception mechanism, but serve a different purpose. Their central statement is:
<!-- bilingual-en:end -->

```python
assert condition
```

断言启用时会先求值 `condition`；若其真值为真，继续执行；若为假，抛出 `AssertionError`。条件表达式本身若求值失败，也可能先抛出别的异常。
<!-- bilingual-en:start -->
With assertions enabled, Python evaluates `condition` and tests its truth value. A true result continues execution; a false result raises `AssertionError`. Evaluating the condition can itself raise another exception first.
<!-- bilingual-en:end -->

课堂里老师把它解释成：

- 程序员在代码中声明“这里我认为某个条件必须成立”
- 一旦不成立，立刻暴露
<!-- bilingual-en:start -->
An assertion records a programmer's claim that a condition must hold at that point and makes any violation fail immediately.
<!-- bilingual-en:end -->

`assert condition, "message"` 可以提供失败信息。`AssertionError` 同样遵循异常传播和捕获规则；slides 中“停止执行”指本例没有处理器接住它，不是断言拥有不可捕获的特殊退出能力。
<!-- bilingual-en:start -->
`assert condition, "message"` supplies a failure message. `AssertionError` follows the same propagation and matching rules as other exceptions. The slides' halted execution assumes that no handler catches it; an assertion is not an uncatchable exit mechanism.
<!-- bilingual-en:end -->

### 11. assertion 更像开发时的自我检查
<!-- bilingual-en:start -->
*11. Assertions as Development-Time Self-Checks*
<!-- bilingual-en:end -->
和 try/except 不同，assert 往往不是用来优雅处理用户输入，而是用来保护程序内部逻辑。
<!-- bilingual-en:start -->
Unlike `try`/`except`, `assert` is generally not a graceful way to handle user input. It protects assumptions internal to the program.
<!-- bilingual-en:end -->

例如：

- 某长度不该为 0
- 某分母在这里必须非零
- 某中间变量应满足你前面推导出的不变量
<!-- bilingual-en:start -->
Examples include a length that cannot be zero, a denominator that must be nonzero at this stage, or an intermediate value that must satisfy an invariant established earlier.
<!-- bilingual-en:end -->

这类条件如果不成立，往往说明：

- 程序逻辑已经走偏
- 或某个前置函数没有兑现承诺
<!-- bilingual-en:start -->
If such a condition fails, the program's logic has gone astray or an earlier function has failed to honor its promise.
<!-- bilingual-en:end -->

所以 assert 更像一种开发时的 guardrail。
<!-- bilingual-en:start -->
An assertion is therefore best understood as a development-time guardrail.
<!-- bilingual-en:end -->

> [!warning] 课堂合同检查与优化边界
> Slides pp. 13–15 用 `assert` 检查非空、等长等输入假设，展示的是断言启用时的开发期检查。Python 的 `-O` 可以在编译时删除整条断言，包括条件和消息的求值。因此，[[断言不替代输入验证|必须执行的不可信输入验证或安全检查要用普通条件与显式异常]]，必要状态更新也不能藏进断言。这里的实践边界依据 Python 语言参考补充，不把课堂简化示例当作无条件保证。
> <!-- bilingual-en:start -->
> Slides pp. 13–15 demonstrate developmental checks of nonempty and equal-length input assumptions with assertions enabled. Python's `-O` can remove the entire assertion at compilation, including evaluation of its condition and message. [[断言不替代输入验证|Use ordinary conditionals and explicit exceptions for mandatory input validation or security checks]], and keep required state changes outside assertions. This qualification comes from the Python language reference, not an unconditional guarantee in the classroom example.
> <!-- bilingual-en:end -->

对应课堂 `pairwise_div` 的下一步，是在第 7 节的零分母拒绝之前增加两条断言。完整教学版如下，仍假设列表元素是数值：
<!-- bilingual-en:start -->
The next classroom step adds two assertions before section 7's rejection of a zero denominator. This complete instructional version still assumes numeric elements:
<!-- bilingual-en:end -->

```python
def pairwise_div(Lnum, Ldenom):
    assert len(Lnum) == len(Ldenom), "lists not equal length"
    assert Lnum != [], "list is empty"
    if 0 in Ldenom:
        raise ValueError("denominator cannot be 0")
    return [Lnum[i] / Ldenom[i] for i in range(len(Lnum))]
```

原课堂例子依次是 `[4,5,6] / [1,2,3]` 返回 `[4.0,2.5,2.0]`，分母换为 `[1,0,3]` 抛 `ValueError`；分子 `[4,5,6,7,8]` 配分母 `[1,8,3]`、以及两边都是 `[]`，在断言启用时分别触发等长与非空检查。前两类属于正常结果/指定失败行为，后两类展示开发假设检查，不应混为一种测试预期。
<!-- bilingual-en:start -->
The original examples return `[4.0,2.5,2.0]` for `[4,5,6] / [1,2,3]` and raise `ValueError` for denominator `[1,0,3]`. Numerator `[4,5,6,7,8]` with denominator `[1,8,3]`, and two empty lists, trigger the length and nonempty assertions respectively when enabled. The first cases test normal results and specified failure behavior; the latter demonstrate checks of developmental assumptions.
<!-- bilingual-en:end -->

#### 课堂长例：空成绩列表
<!-- bilingual-en:start -->
*Extended Classroom Example: `get_stats` and `avg1`–`avg4`*
<!-- bilingual-en:end -->

Slides pp. 17–22 把同一政策问题放到成绩表中：每条学生记录含姓名列表和成绩列表，新记录在末尾增加平均分。原 `.py` 把求均值函数作为 `avg_func` 参数，方便在同一份数据上比较四个版本，直接复用[[一等函数对象]]。
<!-- bilingual-en:start -->
Slides pp. 17–22 put the policy question into a gradebook. Each record contains a name list and a grade list; the new record appends an average. The original `.py` accepts the averaging function as `avg_func`, using [[一等函数对象|first-class functions]] to compare four versions on the same data.
<!-- bilingual-en:end -->

```python
def get_stats(class_list, avg_func):
    new_stats = []
    for person in class_list:
        new_stats.append([person[0], person[1], avg_func(person[1])])
    return new_stats

test_grades = [[['peter', 'parker'], [10.0, 55.0, 85.0]],
               [['bruce', 'wayne'], [10.0, 80.0, 75.0]],
               [['captain', 'america'], [80.0, 10.0, 96.0]],
               [['thor'], []]]

def avg1(grades):
    return sum(grades) / len(grades)

def avg2(grades):
    try:
        return sum(grades) / len(grades)
    except ZeroDivisionError:
        print('warning: no grades data')

def avg3(grades):
    try:
        return sum(grades) / len(grades)
    except ZeroDivisionError:
        print('warning: no grades data')
        return 0.0

def avg4(grades):
    assert len(grades) != 0, 'warning: no grades data'
    return sum(grades) / len(grades)
```

依次把 `avg1`、`avg2`、`avg3`、`avg4` 传给 `get_stats(test_grades, ...)`，前三位学生的均值均为 `50.0`、`55.0`、`62.0`；最后一条空成绩记录区分了四种行为：
<!-- bilingual-en:start -->
Passing each of `avg1` through `avg4` to `get_stats(test_grades, ...)` gives averages `50.0`, `55.0`, and `62.0` for the first three students. The final empty grade list distinguishes the four behaviors:
<!-- bilingual-en:end -->

- `avg1`：抛出 `ZeroDivisionError`，当前 `get_stats` 调用没有正常返回整个新表。
- `avg2`：打印警告后走到函数末尾，隐式返回 `None`，新表的最后一条为 `[['thor'], [], None]`。这是[[返回值与打印]]的区别，不是“打印了警告就返回了分数”。
- `avg3`：打印警告并显式返回 `0.0`，最后一条为 `[['thor'], [], 0.0]`；它选择的是“无成绩记零分”的政策。
- `avg4`：断言启用时抛 `AssertionError`，不返回新表；若删除断言，空列表仍会在后续除法抛 `ZeroDivisionError`，并不会因此变成合法平均值。
<!-- bilingual-en:start -->
- `avg1` raises `ZeroDivisionError`, so the current `get_stats` call does not normally return the completed table.
- `avg2` prints a warning, reaches the function's end, and implicitly returns `None`; the final record becomes `[['thor'], [], None]`. This illustrates [[返回值与打印|return values versus printing]], not a returned grade.
- `avg3` prints a warning and explicitly returns `0.0`; the final record is `[['thor'], [], 0.0]`, implementing a policy of assigning zero when grades are absent.
- `avg4` raises `AssertionError` when assertions are enabled and does not return the table. Removing the assertion still leaves a later `ZeroDivisionError`; it does not make the empty mean valid.
<!-- bilingual-en:end -->

新表的外层记录是新建的，但姓名、成绩仍引用原来的子列表，不能据此宣称深拷贝；这里没有修改这些子列表。原 slides 用 `deadpool`，`.py` 用 `thor` 作无成绩学生，以上保留 `.py` 数据。核心对照是同一空输入在不同[[函数契约]]下的可观察结果。
<!-- bilingual-en:start -->
The result creates new outer records but retains references to the original name and grade lists; it is not a deep copy, and this code does not mutate those sublists. The slides use `deadpool` while the `.py` uses `thor` for the student without grades; the data above follows the `.py`. The essential comparison is the observable behavior of the same empty input under different [[函数契约|function contracts]].
<!-- bilingual-en:end -->

### 12. exceptions 和 assertions 的课堂分工
<!-- bilingual-en:start -->
*12. The Classroom Division of Labor Between Exceptions and Assertions*
<!-- bilingual-en:end -->
老师后面其实一直在帮助大家区分这两套机制：

- exceptions：用带类型的信号表达正常流程不能继续的情况，并允许外层决定处理方式
- assertions：断言启用时检查本应成立的假设，失败时也通过异常机制报告
<!-- bilingual-en:start -->
- Exceptions use typed signals when normal execution cannot continue and allow an outer layer to select a response.
- Assertions check assumptions that should hold when enabled and report failure through that same exception mechanism.
<!-- bilingual-en:end -->

更口语化地说：

- `try` / `except` 更像“外部世界可能出错，我要怎么应对”
- `assert` 更像“如果这里都不成立，那说明我的程序自己有问题”
<!-- bilingual-en:start -->
In plain language, `try`/`except` says, “the outside world may fail; how should I respond?” An `assert` says, “if this condition is false, my own program is inconsistent.”
<!-- bilingual-en:end -->

### 13. 这节课并不是“让程序不报错”，而是让错误变得有语义
<!-- bilingual-en:start -->
*13. The Goal Is Not to Eliminate Errors, but to Give Them Meaning*
<!-- bilingual-en:end -->
Lecture 13 最容易被误读成“学会别让程序崩”。  
但课堂真正目标更高：

- 不是简单压制错误
- 而是让错误处理更有语义、更靠近接口边界
<!-- bilingual-en:start -->
Lecture 13 can be misread as a lesson in preventing crashes. Its actual aim is not to suppress errors, but to make error handling meaningful and place it near the relevant interface boundary.
<!-- bilingual-en:end -->

好的 exception / assertion 使用，会让代码回答下面这些问题：

- 什么算合法输入
- 什么算调用者犯错
- 什么算程序内部逻辑违例
- 出问题后是继续、跳过、默认、还是立刻停止
<!-- bilingual-en:start -->
Good uses of exceptions and assertions make four questions explicit: what counts as valid input, what counts as caller error, what indicates an internal logic violation, and whether failure should lead to continuation, skipping, a default response, or immediate termination.
<!-- bilingual-en:end -->

## Exercise log

> [!example] Finger exercise 13
> 官方题目是 `sum_str_lengths(L)`：
> - `L` 是非空列表
> - 元素要么是字符串，要么是“非空字符串子列表”
> - 返回所有字符串长度之和
> - 顶层元素既非字符串也非列表时，或子列表中出现任何非字符串元素时，要 `raise ValueError`
> <!-- bilingual-en:start -->
> The official task is `sum_str_lengths(L)`:
> - `L` is a nonempty list.
> - Each element is either a string or a nonempty sublist of strings.
> - The function returns the total length of all strings.
> - It must `raise ValueError` if a top-level element is neither a string nor a list, or if any element of a sublist is not a string.
> <!-- bilingual-en:end -->

这题正好卡在本讲的核心点上：

- 你不只是遍历和计数
- 你还要在发现非法结构时主动抛异常
<!-- bilingual-en:start -->
This exercise sits directly on the lecture's central idea: besides traversing and counting, you must raise an exception when the input structure is invalid.
<!-- bilingual-en:end -->

官方思路就是：

- 遇到 `str` 就累加长度
- 遇到 `list` 就逐一检查其元素必须是 `str`，不继续接受更深的列表
- 遇到别的类型就 `raise ValueError`
<!-- bilingual-en:start -->
The official approach adds the length of a `str`, checks that every element of a `list` is a `str` without allowing a deeper list, and calls `raise ValueError` for other types.
<!-- bilingual-en:end -->

所以这题本质上在训练你把指定的非法类型写成代码里的显式错误路径；非空顶层列表、非空子列表仍是题目的输入假设，官方解答没有额外检查它们。
<!-- bilingual-en:start -->
The exercise turns the specified invalid types into explicit error paths. A nonempty top-level list and nonempty sublists remain input assumptions: the official solution does not add checks for them.
<!-- bilingual-en:end -->

官方三个例子应同时保留：`["abcd", ["e", "fg"]]` 返回 `7`；`[12, ["e", "fg"]]` 在顶层抛 `ValueError`；`["abcd", [3, "fg"]]` 在子列表中抛 `ValueError`。`["abcd", [["e"]]]` 也应因子列表含列表而拒绝，不应擅自递归展开。来源是题解 PDF p. 1 的完整合同与 p. 2 的两层类型检查。
<!-- bilingual-en:start -->
Retain all three official examples: `["abcd", ["e", "fg"]]` returns `7`; `[12, ["e", "fg"]]` raises `ValueError` at the top level; and `["abcd", [3, "fg"]]` raises it inside the sublist. `["abcd", [["e"]]]` must likewise reject a list inside the sublist rather than recursively flattening it. The solution PDF states the contract on p. 1 and implements two-level type checks on p. 2.
<!-- bilingual-en:end -->

## Links to follow-up practice
- Slides: [[MIT 6.100L-slides/mit6_100l_lec13.pdf|Lecture 13 slides]]；[[MIT 6.100L-slides/mit6_100l_lec13.pdf#page=3|pp. 3–11: exceptions and pairwise division]]；[[MIT 6.100L-slides/mit6_100l_lec13.pdf#page=13|pp. 13–23: assertions and the gradebook policies]]
- Lecture code: [[MIT 6.100L-lecture-code/mit6_100l_lec13_code.py|Lecture 13 code (py)]]
- Finger exercise: [[MIT 6.100L-finger-exercises/mit6_100l_ex13_sol.pdf|Lecture 13 finger exercise solution]]；[[MIT 6.100L-finger-exercises/mit6_100l_ex13_sol.pdf#page=1|pp. 1–2: the two-level input contract and solution]]
- Transcript: [[MIT 6.100L-transcripts/mit6_100l_lec13_transcript.pdf|Lecture 13 transcript]]
- Recitation 7: [[MIT 6.100L-OCW-offline-site/static_resources/mit6_100l_f22_rec07.zip|Recitation 07 materials]]
- Problem set milestone: none directly scheduled on this lecture
- Textbook: [[Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Guttag textbook]]；本地 2013 Revised and Expanded 版对应 [[Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf#page=101|Ch 7 Exceptions and Assertions，印刷 pp. 84–90 / PDF pp. 101–107]]。
  <!-- bilingual-en:start -->
  In the local 2013 revised and expanded edition, this material is Chapter 7, printed pp. 84–90 / PDF pp. 101–107.
  <!-- bilingual-en:end -->
- [Python Language Reference — try](https://docs.python.org/3/reference/compound_stmts.html#the-try-statement)：处理器按类型匹配、未匹配异常沿调用栈传播，以及 else/finally 的控制边界。
  <!-- bilingual-en:start -->
  Supports handler matching, propagation, and else/finally control flow.
  <!-- bilingual-en:end -->
- [Python Language Reference — assert](https://docs.python.org/3/reference/simple_stmts.html#the-assert-statement)：断言求值、`AssertionError` 与优化删除断言的边界。
  <!-- bilingual-en:start -->
  Supports assertion evaluation, `AssertionError`, and removal under optimization.
  <!-- bilingual-en:end -->
- [Python Tutorial — Errors and Exceptions](https://docs.python.org/3/tutorial/errors.html)：具体捕获、`Exception` 与 `BaseException`、重新抛出。
  <!-- bilingual-en:start -->
  Supports specific handling, the exception hierarchy, and re-raising.
  <!-- bilingual-en:end -->

> [!warning] Recitation 7 的语法旁注
> [[MIT 6.100L-recitations/mit6_100l_rec07/1_rec7/R07_summary.pdf#page=1|Recitation 7 summary p. 1]] 把捕获写成 `except ZeroDivisionError("Cannot divide by zero")`，不是有效的处理器写法：应匹配异常类，如 `except ZeroDivisionError as err:`，再在处理器中输出信息，或用 `raise ...` 主动创建异常。即使补上冒号，异常实例也不能充当 except 的匹配类型。原 PDF 保留不改。
> <!-- bilingual-en:start -->
> [[MIT 6.100L-recitations/mit6_100l_rec07/1_rec7/R07_summary.pdf#page=1|Recitation 7 summary p. 1]] writes `except ZeroDivisionError("Cannot divide by zero")`, which is not a valid handler. Match an exception class, for example `except ZeroDivisionError as err:`, then emit a message inside the handler, or create an exception with `raise ...`. Adding a colon does not make an exception instance a valid matching type. The original PDF is retained unchanged.
> <!-- bilingual-en:end -->

## Review checklist
- [ ] 我能解释 exception 的基本含义和它与普通“报错信息”的关系。
- [ ] 我能说明 `try` / `except` 的执行流程。
- [ ] 我能区分具体异常类型和裸 `except` 的差别。
- [ ] 我能说明什么时候应该吞掉异常、什么时候应该重新抛出。
- [ ] 我能解释 `raise ValueError(...)` 为什么是在表达接口边界。
- [ ] 我能解释 assertion 的作用以及它和 exception handling 的区别。
- [ ] 我能判断某个条件更适合写成 `assert` 还是更适合写成 `if ...: raise ...`。
- [ ] 我能把 finger exercise 13 与“主动 raise 异常”联系起来。
- [ ] 我能说出本讲不是为了“消灭错误”，而是让错误处理有语义。
- [ ] 我能按课堂顺序复述：认识异常 -> try/except -> specific exceptions -> raise -> assert。
<!-- bilingual-en:start -->
- [ ] I can explain what an exception means and how it relates to an ordinary error message.
- [ ] I can describe the control flow of `try`/`except`.
- [ ] I can distinguish catching a specific exception type from using a bare `except`.
- [ ] I can explain when an exception should be handled and when it should propagate.
- [ ] I can explain how `raise ValueError(...)` expresses an interface boundary.
- [ ] I can explain the purpose of an assertion and distinguish it from exception handling.
- [ ] I can decide whether a condition belongs in `assert` or in `if ...: raise ...`.
- [ ] I can connect finger exercise 13 to raising an exception deliberately.
- [ ] I can explain why the lecture seeks meaningful error handling rather than the elimination of all errors.
- [ ] I can reconstruct the lecture sequence: recognizing exceptions -> `try`/`except` -> specific exception types -> `raise` -> `assert`.
<!-- bilingual-en:end -->

> [!warning] Common mistakes
> - 用裸 `except` 把本应暴露的 bug 一起吞掉。
> - 看到异常就想强行继续运行，而不思考接口是不是已经被破坏。
> - 把 assertion 当成用户输入校验的唯一手段。
> - 不区分“调用者传错了参数”和“程序内部逻辑自己坏了”。
> <!-- bilingual-en:start -->
> - Using a bare `except` and thereby hiding bugs that should remain visible.
> - Trying to force execution to continue after every exception without asking whether the interface contract has already been broken.
> - Treating assertions as the sole mechanism for validating user input.
> - Failing to distinguish invalid arguments supplied by a caller from a defect in the program's own internal logic.
> <!-- bilingual-en:end -->
