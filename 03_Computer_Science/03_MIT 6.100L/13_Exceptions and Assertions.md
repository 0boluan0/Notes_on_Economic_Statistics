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

> [!tip] Hint
> - 这节课开场先从 “scary red errors” 讲起，目标不是怕错误，而是学会把错误纳入程序逻辑。
> - 异常是程序遇到 unexpected condition 时抛出的信号，不同错误类型对应不同异常。
> - `try`/`except` 的课堂语义很明确：先试正常路径，出错了再走备用处理逻辑。
> - `except ValueError` 和裸 `except` 的区别，是“你到底想捕获什么问题”。
> - 这节课反复强调：不是所有错误都该被吞掉，有时应该返回默认值，有时应该重新 raise。
> - `raise ValueError(...)` 的意义是把“这个输入不符合我函数约定”明确表达出来。
> - assertions 也是异常，但更像是给程序员自己的内在假设加护栏。
> - `assert condition` 失败时抛的是 `AssertionError`，适合放在“不该被破坏的前提”上。
> - exceptions 是处理运行时异常情况，assertions 更偏向开发时抓逻辑违例。
> - 听完这节课，你应该能判断某个问题该 try/except、该 raise、还是该 assert。
> <!-- bilingual-en:start -->
> - The lecture opens with “scary red errors.” Its goal is not to make errors frightening, but to bring them into the program's logic.
> - An exception is a signal that the program has encountered an unexpected condition; different kinds of failure produce different exception types.
> - The classroom meaning of `try`/`except` is straightforward: attempt the normal path first, then use a fallback path if an error occurs.
> - The difference between `except ValueError` and a bare `except` is the difference between naming the failure you intend to handle and indiscriminately catching everything.
> - The lecture repeatedly stresses that not every error should be suppressed. Sometimes a default value is appropriate; at other times the exception should be raised again.
> - `raise ValueError(...)` explicitly states that an input violates the function's contract.
> - Assertions are also implemented through exceptions, but they chiefly guard assumptions made by the programmer.
> - A failed `assert condition` raises `AssertionError`; use it for an invariant that should never have been broken.
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
如果 try 里抛了异常，Python 会跳出 try，转去执行对应的 except。
<!-- bilingual-en:start -->
If the `try` block raises no exception, the `except` block is skipped. If an exception is raised, Python leaves the `try` block and transfers control to the matching handler.
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

### 7. `pairwise_div`：异常处理可以保护整个循环继续跑
<!-- bilingual-en:start -->
*7. `pairwise_div`: Exception Handling Can Keep a Loop Running*
<!-- bilingual-en:end -->
老师接着用列表分母相除之类的例子说明：

- 如果没有异常处理，循环中途遇到一个坏元素可能整个函数就崩掉
- 加了 try/except，你可以选择记录问题、跳过坏输入，或对该位置给默认值
<!-- bilingual-en:start -->
Using an example such as pairwise division of list elements, the instructor shows that one invalid element can terminate the entire function without exception handling. With `try`/`except`, the program can record the problem, skip the invalid input, or place a default value at that position.
<!-- bilingual-en:end -->

这时异常处理的一个实际价值就很明显了：

- 它不只是“报错更好看”
- 而是能决定整个程序是“全盘崩掉”还是“局部容错继续工作”
<!-- bilingual-en:start -->
The value of exception handling is therefore not prettier error output. It determines whether one local failure brings down the whole computation or is contained so that useful work can continue.
<!-- bilingual-en:end -->

### 8. 什么时候应该 `raise` 自己的异常
<!-- bilingual-en:start -->
*8. When to `raise` an Exception Yourself*
<!-- bilingual-en:end -->
讲到这里，老师把方向再推进一步：

- 你不只会“接住” Python 自带异常
- 你还可以主动 `raise` 自己的异常
<!-- bilingual-en:start -->
The lecture now moves from catching built-in exceptions to raising an exception deliberately.
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

如果某个输入违背了函数承诺，你通常有两种选择：

- 悄悄给个默认值，继续跑
- 明确拒绝它，并 raise 一个异常
<!-- bilingual-en:start -->
When an input violates the function's promise, the implementation can either continue quietly with a default or reject the input explicitly by raising an exception.
<!-- bilingual-en:end -->

当错误真的代表“调用者违反了接口约定”时，后者往往更合适。  
这让函数接口边界更清楚，也让 bug 更早暴露出来。
<!-- bilingual-en:start -->
Explicit rejection is often preferable when the caller has violated the contract. It sharpens the function boundary and exposes bugs earlier.
<!-- bilingual-en:end -->

### 10. assertions：给程序自己的假设加护栏
<!-- bilingual-en:start -->
*10. Assertions: Guardrails for the Program's Own Assumptions*
<!-- bilingual-en:end -->
讲完 exceptions 之后，老师再引入 assertions。

assertion 也是一种异常机制，但角度不一样。  
它的核心语句是：
<!-- bilingual-en:start -->
After exceptions, the instructor introduces assertions. They also use the exception mechanism, but serve a different purpose. Their central statement is:
<!-- bilingual-en:end -->

```python
assert condition
```

如果 `condition` 为真，什么都不发生；  
如果为假，就抛出 `AssertionError`。
<!-- bilingual-en:start -->
If `condition` is true, execution continues unchanged; if it is false, Python raises `AssertionError`.
<!-- bilingual-en:end -->

课堂里老师把它解释成：

- 程序员在代码中声明“这里我认为某个条件必须成立”
- 一旦不成立，立刻暴露
<!-- bilingual-en:start -->
An assertion records a programmer's claim that a condition must hold at that point and makes any violation fail immediately.
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

### 12. exceptions 和 assertions 的课堂分工
<!-- bilingual-en:start -->
*12. The Classroom Division of Labor Between Exceptions and Assertions*
<!-- bilingual-en:end -->
老师后面其实一直在帮助大家区分这两套机制：

- exceptions：处理运行时可能出现的异常情况
- assertions：检查程序内部本不该被破坏的假设
<!-- bilingual-en:start -->
- Exceptions handle exceptional conditions that may arise at runtime.
- Assertions check internal assumptions that should never be violated.
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
> - 如果出现非字符串或非列表元素，要 `raise ValueError`
> <!-- bilingual-en:start -->
> The official task is `sum_str_lengths(L)`:
> - `L` is a nonempty list.
> - Each element is either a string or a nonempty sublist of strings.
> - The function returns the total length of all strings.
> - It must `raise ValueError` if it encounters an element that is neither a string nor a list.
> <!-- bilingual-en:end -->

这题正好卡在本讲的核心点上：

- 你不只是遍历和计数
- 你还要在发现非法结构时主动抛异常
<!-- bilingual-en:start -->
This exercise sits directly on the lecture's central idea: besides traversing and counting, you must raise an exception when the input structure is invalid.
<!-- bilingual-en:end -->

官方思路就是：

- 遇到 `str` 就累加长度
- 遇到 `list` 就继续检查其元素
- 遇到别的类型就 `raise ValueError`
<!-- bilingual-en:start -->
The official approach adds the length of a `str`, inspects the elements of a `list`, and calls `raise ValueError` for any other type.
<!-- bilingual-en:end -->

所以这题本质上在训练你把“输入前提”写成代码里的显式错误路径。
<!-- bilingual-en:start -->
In essence, the exercise turns an input precondition into an explicit error path in the code.
<!-- bilingual-en:end -->

## Links to follow-up practice
- Slides: [[MIT 6.100L-slides/mit6_100l_lec13.pdf|Lecture 13 slides]]
- Lecture code: [[MIT 6.100L-lecture-code/mit6_100l_lec13_code.py|Lecture 13 code (py)]]
- Finger exercise: [[MIT 6.100L-finger-exercises/mit6_100l_ex13_sol.pdf|Lecture 13 finger exercise solution]]
- Transcript: [[MIT 6.100L-transcripts/mit6_100l_lec13_transcript.pdf|Lecture 13 transcript]]
- Recitation 7: [[MIT 6.100L-recitations/mit6_100l_rec07.zip|Recitation 07 materials]]
- Problem set milestone: none directly scheduled on this lecture
- Textbook: [[Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Guttag textbook]] (Ch 9)

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
