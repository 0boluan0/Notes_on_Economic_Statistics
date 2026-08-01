---
aliases:
  - MIT 6.100L Lecture 07
  - 6.100L L07
  - Decomposition, Abstraction, and Functions
tags:
  - computer-science
  - python
  - mit-6.100l
  - lecture-note
科目: Computer Science
course: MIT 6.100L Introduction to CS and Programming Using Python
lecture: 07
---

# Lecture 07: Decomposition, Abstraction, and Functions

> [!tip] Hint
> - 这节课不是一上来写 `def`，而是先拿 smartphone 当黑箱，说明“会用”和“知道内部怎么实现”可以分开。
> - abstraction 先把实现细节藏起来，decomposition 才有可能把大系统拆给不同的人做。
> - 课程这里第一次强调 interface：别人只需要知道输入、输出和承诺，不需要知道你内部的循环和变量名。
> - docstring 在这节课不是装饰，而是函数的 specification，是写函数的人和调用者之间的 contract。
> - `def ...:` 只是语法外壳，真正重要的是你有没有把“这段动作”起一个稳定名字。
> - `is_even` 这种函数看起来很小，但它第一次把“判断偶数”从一段代码变成了一个可复用部件。
> - 老师反复在讲 return value，而不是只讲“打印出来”；函数是为了把结果交给别的代码继续用。
> - `sum_odd` 的 for 版和 while 版在提醒你：同一个 specification 可以有不同 implementation。
> - palindrome、keep_consonants、first_to_last_diff 这些例子都是在练“先想清楚接口，再决定怎么扫字符串”。
> - 听完这节课，最该记住的不是函数语法，而是“把重复逻辑命名并封装起来”这件事。
> <!-- bilingual-en:start -->
> - Rather than beginning immediately with `def`, the lecture uses a smartphone as a black box to separate knowing how to use something from knowing how it is implemented.
> - Abstraction hides implementation details behind a stable boundary; that boundary makes it possible to decompose a large system among several people or components.
> - The interface tells callers the required inputs, promised outputs, and behavior without exposing internal loops or variable names.
> - A docstring is not decoration here. It records the function's specification and acts as a contract between implementer and caller.
> - `def ...:` is only the syntactic shell; the deeper act is giving a stable name and interface to a coherent operation.
> - `is_even` is tiny, but it turns an inline evenness test into a reusable component.
> - The instructor emphasizes returned values rather than merely printed output because functions must make results available to subsequent code.
> - The `for` and `while` implementations of `sum_odd` show that one specification can be realized in different ways.
> - `is_palindrome`, `keep_consonants`, and `first_to_last_diff` all practice designing the interface before choosing how to scan the string.
> - The enduring lesson is not function syntax, but naming and encapsulating logic so that it can be reused.
> <!-- bilingual-en:end -->

## Lecture flow

### 1. 先从 smartphone 黑箱讲 abstraction
<!-- bilingual-en:start -->
*1. Beginning with the Smartphone as a Black Box*
<!-- bilingual-en:end -->
这节课的进入点不是代码，而是现实世界里的 black box。
<!-- bilingual-en:start -->
The entry point is a real-world black box rather than code.
<!-- bilingual-en:end -->

老师拿 smartphone 举例：  
大多数用户并不知道手机内部的电路、传感器、驱动、操作系统是怎么工作的，但这并不妨碍他们使用手机。用户真正需要知道的是：
<!-- bilingual-en:start -->
Most users do not know how a smartphone's circuits, sensors, drivers, or operating system work, yet can still use the device. They need to know:
<!-- bilingual-en:end -->

- 我做什么输入
- 系统给我什么输出
- 哪些按钮、滑动、触摸会触发哪些功能
<!-- bilingual-en:start -->
- Which inputs they provide.
- Which outputs the system produces.
- Which buttons, swipes, and touches invoke which behavior.
<!-- bilingual-en:end -->

这就是 **abstraction** 的第一层直觉：
<!-- bilingual-en:start -->
This gives the first intuition for abstraction:
<!-- bilingual-en:end -->

> [!note]
> abstraction 不是“忽略细节”，而是把不属于当前使用者职责的细节藏到界面后面。
> <!-- bilingual-en:start -->
> Abstraction does not deny that details exist; it hides details outside the current user's responsibility behind an interface.
> <!-- bilingual-en:end -->

对用户来说，手机的实现细节被隐藏起来了；  
对程序员来说，函数内部的实现细节也可以被隐藏起来。
<!-- bilingual-en:start -->
The phone hides implementation details from its user; a function can hide its implementation from its caller in the same way.
<!-- bilingual-en:end -->

### 2. abstraction 之后，才能谈 decomposition
<!-- bilingual-en:start -->
*2. Abstraction Makes Decomposition Possible*
<!-- bilingual-en:end -->
老师接着把手机例子推进到制造过程。
<!-- bilingual-en:start -->
The instructor extends the smartphone example to its manufacture.
<!-- bilingual-en:end -->

如果一个系统足够复杂，不可能由一个人从头到尾完成。现实中的做法是：
<!-- bilingual-en:start -->
A sufficiently complex system cannot be built from end to end by one person. In practice:
<!-- bilingual-en:end -->

- 不同团队负责不同组件
- 每个组件只需要遵守接口规范
- 最后再把这些组件拼起来
<!-- bilingual-en:start -->
- Different teams own different components.
- Every component adheres to an interface specification.
- The components are then assembled into the full system.
<!-- bilingual-en:end -->

这就是 **decomposition**。
<!-- bilingual-en:start -->
This is called **decomposition**.
<!-- bilingual-en:end -->

这里的课堂重点不是“拆分”两个字本身，而是：
<!-- bilingual-en:start -->
The important point is not merely “splitting” work:
<!-- bilingual-en:end -->

- 没有 abstraction，就没有稳定接口
- 没有稳定接口，就没法把工作拆给别人做
- 所以 abstraction 和 decomposition 是配套出现的
<!-- bilingual-en:start -->
- Without abstraction, there is no stable interface.
- Without a stable interface, components cannot be delegated and combined reliably.
- Abstraction and decomposition therefore work together.
<!-- bilingual-en:end -->

老师在这里其实已经在为函数做铺垫了：  
函数就是程序里最基本的 decomposition 单位。
<!-- bilingual-en:start -->
This prepares the idea of a function as the program's basic unit of decomposition.
<!-- bilingual-en:end -->

### 3. 从现实黑箱转到代码黑箱：函数就是可命名的部件
<!-- bilingual-en:start -->
*3. From Real-World Black Boxes to Named Program Components*
<!-- bilingual-en:end -->
进入编程语境后，老师把前面那套逻辑直接落到函数上。
<!-- bilingual-en:start -->
The same reasoning now transfers directly to functions.
<!-- bilingual-en:end -->

一个函数最重要的价值不是“把几行代码包起来”，而是：
<!-- bilingual-en:start -->
A function's principal value is not merely wrapping several lines of code, but:
<!-- bilingual-en:end -->

- 给一个动作起名字
- 规定它接受什么输入
- 规定它产生什么输出
- 让调用者不必关心内部细节
<!-- bilingual-en:start -->
- Naming an operation.
- Specifying the inputs it accepts.
- Specifying the output it produces.
- Allowing callers to ignore its internal details.
<!-- bilingual-en:end -->

所以函数让程序从“从上到下的一长段脚本”变成“若干可组合的部件”。
<!-- bilingual-en:start -->
Functions turn a long top-to-bottom script into a collection of composable components.
<!-- bilingual-en:end -->

这一点和前面几讲最大的区别是：
<!-- bilingual-en:start -->
This changes the course's concern:
<!-- bilingual-en:end -->

- 之前我们主要在写单段算法
- 这节课开始，课程要你学会组织程序
<!-- bilingual-en:start -->
- Earlier lectures mainly wrote one algorithmic block at a time.
- This lecture begins the organization of a program into parts.
<!-- bilingual-en:end -->

### 4. `def` 语法只是外壳，docstring 才是 contract
<!-- bilingual-en:start -->
*4. `def` Is the Shell; the Docstring States the Contract*
<!-- bilingual-en:end -->
老师随后正式拆开函数定义的结构。
<!-- bilingual-en:start -->
The instructor then decomposes a function definition itself.
<!-- bilingual-en:end -->

一个典型函数定义包含：
<!-- bilingual-en:start -->
A typical definition contains:
<!-- bilingual-en:end -->

- `def`：告诉 Python 我现在要定义函数
- 函数名：给这个动作一个名字
- 参数列表：声明调用者必须提供哪些输入
- 冒号和缩进块：函数体
- docstring：写 specification
<!-- bilingual-en:start -->
- `def`, which begins a function definition.
- A function name, which names the operation.
- A parameter list, which identifies the inputs supplied by the caller.
- A colon and indented function body.
- A docstring, which states the specification.
<!-- bilingual-en:end -->

课堂里的核心观念是 docstring 的角色。它不是“可有可无的注释”，而是：
<!-- bilingual-en:start -->
The docstring is not an optional aside. It states:
<!-- bilingual-en:end -->

- 输入类型和前提条件
- 输出是什么
- 函数完成什么任务
<!-- bilingual-en:start -->
- Input types and preconditions.
- The returned output.
- The task the function performs.
<!-- bilingual-en:end -->

如果把它写成一句更实用的话，就是：
<!-- bilingual-en:start -->
In practical terms:
<!-- bilingual-en:end -->

> [!note]
> 函数名负责“这是什么动作”，docstring 负责“调用这个动作时你能依赖什么”。
> <!-- bilingual-en:start -->
> The function name answers “What operation is this?” The docstring answers “What may a caller rely on?”
> <!-- bilingual-en:end -->

### 5. 第一个完整例子：`is_even`
<!-- bilingual-en:start -->
*5. First Complete Example: `is_even`*
<!-- bilingual-en:end -->
老师用 `is_even(i)` 做第一个完整例子，因为它足够简单，能把函数最核心的结构暴露出来。
<!-- bilingual-en:start -->
`is_even(i)` is simple enough to expose the essential structure of a function without distracting algorithmic complexity.
<!-- bilingual-en:end -->

```python
def is_even(i):
    """Assumes i, a positive int
    Returns True if i is even, otherwise False"""
    if i % 2 == 0:
        return True
    else:
        return False
```

这里真正值得记的不是“偶数怎么判断”，而是：
<!-- bilingual-en:start -->
The important lesson is not the evenness test itself:
<!-- bilingual-en:end -->

- 参数 `i` 是输入占位符
- 函数体内部可以使用局部变量和分支
- `return` 会把结果交还给调用点
<!-- bilingual-en:start -->
- The parameter `i` is a placeholder for an input.
- The body may contain local computations and branches.
- `return` sends the result back to the call site.
<!-- bilingual-en:end -->

老师在这里也顺手提醒了 definition 和 call 的区别：
<!-- bilingual-en:start -->
The example also separates definition from invocation:
<!-- bilingual-en:end -->

- 写 `def is_even(i): ...` 是在定义函数
- 写 `is_even(3)` 才是在调用函数
<!-- bilingual-en:start -->
- `def is_even(i): ...` defines the function.
- `is_even(3)` calls it.
<!-- bilingual-en:end -->

### 6. 函数的第一价值：复用同一个判断
<!-- bilingual-en:start -->
*6. The First Benefit of a Function: Reusing the Same Test*
<!-- bilingual-en:end -->
写完 `is_even` 之后，老师马上把它放回更大的程序里使用。
<!-- bilingual-en:start -->
The instructor immediately uses `is_even` inside a larger program.
<!-- bilingual-en:end -->

例如判断 `1` 到 `10` 中每个数字是 even 还是 odd。  
如果没有函数，你会把“余数是否为 0”的逻辑反复写在循环里面；  
有了函数之后，主程序只需要关心更高层的表达：
<!-- bilingual-en:start -->
To classify the numbers from `1` to `10`, an inline program would repeat remainder logic inside the loop. With the function, the main program can state the higher-level question directly:
<!-- bilingual-en:end -->

```python
for i in range(1, 10):
    if is_even(i):
        print(i, "even")
    else:
        print(i, "odd")
```

这时主程序读起来已经更像人类语言：
<!-- bilingual-en:start -->
The main program now reads closer to its intended meaning:
<!-- bilingual-en:end -->

- 对每个数
- 判断它是不是偶数
- 再决定打印什么
<!-- bilingual-en:start -->
- For each number, ask whether it is even.
- Then choose the appropriate output.
<!-- bilingual-en:end -->

函数把低层判断细节折叠掉了。
<!-- bilingual-en:start -->
The function hides the lower-level remainder test.
<!-- bilingual-en:end -->

### 7. 第一个 you-try-it：`div_by` 训练的是“按 specification 写函数”
<!-- bilingual-en:start -->
*7. First “You Try It”: Implementing `div_by` from Its Specification*
<!-- bilingual-en:end -->
老师接着给了一个非常短的练习：`div_by(n, d)`。
<!-- bilingual-en:start -->
The instructor next gives the short exercise `div_by(n, d)`.
<!-- bilingual-en:end -->

题目看起来只是在考 `%` 运算，但课堂真正要你训练的是：
<!-- bilingual-en:start -->
Although it uses `%`, its real purpose is to practice:
<!-- bilingual-en:end -->

- 先读 specification
- 再把 specification 翻译成条件判断
- 最后决定 return 什么
<!-- bilingual-en:start -->
- Reading the specification first.
- Translating it into a Boolean condition.
- Returning the promised result.
<!-- bilingual-en:end -->

最直接的写法是：
<!-- bilingual-en:start -->
The direct implementation is:
<!-- bilingual-en:end -->

```python
def div_by(n, d):
    """n and d are ints > 0
    Returns True if d divides n evenly and False otherwise"""
    return n % d == 0
```

> [!example]
> 这个练习很短，但它建立了一个重要习惯：  
> 如果 docstring 已经把行为说清楚，函数体经常只是把一句自然语言翻译成一两行布尔表达式。
> <!-- bilingual-en:start -->
> This short exercise establishes an important habit: when the docstring states the behavior precisely, the body may be only a one-line Boolean translation of that statement.
> <!-- bilingual-en:end -->

### 8. `sum_odd` 把“同一任务，不同实现”这件事讲清楚
<!-- bilingual-en:start -->
*8. `sum_odd`: One Task Can Have Several Implementations*
<!-- bilingual-en:end -->
接下来老师把例子升级到一个更完整的函数：  
求 `a` 到 `b` 之间所有 odd numbers 的和。
<!-- bilingual-en:start -->
The next, fuller function sums all odd numbers between `a` and `b`.
<!-- bilingual-en:end -->

课堂先给出一个版本，再给出另一个版本，目的是让你看到：
<!-- bilingual-en:start -->
Two versions demonstrate that:
<!-- bilingual-en:end -->

- specification 相同
- implementation 可以不同
<!-- bilingual-en:start -->
- The specification can remain the same.
- The implementation can change.
<!-- bilingual-en:end -->

这对后面算法比较很重要。你不能把“题目要做什么”和“我这次刚好怎么写”混为一谈。
<!-- bilingual-en:start -->
This distinction matters for later algorithm comparisons: what the function promises is not the same as how one particular version fulfills that promise.
<!-- bilingual-en:end -->

课程代码里同时出现了 `for` 版和 `while` 版。真正需要记住的是这类函数的思维流程：
<!-- bilingual-en:start -->
The course provides both `for` and `while` versions. Their shared reasoning is:
<!-- bilingual-en:end -->

1. 初始化累计变量
2. 依次访问候选元素
3. 用条件筛出符合要求的元素
4. 更新累计结果
5. return 最终值
<!-- bilingual-en:start -->

&nbsp;
**1.** Initialize an accumulator.<br>
**2.** Visit the candidate values.<br>
**3.** Select the values satisfying the condition.<br>
**4.** Update the accumulated result.<br>
**5.** Return the final value.<br>
<!-- bilingual-en:end -->

### 9. `return` 的位置决定函数什么时候结束
<!-- bilingual-en:start -->
*9. The Position of `return` Determines When a Call Ends*
<!-- bilingual-en:end -->
虽然 Lecture 8 会更系统地讲 `return`，但这节课里已经埋下了一个关键点：
<!-- bilingual-en:start -->
Lecture 8 develops `return` more fully, but this lecture already establishes that:
<!-- bilingual-en:end -->

- 一旦执行到 `return`
- 当前函数调用就结束
- 结果被送回调用处
<!-- bilingual-en:start -->
- Executing `return` ends the current call immediately.
- The returned value is sent to the caller.
<!-- bilingual-en:end -->

所以函数设计时要想清楚：
<!-- bilingual-en:start -->
Function design must therefore decide:
<!-- bilingual-en:end -->

- 什么时候已经得到最终答案
- 哪些路径应该提前结束
- 哪些路径应该继续扫描
<!-- bilingual-en:start -->
- When the final answer is already known.
- Which paths should terminate early.
- Which paths must continue scanning.
<!-- bilingual-en:end -->

这一点在后面的 palindrome 中会变得很直观。
<!-- bilingual-en:start -->
The palindrome example makes this concrete.
<!-- bilingual-en:end -->

### 10. palindrome：第一次认真练“提前发现反例就返回”
<!-- bilingual-en:start -->
*10. Palindromes: Returning Early on the First Counterexample*
<!-- bilingual-en:end -->
课堂后半段的字符串例子是 `is_palindrome(s)`。
<!-- bilingual-en:start -->
The second half uses the string function `is_palindrome(s)`.
<!-- bilingual-en:end -->

它的典型思路不是“把字符串倒过来”这种捷径，而是：
<!-- bilingual-en:start -->
Rather than reverse the string, the classroom approach is to:
<!-- bilingual-en:end -->

- 只检查前半段
- 把左边第 `i` 个字符和右边对称位置比较
- 一旦发现不一样，立刻 `return False`
- 如果一路都没出错，最后 `return True`
<!-- bilingual-en:start -->
- Inspect only the first half.
- Compare each left-side character with its symmetric partner on the right.
- `return False` at the first mismatch.
- `return True` only after all required comparisons pass.
<!-- bilingual-en:end -->

```python
def is_palindrome(s):
    for i in range(len(s) // 2):
        if s[i] != s[len(s) - i - 1]:
            return False
    return True
```

这里课堂在训练三件事：
<!-- bilingual-en:start -->
The example practices:
<!-- bilingual-en:end -->

- 如何把 index 运算写对
- 如何利用对称性少做一半工作
- 如何用 early return 让逻辑更干净
<!-- bilingual-en:start -->
- Correct symmetric indexing.
- Using symmetry to avoid half of the work.
- Using an early return to express the logic cleanly.
<!-- bilingual-en:end -->

### 11. 课后字符串练习继续强化 decomposition
<!-- bilingual-en:start -->
*11. Follow-Up String Exercises Reinforce Decomposition*
<!-- bilingual-en:end -->
老师最后又给了两个 at-home 风格的函数：
<!-- bilingual-en:start -->
The lecture ends with two take-home-style functions:
<!-- bilingual-en:end -->

- `keep_consonants(word)`
- `first_to_last_diff(s, c)`

它们不像前面的例子那样只是讲语法，而是在要求你把一个模糊目标拆成清楚步骤。
<!-- bilingual-en:start -->
They require a vague objective to be decomposed into explicit steps rather than merely exercising syntax.
<!-- bilingual-en:end -->

例如 `keep_consonants` 的自然拆法就是：
<!-- bilingual-en:start -->
For `keep_consonants`, a natural decomposition is:
<!-- bilingual-en:end -->

1. 先定义什么算 vowel
2. 建一个空字符串作为答案
3. 逐字符扫描输入
4. 只把 consonant 接到答案里
<!-- bilingual-en:start -->

&nbsp;
**1.** Define which characters count as vowels.<br>
**2.** Initialize an empty output string.<br>
**3.** Traverse the input one character at a time.<br>
**4.** Append only consonants to the output.<br>
<!-- bilingual-en:end -->

而 `first_to_last_diff` 更像是训练“先找第一个，再找最后一个，再组合结果”这种程序分解能力。
<!-- bilingual-en:start -->
`first_to_last_diff` similarly decomposes into finding the first occurrence, finding the last, and combining those results.
<!-- bilingual-en:end -->

### 12. 这节课真正完成了什么
<!-- bilingual-en:start -->
*12. What the Lecture Actually Accomplishes*
<!-- bilingual-en:end -->
如果把 Lecture 7 压缩成一句话，它做的不是“介绍新语法”，而是：
<!-- bilingual-en:start -->
Compressed to one sentence, Lecture 7 is not mainly new syntax:
<!-- bilingual-en:end -->

> [!note]
> 让你第一次把程序看成由若干部件组成，而函数是这些部件最基本的封装形式。
> <!-- bilingual-en:start -->
> It introduces a program as a composition of parts, with functions as the basic unit of encapsulation.
> <!-- bilingual-en:end -->

从这节课开始，后面所有内容都会默认你接受这套思路：
<!-- bilingual-en:start -->
Later material assumes this workflow:
<!-- bilingual-en:end -->

- 先说清楚接口
- 再隐藏实现
- 然后复用部件
<!-- bilingual-en:start -->
- State the interface clearly.
- Hide the implementation behind it.
- Reuse and compose the resulting component.
<!-- bilingual-en:end -->

## Exercise log

> [!example] Finger exercise 07
> 官方练习分成两步：
> - `eval_quadratic(a, b, c, x)`：返回二次式 `a*x^2 + b*x + c` 的值。
> - `two_quadratics(...)`：分别计算两个二次式，并 `print` 它们的和，不返回值。
> <!-- bilingual-en:start -->
> The official exercise has two parts:
> - `eval_quadratic(a, b, c, x)` returns the value of `a*x^2 + b*x + c`.
> - `two_quadratics(...)` evaluates two quadratics and prints their sum without returning a value.
> <!-- bilingual-en:end -->

这套题很适合放在本讲后面，因为它正好检查你有没有真的分清下面三件事：
<!-- bilingual-en:start -->
The pair checks whether three roles remain distinct:
<!-- bilingual-en:end -->

- 一个函数负责 **计算并返回值**
- 另一个函数负责 **调用已有函数并组织结果**
- `print` 和 `return` 的语义并不相同
<!-- bilingual-en:start -->
- One function **computes and returns a value**.
- Another **calls an existing function and organizes its results**.
- `print` and `return` have different semantics.
<!-- bilingual-en:end -->

第一问几乎只是在检查你会不会把 specification 准确翻译成表达式。  
第二问开始要求你把“已有函数当部件”重新拼起来，这正是 decomposition 的核心。
<!-- bilingual-en:start -->
The first task translates a specification into an expression. The second composes an existing function into a larger task, which is the essence of decomposition.
<!-- bilingual-en:end -->

如果第二问做得别扭，通常不是二次函数不会算，而是这两个概念还没分开：
<!-- bilingual-en:start -->
If the second part feels awkward, the difficulty is often not quadratics but the distinction between:
<!-- bilingual-en:end -->

- “一个函数自己完成所有计算”
- “一个函数调用别的函数来完成更大的任务”
<!-- bilingual-en:start -->
- A function that performs every calculation itself.
- A function that delegates part of a larger task to another function.
<!-- bilingual-en:end -->

## Links to follow-up practice
- Slides: [[MIT 6.100L-slides/mit6_100l_lec07.pdf|Lecture 07 slides]]
- Lecture code: [[MIT 6.100L-lecture-code/mit6_100l_lec07_code.py|Lecture 07 code (py)]]
- Finger exercise: [[MIT 6.100L-finger-exercises/mit6_100l_ex07_sol.pdf|Lecture 07 finger exercise solution]]
- Transcript: [[MIT 6.100L-transcripts/mit6_100l_lec07_transcript.pdf|Lecture 07 transcript]]
- Recitation 3: [[MIT 6.100L-recitations/mit6_100l_rec03.zip|Recitation 03 materials]]
- Problem set milestone: none directly scheduled on this lecture
- Textbook: [[Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Guttag textbook]] (Ch 4.1-4.2)

## Review checklist
- [ ] 我能用 smartphone 黑箱例子解释 abstraction 和 decomposition 的关系。
- [ ] 我能说明为什么调用函数的人只需要 interface，不需要知道内部实现。
- [ ] 我能说出函数定义里函数名、参数、docstring、body、return 各自承担什么角色。
- [ ] 我能解释 specification 和 implementation 为什么不能混在一起理解。
- [ ] 我能手写 `is_even` 这类最小函数，并解释为什么它比把判断逻辑写死在循环里更好。
- [ ] 我能说明同一个 specification 为什么可以有不同实现，比如 `sum_odd` 的 for 版和 while 版。
- [ ] 我能解释 palindrome 例子里为什么只需要扫描前半段。
- [ ] 我能说明什么时候应该 `return False` 提前结束，而不是一直把循环跑完。
- [ ] 我能把 finger exercise 07 的两问联系到“函数复用”和“print/return 区分”上，而不是只把它当算式题。
- [ ] 我能不看 slides，只根据这份笔记把整节课的推进顺序讲出来。
<!-- bilingual-en:start -->
- [ ] I can explain the relationship between abstraction and decomposition using the smartphone black box example.
- [ ] I can explain why someone calling a function only needs the interface, not the internal implementation.
- [ ] I can identify the roles of function name, parameters, docstring, body, and return in a function definition.
- [ ] I can explain why a specification must be distinguished from its implementation.
- [ ] I can write a minimal function such as `is_even` and explain why encapsulating its logic is preferable to repeating it inside a loop.
- [ ] I can explain why the same specification can have different implementations, such as the `sum_odd` for-loop version and while-loop version.
- [ ] I can explain why only the first half needs to be scanned in the palindrome example.
- [ ] I can explain when it is better to `return False` early rather than running the loop until completion.
- [ ] I can connect the two parts of finger exercise 07 to function reuse and the distinction between `print` and `return`.
- [ ] I can reconstruct the lecture's progression from these notes without looking at the slides.
<!-- bilingual-en:end -->

> [!warning] Common mistakes
> - 把函数理解成“缩进起来的代码块”，却没有真正写清楚接口。
> - 写了 docstring，但函数体并没有兑现 docstring 里的承诺。
> - 需要返回结果时只 `print`，导致后续代码拿不到值。
> - 还没想清楚 specification 就直接开写，最后函数名、参数和行为互相打架。
> <!-- bilingual-en:start -->
> - Treating a function as merely an indented block without defining its interface.
> - Writing a docstring whose promises the body does not fulfill.
> - Using only `print` when a result must be returned for subsequent code.
> - Coding before clarifying the specification, leaving the name, parameters, and behavior inconsistent.
> <!-- bilingual-en:end -->
