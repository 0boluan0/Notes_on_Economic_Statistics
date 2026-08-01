---
aliases:
  - MIT 6.100L Lecture 01
  - 6.100L L01
  - Introduction
tags:
  - computer-science
  - python
  - mit-6.100l
  - lecture-note
科目: Computer Science
course: MIT 6.100L Introduction to CS and Programming Using Python
lecture: 01
---

# Lecture 01: Introduction

> [!tip] Hint
> - 这节课开头先讲的不是 Python 语法，而是为什么这门课必须把 lecture、finger exercise、problem set 当成三种不同训练。
> - Ana Bell 先把课程目标定成 computational thinking：以后看到问题，先想能不能交给 computation，而不是先手算。
> - declarative knowledge 只是陈述事实；imperative knowledge 才是机器能执行的 recipe。
> - 平方根例子不是为了教开方，而是第一次展示 guess、update、close enough、repeat 这些程序思维。
> - 一个算法至少要能回答三件事：做哪些步骤、什么时候改变流程、什么时候停。
> - “The computer only does what you tell it to do” 是本讲最重要的句子之一，后面大部分 bug 都能回到这里理解。
> - 老师先讲 fixed-program vs stored-program computer，再讲 interpreter、memory、ALU、control unit，是为了让你知道高级语言省掉了多少机械细节。
> - English 和 Python 的类比，重点不在语言学，而在 primitive、syntax、static semantics、semantics 这四层。
> - 今天真正进入 Python 时，顺序是 shell -> object -> type -> cast -> expression -> variable binding，而不是先背一堆语法清单。
> - 结尾的 `radius = radius + 1` 和 swap bug 都在强调同一件事：`=` 是 assignment，不是数学等号；程序逐行执行，不会自动替你“理解上下文”。
> <!-- bilingual-en:start -->
> - The lecture begins not with Python syntax, but with why lectures, finger exercises, and problem sets train different aspects of programming.
> - Ana Bell frames the goal as computational thinking: when facing a problem, first ask whether its solution can be expressed as a computation rather than reaching immediately for manual calculation.
> - Declarative knowledge states a fact; imperative knowledge supplies an executable recipe.
> - The square-root example is not primarily about roots. It first introduces the programmatic structure of a guess, an update, a close-enough test, and repetition.
> - An algorithm must at least state the steps, specify when control changes, and determine when execution stops.
> - “The computer only does what you tell it to do” is one of the lecture's central claims and explains many later bugs.
> - The progression from fixed-program and stored-program computers to the interpreter, memory, ALU, and control unit reveals how much mechanical detail a high-level language hides.
> - The analogy between English and Python introduces four levels: primitives, syntax, static semantics, and semantics.
> - The first Python material proceeds from shell to object, type, cast, expression, and name binding rather than presenting a disconnected syntax list.
> - `radius = radius + 1` and the closing swap bug emphasize the same model: `=` is assignment, not mathematical equality, and execution proceeds line by line without inferring the programmer's unstated intent.
> <!-- bilingual-en:end -->

## Lecture flow

### 1. 开场先说明这门课怎么学，而不是只说“学什么”
<!-- bilingual-en:start -->
*1. Beginning with How the Course Is Learned, Not Only What It Covers*
<!-- bilingual-en:end -->
老师一开始先花了一点时间讲课程运行方式，但这段其实很重要，因为它决定了你后面应该怎么用这门课的材料。
<!-- bilingual-en:start -->
The opening explanation of course structure determines how each kind of material should be used.
<!-- bilingual-en:end -->

- lecture 负责建立概念直觉，让你看见一个概念为什么会被发明出来。
- finger exercise 负责把最小代码动作练熟，不让你只停留在“听懂了”。
- problem set 负责把英文题意翻译成计算问题，再翻译成程序。
<!-- bilingual-en:start -->
- Lectures build conceptual intuition and show why an idea is needed.
- Finger exercises make the smallest coding actions fluent rather than merely familiar.
- Problem sets translate a verbal task into a computational problem and then into a program.
<!-- bilingual-en:end -->

> [!note]
> 这一讲从一开始就在强调：编程是技能，不是纯观看型知识。你不能只看老师敲代码，就默认自己已经会写。
> <!-- bilingual-en:start -->
> Programming is a practiced skill, not knowledge acquired by observation alone. Watching an instructor type code does not establish the ability to write it.
> <!-- bilingual-en:end -->

老师还提前告诉你两件事：

- 课堂里会不断出现 “you try it” 的停顿，这不是插曲，而是课程设计的一部分。
- 这门课的最终目标不只是会写 Python，而是获得 `knowledge of concepts`、`programming skill` 和 `problem solving` 三层能力。
<!-- bilingual-en:start -->
The recurring “you try it” pauses are deliberate parts of the course design. The final goal includes conceptual knowledge, programming skill, and problem solving—not Python syntax alone.
<!-- bilingual-en:end -->

### 2. 课程主轴先落在 computation，而不是某条 Python 语法
<!-- bilingual-en:start -->
*2. Centering Computation Rather Than a Particular Python Feature*
<!-- bilingual-en:end -->
正式进入内容后，老师先问的其实是“这门课到底在做什么”。答案不是“学一个编程语言”，而是学会用 computation 解决问题。
<!-- bilingual-en:start -->
The course first asks what it is fundamentally doing. The answer is learning to solve problems through computation, not merely learning one programming language.
<!-- bilingual-en:end -->

她给出的课程主题顺序很清楚：

1. computational thinking
2. Python programming language
3. 组织更整洁、更模块化的代码
4. 一些基础但重要的 algorithms
5. algorithmic complexity
<!-- bilingual-en:start -->
Its themes progress through computational thinking, Python, cleaner and more modular program organization, foundational algorithms, and algorithmic complexity.
<!-- bilingual-en:end -->

这段的重点在于心态切换。以后碰到问题时，第一反应不该只是“我能不能手算出来”，而应该是“我能不能把求解过程写成机器能执行的步骤”。
<!-- bilingual-en:start -->
The required shift is from asking whether a result can be calculated by hand to asking whether the solution process can be written as steps executable by a machine.
<!-- bilingual-en:end -->

### 3. 先区分两种知识：declarative vs imperative
<!-- bilingual-en:start -->
*3. Distinguishing Declarative from Imperative Knowledge*
<!-- bilingual-en:end -->
真正进入计算机科学之前，老师先把 knowledge 分成两类：
<!-- bilingual-en:start -->
Before computer science proper, the instructor distinguishes two kinds of knowledge:
<!-- bilingual-en:end -->

- `declarative knowledge`：陈述事实
- `imperative knowledge`：给出做法
<!-- bilingual-en:start -->
- `declarative knowledge` states facts.
- `imperative knowledge` gives a procedure.
<!-- bilingual-en:end -->

数学里你很常见的是 declarative knowledge。例如：

> square root of x is y such that `y*y = x`
> <!-- bilingual-en:start -->
> The square root of `x` is the value `y` such that `y*y = x`.
> <!-- bilingual-en:end -->

这句话对人是有意义的，因为人知道它在定义什么；但对计算机来说，这句话只是一个事实陈述，没有告诉它下一步怎么做。
<!-- bilingual-en:start -->
The statement defines the answer for a person but gives a computer no next action.
<!-- bilingual-en:end -->

编程恰恰相反。你写程序时，给计算机的不是“答案的定义”，而是“得到答案的步骤”。所以老师说 programming is about writing recipes to generate facts。
<!-- bilingual-en:start -->
Programming supplies the steps that generate an answer rather than only its definition; in that sense, it writes recipes for producing facts.
<!-- bilingual-en:end -->

### 4. 用平方根例子第一次看到 algorithm 长什么样
<!-- bilingual-en:start -->
*4. The Square-Root Example Shows the Shape of an Algorithm*
<!-- bilingual-en:end -->
接下来老师没有抽象空谈，而是立刻拿平方根举例，把 declarative statement 转成机器可执行的 recipe。
<!-- bilingual-en:start -->
The instructor immediately turns the declarative square-root statement into an executable recipe.
<!-- bilingual-en:end -->

> [!example] 从“定义”走到“步骤”
> 已知要找 `sqrt(16)`，但计算机并不知道答案是 4。  
> 于是先给一个初始 guess，例如 `g = 3`：
> 1. 如果 `g*g` 已经离 16 足够近，就停下。
> 2. 否则把新猜测更新成 `(g + x/g) / 2`。
> 3. 用新 guess 重复这个过程。
> <!-- bilingual-en:start -->
> To find `sqrt(16)` without knowing that the answer is 4, begin with a guess such as `g = 3`:
> **1.** Stop if `g*g` is sufficiently close to 16.<br>
> **2.** Otherwise update the guess to `(g + x/g) / 2`.<br>
> **3.** Repeat using the new guess.<br>
> <!-- bilingual-en:end -->

老师现场按这个流程走了几轮：

- `g = 3`，`g*g = 9`，不够接近
- 新 guess 变成 `4.17`
- `4.17^2 = 17.36`，还不够接近
- 再更新，得到 `4.0035`
- `4.0035^2` 已经非常接近 16
<!-- bilingual-en:start -->
The trace moves from `g = 3` and a square of `9`, to `4.17` and approximately `17.36`, and then to `4.0035`, whose square is already very close to 16.
<!-- bilingual-en:end -->

这段是整讲最核心的第一个“程序视角”。
<!-- bilingual-en:start -->
This is the lecture's first central computational perspective.
<!-- bilingual-en:end -->

你要记住的不是数字，而是这种结构：

- 先有一个 guess
- 再有一个 update rule
- 再有一个 stopping condition
<!-- bilingual-en:start -->
The reusable structure is an initial guess, an update rule, and a stopping condition.
<!-- bilingual-en:end -->

也就是后来所有近似法、搜索法、优化法都会反复出现的基本骨架。
<!-- bilingual-en:start -->
Approximation, search, and optimization methods repeatedly reuse this skeleton.
<!-- bilingual-en:end -->

### 5. 由平方根例子抽出 algorithm 的三要素
<!-- bilingual-en:start -->
*5. Extracting Three Components of an Algorithm*
<!-- bilingual-en:end -->
老师随后把这个例子抽象成 algorithm 的定义。一个算法至少要有三部分：
<!-- bilingual-en:start -->
The square-root example is generalized into three required components:
<!-- bilingual-en:end -->

1. sequence of simple steps
2. flow of control
3. a means of determining when to stop
<!-- bilingual-en:start -->

&nbsp;
**1.** A sequence of simple steps.<br>
**2.** A flow of control.<br>
**3.** A means of determining when to stop.<br>
<!-- bilingual-en:end -->

这三个要素分别对应刚才平方根例子的三个面向：

- “先算什么、后算什么” 是步骤顺序
- “如果 close enough 就停，否则继续” 是控制流
- “close enough” 本身就是停止准则
<!-- bilingual-en:start -->
The order of calculations supplies the sequence, the choice between stopping and updating supplies control flow, and “close enough” supplies the stopping rule.
<!-- bilingual-en:end -->

> [!note]
> 很多人一开始会把“算法”理解成“有一串步骤”。这不够。  
> 如果没有 flow of control 和 stopping condition，那还只是说明书，不是对机器真正友好的算法。
> <!-- bilingual-en:start -->
> A list of steps alone is insufficient. Without control flow and a stopping condition, it remains an instruction sheet rather than a machine-ready algorithm.
> <!-- bilingual-en:end -->

### 6. Recipe 只是日常版 algorithm，computer 只是执行 recipe 的机器
<!-- bilingual-en:start -->
*6. A Recipe Is an Everyday Algorithm, and a Computer Executes Recipes*
<!-- bilingual-en:end -->
老师马上把算法和日常生活连了起来，说 recipe 本质上也是 algorithm。比如烤蛋糕时：
<!-- bilingual-en:start -->
The instructor connects algorithms to an ordinary recipe, such as baking a cake:
<!-- bilingual-en:end -->

- 先混合什么
- 如果没有鸡蛋怎么替代
- 每隔多久用牙签检查一次
- 什么时候算烤好
<!-- bilingual-en:start -->
It specifies what to mix first, what to do if an ingredient is missing, how often to test the cake, and when baking is complete.
<!-- bilingual-en:end -->

这其实已经完整包含：

- sequence
- condition
- repetition
- stop rule
<!-- bilingual-en:start -->
That already contains sequence, condition, repetition, and a stopping rule.
<!-- bilingual-en:end -->

然后她顺势强调本讲最重要的一句话：

> [!warning]
> `The computer only does what you tell it to do.`
> <!-- bilingual-en:start -->
> The computer only does what you tell it to do.
> <!-- bilingual-en:end -->

计算机擅长的是两件事：

- 存很多数据
- 很快地执行操作
<!-- bilingual-en:start -->
Computers store large amounts of data and execute operations quickly.
<!-- bilingual-en:end -->

它不擅长“自动理解你本来想表达什么”。所以只要程序行为和你想的不一样，第一反应应该是：我到底告诉它做了什么，而不是我本来想让它做什么。
<!-- bilingual-en:start -->
They do not infer what the programmer intended. When behavior differs from intention, inspect the instructions actually given rather than the unstated goal.
<!-- bilingual-en:end -->

### 7. 为什么 stored-program computer 重要
<!-- bilingual-en:start -->
*7. Why the Stored-Program Computer Matters*
<!-- bilingual-en:end -->
从这里开始，老师把视角往机器底层压了一层。她先对比了两类计算机：
<!-- bilingual-en:start -->
The instructor now moves one level closer to the machine by comparing:
<!-- bilingual-en:end -->

- `fixed-program computer`：像传统计算器，每按一个按钮就是一个操作，但无法把一整套步骤存起来以后反复复用
- `stored-program computer`：可以把 instructions 当成 data 存起来，再由 interpreter 去执行
<!-- bilingual-en:start -->
- A `fixed-program computer`, such as a traditional calculator, executes operations selected directly but does not store a reusable sequence of steps.
- A `stored-program computer` stores instructions as data and executes them through an interpreter.
<!-- bilingual-en:end -->

这个过渡是为了说明：现代编程语言之所以有意义，是因为机器已经不只是“临时按键计算”，而是能把步骤保存下来并解释执行。
<!-- bilingual-en:start -->
Modern programming languages matter because the machine can preserve and interpret a procedure rather than perform only one manually selected operation at a time.
<!-- bilingual-en:end -->

老师给出 stored-program computer 的三个直观部件：

- memory
- arithmetic logic unit, ALU
- control unit / program counter
<!-- bilingual-en:start -->
The intuitive components are memory, an arithmetic logic unit (ALU), and a control unit or program counter.
<!-- bilingual-en:end -->

> [!example] 老师举的低层执行示意
> 程序先从 memory 里取两个位置的值，例如 `3` 和 `4`。  
> ALU 做加法，得到 `7`。  
> 程序再把 `7` 存回某个 memory location。  
> 再做另一组加法，最后比较两个结果是否相等，并打印 `True`。
> <!-- bilingual-en:start -->
> At a low level, a program loads values such as `3` and `4` from memory, asks the ALU to add them, stores the resulting `7`, performs another addition, compares the two results, and prints `True`.
> <!-- bilingual-en:end -->

这段不是要你记机器结构细节，而是要你意识到：哪怕一个很普通的“先算两个和，再比较”的动作，在更低层都非常机械、繁琐。高级语言的价值，就是把这些细节封装掉。
<!-- bilingual-en:start -->
The purpose is not hardware memorization, but recognition that even an ordinary comparison of two sums becomes mechanical and verbose at a lower level. A high-level language hides that machinery.
<!-- bilingual-en:end -->

### 8. 从 English 到 Python：primitive、syntax、static semantics、semantics
<!-- bilingual-en:start -->
*8. From English to Python: Primitives, Syntax, Static Semantics, and Semantics*
<!-- bilingual-en:end -->
接下来老师用 English 和 programming language 做类比，帮助你理解语言的四层结构。
<!-- bilingual-en:start -->
An analogy with English introduces four layers of a programming language.
<!-- bilingual-en:end -->

第一层是 primitive。

- English 的 primitive 可以看成词
- Python 的 primitive 可以看成数字、字符序列、运算符等基础对象
<!-- bilingual-en:start -->
English primitives can be viewed as words; Python primitives include basic objects and operators such as numbers and character sequences.
<!-- bilingual-en:end -->

有了 primitive 以后，才谈得上 syntax。比如：

- English 里 `noun verb noun` 合法，`cat dog boy` 不像句子
- Python 里 `"hi" 5` 没有意义，但 `"hi" * 5` 有意义
<!-- bilingual-en:start -->
Syntax combines primitives legally: `noun verb noun` resembles an English sentence, while Python accepts `"hi" * 5` but not `"hi" 5`.
<!-- bilingual-en:end -->

> [!example]
> `"hi" * 5` 的含义是把字符串 `"hi"` 重复五次。  
> 但 `"hi" + 5` 虽然看起来也是“对象 运算符 对象”，却没有合法意义。
> <!-- bilingual-en:start -->
> `"hi" * 5` repeats the string five times. Although `"hi" + 5` has an object–operator–object shape, its operand types do not form a valid operation.
> <!-- bilingual-en:end -->

再往后是 static semantics。老师用 `"I are hungry"` 举例：它长得像一句话，但语义搭配不对。Python 里也一样，某些写法形式上像表达式，但类型组合不合法。
<!-- bilingual-en:start -->
Static semantics concerns combinations that have valid form but invalid relationships, like the English sentence `"I are hungry"` or a Python operation applied to incompatible types.
<!-- bilingual-en:end -->

最后是 semantics。这里老师特别强调，程序没有自然语言那种“模糊多义”。程序只有一个确定含义，只不过那个含义经常不是你原本想要的。
<!-- bilingual-en:start -->
Semantics gives a valid program its determined meaning. Unlike natural language, the program does not resolve ambiguity according to human intent; its meaning may simply differ from the intended one.
<!-- bilingual-en:end -->

所以程序错误大致分三层：

- syntax error：写法不合法
- static semantic error：类型/组合不合法
- semantic bug：程序能跑，但结果不符合你的意图
<!-- bilingual-en:start -->
- A syntax error violates grammatical form.
- A static-semantic error combines types or operations invalidly.
- A semantic bug runs successfully but produces behavior different from the programmer's intention.
<!-- bilingual-en:end -->

### 9. 真正开始写 Python 前，先理解 shell 是干什么的
<!-- bilingual-en:start -->
*9. Understanding the Shell Before Writing Larger Python Programs*
<!-- bilingual-en:end -->
老师在进入 Python 示例前，专门讲了一下 editor 和 shell 的区别。
<!-- bilingual-en:start -->
Before the Python examples, the instructor distinguishes the editor from the shell.
<!-- bilingual-en:end -->

- 今天主要在 shell 里做快速实验
- 以后多行程序会写在 editor 里
- shell 的作用是快速检查一条表达式、一个类型、一个小想法
<!-- bilingual-en:start -->
The shell supports quick experiments with one expression, type, or small idea; longer programs will be written in an editor.
<!-- bilingual-en:end -->

这个顺序很像数学里先在草稿纸上验算，再把正式推导写整齐。后面整门课里，shell 都会是一个很重要的“低成本试错区”。
<!-- bilingual-en:start -->
Like checking a calculation on scratch paper before writing a formal derivation, the shell provides a low-cost space for experimentation throughout the course.
<!-- bilingual-en:end -->

### 10. 程序在最底层就是：创建对象并操作对象
<!-- bilingual-en:start -->
*10. At the Base Level, Programs Create and Manipulate Objects*
<!-- bilingual-en:end -->
老师随后把“写程序”压缩成一句话：
<!-- bilingual-en:start -->
The instructor compresses programming into one statement:
<!-- bilingual-en:end -->

> at the base of it, we create objects and manipulate them
> <!-- bilingual-en:start -->
> At the base of it, we create objects and manipulate them.
> <!-- bilingual-en:end -->

然后她立刻说明 type 为什么重要。因为 type 决定 Python 允许你对对象做什么。
<!-- bilingual-en:start -->
An object's type matters because it determines which operations Python permits.
<!-- bilingual-en:end -->

两个对照例子是：

- `30` 这样的 number
- `"Ana"` 这样的 string

数字允许的操作，和字符串允许的操作并不相同。你可以对 number 做加减乘方，但不能把字符串随便拿去做相同的算术。
<!-- bilingual-en:start -->
Numbers and strings support different operations: arithmetic applies to numbers, not arbitrarily to strings.
<!-- bilingual-en:end -->

接着她给出了本讲涉及的 scalar object types：

- `int`
- `float`
- `bool`
- `NoneType`
<!-- bilingual-en:start -->
The scalar types introduced are `int`, `float`, `bool`, and `NoneType`.
<!-- bilingual-en:end -->

并反复强调：

- `True` 和 `False` 必须大写
- `type(...)` 是检查对象类型的最直接工具
<!-- bilingual-en:start -->
`True` and `False` require capitalization, and `type(...)` is the direct way to inspect an object's type.
<!-- bilingual-en:end -->

> [!warning]
> `true` 不是布尔值，而是 `NameError`。  
> 这类错误在第一周就出现，是为了尽早让你习惯 Python 对大小写非常敏感。
> <!-- bilingual-en:start -->
> `true` is not a Boolean literal; it is treated as an undefined name and raises `NameError`. Python is case-sensitive from the outset.
> <!-- bilingual-en:end -->

### 11. cast 不是修改原对象，而是拿到另一个新对象
<!-- bilingual-en:start -->
*11. Casting Produces Another Value Rather Than Mutating the Original Object*
<!-- bilingual-en:end -->
对象有了 type 之后，老师继续讲 casting。
<!-- bilingual-en:start -->
After types, the instructor introduces casting.
<!-- bilingual-en:end -->

几个课堂上的关键例子是：

```python
float(3)
int(5.2)
int(5.9)
round(5.9)
```

这里最容易混的点有两个：

1. `int(5.9)` 是 truncation，不是四舍五入
2. cast 不会“改变原来的对象”，而是产生一个新的值
<!-- bilingual-en:start -->
Two points are easy to confuse: `int(5.9)` truncates rather than rounds, and a cast produces a new value rather than changing the original object.
<!-- bilingual-en:end -->

也就是说，原来的 `3` 还是那个 `3`；`float(3)` 只是额外给你一个 `3.0`。
<!-- bilingual-en:start -->
The original `3` remains the same integer; `float(3)` additionally produces the floating-point value `3.0`.
<!-- bilingual-en:end -->

### 12. expressions 会被求值成一个 value，Python 存的是 value 不是表达式
<!-- bilingual-en:start -->
*12. Expressions Evaluate to Values; Python Stores the Value, Not the Expression*
<!-- bilingual-en:end -->
讲完 type 和 cast 后，老师才系统讲 expression。
<!-- bilingual-en:start -->
Only after types and casts does the instructor treat expressions systematically.
<!-- bilingual-en:end -->

她先从简单的算术表达式讲起：

```python
3 + 2
(4 + 2) * 6 - 1
type((4 + 2) * 6 - 1)
float((4 + 2) * 6 - 1)
```

这部分真正要记住的一句话是：
<!-- bilingual-en:start -->
The central statement is:
<!-- bilingual-en:end -->

> [!note]
> Python reads an expression, evaluates it to one value, and stores the value.  
> 它不会把整条表达式原样“存起来”。
> <!-- bilingual-en:start -->
> Python reads an expression, evaluates it to one value, and stores that value rather than preserving the unevaluated expression.
> <!-- bilingual-en:end -->

这就是为什么右边可以放很复杂的东西，最后左边拿到的仍然只是一个结果值。
<!-- bilingual-en:start -->
The right-hand side may therefore be complex while the left-hand name receives only its resulting value.
<!-- bilingual-en:end -->

老师也顺带讲了几个运算规则：

- 纯整数的 `+ - *` 结果还是 `int`
- 只要混入 `float`，很多结果会变成 `float`
- `/` 在 Python 里总是给 `float`
- `//` 是整除后的整数部分
- `%` 是 remainder
- `**` 表示乘方
<!-- bilingual-en:start -->
- Integer-only `+`, `-`, and `*` operations produce integers.
- Introducing a float commonly produces a float result.
- `/` always produces a float in Python.
- `//` gives the integer quotient.
- `%` gives the remainder.
- `**` denotes exponentiation.
<!-- bilingual-en:end -->

### 13. variable 是 binding，不是数学里的未知数
<!-- bilingual-en:start -->
*13. A Variable Is a Binding, Not a Mathematical Unknown*
<!-- bilingual-en:end -->
接下来，整讲进入另一个真正重要的点：assignment。
<!-- bilingual-en:start -->
The lecture now turns to assignment.
<!-- bilingual-en:end -->

老师先说，数学变量和程序变量不是一回事。数学里的 `x` 常常是“满足某条件的未知量”；而程序里的 variable 更像“给某个值起一个可复用的名字”。
<!-- bilingual-en:start -->
A mathematical variable often denotes an unknown satisfying a condition. A program variable is better understood as a reusable name bound to a value.
<!-- bilingual-en:end -->

所以：

```python
x = 6
xy = 3 + 4
```

是合法的 assignment；但

```python
6 = x
x * y = 3 + 4
```

不合法。
<!-- bilingual-en:start -->
The former statements are valid assignments; the latter are not.
<!-- bilingual-en:end -->

因为 assignment 的格式是：

- 左边只能是一个 variable name
- 右边必须先被求值成某个 value
<!-- bilingual-en:start -->
The left side of assignment must be an assignable name, while the right side is evaluated to a value before that value is bound to the name.
<!-- bilingual-en:end -->

然后这个 value 才会被 bind 到左边的名字上。
<!-- bilingual-en:start -->
Binding occurs only after evaluation.
<!-- bilingual-en:end -->

### 14. 为什么好变量名和注释不是“形式主义”
<!-- bilingual-en:start -->
*14. Why Good Names and Comments Are Not Formalities*
<!-- bilingual-en:end -->
老师紧接着拿圆面积和周长的程序，对比了坏风格和好风格。
<!-- bilingual-en:start -->
A circle-area and circumference program contrasts poor and good style.
<!-- bilingual-en:end -->

坏风格的问题包括：

- 重复写同一数值
- 变量名只有单字母
- 注释只是把代码翻译成英文
<!-- bilingual-en:start -->
Poor style repeats literal values, uses uninformative one-letter names, and comments by merely restating each line in English.
<!-- bilingual-en:end -->

好风格的特点则是：

- 把反复出现的值抽成变量，例如 `pi`
- 用 `radius`、`area`、`circumference` 这种有角色的名字
- 注释说明“这一段代码在干什么”，而不是逐行复述
<!-- bilingual-en:start -->
Better style names repeated values such as `pi`, uses role-bearing names such as `radius`, `area`, and `circumference`, and comments on the purpose of a block rather than paraphrasing its syntax.
<!-- bilingual-en:end -->

> [!note]
> 这段看起来像 coding style 的边角料，其实是在提前训练你未来调试和阅读程序的能力。  
> 代码不是只写给电脑看，也写给未来的自己看。
> <!-- bilingual-en:start -->
> These are not peripheral style concerns; they prepare debugging and program comprehension. Code is written for a future reader as well as for the computer.
> <!-- bilingual-en:end -->

### 15. `radius = radius + 1` 为什么合法
<!-- bilingual-en:start -->
*15. Why `radius = radius + 1` Is Valid*
<!-- bilingual-en:end -->
讲到这里，老师开始用 memory diagram 演示 binding 的变化。
<!-- bilingual-en:start -->
The memory diagram now shows bindings changing over time.
<!-- bilingual-en:end -->

先有：

```python
pi = 3.14
radius = 2.2
area = pi * (radius ** 2)
radius = radius + 1
```

然后她重点解释第四行：

- 先看右边的 `radius + 1`
- 当前 `radius` 绑定的是 `2.2`
- 右边求值后得到 `3.2`
- 再把左边的 `radius` 重新绑定到 `3.2`
<!-- bilingual-en:start -->
For `radius = radius + 1`, Python first reads the current binding `2.2`, evaluates the right side to `3.2`, and then rebinds `radius` to that new value.
<!-- bilingual-en:end -->

旧的 `2.2` 没有“神秘消失”，只是 `radius` 这个名字不再指向它。
<!-- bilingual-en:start -->
The old value has not been altered mysteriously; the name simply no longer refers to it.
<!-- bilingual-en:end -->

> [!warning]
> `area` 不会自动更新。  
> 因为程序没有被告知“半径变了以后，请重新计算面积”。  
> 计算机只按你写出的行顺序执行，不会替你推断依赖关系。
> <!-- bilingual-en:start -->
> `area` does not update automatically. The program was never instructed to recompute it after `radius` changed; execution follows the written lines rather than inferring dependencies.
> <!-- bilingual-en:end -->

这也是本讲第二个特别重要的程序观：程序到目前为止都是 `line by line` 执行，还没有任何“跳过”“回头”“自动联动”的能力。
<!-- bilingual-en:start -->
At this stage, programs execute line by line without branching, repetition, or automatic propagation of earlier calculations.
<!-- bilingual-en:end -->

### 16. 最后的 swap bug：为什么需要 temporary variable
<!-- bilingual-en:start -->
*16. The Closing Swap Bug: Why a Temporary Variable Is Necessary*
<!-- bilingual-en:end -->
本讲最后一个例子是交换 `x` 和 `y` 的值。
<!-- bilingual-en:start -->
The final example attempts to swap the values bound to `x` and `y`.
<!-- bilingual-en:end -->

错误写法是：

```python
x = 1
y = 2
y = x
x = y
```

第一步 `y = x` 执行后，原来 `y` 指向的 `2` 就丢失了；第二步再写 `x = y`，两边都只剩 `1`。
<!-- bilingual-en:start -->
After `y = x`, no remaining name preserves the original value `2`; the next line can only bind both names to `1`.
<!-- bilingual-en:end -->

所以老师引出正确做法：
<!-- bilingual-en:start -->
A temporary name preserves the value before either binding is overwritten:
<!-- bilingual-en:end -->

```python
x = 1
y = 2
temp = y
y = x
x = temp
```

> [!example]
> 这个例子表面上在教“交换两个变量”，实际是在教更底层的事情：  
> variable 保存的是 binding，不是你脑中对“这个变量原本代表什么”的意图。
> <!-- bilingual-en:start -->
> The exercise appears to teach swapping, but its deeper lesson is that variables preserve bindings, not the programmer's memory of what each name used to represent.
> <!-- bilingual-en:end -->

老师也借这个例子顺手介绍了 `Python Tutor` 这类逐步 trace 工具。对初学阶段来说，它最大的价值不是炫技，而是让你真正看到“每执行一行，绑定到底发生了什么变化”。
<!-- bilingual-en:start -->
Python Tutor makes those changes visible one line at a time, which is its practical value for a beginner.
<!-- bilingual-en:end -->

## Exercise log
> [!example] Finger exercise 01
> 题目要求：已知 `a`、`b`、`c`，创建 `total = (a+b)*c`，并打印 `total`。
> <!-- bilingual-en:start -->
> Given `a`, `b`, and `c`, the exercise creates `total = (a+b)*c` and prints `total`.
> <!-- bilingual-en:end -->
>
> ```python
> total = (a + b) * c
> print(total)
> ```
>
> 这题对应课堂中的两个最基础动作：
> - 先把 expression 按优先级求值
> - 再把结果绑定到 variable
> <!-- bilingual-en:start -->
> It practices two foundational actions:
> - Evaluate an expression according to precedence.
> - Bind the result to a variable.
> <!-- bilingual-en:end -->
>
> 如果这题做不顺，应该回看本讲的这两个位置：
> - “expressions 会先求值成一个 value”
> - “assignment 是把右边结果绑定给左边名字”
> <!-- bilingual-en:start -->
> If it is difficult, revisit the claims that an expression evaluates to one value and that assignment binds the right-hand result to the left-hand name.
> <!-- bilingual-en:end -->
>
> 它表面很短，但真正检查的是你有没有把“表达式”和“赋值”分开理解。
> <!-- bilingual-en:start -->
> Its brevity conceals the real test: keeping expression evaluation distinct from assignment.
> <!-- bilingual-en:end -->

## Links to follow-up practice
- Slides: [[MIT 6.100L-slides/mit6_100l_lec01.pdf|Lecture 01 slides]]
- Lecture code: [[MIT 6.100L-lecture-code/mit6_100l_lec01_code.py|Lecture 01 code (py)]]
- Finger exercise: [[MIT 6.100L-finger-exercises/mit6_100l_ex01_sol.pdf|Lecture 01 finger exercise solution]]
- Transcript: [[MIT 6.100L-transcripts/mit6_100l_lec01_transcript.pdf|Lecture 01 transcript]]
- Recitation 1: [[MIT 6.100L-recitations/mit6_100l_rec01.pdf|Recitation 01 materials]]
- PS 0 out (not graded): [[MIT 6.100L-problem-sets/mit6_100l_ps0.pdf|PS0 statement]], [[MIT 6.100L-problem-sets/mit6_100l_ps0_code.zip|PS0 starter code]]
- Textbook: [[Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Guttag textbook]] (Ch 1, Ch 2.1-2.2)

## Review checklist
- [ ] 我能复述为什么这门课把 lecture、finger exercise、problem set 看成不同训练，而不是同一件事的重复。
- [ ] 我能区分 declarative knowledge 和 imperative knowledge，并说明为什么程序必须写成后者。
- [ ] 我能不用看 slides，重新讲出平方根近似例子里的 guess、update、repeat、stop。
- [ ] 我能准确说出 algorithm 至少需要哪三部分。
- [ ] 我能解释 fixed-program computer 和 stored-program computer 的差别，不只背术语。
- [ ] 我能用自己的话解释 primitive、syntax、static semantics、semantics 分别在说什么。
- [ ] 我能区分 `int`、`float`、`bool`、`NoneType`，并知道 `True` / `False` 的大小写要求。
- [ ] 我能解释 cast 为什么不是“改原对象”，而是生成新值。
- [ ] 我能解释 `radius = radius + 1` 为什么在编程里合法，以及为什么 `area` 不会自动更新。
- [ ] 我能说出 swap bug 为什么会失败，以及 temporary variable 在保护什么信息。
<!-- bilingual-en:start -->
- [ ] I can explain why lectures, finger exercises, and problem sets provide different kinds of training.
- [ ] I can distinguish declarative and imperative knowledge and explain why a program must supply the latter.
- [ ] I can reconstruct the guess, update, repeat, and stop structure of the square-root example without the slides.
- [ ] I can state the three minimum components of an algorithm.
- [ ] I can explain the difference between fixed-program and stored-program computers rather than only name them.
- [ ] I can explain primitives, syntax, static semantics, and semantics in my own words.
- [ ] I can distinguish `int`, `float`, `bool`, and `NoneType`, including the capitalization of `True` and `False`.
- [ ] I can explain why a cast produces a new value rather than mutating the original object.
- [ ] I can explain why `radius = radius + 1` is valid and why `area` does not update automatically.
- [ ] I can explain the swap bug and identify the information protected by the temporary variable.
<!-- bilingual-en:end -->

> [!warning] Common mistakes
> - 把 `=` 当成数学里的“相等”，而不是 assignment。
> - 以为变量绑定变了，所有依赖它算出来的旧结果也会自动刷新。
> - 只记住 `type()`、`int()`、`float()` 这些命令，却没建立“对象先有类型，类型再决定操作”的顺序。
> <!-- bilingual-en:start -->
> - Treating `=` as mathematical equality rather than assignment.
> - Assuming that rebinding a variable automatically refreshes values computed from its previous binding.
> - Memorizing commands such as `type()`, `int()`, and `float()` without the underlying order: an object has a type, and its type determines the permitted operations.
> <!-- bilingual-en:end -->
