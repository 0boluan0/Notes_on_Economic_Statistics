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

## Lecture flow

### 1. 先从 smartphone 黑箱讲 abstraction
这节课的进入点不是代码，而是现实世界里的 black box。

老师拿 smartphone 举例：  
大多数用户并不知道手机内部的电路、传感器、驱动、操作系统是怎么工作的，但这并不妨碍他们使用手机。用户真正需要知道的是：

- 我做什么输入
- 系统给我什么输出
- 哪些按钮、滑动、触摸会触发哪些功能

这就是 **abstraction** 的第一层直觉：

> [!note]
> abstraction 不是“忽略细节”，而是把不属于当前使用者职责的细节藏到界面后面。

对用户来说，手机的实现细节被隐藏起来了；  
对程序员来说，函数内部的实现细节也可以被隐藏起来。

### 2. abstraction 之后，才能谈 decomposition
老师接着把手机例子推进到制造过程。

如果一个系统足够复杂，不可能由一个人从头到尾完成。现实中的做法是：

- 不同团队负责不同组件
- 每个组件只需要遵守接口规范
- 最后再把这些组件拼起来

这就是 **decomposition**。

这里的课堂重点不是“拆分”两个字本身，而是：

- 没有 abstraction，就没有稳定接口
- 没有稳定接口，就没法把工作拆给别人做
- 所以 abstraction 和 decomposition 是配套出现的

老师在这里其实已经在为函数做铺垫了：  
函数就是程序里最基本的 decomposition 单位。

### 3. 从现实黑箱转到代码黑箱：函数就是可命名的部件
进入编程语境后，老师把前面那套逻辑直接落到函数上。

一个函数最重要的价值不是“把几行代码包起来”，而是：

- 给一个动作起名字
- 规定它接受什么输入
- 规定它产生什么输出
- 让调用者不必关心内部细节

所以函数让程序从“从上到下的一长段脚本”变成“若干可组合的部件”。

这一点和前面几讲最大的区别是：

- 之前我们主要在写单段算法
- 这节课开始，课程要你学会组织程序

### 4. `def` 语法只是外壳，docstring 才是 contract
老师随后正式拆开函数定义的结构。

一个典型函数定义包含：

- `def`：告诉 Python 我现在要定义函数
- 函数名：给这个动作一个名字
- 参数列表：声明调用者必须提供哪些输入
- 冒号和缩进块：函数体
- docstring：写 specification

课堂里的核心观念是 docstring 的角色。它不是“可有可无的注释”，而是：

- 输入类型和前提条件
- 输出是什么
- 函数完成什么任务

如果把它写成一句更实用的话，就是：

> [!note]
> 函数名负责“这是什么动作”，docstring 负责“调用这个动作时你能依赖什么”。

### 5. 第一个完整例子：`is_even`
老师用 `is_even(i)` 做第一个完整例子，因为它足够简单，能把函数最核心的结构暴露出来。

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

- 参数 `i` 是输入占位符
- 函数体内部可以使用局部变量和分支
- `return` 会把结果交还给调用点

老师在这里也顺手提醒了 definition 和 call 的区别：

- 写 `def is_even(i): ...` 是在定义函数
- 写 `is_even(3)` 才是在调用函数

### 6. 函数的第一价值：复用同一个判断
写完 `is_even` 之后，老师马上把它放回更大的程序里使用。

例如判断 `1` 到 `10` 中每个数字是 even 还是 odd。  
如果没有函数，你会把“余数是否为 0”的逻辑反复写在循环里面；  
有了函数之后，主程序只需要关心更高层的表达：

```python
for i in range(1, 10):
    if is_even(i):
        print(i, "even")
    else:
        print(i, "odd")
```

这时主程序读起来已经更像人类语言：

- 对每个数
- 判断它是不是偶数
- 再决定打印什么

函数把低层判断细节折叠掉了。

### 7. 第一个 you-try-it：`div_by` 训练的是“按 specification 写函数”
老师接着给了一个非常短的练习：`div_by(n, d)`。

题目看起来只是在考 `%` 运算，但课堂真正要你训练的是：

- 先读 specification
- 再把 specification 翻译成条件判断
- 最后决定 return 什么

最直接的写法是：

```python
def div_by(n, d):
    """n and d are ints > 0
    Returns True if d divides n evenly and False otherwise"""
    return n % d == 0
```

> [!example]
> 这个练习很短，但它建立了一个重要习惯：  
> 如果 docstring 已经把行为说清楚，函数体经常只是把一句自然语言翻译成一两行布尔表达式。

### 8. `sum_odd` 把“同一任务，不同实现”这件事讲清楚
接下来老师把例子升级到一个更完整的函数：  
求 `a` 到 `b` 之间所有 odd numbers 的和。

课堂先给出一个版本，再给出另一个版本，目的是让你看到：

- specification 相同
- implementation 可以不同

这对后面算法比较很重要。你不能把“题目要做什么”和“我这次刚好怎么写”混为一谈。

课程代码里同时出现了 `for` 版和 `while` 版。真正需要记住的是这类函数的思维流程：

1. 初始化累计变量
2. 依次访问候选元素
3. 用条件筛出符合要求的元素
4. 更新累计结果
5. return 最终值

### 9. `return` 的位置决定函数什么时候结束
虽然 Lecture 8 会更系统地讲 `return`，但这节课里已经埋下了一个关键点：

- 一旦执行到 `return`
- 当前函数调用就结束
- 结果被送回调用处

所以函数设计时要想清楚：

- 什么时候已经得到最终答案
- 哪些路径应该提前结束
- 哪些路径应该继续扫描

这一点在后面的 palindrome 中会变得很直观。

### 10. palindrome：第一次认真练“提前发现反例就返回”
课堂后半段的字符串例子是 `is_palindrome(s)`。

它的典型思路不是“把字符串倒过来”这种捷径，而是：

- 只检查前半段
- 把左边第 `i` 个字符和右边对称位置比较
- 一旦发现不一样，立刻 `return False`
- 如果一路都没出错，最后 `return True`

```python
def is_palindrome(s):
    for i in range(len(s) // 2):
        if s[i] != s[len(s) - i - 1]:
            return False
    return True
```

这里课堂在训练三件事：

- 如何把 index 运算写对
- 如何利用对称性少做一半工作
- 如何用 early return 让逻辑更干净

### 11. 课后字符串练习继续强化 decomposition
老师最后又给了两个 at-home 风格的函数：

- `keep_consonants(word)`
- `first_to_last_diff(s, c)`

它们不像前面的例子那样只是讲语法，而是在要求你把一个模糊目标拆成清楚步骤。

例如 `keep_consonants` 的自然拆法就是：

1. 先定义什么算 vowel
2. 建一个空字符串作为答案
3. 逐字符扫描输入
4. 只把 consonant 接到答案里

而 `first_to_last_diff` 更像是训练“先找第一个，再找最后一个，再组合结果”这种程序分解能力。

### 12. 这节课真正完成了什么
如果把 Lecture 7 压缩成一句话，它做的不是“介绍新语法”，而是：

> [!note]
> 让你第一次把程序看成由若干部件组成，而函数是这些部件最基本的封装形式。

从这节课开始，后面所有内容都会默认你接受这套思路：

- 先说清楚接口
- 再隐藏实现
- 然后复用部件

## Exercise log

> [!example] Finger exercise 07
> 官方练习分成两步：
> - `eval_quadratic(a, b, c, x)`：返回二次式 `a*x^2 + b*x + c` 的值。
> - `two_quadratics(...)`：分别计算两个二次式，并 `print` 它们的和，不返回值。

这套题很适合放在本讲后面，因为它正好检查你有没有真的分清下面三件事：

- 一个函数负责 **计算并返回值**
- 另一个函数负责 **调用已有函数并组织结果**
- `print` 和 `return` 的语义并不相同

第一问几乎只是在检查你会不会把 specification 准确翻译成表达式。  
第二问开始要求你把“已有函数当部件”重新拼起来，这正是 decomposition 的核心。

如果第二问做得别扭，通常不是二次函数不会算，而是这两个概念还没分开：

- “一个函数自己完成所有计算”
- “一个函数调用别的函数来完成更大的任务”

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

> [!warning] Common mistakes
> - 把函数理解成“缩进起来的代码块”，却没有真正写清楚接口。
> - 写了 docstring，但函数体并没有兑现 docstring 里的承诺。
> - 需要返回结果时只 `print`，导致后续代码拿不到值。
> - 还没想清楚 specification 就直接开写，最后函数名、参数和行为互相打架。
