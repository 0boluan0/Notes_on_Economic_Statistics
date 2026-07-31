---
aliases:
  - MIT 6.100L Lecture 15
  - 6.100L L15
  - Recursion
tags:
  - computer-science
  - python
  - mit-6.100l
  - lecture-note
科目: Computer Science
course: MIT 6.100L Introduction to CS and Programming Using Python
lecture: 15
---

# Lecture 15: Recursion

> [!tip] Hint
> - 这节课一开始就提醒你：recursion 不是更高级的 loop，而是换一种想问题的方式。
> - 老师先回顾 iterative algorithms：状态变量、循环更新、停止条件、累计结果。
> - 然后故意用“乘法只允许靠加法实现”这个例子，把 iterative 写法和 recursive 写法并排比较。
> - 递归最重要的两个部分是 base case 和 recursive step，缺一个都不行。
> - `mult_recur(a, b)` 里真正变化的是问题规模：`b` 在不断减小，直到碰到 base case。
> - 课堂花很长时间 trace function calls，是为了让你接受“每次调用都是独立的一层工作”。
> - 老师用了批改作业/层层转交任务的类比来解释为什么递归会形成调用链。
> - `power_recur` 让你看到一个递归函数可以有不止一个 base case。
> - factorial 例子把“调用链下去，再沿调用链返回”这件事讲得最完整。
> - 听完这节课，你应该能解释 recursion 的执行过程，而不只是会照抄 `return n * f(n-1)`。
> <!-- bilingual-en:start -->
> - The lecture begins with a warning: recursion is not a more advanced loop, but a different way to frame a problem.
> - The instructor first reviews iterative algorithms—their state variables, repeated updates, stopping conditions, and accumulated results.
> - Multiplication implemented only through addition then provides a direct comparison between iterative and recursive formulations.
> - Every working recursion needs both a base case and a recursive step.
> - In `mult_recur(a, b)`, the size of the problem changes: `b` decreases until it reaches the base case.
> - The extended tracing of function calls is designed to show that every call is a separate layer of work.
> - An analogy involving grading and passing tasks down a chain explains how recursive calls form a call chain.
> - `power_recur` demonstrates that a recursive function may have more than one base case.
> - The factorial example gives the fullest account of descending through a call chain and then returning through it.
> - By the end, you should be able to explain how recursion executes rather than merely copy `return n * f(n-1)`.
> <!-- bilingual-en:end -->

## Lecture flow

### 1. 先承认：recursion 会迫使你换脑子
<!-- bilingual-en:start -->
*1. Accepting That Recursion Requires a Different Mental Model*
<!-- bilingual-en:end -->
Lecture 15 一开始老师就把预期管理说得很直白：

- recursion 是一种 programming technique
- 它不一定一上手就自然
- 你需要暂时放下之前对 loops 的默认思路
<!-- bilingual-en:start -->
At the opening of Lecture 15, the instructor sets expectations plainly: recursion is a programming technique, it may not feel natural at first, and you must temporarily set aside the default loop-based approach developed so far.
<!-- bilingual-en:end -->

这段开场很重要，因为课程不是把 recursion 当成“更快的 loop”，而是把它当成另一种 problem-solving lens。
<!-- bilingual-en:start -->
The course treats recursion not as a faster loop, but as another lens for solving problems.
<!-- bilingual-en:end -->

### 2. 先复习 iterative algorithms 是怎么想的
<!-- bilingual-en:start -->
*2. Reviewing the Iterative Way of Thinking First*
<!-- bilingual-en:end -->
为了让对比更清楚，老师没有直接写递归，而是先回顾迭代算法的思维方式。
<!-- bilingual-en:start -->
To make the contrast clear, the instructor reviews iterative thinking before writing recursive code.
<!-- bilingual-en:end -->

迭代算法通常依赖：

- 某些状态变量
- 每轮循环更新这些状态
- 一个停止条件
- 最后累积出答案
<!-- bilingual-en:start -->
An iterative algorithm usually maintains state variables, updates them on each pass, stops when a condition is met, and accumulates the final result.
<!-- bilingual-en:end -->

老师这里其实是在提醒你，前面十几讲你已经非常熟悉这种模式了：

- running sum
- counters
- while 条件
- for exhaust sequence
<!-- bilingual-en:start -->
Running sums, counters, `while` conditions, and `for` loops that exhaust a sequence have made this pattern familiar over the preceding lectures.
<!-- bilingual-en:end -->

所以 recursion 的进入点必须通过比较，才能看出差异。
<!-- bilingual-en:start -->
Recursion is therefore introduced by comparison so that the change in reasoning becomes visible.
<!-- bilingual-en:end -->

### 3. 先用乘法的迭代版建立共同问题
<!-- bilingual-en:start -->
*3. Establishing a Common Problem with Iterative Multiplication*
<!-- bilingual-en:end -->
老师选的例子是：  
假装我们不会用 `*`，只会用加法，如何算 `a * b`。
<!-- bilingual-en:start -->
The chosen problem asks how to compute `a * b` using addition alone, pretending that `*` is unavailable.
<!-- bilingual-en:end -->

先给出迭代版：
<!-- bilingual-en:start -->
The iterative version comes first:
<!-- bilingual-en:end -->

```python
def mult_iter(a, b):
    result = 0
    while b > 0:
        result += a
        b -= 1
    return result
```

这里完全是熟悉的 iterative thinking：

- `result` 记录当前累积值
- `b` 记录还要加几次
- 每轮做一次相同动作
- 直到 `b == 0`
<!-- bilingual-en:start -->
This is familiar iterative reasoning: `result` stores the running total, `b` records how many additions remain, each pass repeats the same action, and the loop stops when `b == 0`.
<!-- bilingual-en:end -->

### 4. 递归版不是“循环换写法”，而是“问题换表述”
<!-- bilingual-en:start -->
*4. The Recursive Version Reframes the Problem Rather Than Rewriting the Loop*
<!-- bilingual-en:end -->
老师接着问：  
如果用递归想同一个问题，会怎么想？
<!-- bilingual-en:start -->
The instructor then asks how the same problem can be formulated recursively.
<!-- bilingual-en:end -->

关键观察是：

- `a * b`
- 可以看成 `a + a * (b - 1)`
<!-- bilingual-en:start -->
The key observation is that `a * b` can be written as `a + a * (b - 1)`.
<!-- bilingual-en:end -->

也就是说，你不再想“我要循环 b 次”，而是想：

- 如果我已经知道 `a * (b - 1)` 怎么求
- 那 `a * b` 只差再加一个 `a`
<!-- bilingual-en:start -->
Instead of planning to repeat an action `b` times, assume the smaller product `a * (b - 1)` can be obtained; then the current product requires only one additional `a`.
<!-- bilingual-en:end -->

这就是递归思维的入口：

> [!note]
> 把当前问题写成“一个更小的同类问题 + 一点额外工作”。
> <!-- bilingual-en:start -->
> Express the current problem as a smaller instance of the same problem plus a small amount of additional work.
> <!-- bilingual-en:end -->

### 5. base case：递归必须有一个不用再递归的地方
<!-- bilingual-en:start -->
*5. The Base Case: A Point at Which Recursion Stops*
<!-- bilingual-en:end -->
一旦有了递归表达式，还差最关键的一步：base case。
<!-- bilingual-en:start -->
Once the recursive relation is available, one essential component remains: the base case.
<!-- bilingual-en:end -->

例如：
<!-- bilingual-en:start -->
For example:
<!-- bilingual-en:end -->

```python
def mult_recur(a, b):
    if b == 1:
        return a
    else:
        return a + mult_recur(a, b - 1)
```

这里：

- `b == 1` 是 base case
- `a + mult_recur(a, b - 1)` 是 recursive step
<!-- bilingual-en:start -->
Here, `b == 1` is the base case and `a + mult_recur(a, b - 1)` is the recursive step.
<!-- bilingual-en:end -->

为什么 base case 必须存在？
<!-- bilingual-en:start -->
Why is a base case necessary?
<!-- bilingual-en:end -->

- 否则问题规模只会一直缩小
- 但永远没有真正停下来的点
- 最后递归会无限展开
<!-- bilingual-en:start -->
Without it, the problem may keep shrinking but never reach a point that returns an answer, so the recursive calls continue without termination.
<!-- bilingual-en:end -->

### 6. 递归还有第二个要求：每一步都要朝 base case 靠近
<!-- bilingual-en:start -->
*6. A Second Requirement: Every Step Must Approach the Base Case*
<!-- bilingual-en:end -->
老师在讲完 base case 之后，又单独强调了另一个常见失误：

- 不只是要有 base case
- recursive step 还必须真正 **推进 toward base case**
<!-- bilingual-en:start -->
The instructor emphasizes a second common failure: having a base case is insufficient unless the recursive step actually moves toward it.
<!-- bilingual-en:end -->

在 `mult_recur(a, b)` 里，体现为：

- `b` 每次减 1
- 迟早会到 `1`
<!-- bilingual-en:start -->
In `mult_recur(a, b)`, `b` decreases by one on every call and must eventually reach `1`.
<!-- bilingual-en:end -->

如果参数没朝 base case 靠近，即使你写了 base case，也可能永远到不了。
<!-- bilingual-en:start -->
If the arguments do not progress toward the base case, the function may never reach it even though it appears in the code.
<!-- bilingual-en:end -->

### 7. 课堂花很大篇幅 trace 调用链
<!-- bilingual-en:start -->
*7. Spending Substantial Time Tracing the Call Chain*
<!-- bilingual-en:end -->
这节课最容易让人着急，但又最重要的部分，就是老师手动 trace `mult_recur(5, 4)` 之类的调用。
<!-- bilingual-en:start -->
The most demanding—and most important—part of the lecture is the manual tracing of calls such as `mult_recur(5, 4)`.
<!-- bilingual-en:end -->

她不断追问：

- 当前函数调用知道什么
- 它不知道什么
- 它下一步要把工作交给谁
- 谁会把结果返回给它
<!-- bilingual-en:start -->
At each layer, the instructor asks what the current call knows, what it does not know, which call receives the smaller task, and which call will eventually return the result.
<!-- bilingual-en:end -->

这是为了让你真正建立对 recursion execution 的感觉：

- 每次函数调用都是独立的一层
- 每层有自己的参数和局部环境
- 当前层只负责“做自己这一步，再等更小问题回来”
<!-- bilingual-en:start -->
The resulting execution model is that every call is a separate layer with its own arguments and local environment. It performs the current step and waits for the smaller problem to return.
<!-- bilingual-en:end -->

### 8. 调用链怎么“下去”又怎么“回来”
<!-- bilingual-en:start -->
*8. How the Call Chain Descends and Returns*
<!-- bilingual-en:end -->
如果 trace `mult_recur(5, 4)`：
<!-- bilingual-en:start -->
Tracing `mult_recur(5, 4)` gives:
<!-- bilingual-en:end -->

```python
mult_recur(5, 4)
= 5 + mult_recur(5, 3)
= 5 + (5 + mult_recur(5, 2))
= 5 + (5 + (5 + mult_recur(5, 1)))
= 5 + (5 + (5 + 5))
```

你会看到两个阶段：

- 向下展开：不断制造更小调用
- 向上返回：base case 出现后，结果一层层带回来
<!-- bilingual-en:start -->
Two phases are visible: the downward expansion creates successively smaller calls; after the base case returns, the result travels upward through the suspended calls.
<!-- bilingual-en:end -->

这也是为什么 recursion 往往让人感觉“先悬空着一堆没完成的工作”。
<!-- bilingual-en:start -->
This explains why recursion can feel as though it leaves a stack of unfinished work suspended before resolving it.
<!-- bilingual-en:end -->

### 9. 课堂中的类比：递归像一层层把任务交下去
<!-- bilingual-en:start -->
*9. Classroom Analogy: Passing a Task Down One Layer at a Time*
<!-- bilingual-en:end -->
老师在课堂中用了一个非常形象的类比：  
像批改作业或层层找助教帮忙那样。
<!-- bilingual-en:start -->
The instructor compares recursion to grading work or passing a question down through successive teaching assistants.
<!-- bilingual-en:end -->

直觉上可以理解为：

- 当前人接到任务
- 发现还得先解决一个更小的子任务
- 把子任务继续交下去
- 最底层的人先完成最小任务
- 然后答案再一层层返回
<!-- bilingual-en:start -->
One person receives a task, discovers a smaller prerequisite task, and passes it down. The bottom layer completes the smallest task first, after which the answer returns one layer at a time.
<!-- bilingual-en:end -->

这个类比虽然不影响写代码，但对理解“为什么每层调用都像在等待下一级完成”非常有帮助。
<!-- bilingual-en:start -->
The analogy does not change the code, but makes clear why each call waits for the next call to finish.
<!-- bilingual-en:end -->

### 10. `power_recur`：一个递归函数可以有多个 base case
<!-- bilingual-en:start -->
*10. `power_recur`: A Recursive Function May Have Several Base Cases*
<!-- bilingual-en:end -->
讲完乘法之后，老师把递归模式推广到幂函数：
<!-- bilingual-en:start -->
After multiplication, the instructor extends the recursive pattern to exponentiation:
<!-- bilingual-en:end -->

```python
def power_recur(n, p):
    if p == 0:
        return 1
    elif p == 1:
        return n
    else:
        return n * power_recur(n, p - 1)
```

这个例子的重要性在于：

- 让你看到 base case 不一定只有一个
- 让你开始把“数学定义”直接翻译成递归结构
<!-- bilingual-en:start -->
The example shows that a recursive function can have more than one base case and begins the practice of translating a mathematical definition directly into recursive structure.
<!-- bilingual-en:end -->

这里课堂的视角已经从“trace 一次递归”转成“如何设计递归函数”。
<!-- bilingual-en:start -->
The focus has now shifted from tracing a given recursion to designing one.
<!-- bilingual-en:end -->

### 11. factorial：最标准的递归定义
<!-- bilingual-en:start -->
*11. Factorial: The Standard Recursive Definition*
<!-- bilingual-en:end -->
接着老师把焦点放到 factorial。
<!-- bilingual-en:start -->
The instructor next turns to factorial.
<!-- bilingual-en:end -->

数学定义本身就很递归：

- `0! = 1` 或 `1! = 1`
- `n! = n * (n - 1)!`
<!-- bilingual-en:start -->
Its mathematical definition is already recursive: `0! = 1` or `1! = 1`, and `n! = n * (n - 1)!`.
<!-- bilingual-en:end -->

所以它特别适合拿来讲 recursion。
<!-- bilingual-en:start -->
That makes factorial a natural example of recursion.
<!-- bilingual-en:end -->

代码形式自然写成：
<!-- bilingual-en:start -->
The code follows the definition directly:
<!-- bilingual-en:end -->

```python
def fact_recur(n):
    if n == 1:
        return 1
    else:
        return n * fact_recur(n - 1)
```

课堂这里又一次长时间 trace，是为了让你看到 factorial 的调用链如何层层展开、再层层返回。
<!-- bilingual-en:start -->
Another extended trace shows the factorial call chain expanding and then returning layer by layer.
<!-- bilingual-en:end -->

### 12. 每个函数调用都是独立实例
<!-- bilingual-en:start -->
*12. Every Function Call Is a Separate Instance*
<!-- bilingual-en:end -->
老师在 factorial 追踪里不断强调一个容易忽视的事实：
<!-- bilingual-en:start -->
During that trace, the instructor repeatedly emphasizes an easily missed fact:
<!-- bilingual-en:end -->

- 虽然每层调用名字都叫 `fact_recur`
- 但每一层都是不同的 function call
- 每一层只知道自己的 `n`
<!-- bilingual-en:start -->
Although every layer calls the function named `fact_recur`, each is a distinct function call and knows only its own value of `n`.
<!-- bilingual-en:end -->

所以递归不是“一个函数在一张纸上自言自语”，而是“一连串独立调用形成的链”。
<!-- bilingual-en:start -->
Recursion is therefore not one function talking to itself on a page, but a chain of independent calls.
<!-- bilingual-en:end -->

这也是为什么 trace 时要认真区分：

- 当前层参数是什么
- 谁调用了我
- 我还欠着谁一个返回值
<!-- bilingual-en:start -->
A careful trace distinguishes the current call's arguments, the caller to which it belongs, and the return value it still owes that caller.
<!-- bilingual-en:end -->

### 13. 递归与迭代比较：不一定更快，但有时更自然
<!-- bilingual-en:start -->
*13. Recursion and Iteration: Sometimes More Natural, Not Necessarily Faster*
<!-- bilingual-en:end -->
课程最后没有把 recursion 神化。  
老师很明确地说：
<!-- bilingual-en:start -->
The lecture does not present recursion as universally superior.
<!-- bilingual-en:end -->

- 有些问题递归写起来更自然
- 但迭代版往往更直接、更高效
- 不是每个问题都必须递归
<!-- bilingual-en:start -->
Some problems have a more natural recursive formulation, but iterative solutions are often more direct and efficient, and not every problem calls for recursion.
<!-- bilingual-en:end -->

因此这节课的真正目标不是让你以后“逢题必递归”，而是多掌握一种拆问题的方式。
<!-- bilingual-en:start -->
The goal is to add another way of decomposing problems, not to force recursion into every solution.
<!-- bilingual-en:end -->

## Exercise log

> [!example] Finger exercise 15
> 官方练习是 `recur_power(base, exp)`：
> - `exp >= 0`
> - 用 recursion 计算 `base ** exp`
> - 提示明确给出：base case 在 `exp == 0`
> <!-- bilingual-en:start -->
> The official exercise is `recur_power(base, exp)`:
> - `exp >= 0`.
> - Compute `base ** exp` recursively.
> - The hint explicitly places the base case at `exp == 0`.
> <!-- bilingual-en:end -->

这题之所以合适，是因为它几乎是课堂上 `power_recur` 的直接练习版。  
它会检查你有没有真正掌握：
<!-- bilingual-en:start -->
The exercise is a direct application of the classroom `power_recur` example and checks whether the recursive structure has become usable.
<!-- bilingual-en:end -->

- base case 写什么
- recursive step 如何朝 base case 缩小
<!-- bilingual-en:start -->
You must choose the base case and make the recursive step reduce the exponent toward it.
<!-- bilingual-en:end -->

官方解法是：
<!-- bilingual-en:start -->
The official solution is:
<!-- bilingual-en:end -->

```python
if exp <= 0:
    return 1
return base * recur_power(base, exp - 1)
```

如果你写这题时还是下意识想上 `for` 或 `while`，说明你的脑子还没切进递归模式。
<!-- bilingual-en:start -->
If your first impulse is still to reach for `for` or `while`, the recursive formulation has not yet become intuitive.
<!-- bilingual-en:end -->

## Links to follow-up practice
- Slides: [[MIT 6.100L-slides/mit6_100l_lec15.pdf|Lecture 15 slides]]
- Lecture code: [[MIT 6.100L-lecture-code/mit6_100l_lec15_code.py|Lecture 15 code (py)]]
- Finger exercise: [[MIT 6.100L-finger-exercises/mit6_100l_ex15_sol.pdf|Lecture 15 finger exercise solution]]
- Transcript: [[MIT 6.100L-transcripts/mit6_100l_lec15_transcript.pdf|Lecture 15 transcript]]
- Recitation 8: [[MIT 6.100L-recitations/mit6_100l_rec08.zip|Recitation 08 materials]]
- Problem set milestone: none directly scheduled on this lecture
- Textbook: [[Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Guttag textbook]] (Ch 6.1)

## Review checklist
- [ ] 我能解释 recursion 和 iteration 在思考问题方式上的差别。
- [ ] 我能说清 iterative algorithm 常见的状态变量、更新和停止条件。
- [ ] 我能把乘法问题分别写成 iterative 和 recursive 的表述。
- [ ] 我能解释 base case 为什么不可缺少。
- [ ] 我能说明 recursive step 为什么必须朝 base case 推进。
- [ ] 我能手动 trace `mult_recur` 或 `fact_recur` 的调用链。
- [ ] 我能解释“每个函数调用都是独立的一层”这件事。
- [ ] 我能设计一个有两个 base case 的递归函数，比如 `power_recur`。
- [ ] 我能比较 factorial 的 iterative 和 recursive 版本各自的直观优缺点。
- [ ] 我能按课堂顺序复述：iterative recap -> multiplication -> base case/recursive step -> call tracing -> power -> factorial。
<!-- bilingual-en:start -->
- [ ] I can explain how recursion and iteration frame a problem differently.
- [ ] I can identify the state variables, updates, and stopping condition in a typical iterative algorithm.
- [ ] I can formulate multiplication both iteratively and recursively.
- [ ] I can explain why a base case is indispensable.
- [ ] I can explain why the recursive step must make progress toward the base case.
- [ ] I can manually trace the call chain of `mult_recur` or `fact_recur`.
- [ ] I can explain why every function call is a separate layer.
- [ ] I can design a recursive function with two base cases, such as `power_recur`.
- [ ] I can compare the intuitive strengths and weaknesses of iterative and recursive factorial implementations.
- [ ] I can reconstruct the lecture sequence: iterative review -> multiplication -> base case and recursive step -> call tracing -> powers -> factorial.
<!-- bilingual-en:end -->

> [!warning] Common mistakes
> - 只写 recursive step，不写 base case。
> - 写了 base case，但参数变化方向根本到不了它。
> - 递归 trace 时把所有调用混成同一层。
> - 一看到 recursion 就把它当成“更短的 loop”，却没有真正换问题表述。
> <!-- bilingual-en:start -->
> - Writing a recursive step without a base case.
> - Including a base case that the changing arguments can never reach.
> - Collapsing all calls into one layer while tracing recursion.
> - Treating recursion as a shorter loop without reframing the problem.
> <!-- bilingual-en:end -->
