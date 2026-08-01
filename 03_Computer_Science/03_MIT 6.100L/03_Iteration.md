---
aliases:
  - MIT 6.100L Lecture 03
  - 6.100L L03
  - Iteration
tags:
  - computer-science
  - python
  - mit-6.100l
  - lecture-note
科目: Computer Science
course: MIT 6.100L Introduction to CS and Programming Using Python
lecture: 03
---

# Lecture 03: Iteration

> [!tip] Hint
> - 这节课先复习了上一讲的 branching，因为 iteration 被引入时，就是作为“另一种 control flow”出现的。
> - Lost Woods 例子先故意用重复嵌套 `if` 写坏，目的是让你感到：当重复次数未知时，branching 不够用。
> - while loop 的定义不是“会重复”，而是“只要 condition 仍然为真，就继续执行同一块代码”。
> - 老师反复强调：while 的危险不在语法，而在你可能没有让 condition 朝着 False 前进。
> - `RIGHT` 这个小问题是在提醒你，字符串比较是 case sensitive，loop guard 也因此可能和人类直觉不一致。
> - 打印 `x` 那个例子真正想教的是：loop body 里必须更新和 guard 相关的变量，否则就会 infinite loop。
> - sad face 练习第一次把 counter 放进 while loop，让你看到“循环次数”本身也可以成为状态。
> - factorial 的 while 版本不是为了刷乘法，而是为了让你分清 loop variable 和 running product。
> - for loop 是从 while 的“数数模式”自然压缩出来的，range 的 start/stop/step 和 slicing 的思路是平行的。
> - 本讲最后真正想让你带走的是：while 更适合“不知道要重复多少次”，for 更适合“已经有一个 sequence 要走完”。
> <!-- bilingual-en:start -->
> - The lecture reviews branching because iteration is introduced as another form of control flow.
> - The Lost Woods example deliberately expands repeated nested `if`s until their inadequacy becomes obvious: branching alone cannot express an unknown number of repetitions.
> - A `while` loop means more than “repeat”; it executes the same block again as long as its condition remains true.
> - The danger is not the syntax of `while`, but failing to change the state on which its condition depends so that the condition can eventually become false.
> - The `RIGHT` example reminds you that string comparison is case-sensitive, so a loop guard follows exact program semantics rather than a person's intended meaning.
> - In the repeated-`x` example, the guard variable must be updated inside the body or the loop never terminates.
> - The sad-face exercise adds a counter and shows that the number of completed iterations can itself be part of program state.
> - The `while` implementation of factorial separates the loop variable from the running product; multiplication practice is secondary.
> - A `for` loop compresses a common counting pattern written with `while`, and `range(start, stop, step)` parallels slicing.
> - As a practical first distinction, use `while` when the number of repetitions is unknown but the stopping condition is known, and `for` when a sequence is already available for traversal.
> <!-- bilingual-en:end -->

## Lecture flow

### 1. 一开始先回顾 branching，因为 iteration 是另一种 control flow
<!-- bilingual-en:start -->
*1. Reviewing Branching Because Iteration Is Another Form of Control Flow*
<!-- bilingual-en:end -->
Lecture 3 开头没有直接写循环，而是先复习上一讲的 branching。
<!-- bilingual-en:start -->
Lecture 3 reviews branching before writing a loop.
<!-- bilingual-en:end -->

老师先提醒你，上一讲学到的是：
<!-- bilingual-en:start -->
The previous lecture introduced:
<!-- bilingual-en:end -->

- string
- input / output
- `if / elif / else`

并特别强调 control flow 这个词。  
branching 的作用，是让程序不再机械地一行一行走，而是可以根据条件跳进某个 block 或跳过某个 block。
<!-- bilingual-en:start -->
The important unifying term is control flow. Branching lets execution enter or skip a block according to a condition rather than proceeding mechanically from top to bottom.
<!-- bilingual-en:end -->

这一句很重要，因为 iteration 不是和 branching 无关的新知识，而是同一个主题下的下一步：
<!-- bilingual-en:start -->
Iteration is therefore the next step in the same theme, not an unrelated topic:
<!-- bilingual-en:end -->

> [!note]
> branching 解决的是“走哪条路”；  
> iteration 解决的是“同一条路要不要再走一遍”。
> <!-- bilingual-en:start -->
> Branching answers “Which path should execute?” Iteration answers “Should this path execute again?”
> <!-- bilingual-en:end -->

### 2. Lost Woods 先故意用 nested `if` 写坏，暴露出重复问题
<!-- bilingual-en:start -->
*2. Lost Woods: Deliberately Bad Nested `if`s Expose the Repetition Problem*
<!-- bilingual-en:end -->
老师接着用了 Zelda 里的 Lost Woods 作为动机例子。
<!-- bilingual-en:start -->
Zelda's Lost Woods provides the motivating example.
<!-- bilingual-en:end -->

问题设定很简单：
<!-- bilingual-en:start -->
The setup is simple:
<!-- bilingual-en:end -->

- 用户进入森林
- 如果一直往右走，就会一直回到同一个场景
- 只有某次改成往左走，才能离开
<!-- bilingual-en:start -->
- The user enters the forest.
- Going right returns to the same scene.
- Choosing left eventually permits an exit.
<!-- bilingual-en:end -->

如果只用上一讲会的 branching，代码会变成：
<!-- bilingual-en:start -->
With branching alone, the code must say:
<!-- bilingual-en:end -->

- 如果向右，就再问一次
- 如果又向右，就再问一次
- 如果又又向右，就再问一次
<!-- bilingual-en:start -->
- If the answer is right, ask again.
- If the next answer is right, ask again.
- Repeat by adding another nested `if` for every possible answer.
<!-- bilingual-en:end -->

也就是不断嵌套 `if`。
<!-- bilingual-en:start -->
The program expands into an indefinite nesting of conditionals.
<!-- bilingual-en:end -->

> [!warning]
> 这种写法的问题不是“能不能写出来”，而是你根本不知道用户会连续说多少次 `right`。  
> 当重复次数未知时，手工展开分支会立刻崩掉。
> <!-- bilingual-en:start -->
> The problem is not that a few nested branches cannot be written, but that the number of consecutive `right` responses is unknown. Manual expansion cannot cover an unbounded repetition count.
> <!-- bilingual-en:end -->

这就是 iteration 的真正动机：  
你需要的不是更多层 `if`，而是“只要条件仍为真，就重复同一块代码”。
<!-- bilingual-en:start -->
This motivates iteration: repeat one block while a condition remains true instead of adding more layers of `if`.
<!-- bilingual-en:end -->

### 3. while loop 先用 Netflix 类比，再给出正式语法
<!-- bilingual-en:start -->
*3. From a Netflix Analogy to the Formal `while` Syntax*
<!-- bilingual-en:end -->
为了让“重复直到条件失效”的直觉更自然，老师先用了 binge-watch Netflix 的类比：
<!-- bilingual-en:start -->
The instructor first uses binge-watching to make “repeat until the condition fails” intuitive:
<!-- bilingual-en:end -->

- 只要这个 show 还有下一集
- 就继续播放下一集
- 没有下一集时，退出这个过程
<!-- bilingual-en:start -->
- While another episode exists, play it.
- When there is no next episode, stop.
<!-- bilingual-en:end -->

然后她给出正式语法：
<!-- bilingual-en:start -->
The formal syntax is:
<!-- bilingual-en:end -->

```python
while <condition>:
    <code>
    <code>
```

执行机制是：
<!-- bilingual-en:start -->
Execution proceeds as follows:
<!-- bilingual-en:end -->

1. 检查 condition
2. 如果为 `True`，执行缩进块
3. 执行完后自动回去再检查同一个 condition
4. 直到 condition 为 `False` 才离开循环
<!-- bilingual-en:start -->

&nbsp;
**1.** Evaluate the condition.<br>
**2.** If it is `True`, execute the indented body.<br>
**3.** Return automatically to the same condition.<br>
**4.** Exit only when the condition is `False`.<br>
<!-- bilingual-en:end -->

这四步里，最关键的是第 3 步。  
你不需要自己写“跳回去”的语句，`while` 本身就已经把“回头再检查”这件事内建好了。
<!-- bilingual-en:start -->
The third step is essential: `while` itself provides the return to the guard, so no explicit “go back” statement is needed.
<!-- bilingual-en:end -->

### 4. 用 Lost Forest 程序第一次看见 while 的实际效果
<!-- bilingual-en:start -->
*4. Seeing `while` Operate in the Lost Forest Program*
<!-- bilingual-en:end -->
老师马上把 while 用回 Lost Forest：
<!-- bilingual-en:start -->
The instructor immediately rewrites Lost Forest with `while`:
<!-- bilingual-en:end -->

```python
where = input("You're in the Lost Forest. Go left or right? ")
while where == "right":
    where = input("You're in the Lost Forest. Go left or right? ")
print("You got out of the Lost Forest!")
```

这个程序的逻辑是：
<!-- bilingual-en:start -->
Its logic is:
<!-- bilingual-en:end -->

- 先问一次
- 只要答案还是 `"right"`，就继续问
- 一旦答案不再是 `"right"`，就跳出循环，打印“你出去了”
<!-- bilingual-en:start -->
- Ask once before the loop.
- Continue asking while the answer is `"right"`.
- When the answer differs, exit and print the success message.
<!-- bilingual-en:end -->

课堂里老师还专门问了一个很小但很有代表性的问题：
<!-- bilingual-en:start -->
The instructor then asks a small but representative question:
<!-- bilingual-en:end -->

- 如果用户输入 `"RIGHT"` 会怎样？
<!-- bilingual-en:start -->
- If the user inputs `"RIGHT"`, what happens?
<!-- bilingual-en:end -->

答案是：会直接跳出循环。  
因为 string comparison 是 case sensitive，`"RIGHT" != "right"`。
<!-- bilingual-en:start -->
It exits immediately because string comparison is case-sensitive and `"RIGHT" != "right"`.
<!-- bilingual-en:end -->

> [!note]
> 这说明 loop guard 不是“按人类意图判断”，而是按精确的表达式结果判断。  
> 如果你想兼容大小写，必须自己在程序里加规则，例如 `.lower()`。
> <!-- bilingual-en:start -->
> A loop guard follows the exact value of its expression, not human intent. Supporting case-insensitive input requires an explicit normalization rule such as `.lower()`.
> <!-- bilingual-en:end -->

### 5. 第二个 while 例子：打印 `x` 若干次，第一次明确看到 loop update
<!-- bilingual-en:start -->
*5. A Repeated-`x` Example Makes the Loop Update Explicit*
<!-- bilingual-en:end -->
Lost Forest 之后，老师换了一个更容易 trace 的数字例子：
<!-- bilingual-en:start -->
After Lost Forest, the lecture uses a numerical loop that is easier to trace:
<!-- bilingual-en:end -->

```python
n = int(input("Enter a non-negative integer: "))
while n > 0:
    print("x")
    n = n - 1
```

这个例子在课堂上是逐步 trace 的：
<!-- bilingual-en:start -->
For an initial `n = 4`, the trace is:
<!-- bilingual-en:end -->

- 如果最初 `n = 4`
- 第一次进循环打印一个 `x`，然后把 `n` 改成 `3`
- 第二次进循环再打印一个 `x`，再改成 `2`
- 最终 `n` 变成 `0`
- 再检查 guard 时，`0 > 0` 为假，循环结束
<!-- bilingual-en:start -->
- Print one `x` and reduce `n` to `3`.
- Print another and reduce `n` to `2`.
- Continue until `n` becomes `0`.
- The next guard test, `0 > 0`, is false, so the loop ends.
<!-- bilingual-en:end -->

这里第一次清楚地暴露了 while loop 的三个部件：
<!-- bilingual-en:start -->
The example exposes three loop components:
<!-- bilingual-en:end -->

- `guard`：`n > 0`
- `body`：打印并更新
- `state update`：`n = n - 1`
<!-- bilingual-en:start -->
- Guard: `n > 0`.
- Body: printing and updating.
- State update: `n = n - 1`.
<!-- bilingual-en:end -->

### 6. 为什么会 infinite loop：因为 guard 相关状态根本没变
<!-- bilingual-en:start -->
*6. Infinite Loops Occur When Guard-Related State Does Not Change*
<!-- bilingual-en:end -->
老师接着立刻故意把上一段代码里的更新删掉：
<!-- bilingual-en:start -->
The instructor deliberately removes the update:
<!-- bilingual-en:end -->

```python
n = int(input("Enter a non-negative integer: "))
while n > 0:
    print("x")
```

这时会发生什么？程序会一直打印下去。
<!-- bilingual-en:start -->
The program now prints indefinitely.
<!-- bilingual-en:end -->

原因非常直接：
<!-- bilingual-en:start -->
The reason is direct:
<!-- bilingual-en:end -->

- guard 依赖 `n > 0`
- 但 loop body 里没有任何语句改变 `n`
- 所以 guard 永远保持真
<!-- bilingual-en:start -->
- The guard depends on `n > 0`.
- The body never changes `n`.
- The guard therefore remains true forever.
<!-- bilingual-en:end -->

> [!warning]
> while loop 最大的坑不是不会写 `while` 这个单词。  
> 真正的坑是：你忘了让程序朝着“退出循环”前进。
> <!-- bilingual-en:start -->
> The main danger of a `while` loop is not its syntax, but forgetting to make program state progress toward termination.
> <!-- bilingual-en:end -->

老师还顺手教了两个处理方式：
<!-- bilingual-en:start -->
The lecture also shows how to interrupt a runaway program:
<!-- bilingual-en:end -->

- `Ctrl-C` / `Command-C`
- shell 里的停止按钮 / interrupt kernel
<!-- bilingual-en:start -->
- `Ctrl-C` / `Command-C`
- The shell's stop button or an interrupt of the kernel.
<!-- bilingual-en:end -->

这类操作在初学阶段很实用，因为写出无限循环几乎是必经阶段。
<!-- bilingual-en:start -->
These controls are practical because accidental infinite loops are a normal part of learning.
<!-- bilingual-en:end -->

### 7. 第一个带额外状态的循环：用 counter 统计进了几次森林
<!-- bilingual-en:start -->
*7. Adding State: A Counter Tracks Repeated Visits to the Forest*
<!-- bilingual-en:end -->
然后老师把 Lost Forest 程序做了一个小升级：
<!-- bilingual-en:start -->
The Lost Forest program is extended so that:
<!-- bilingual-en:end -->

- 如果用户说 `right` 超过两次
- 之后每次再问方向时，先打印一个 sad face
<!-- bilingual-en:start -->
- After more than two `right` responses, each new prompt is preceded by a sad face.
<!-- bilingual-en:end -->

这个版本第一次让你看到：  
循环里除了“决定是否继续”的变量外，还可以维护别的状态，例如 counter。
<!-- bilingual-en:start -->
The loop can maintain state beyond the variable that controls continuation; here it also maintains a counter.
<!-- bilingual-en:end -->

典型写法是：
<!-- bilingual-en:start -->
A typical implementation is:
<!-- bilingual-en:end -->

```python
where = input("Go left or right? ")
counter = 0
while where == "right":
    counter = counter + 1
    if counter > 2:
        print(":(")
    where = input("Go left or right? ")
print("You got out!")
```

这里 `counter` 的作用和 `where` 不一样：
<!-- bilingual-en:start -->
The two variables have distinct roles:
<!-- bilingual-en:end -->

- `where` 决定循环继续还是结束
- `counter` 记录我们已经重复了多少次
<!-- bilingual-en:start -->
- `where` determines whether the loop continues.
- `counter` records how many repetitions have occurred.
<!-- bilingual-en:end -->

这其实是在悄悄为后面的 accumulator、loop variable 做准备。
<!-- bilingual-en:start -->
This prepares the later distinction among counters, accumulators, and loop variables.
<!-- bilingual-en:end -->

### 8. 从具体例子抽出 while 的通用“数数模式”
<!-- bilingual-en:start -->
*8. Extracting a General Counting Pattern for `while` Loops*
<!-- bilingual-en:end -->
到这里，老师开始把 while 循环抽象成一个常见 pattern：
<!-- bilingual-en:start -->
The instructor now abstracts a common pattern:
<!-- bilingual-en:end -->

1. 在循环前初始化一个 loop variable
2. 在 guard 里用这个 variable 做判断
3. 在 body 里做工作
4. 同时更新这个 variable
<!-- bilingual-en:start -->

&nbsp;
**1.** Initialize a loop variable before the loop.<br>
**2.** Use it in the guard.<br>
**3.** Perform the work in the body.<br>
**4.** Update the variable so that the next guard test changes appropriately.<br>
<!-- bilingual-en:end -->

例如：
<!-- bilingual-en:start -->
For example:
<!-- bilingual-en:end -->

```python
n = 0
while n < 5:
    print(n)
    n = n + 1
```

这个 pattern 后来会反复出现。  
无论你是在计数、枚举、积累结果，很多 while loop 本质上都在做这四件事。
<!-- bilingual-en:start -->
Counting, enumeration, and accumulation repeatedly use these four steps.
<!-- bilingual-en:end -->

### 9. factorial 的 while 版本：区分 loop variable 和 running product
<!-- bilingual-en:start -->
*9. Factorial with `while`: Separating the Loop Variable from the Running Product*
<!-- bilingual-en:end -->
老师接着用 factorial 展示一个更完整的 while 程序：
<!-- bilingual-en:start -->
Factorial provides a fuller `while` program:
<!-- bilingual-en:end -->

```python
x = 4
i = 1
factorial = 1
while i <= x:
    factorial *= i
    i += 1
print(f"{x} factorial is {factorial}")
```

这段代码里有两个角色必须分清：
<!-- bilingual-en:start -->
Two roles must remain distinct:
<!-- bilingual-en:end -->

- `i`：loop variable，控制走到第几步
- `factorial`：running product，累计当前结果
<!-- bilingual-en:start -->
- `i` is the loop variable that controls progress.
- `factorial` is the running product that accumulates the answer.
<!-- bilingual-en:end -->

老师在课堂上逐轮 trace：
<!-- bilingual-en:start -->
The classroom trace is:
<!-- bilingual-en:end -->

- `i = 1` 时，`factorial = 1`
- `i = 2` 时，`factorial = 2`
- `i = 3` 时，`factorial = 6`
- `i = 4` 时，`factorial = 24`
- `i = 5` 时，guard 失败，退出
<!-- bilingual-en:start -->
- At `i = 1`, `factorial = 1`.
- At `i = 2`, `factorial = 2`.
- At `i = 3`, `factorial = 6`.
- At `i = 4`, `factorial = 24`.
- At `i = 5`, the guard fails and the loop exits.
<!-- bilingual-en:end -->

这段 trace 的目的不是死记 factorial，而是让你习惯区分：
<!-- bilingual-en:start -->
The trace is meant to distinguish:
<!-- bilingual-en:end -->

- 哪个变量在控制循环边界
- 哪个变量在积累答案
<!-- bilingual-en:start -->
- The variable controlling the loop boundary.
- The variable accumulating the answer.
<!-- bilingual-en:end -->

老师还顺带介绍了 shorthand notation：
<!-- bilingual-en:start -->
The instructor also introduces shorthand assignment:
<!-- bilingual-en:end -->

```python
i += 1
factorial *= i
```

它们分别等价于：
<!-- bilingual-en:start -->
These are equivalent to:
<!-- bilingual-en:end -->

```python
i = i + 1
factorial = factorial * i
```

### 10. Python Tutor 的价值：把“每一轮变量怎么变”可视化
<!-- bilingual-en:start -->
*10. Python Tutor Visualizes State Changes Across Iterations*
<!-- bilingual-en:end -->
讲 factorial 时，老师特意提到 `Python Tutor`。  
这一点在本讲很重要，因为循环的难点往往不是“最终答案是什么”，而是“中间每一轮发生了什么”。
<!-- bilingual-en:start -->
During factorial, the instructor recommends `Python Tutor` because the difficulty of a loop often lies in its intermediate states rather than its final answer.
<!-- bilingual-en:end -->

如果你看 while/for 经常觉得脑内 trace 不稳，Python Tutor 的作用就是把这些绑定变化外显出来：
<!-- bilingual-en:start -->
It externalizes a difficult mental trace by showing:
<!-- bilingual-en:end -->

- 当前执行到哪一行
- 每个变量此刻的值是什么
- 哪个值是在这一步刚变化的
<!-- bilingual-en:start -->
- The line currently executing.
- Every variable's current value.
- The binding or value changed by the latest step.
<!-- bilingual-en:end -->

初学循环时，这类工具非常有价值。
<!-- bilingual-en:start -->
That visualization is particularly valuable when loops are new.
<!-- bilingual-en:end -->

### 11. for loop 不是另一种“神秘循环”，而是 while 数数模式的压缩写法
<!-- bilingual-en:start -->
*11. A `for` Loop Compresses a Common `while` Counting Pattern*
<!-- bilingual-en:end -->
老师接下来引入 for loop，但她不是把它当成平行概念突然塞进来，而是明确说明：
<!-- bilingual-en:start -->
The instructor introduces `for` as a compression of a special kind of `while` loop rather than as a mysterious unrelated construct:
<!-- bilingual-en:end -->

> for loop 是对某类特殊 while loop 的压缩
> <!-- bilingual-en:start -->
> A `for` loop is a compressed form of a particular `while` pattern.
> <!-- bilingual-en:end -->

也就是这种模式：
<!-- bilingual-en:start -->
That pattern repeatedly:
<!-- bilingual-en:end -->

- 先初始化变量
- 判断变量是否还在区间里
- 每次把变量更新成下一个值
<!-- bilingual-en:start -->
- Initializes a variable.
- Checks whether it remains within a sequence or interval.
- Advances it to the next value.
<!-- bilingual-en:end -->

用 while 写会很啰嗦，用 for 则更紧凑：
<!-- bilingual-en:start -->
Writing those mechanics explicitly with `while` is verbose; `for` expresses them concisely:
<!-- bilingual-en:end -->

```python
for n in range(5):
    print(n)
```

这个循环会依次让 `n` 取值：
<!-- bilingual-en:start -->
The loop binds `n` successively to:
<!-- bilingual-en:end -->

- `0`
- `1`
- `2`
- `3`
- `4`

老师在这里特别强调，for loop 的重点是：
<!-- bilingual-en:start -->
The defining perspective is:
<!-- bilingual-en:end -->

- 你已经有一个 sequence
- loop variable 会自动依次绑定到 sequence 里的每个值
<!-- bilingual-en:start -->
- A sequence is already available.
- The loop variable binds automatically to each of its values in turn.
<!-- bilingual-en:end -->

### 12. range 的思路和 slicing 平行：start、stop、step
<!-- bilingual-en:start -->
*12. `range` Parallels Slicing Through Start, Stop, and Step*
<!-- bilingual-en:end -->
随后老师系统讲了 `range`。
<!-- bilingual-en:start -->
The instructor then develops `range` systematically.
<!-- bilingual-en:end -->

最基本的是：
<!-- bilingual-en:start -->
The simplest form is:
<!-- bilingual-en:end -->

```python
range(5)
```

表示从 `0` 到 `4`。  
然后她说明 `range` 也可以写成三参数版本：
<!-- bilingual-en:start -->
It represents the integers from `0` through `4`. The three-argument form is:
<!-- bilingual-en:end -->

```python
range(start, stop, step)
```

这和上一讲 slicing 的直觉几乎是平行的：
<!-- bilingual-en:start -->
Its interpretation parallels slicing:
<!-- bilingual-en:end -->

- 有起点
- 有终点
- stop 不包含自己
- 可以正着走，也可以倒着走
<!-- bilingual-en:start -->
- It has a starting value.
- It has a stopping boundary that is excluded.
- Its step can move forward or backward.
<!-- bilingual-en:end -->

课堂上的例子包括：
<!-- bilingual-en:start -->
Classroom examples include:
<!-- bilingual-en:end -->

```python
for i in range(1, 4, 1):
    print(i)         # 1 2 3

for j in range(1, 4, 2):
    print(j * 2)     # 2 6

for me in range(4, 0, -1):
    print("$" * me)  # $$$$, $$$, $$, $
```

> [!note]
> `range` 不只是“拿来数数”。  
> 它真正做的是生成一个你要依次走过的整数 sequence。
> <!-- bilingual-en:start -->
> `range` does more than “count”; it generates the integer sequence that a loop traverses.
> <!-- bilingual-en:end -->

### 13. accumulator 例子：用 for loop 求和
<!-- bilingual-en:start -->
*13. Accumulator Example: Summing with a For Loop*
<!-- bilingual-en:end -->
老师接着给了一个标准 accumulator 模式：
<!-- bilingual-en:start -->
The instructor next gives a standard accumulator pattern:
<!-- bilingual-en:end -->

```python
mysum = 0
for i in range(10):
    mysum += i
print(mysum)
```

这里的角色分工和 factorial 一样清晰：
<!-- bilingual-en:start -->
As in factorial, the roles are distinct:
<!-- bilingual-en:end -->

- `i` 是 loop variable
- `mysum` 是 accumulator
<!-- bilingual-en:start -->
- `i` is the loop variable.
- `mysum` is the accumulator.
<!-- bilingual-en:end -->

课堂上老师逐步讲了 `mysum` 如何从：
<!-- bilingual-en:start -->
The trace shows `mysum` moving from:
<!-- bilingual-en:end -->

- `0`
- 到 `1`
- 到 `3`
- 到 `6`
- ...
- 最终到 `45`
<!-- bilingual-en:start -->
- `0`
- to `1`
- to `3`
- to `6`
- ...
- finally to `45`.
<!-- bilingual-en:end -->

这段在训练一种很重要的阅读能力：  
看见一个循环时，你要先问自己，“这个变量是在控制循环，还是在累计答案？”
<!-- bilingual-en:start -->
This develops an important reading habit: for each variable in a loop, ask whether it controls traversal or accumulates the result.
<!-- bilingual-en:end -->

### 14. 用 print debugging 抓 off-by-one：为什么没把 end 加进去
<!-- bilingual-en:start -->
*14. Using Print Debugging to Find an Off-by-One Error*
<!-- bilingual-en:end -->
本讲后半段还有一个很典型的调试例子：  
写一个程序，把 `start` 到 `end` 之间的整数求和，而且要包含 `end` 本身。
<!-- bilingual-en:start -->
The task is to sum the integers from `start` through `end`, including `end`.
<!-- bilingual-en:end -->

错误代码大致是：
<!-- bilingual-en:start -->
The faulty code is:
<!-- bilingual-en:end -->

```python
mysum = 0
start = 3
end = 5
for i in range(start, end):
    mysum += i
print(mysum)
```

它算出来是 `7`，因为只加了 `3 + 4`。
<!-- bilingual-en:start -->
It returns `7` because it adds only `3 + 4`.
<!-- bilingual-en:end -->

老师这里的重点不只是“答案要改成 `end + 1`”，而是展示一种调试思路：
<!-- bilingual-en:start -->
The point is not merely the correction `end + 1`, but a debugging method:
<!-- bilingual-en:end -->

1. 先怀疑 sequence 到底生成了哪些值
2. 在 loop body 里临时加 `print(i)`
3. 看程序实际走了哪些步
<!-- bilingual-en:start -->

&nbsp;
**1.** Question which values the sequence actually generates.<br>
**2.** Add a temporary `print(i)` inside the body.<br>
**3.** Observe the values the loop actually visits.<br>
<!-- bilingual-en:end -->

于是问题就暴露了：`range(start, end)` 不会包含 `end`。
<!-- bilingual-en:start -->
The trace reveals that `range(start, end)` excludes `end`.
<!-- bilingual-en:end -->

正确写法是：
<!-- bilingual-en:start -->
The correction is:
<!-- bilingual-en:end -->

```python
for i in range(start, end + 1):
    mysum += i
```

> [!warning]
> 这就是最经典的 off-by-one。  
> 错的不是加法逻辑，而是你对 sequence 边界的理解差了 1。
> <!-- bilingual-en:start -->
> This is a classic off-by-one error: the addition is correct, but the understood sequence boundary differs from the actual one by one value.
> <!-- bilingual-en:end -->

### 15. 再回头看 factorial：for 版本更贴合“已知要乘 1 到 x”
<!-- bilingual-en:start -->
*15. Factorial Revisited: `for` Fits the Known Sequence from 1 to `x`*
<!-- bilingual-en:end -->
最后老师回到 factorial，把 while 版改写成 for 版：
<!-- bilingual-en:start -->
The instructor rewrites factorial with `for`:
<!-- bilingual-en:end -->

```python
x = 4
factorial = 1
for i in range(1, x + 1, 1):
    factorial *= i
print(f"{x} factorial is {factorial}")
```

这段改写的意义很明确：
<!-- bilingual-en:start -->
The reason is structural:
<!-- bilingual-en:end -->

- factorial 需要遍历一个明确的整数序列 `1, 2, ..., x`
- 所以 for loop 比 while 更直接
<!-- bilingual-en:start -->
- Factorial traverses the explicit sequence `1, 2, ..., x`.
- A `for` loop therefore expresses the task more directly than `while`.
<!-- bilingual-en:end -->

老师最后给出的高层区分也很实用：
<!-- bilingual-en:start -->
A useful high-level distinction is:
<!-- bilingual-en:end -->

- `while`：不知道要做多少次，只知道何时该停
- `for`：已经知道要走过哪个 sequence
<!-- bilingual-en:start -->
- `while`: the repetition count is not known in advance, but a stopping condition is.
- `for`: the sequence to traverse is already known.
<!-- bilingual-en:end -->

这不是绝对规则，但对初学阶段非常好用。
<!-- bilingual-en:start -->
This is not absolute, but it is a useful beginner's design rule.
<!-- bilingual-en:end -->

## Exercise log
> [!example] Finger exercise 03
> 题目要求：给定正整数 `N`，把 `hello world` 单独成行打印 `N` 次，可以用 `while` 或 `for`。
>
> 官方解答写的是：
>
> ```python
> for i in range(N):
>     print("hello world")
> ```
>
> 这题对应本讲中段到后段的核心迁移：
> - 先理解“重复执行同一块代码”
> - 再识别这是一个“走完固定长度 sequence”的问题
> - 最后自然选到 `for i in range(N)`
>
> 如果你更想用 `while`，也完全可以，但你就必须自己负责初始化、更新和停止条件。
> <!-- bilingual-en:start -->
> Given a positive integer `N`, print `hello world` on a separate line `N` times using either `while` or `for`.
>
> The official solution uses:
>
> The exercise captures the lecture's progression:
> - First understand repetition of one block.
> - Then recognize a fixed-length sequence.
> - Finally express it naturally as `for i in range(N)`.
>
> A `while` solution is equally valid, but it must state the initialization, update, and stopping condition explicitly.
> <!-- bilingual-en:end -->

## Links to follow-up practice
- Slides: [[MIT 6.100L-slides/mit6_100l_lec03.pdf|Lecture 03 slides]]
- Lecture code: [[MIT 6.100L-lecture-code/mit6_100l_lec03_code.py|Lecture 03 code (py)]]
- Finger exercise: [[MIT 6.100L-finger-exercises/mit6_100l_ex03_sol.pdf|Lecture 03 finger exercise solution]]
- Transcript: [[MIT 6.100L-transcripts/mit6_100l_lec03_transcript.pdf|Lecture 03 transcript]]
- Recitation 2: [[MIT 6.100L-recitations/mit6_100l_rec02.zip|Recitation 02 materials]]
- Problem set milestone: none directly scheduled on this lecture
- Textbook: [[Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Guttag textbook]] (Ch 2.5-2.8)

## Review checklist
- [ ] 我能解释为什么 Lost Woods 例子用 nested `if` 会很笨重，以及 iteration 解决的到底是什么问题。
- [ ] 我能准确说出 while loop 的执行顺序：检查 guard、执行 body、再检查 guard。
- [ ] 我能说明为什么 `RIGHT` 会让 Lost Forest 程序直接退出。
- [ ] 我能解释 infinite loop 的根源，不只会说“程序卡住了”。
- [ ] 我能区分 loop guard、loop variable、counter、accumulator 各自的角色。
- [ ] 我能手动 trace factorial 的 while 版本，并说出每一轮 `i` 和 `factorial` 的值。
- [ ] 我能解释 `+=` 和 `*=` 只是 shorthand，而不是新的语义。
- [ ] 我能说出 `range(start, stop, step)` 和 slicing 的相似点，尤其是 stop 不包含自己。
- [ ] 我能用 print debugging 找出一个 for loop 的 off-by-one bug。
- [ ] 我能判断某个问题更自然地适合 `while` 还是 `for`，并给出理由。
<!-- bilingual-en:start -->
- [ ] I can explain why the Lost Woods example would be cumbersome with nested `if` statements and what exactly iteration solves.
- [ ] I can accurately describe the execution order of a while loop: check guard, execute body, then check guard again.
- [ ] I can explain why setting `RIGHT` causes the Lost Forest program to exit directly.
- [ ] I can explain the cause of an infinite loop rather than merely say that the program is stuck.
- [ ] I can distinguish between the roles of loop guard, loop variable, counter, and accumulator.
- [ ] I can manually trace through the while version of factorial and state the values of `i` and `factorial` in each round.
- [ ] I can explain that `+=` and `*=` are just shorthand, not new semantics.
- [ ] I can describe the similarities between `range(start, stop, step)` and slicing, especially that the `stop` value is excluded.
- [ ] I can use print debugging to find an off-by-one bug in a for loop.
- [ ] I can decide whether a problem is expressed more naturally with `while` or `for` and justify the choice.
<!-- bilingual-en:end -->

> [!warning] Common mistakes
> - 写了 while guard，却没有在 loop body 里更新与 guard 相关的变量。
> - 把 counter、loop variable 和 accumulator 混成一个变量，最后逻辑和语义一起乱掉。
> - 忘记 `range` 的 stop 不包含自己，导致边界少算一个值。
> <!-- bilingual-en:start -->
> - Writing a `while` guard without updating the state on which it depends.
> - Collapsing a counter, loop variable, and accumulator into one variable and confusing their roles.
> - Forgetting that the stop value of `range` is excluded and omitting a boundary value.
> <!-- bilingual-en:end -->
