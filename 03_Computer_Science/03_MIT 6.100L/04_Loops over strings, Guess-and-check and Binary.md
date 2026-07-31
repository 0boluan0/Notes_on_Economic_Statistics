---
aliases:
  - MIT 6.100L Lecture 04
  - 6.100L L04
  - Loops over strings, Guess-and-check and Binary
tags:
  - computer-science
  - python
  - mit-6.100l
  - lecture-note
科目: Computer Science
course: MIT 6.100L Introduction to CS and Programming Using Python
lecture: 04
---

# Lecture 04: Loops over strings, Guess-and-check, and Binary

> [!tip] Hint
> - 这节课先不是直接讲新算法，而是先把 loop 再推进半步：`break` 和 “for loop 不只迭代数字”。
> - 老师一开始用 `break` 展示的是“提前退出循环”，而不是“替代 while/for 的另一种循环”。
> - 从 `range(len(s))` 到 `for char in s` 的过渡，是这节课前半段最重要的代码审美升级。
> - 机器人 cheerleader 例子其实在把三个旧知识串起来：循环、字符串、条件。
> - “我们现在已经有 enough tools to implement algorithms” 是全讲中段的转折点。
> - guess-and-check 的本质不是瞎猜，而是 exhaustive enumeration：先定义候选空间，再系统检查。
> - 平方根、立方根和 word problem 都是在训练同一种结构：候选值、检验条件、停止条件。
> - `break` 在 cube-root 代码里第一次显得合理，因为一旦 overshoot 就没必要继续枚举。
> - 浮点数问题的引入不是新话题，而是在提醒你：有些问题上“完全相等”会失效。
> - 十进制转二进制那段不是离题，而是在为后面 float 近似铺底层表示直觉。
> <!-- bilingual-en:start -->
> - The lecture first extends loops through `break` and through the fact that a `for` loop can traverse more than numbers.
> - `break` means an early exit from the current loop; it is not another kind of loop that replaces `while` or `for`.
> - Moving from `range(len(s))` to `for char in s` is the main improvement in code clarity during the first half.
> - The robot cheerleader combines three existing ideas: loops, strings, and conditionals.
> - “We now have enough tools to implement algorithms” marks the lecture's midpoint transition.
> - Guess-and-check is systematic exhaustive enumeration, not arbitrary guessing: define a candidate space and test every candidate methodically.
> - Square roots, cube roots, and the word problem all instantiate the same structure of candidates, a test, and stopping conditions.
> - `break` becomes algorithmically useful in the cube-root search because no larger candidate can work after an overshoot.
> - The floating-point example is not a disconnected topic; it warns that exact equality can fail in some numerical settings.
> - Decimal-to-binary conversion develops the representation intuition needed for the next lecture's treatment of floating-point approximation.
> <!-- bilingual-en:end -->

## Lecture flow

### 1. 开场先回顾 loops，然后补一个很实用的细节：`break`
<!-- bilingual-en:start -->
*1. Reviewing Loops and Adding a Practical Control: `break`*
<!-- bilingual-en:end -->
Lecture 4 一开始并没有直接进入 guess-and-check，而是先接着上一讲的 while/for 往前推一点。
<!-- bilingual-en:start -->
Lecture 4 extends the preceding treatment of `while` and `for` before introducing guess-and-check.
<!-- bilingual-en:end -->

老师先回顾：

- `while`：只要 condition 为真就继续
- `for`：依次走过一个 sequence of values
<!-- bilingual-en:start -->
- A `while` loop continues as long as its condition is true.
- A `for` loop traverses a sequence of values.
<!-- bilingual-en:end -->

然后她马上提出一个很实际的问题：  
如果我们还没走到 while 的自然结束，或者还没把 for 的序列走完，但已经知道“可以停了”，该怎么办？
<!-- bilingual-en:start -->
The practical question is how to stop when the natural `while` condition has not yet failed or the `for` sequence has not been exhausted, but the needed result is already known.
<!-- bilingual-en:end -->

答案就是 `break`。
<!-- bilingual-en:start -->
The answer is `break`.
<!-- bilingual-en:end -->

> [!note]
> `break` 的语义非常具体：  
> 立刻退出当前所在的最内层循环，并跳过这个循环剩下的代码。
> <!-- bilingual-en:start -->
> `break` immediately exits the innermost loop that contains it and skips the rest of that loop.
> <!-- bilingual-en:end -->

老师先用嵌套循环说明它只会跳出 “surrounding loop”，不会把外层循环也一并终止。这个地方是在纠正常见误解：`break` 不是“结束整个程序”，也不是“让所有循环同时停”。
<!-- bilingual-en:start -->
Nested loops show that it exits only the surrounding loop, not every enclosing loop or the entire program.
<!-- bilingual-en:end -->

### 2. 用一个很小的 for-loop 例子看出 `break` 到底会切掉哪些代码
<!-- bilingual-en:start -->
*2. A Small `for` Loop Shows Exactly What `break` Skips*
<!-- bilingual-en:end -->
接着老师给了一个很短的 for-loop：
<!-- bilingual-en:start -->
The instructor then gives a short loop:
<!-- bilingual-en:end -->

```python
mysum = 0
for i in range(5, 11, 2):
    mysum += i
    if mysum == 5:
        break
        mysum += 1
print(mysum)
```

课堂上这段代码是用 Python Tutor 逐步 trace 的。  
它的重点不是算术结果，而是让你意识到：
<!-- bilingual-en:start -->
Python Tutor is used to trace it step by step. The arithmetic result is secondary to the control flow:
<!-- bilingual-en:end -->

- `range(5, 11, 2)` 只会产生 `5, 7, 9`
- 第一次 `mysum += i` 后，`mysum` 立刻变成 `5`
- 触发 `break` 后，循环体中后面的 `mysum += 1` 永远不会执行
- 也不会回去检查下一个 `i`
<!-- bilingual-en:start -->
- `range(5, 11, 2)` produces only `5, 7, 9`.
- On the first iteration, `mysum += i` makes `mysum` equal to `5`.
- The triggered `break` prevents the later `mysum += 1` from running.
- The loop never advances to another value of `i`.
<!-- bilingual-en:end -->

所以这段代码最后打印 `5`。
<!-- bilingual-en:start -->
The final printed value is therefore `5`.
<!-- bilingual-en:end -->

> [!warning]
> 这一段是用来训练“控制流可视化”的。  
> 只要看到 `break`，你就要立刻问：这一行下面、以及这次循环之后的哪些代码，已经没有机会运行了？
> <!-- bilingual-en:start -->
> This is an exercise in visualizing control flow. At every `break`, identify which later statements in the body and which later iterations have become unreachable.
> <!-- bilingual-en:end -->

### 3. 用“统计偶数个数”的暖身题，把上一讲循环模式再稳一遍
<!-- bilingual-en:start -->
*3. Reinforcing the Previous Loop Pattern by Counting Even Numbers*
<!-- bilingual-en:end -->
在进入新内容前，老师安排了一个非常标准的小练习：  
给一个 `range(...)`，统计其中有多少个 even numbers。
<!-- bilingual-en:start -->
Before the new material, a warm-up counts the even numbers in a `range(...)`.
<!-- bilingual-en:end -->

典型结构是：

```python
even_nums = 0
for i in range(5):
    if i % 2 == 0:
        even_nums += 1
print(even_nums)
```

这个题看起来简单，但它其实把 Lecture 3 的核心循环模式再压了一遍：

- 先有一个 counter
- 循环依次给出候选值
- 用条件筛选
- 满足条件时更新 counter
<!-- bilingual-en:start -->
The exercise condenses Lecture 3's pattern: initialize a counter, traverse candidate values, filter them with a condition, and update the counter when the condition holds.
<!-- bilingual-en:end -->

这也是老师在这门课里反复强调的节奏：  
每次讲新东西前，先用一个小题把上一讲的基础动作重新激活。
<!-- bilingual-en:start -->
This reflects the course's recurring rhythm of reactivating a basic operation before adding a new one.
<!-- bilingual-en:end -->

### 4. for loop 的 sequence 不只可以是数字，也可以直接是字符串字符
<!-- bilingual-en:start -->
*4. A `for` Loop Can Traverse Characters Directly, Not Only Numbers*
<!-- bilingual-en:end -->
这一讲真正的前半段重点，是让你意识到：
<!-- bilingual-en:start -->
The central point of the first half is:
<!-- bilingual-en:end -->

> [!note]
> for loop 不只是在 `range(...)` 上数数。  
> 只要是 sequence，都可以直接迭代。
> <!-- bilingual-en:start -->
> A `for` loop is not limited to counting through `range(...)`; it can traverse any sequence directly.
> <!-- bilingual-en:end -->

老师先从一个很普通的任务开始：检查字符串里是否有 `i` 或 `u`。
<!-- bilingual-en:start -->
The first task checks whether a string contains `i` or `u`.
<!-- bilingual-en:end -->

第一种写法是按 index 走：

```python
for index in range(len(s)):
    if s[index] == 'i' or s[index] == 'u':
        print("There is an i or u")
```

它能工作，但读起来偏绕，因为：

- 你实际上关心的是字符本身
- 却先去迭代 index
- 再用 index 去取字符
<!-- bilingual-en:start -->
The indexed version works, but it iterates over positions even though the task concerns characters themselves, then uses each position merely to recover a character.
<!-- bilingual-en:end -->

于是老师顺势给出第二种、更自然的写法：
<!-- bilingual-en:start -->
The more natural version traverses the characters directly:
<!-- bilingual-en:end -->

```python
for char in s:
    if char == 'i' or char == 'u':
        print("There is an i or u")
```

再进一步，写成更 Pythonic 的版本：
<!-- bilingual-en:start -->
It can then be expressed more idiomatically with membership:
<!-- bilingual-en:end -->

```python
for char in s:
    if char in 'iu':
        print("There is an i or u")
```

这一小段课的真正收获不是 “我会检查元音” 这种表层能力，而是你开始会判断：

- 我到底需要 index，还是只需要 element 本身？
- 如果只需要 element，为什么还要硬写 `range(len(...))`？
<!-- bilingual-en:start -->
The real lesson is deciding whether the algorithm needs an index or only the element itself. If it needs only the element, `range(len(...))` adds unnecessary indirection.
<!-- bilingual-en:end -->

### 5. 机器人 cheerleader：第一次把字符串循环、条件分支和重复输出串成一个小程序
<!-- bilingual-en:start -->
*5. Robot Cheerleader: Combining String Traversal, Branching, and Repetition*
<!-- bilingual-en:end -->
接下来老师用了一个很有课堂感的例子：robot cheerleaders。
<!-- bilingual-en:start -->
The next classroom example is a robot cheerleader.
<!-- bilingual-en:end -->

程序接受：

- 一个单词
- 一个 enthusiasm level
<!-- bilingual-en:start -->
The program receives a word and an enthusiasm level.
<!-- bilingual-en:end -->

然后它会：

1. 按字母逐个拼读这个单词
2. 判断每个字母前该说 `a` 还是 `an`
3. 再把整个单词带着感叹号重复打印若干次
<!-- bilingual-en:start -->
It spells the word one letter at a time, selects `a` or `an` for each letter, and then prints the complete word with exclamation marks the requested number of times.
<!-- bilingual-en:end -->

这段程序为什么重要？因为它第一次把你前几讲零散学到的东西合到一起：

- `for w in word`：循环直接走字符
- `if w in an_letters`：用字符串 membership 做条件判断
- `input(...)` + `int(...)`：交互输入
- 第二个循环 `for i in range(times)`：按用户给的次数重复动作
<!-- bilingual-en:start -->
The example combines `for w in word`, membership in `an_letters`, interactive input converted with `int(...)`, and a second `for i in range(times)` loop for repeated output.
<!-- bilingual-en:end -->

> [!example]
> 这段代码的结构可以概括成：
> <!-- bilingual-en:start -->
> Its structure is:
> <!-- bilingual-en:end -->
>
> ```python
> for w in word:
>     if w in an_letters:
>         print("Give me an " + w + ": " + w)
>     else:
>         print("Give me a " + w + ": " + w)
> print("What does that spell?")
> for i in range(times):
>     print(word, "!!!")
> ```

这类例子很像“你跟着老师听下来会怎么记”的笔记点，因为它保留了课堂里真实的程序组合路径，而不是事后才抽象成几个平级知识点。
<!-- bilingual-en:start -->
The example preserves the classroom's actual path of composing a program rather than retrospectively flattening the ideas into unrelated concepts.
<!-- bilingual-en:end -->

### 6. unique letters 练习：scan 一个 sequence，同时维护“已经见过什么”
<!-- bilingual-en:start -->
*6. Unique-Letters Exercise: Scanning a Sequence While Tracking What Has Been Seen*
<!-- bilingual-en:end -->
然后老师给了一个稍微更算法化一点的练习：  
给定 `s = "abca"`，数出里面有多少 unique letters。
<!-- bilingual-en:start -->
The next, slightly more algorithmic, exercise counts the unique letters in `s = "abca"`.
<!-- bilingual-en:end -->

她给的提示非常关键：

- 一路扫描 `s`
- 维护一个 `seen` 字符串
- 只有当前字符不在 `seen` 里时，才把它加进去
<!-- bilingual-en:start -->
The key hint is to scan `s`, maintain a `seen` string, and add a character only when it is not already present.
<!-- bilingual-en:end -->

这已经非常接近后面会反复出现的“scan + state”模式：

- 你沿着 sequence 一路走
- 每一轮都用当前元素更新某个状态容器
<!-- bilingual-en:start -->
This is an early form of the recurring scan-plus-state pattern: traverse a sequence and update a state container from each element.
<!-- bilingual-en:end -->

只是这里的状态容器还比较朴素，是一个 string；后面会升级成 list、dict、set 风格的思维。
<!-- bilingual-en:start -->
Here the container is only a string; later work develops the same idea with lists, dictionaries, and sets.
<!-- bilingual-en:end -->

### 7. 课堂中段的转折：我们现在终于有 enough tools 去实现 algorithms
<!-- bilingual-en:start -->
*7. Midpoint Transition: Enough Tools to Implement Algorithms*
<!-- bilingual-en:end -->
到这里老师明确停了一下，说：
<!-- bilingual-en:start -->
At this point, the instructor pauses to collect the tools already available:
<!-- bilingual-en:end -->

- objects
- expressions
- branching
- loops
<!-- bilingual-en:start -->
- Objects.
- Expressions.
- Branching.
- Loops.
<!-- bilingual-en:end -->

这些东西加在一起，其实已经足够实现很多算法了。
<!-- bilingual-en:start -->
Together, these are sufficient to implement many algorithms.
<!-- bilingual-en:end -->

这句话标志着这一讲中段的转折。  
前半段是在补工具箱，后半段开始真正讲 “algorithmic method”。
<!-- bilingual-en:start -->
The first half extends the toolbox; the second begins to teach an algorithmic method.
<!-- bilingual-en:end -->

### 8. guess-and-check 不是“随便猜”，而是 exhaustive enumeration
<!-- bilingual-en:start -->
*8. Guess-and-Check Is Systematic Exhaustive Enumeration*
<!-- bilingual-en:end -->
进入算法部分后，老师先讲的是 `guess-and-check`，也就是 `exhaustive enumeration`。
<!-- bilingual-en:start -->
The algorithmic section begins with `guess-and-check`, or `exhaustive enumeration`.
<!-- bilingual-en:end -->

她把这个方法抽象成两件事：

1. 你能生成候选解
2. 你能检查候选解是否正确
<!-- bilingual-en:start -->
The method requires two capabilities: generating candidate solutions and testing whether a candidate is correct.
<!-- bilingual-en:end -->

然后就可以：

- 从某个初始 guess 开始
- 系统地一个一个试
- 找到答案就停
- 候选空间试完还没找到，就宣告失败
<!-- bilingual-en:start -->
Starting from an initial candidate, test candidates systematically, stop when one works, and report failure if the candidate space is exhausted.
<!-- bilingual-en:end -->

> [!note]
> 这里的 “guess” 很容易让人误会成“靠直觉乱试”。  
> 但老师强调的恰恰是 “be systematic”。  
> 它的本质是枚举 candidate space，不是拍脑袋。
> <!-- bilingual-en:start -->
> “Guess” does not mean relying on intuition. The instructor emphasizes being systematic: enumerate a candidate space rather than making arbitrary attempts.
> <!-- bilingual-en:end -->

### 9. perfect square 例子：第一次完整感受 exhaustive enumeration 的骨架
<!-- bilingual-en:start -->
*9. Perfect Squares: The Full Skeleton of Exhaustive Enumeration*
<!-- bilingual-en:end -->
最先讲的是 perfect square root。
<!-- bilingual-en:start -->
The first complete example searches for an integer square root.
<!-- bilingual-en:end -->

代码思路非常直接：
<!-- bilingual-en:start -->
The code is direct:
<!-- bilingual-en:end -->

```python
x = int(input("Enter an integer: "))
guess = 0
while guess**2 < x:
    guess += 1
if guess**2 == x:
    print(f"Square root of {x} is {guess}")
else:
    print(f"{x} is not a perfect square")
```

这里的结构值得记住：

- candidate space：`0, 1, 2, ...`
- test：`guess**2 == x`
- stop rule 1：找到答案
- stop rule 2：已经 overshoot，说明不是 perfect square
<!-- bilingual-en:start -->
- Candidate space: `0, 1, 2, ...`.
- Test: `guess**2 == x`.
- First stopping rule: an answer is found.
- Second stopping rule: the search overshoots, proving that `x` is not a perfect square.
<!-- bilingual-en:end -->

老师随后又加了 negative flag 版本，提醒你输入可能不在算法的默认假设里。
<!-- bilingual-en:start -->
A later negative-flag version reminds you that an input may fall outside the algorithm's initial assumptions.
<!-- bilingual-en:end -->

### 10. 用“secret number 1 到 10”说明 exhaustive enumeration 的一般形态
<!-- bilingual-en:start -->
*10. A Secret Number from 1 to 10 Shows the General Form of Enumeration*
<!-- bilingual-en:end -->
在平方根之后，老师又给了一个更抽象、更容易泛化的小问题：
<!-- bilingual-en:start -->
After square roots, a more abstract exercise makes the pattern easier to generalize:
<!-- bilingual-en:end -->

- secret number 硬编码在程序里
- 从 `1` 到 `10` 逐个检查
- 找到就打印
- 如果走完整个区间还没找到，要么什么都不打印，要么明确打印没找到
<!-- bilingual-en:start -->
The secret number is hard-coded, every integer from `1` to `10` is tested, success is reported when found, and exhausting the interval must either produce no output or an explicit “not found” result.
<!-- bilingual-en:end -->

这一步其实在教你把 guess-and-check 从“数值运算题”迁移到更一般的 candidate search。
<!-- bilingual-en:start -->
This transfers guess-and-check from a numerical root problem to a general candidate search.
<!-- bilingual-en:end -->

也就是说，关键不在 square root，而在你会不会：

- 先定义 search space
- 再决定检查规则
- 最后决定找到/找不到时各做什么
<!-- bilingual-en:start -->
The essential design questions are the search space, the test, and the behavior on success and failure.
<!-- bilingual-en:end -->

### 11. cube root 和 `break`：一旦 overshoot，就可以提前退出
<!-- bilingual-en:start -->
*11. Cube Roots and `break`: Exit as Soon as the Search Overshoots*
<!-- bilingual-en:end -->
随后老师把同一个结构搬到 cube root 上。
<!-- bilingual-en:start -->
The same structure is then applied to cube roots.
<!-- bilingual-en:end -->

对 perfect cube，可以写：

```python
cube = int(input("Enter an integer: "))
for guess in range(abs(cube)+1):
    if guess**3 == abs(cube):
        ...
```

但她又马上展示更高效一点的版本：
<!-- bilingual-en:start -->
The instructor immediately improves it:
<!-- bilingual-en:end -->

- 当 `guess**3` 已经大于目标时
- 就不必继续试更大的 guess
- 于是可以用 `break` 提前退出
<!-- bilingual-en:start -->
Once `guess**3` exceeds the target, no larger guess can work, so `break` can terminate the search early.
<!-- bilingual-en:end -->

这也是为什么 Lecture 4 前半段要先讲 `break`。  
到这里它第一次真正进入算法，而不只是语法演示。
<!-- bilingual-en:start -->
This explains why `break` was introduced earlier: it now serves an algorithmic argument rather than merely demonstrating syntax.
<!-- bilingual-en:end -->

### 12. word problem：guess-and-check 不只解整数根，也可以解约束满足问题
<!-- bilingual-en:start -->
*12. A Word Problem: Guess-and-Check Also Solves Constraint Problems*
<!-- bilingual-en:end -->
老师还给了一个卖票的 word problem：
<!-- bilingual-en:start -->
The instructor also gives a ticket-selling word problem:
<!-- bilingual-en:end -->

- Alyssa, Ben, Cindy 三个人卖票
- 总数满足一个条件
- 人与人之间还有额外关系
<!-- bilingual-en:start -->
Alyssa, Ben, and Cindy sell tickets subject to a total and additional relationships among their numbers.
<!-- bilingual-en:end -->

最直接的写法是三重循环，把所有组合都试一遍；但老师紧接着展示更好的版本：  
如果某些变量能由另一个变量推出来，就没必要继续暴力枚举那么多层。
<!-- bilingual-en:start -->
A direct solution uses three nested loops to test every combination. A better version derives some variables from another one and avoids enumerating unnecessary dimensions.
<!-- bilingual-en:end -->

这一段很重要，因为它在暗示：

- guess-and-check 是通用策略
- 但写得好不好，差别很大
- 候选空间定义得越聪明，算法就越不笨
<!-- bilingual-en:start -->
The example shows that guess-and-check is general, but its efficiency depends strongly on how intelligently the candidate space is defined.
<!-- bilingual-en:end -->

### 13. `x += 0.1` 居然十次后不等于 1：这为下一讲挖了坑
<!-- bilingual-en:start -->
*13. Ten Additions of `0.1` Do Not Equal 1: Motivating the Next Lecture*
<!-- bilingual-en:end -->
在讲完整数枚举以后，老师突然插入一个短例子：
<!-- bilingual-en:start -->
After integer enumeration, the instructor inserts a short example:
<!-- bilingual-en:end -->

```python
x = 0
for i in range(10):
    x += 0.1
print(x == 1)
print(x, 'is the same as?', 10*0.1)
```

结果不是 `True`，而是浮点误差暴露出来。
<!-- bilingual-en:start -->
The result is not `True`; floating-point error becomes visible.
<!-- bilingual-en:end -->

这段内容表面上像离开了 guess-and-check，但实际上是在为下一讲讲 float approximation 做动机铺垫：

- 不是所有数值问题都能靠整数枚举解决
- 也不是所有数值都能精确表示
<!-- bilingual-en:start -->
The example prepares the next lecture on floating-point approximation: integer enumeration does not solve every numerical problem, and not every numerical value has an exact machine representation.
<!-- bilingual-en:end -->

### 14. 讲二进制不是跑题，而是在补“数在机器里怎么存”
<!-- bilingual-en:start -->
*14. Binary Is Not a Digression: It Explains How Numbers Are Stored*
<!-- bilingual-en:end -->
最后一大段内容转到 binary。  
老师先说，既然计算机最后都在用 bits，那就应该知道整数是怎么转成 binary representation 的。
<!-- bilingual-en:start -->
The final major section turns to binary representation. Since computers ultimately use bits, the course asks how an integer is converted to binary.
<!-- bilingual-en:end -->

核心算法是：

- 一直看当前数字除以 2 的 remainder
- remainder 只可能是 `0` 或 `1`
- 把它 prepend 到结果字符串前面
- 再把数字整除 2，继续做
<!-- bilingual-en:start -->
Repeatedly take the remainder on division by 2, prepend that `0` or `1` to the result string, replace the number by its integer quotient, and continue.
<!-- bilingual-en:end -->

```python
result = ''
while num > 0:
    result = str(num % 2) + result
    num = num // 2
```

为什么是 prepend 而不是 append？  
因为你最先得到的是最低位 bit，但最后的 binary string 需要高位在左边。
<!-- bilingual-en:start -->
Prepending is necessary because the least significant bit is discovered first, while the final representation places the most significant bit on the left.
<!-- bilingual-en:end -->

老师还补了 negative flag 版本，说明：

- 可以先把负号剥掉
- 用正数部分跑同样的算法
- 最后再把 `-` 接回去
<!-- bilingual-en:start -->
A negative-flag version removes the sign, runs the same positive-integer algorithm, and restores `-` afterward.
<!-- bilingual-en:end -->

> [!warning]
> 这一段最常见的错，是把 bit 加到字符串后面。  
> 那样你会得到反过来的二进制表示。
> <!-- bilingual-en:start -->
> A common mistake is to append each new bit, which reverses the intended binary representation.
> <!-- bilingual-en:end -->

## Exercise log
> [!example] Finger exercise 04
> 官方题目要求：给定正整数 `N`，找出它的 cube root；如果不是 perfect cube，就打印 `error`。
> <!-- bilingual-en:start -->
> Given a positive integer `N`, the official exercise finds its cube root or prints `error` when `N` is not a perfect cube.
> <!-- bilingual-en:end -->
>
> ```python
> i = 1
> while i**3 < N:
>     i += 1
> if i**3 == N:
>     print(i)
> else:
>     print('error')
> ```
>
> 这题正好对应本讲后半段刚引入的 guess-and-check 结构：
> - `i` 是系统枚举的 candidate
> - `i**3 < N` 是继续搜索的条件
> - `i**3 == N` 是成功判定
> - 否则就进入失败分支
> <!-- bilingual-en:start -->
> It directly instantiates the newly introduced guess-and-check structure:
> - `i` is the systematically enumerated candidate.
> - `i**3 < N` is the continuation condition.
> - `i**3 == N` is the success test.
> - Otherwise, execution enters the failure branch.
> <!-- bilingual-en:end -->
>
> 如果你能做出这题，说明你已经把 “候选空间 + 检验 + 成功/失败出口” 这个算法模板抓住了。
> <!-- bilingual-en:start -->
> Solving it shows that you understand the candidate-space, test, and success/failure-exit pattern.
> <!-- bilingual-en:end -->

## Links to follow-up practice
- Slides: [[MIT 6.100L-slides/mit6_100l_lec04.pdf|Lecture 04 slides]]
- Lecture code: [[MIT 6.100L-lecture-code/mit6_100l_lec04_code.py|Lecture 04 code (py)]]
- Finger exercise: [[MIT 6.100L-finger-exercises/mit6_100l_ex04_sol.pdf|Lecture 04 finger exercise solution]]
- Transcript: [[MIT 6.100L-transcripts/mit6_100l_lec04_transcript.pdf|Lecture 04 transcript]]
- Recitation: none attached to this lecture week
- Problem set milestone: none directly scheduled on this lecture
- Textbook: [[Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Guttag textbook]] (Ch 3.1, Ch 3.3)

## Review checklist
- [ ] 我能解释 `break` 为什么只退出当前最内层循环。
- [ ] 我能说明什么时候该直接 `for char in s`，什么时候真的需要 index。
- [ ] 我能把 robot cheerleader 程序拆成“拼写”和“重复输出”两个循环任务。
- [ ] 我能准确说出 guess-and-check 的核心不是乱猜，而是系统枚举候选空间。
- [ ] 我能写出 perfect square / perfect cube 的 exhaustive enumeration 程序。
- [ ] 我能解释为什么 overshoot 以后就可以 `break`。
- [ ] 我能说明 word problem 里 candidate space 的设计为什么会决定程序快慢。
- [ ] 我能解释为什么 `x += 0.1` 会给后面 float 误差问题埋下伏笔。
- [ ] 我能从头写出十进制正整数转二进制的循环。
- [ ] 我能解释为什么 bit 要 prepend 到结果字符串前面。
<!-- bilingual-en:start -->
- [ ] I can explain why `break` exits only the innermost current loop.
- [ ] I can decide when to write `for char in s` directly and when an index is genuinely needed.
- [ ] I can decompose the robot cheerleader into spelling and repeated-output loops.
- [ ] I can explain why guess-and-check is systematic enumeration rather than arbitrary guessing.
- [ ] I can write exhaustive searches for perfect squares and perfect cubes.
- [ ] I can justify using `break` after an overshoot.
- [ ] I can explain how the design of a candidate space controls the speed of the word-problem solution.
- [ ] I can explain how `x += 0.1` foreshadows floating-point error.
- [ ] I can write decimal-positive-integer to binary conversion from scratch.
- [ ] I can explain why each new bit must be prepended to the result string.
<!-- bilingual-en:end -->

> [!warning] Common mistakes
> - 明明不需要 index，却硬写 `range(len(s))`，让代码变得更绕。
> - 把 guess-and-check 写成随意尝试，而不是系统枚举候选空间。
> - 做二进制转换时把新 bit 加在字符串后面，导致结果顺序反了。
> <!-- bilingual-en:start -->
> - Using `range(len(s))` when no index is needed, making the code unnecessarily indirect.
> - Treating guess-and-check as arbitrary trial rather than systematic enumeration of a candidate space.
> - Appending each new bit during binary conversion and thereby reversing the result.
> <!-- bilingual-en:end -->
