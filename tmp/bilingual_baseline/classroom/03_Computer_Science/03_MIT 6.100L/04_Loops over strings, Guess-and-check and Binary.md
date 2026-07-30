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

## Lecture flow

### 1. 开场先回顾 loops，然后补一个很实用的细节：`break`
Lecture 4 一开始并没有直接进入 guess-and-check，而是先接着上一讲的 while/for 往前推一点。

老师先回顾：

- `while`：只要 condition 为真就继续
- `for`：依次走过一个 sequence of values

然后她马上提出一个很实际的问题：  
如果我们还没走到 while 的自然结束，或者还没把 for 的序列走完，但已经知道“可以停了”，该怎么办？

答案就是 `break`。

> [!note]
> `break` 的语义非常具体：  
> 立刻退出当前所在的最内层循环，并跳过这个循环剩下的代码。

老师先用嵌套循环说明它只会跳出 “surrounding loop”，不会把外层循环也一并终止。这个地方是在纠正常见误解：`break` 不是“结束整个程序”，也不是“让所有循环同时停”。

### 2. 用一个很小的 for-loop 例子看出 `break` 到底会切掉哪些代码
接着老师给了一个很短的 for-loop：

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

- `range(5, 11, 2)` 只会产生 `5, 7, 9`
- 第一次 `mysum += i` 后，`mysum` 立刻变成 `5`
- 触发 `break` 后，循环体中后面的 `mysum += 1` 永远不会执行
- 也不会回去检查下一个 `i`

所以这段代码最后打印 `5`。

> [!warning]
> 这一段是用来训练“控制流可视化”的。  
> 只要看到 `break`，你就要立刻问：这一行下面、以及这次循环之后的哪些代码，已经没有机会运行了？

### 3. 用“统计偶数个数”的暖身题，把上一讲循环模式再稳一遍
在进入新内容前，老师安排了一个非常标准的小练习：  
给一个 `range(...)`，统计其中有多少个 even numbers。

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

这也是老师在这门课里反复强调的节奏：  
每次讲新东西前，先用一个小题把上一讲的基础动作重新激活。

### 4. for loop 的 sequence 不只可以是数字，也可以直接是字符串字符
这一讲真正的前半段重点，是让你意识到：

> [!note]
> for loop 不只是在 `range(...)` 上数数。  
> 只要是 sequence，都可以直接迭代。

老师先从一个很普通的任务开始：检查字符串里是否有 `i` 或 `u`。

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

于是老师顺势给出第二种、更自然的写法：

```python
for char in s:
    if char == 'i' or char == 'u':
        print("There is an i or u")
```

再进一步，写成更 Pythonic 的版本：

```python
for char in s:
    if char in 'iu':
        print("There is an i or u")
```

这一小段课的真正收获不是 “我会检查元音” 这种表层能力，而是你开始会判断：

- 我到底需要 index，还是只需要 element 本身？
- 如果只需要 element，为什么还要硬写 `range(len(...))`？

### 5. 机器人 cheerleader：第一次把字符串循环、条件分支和重复输出串成一个小程序
接下来老师用了一个很有课堂感的例子：robot cheerleaders。

程序接受：

- 一个单词
- 一个 enthusiasm level

然后它会：

1. 按字母逐个拼读这个单词
2. 判断每个字母前该说 `a` 还是 `an`
3. 再把整个单词带着感叹号重复打印若干次

这段程序为什么重要？因为它第一次把你前几讲零散学到的东西合到一起：

- `for w in word`：循环直接走字符
- `if w in an_letters`：用字符串 membership 做条件判断
- `input(...)` + `int(...)`：交互输入
- 第二个循环 `for i in range(times)`：按用户给的次数重复动作

> [!example]
> 这段代码的结构可以概括成：
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

### 6. unique letters 练习：scan 一个 sequence，同时维护“已经见过什么”
然后老师给了一个稍微更算法化一点的练习：  
给定 `s = "abca"`，数出里面有多少 unique letters。

她给的提示非常关键：

- 一路扫描 `s`
- 维护一个 `seen` 字符串
- 只有当前字符不在 `seen` 里时，才把它加进去

这已经非常接近后面会反复出现的“scan + state”模式：

- 你沿着 sequence 一路走
- 每一轮都用当前元素更新某个状态容器

只是这里的状态容器还比较朴素，是一个 string；后面会升级成 list、dict、set 风格的思维。

### 7. 课堂中段的转折：我们现在终于有 enough tools 去实现 algorithms
到这里老师明确停了一下，说：

- objects
- expressions
- branching
- loops

这些东西加在一起，其实已经足够实现很多算法了。

这句话标志着这一讲中段的转折。  
前半段是在补工具箱，后半段开始真正讲 “algorithmic method”。

### 8. guess-and-check 不是“随便猜”，而是 exhaustive enumeration
进入算法部分后，老师先讲的是 `guess-and-check`，也就是 `exhaustive enumeration`。

她把这个方法抽象成两件事：

1. 你能生成候选解
2. 你能检查候选解是否正确

然后就可以：

- 从某个初始 guess 开始
- 系统地一个一个试
- 找到答案就停
- 候选空间试完还没找到，就宣告失败

> [!note]
> 这里的 “guess” 很容易让人误会成“靠直觉乱试”。  
> 但老师强调的恰恰是 “be systematic”。  
> 它的本质是枚举 candidate space，不是拍脑袋。

### 9. perfect square 例子：第一次完整感受 exhaustive enumeration 的骨架
最先讲的是 perfect square root。

代码思路非常直接：

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

老师随后又加了 negative flag 版本，提醒你输入可能不在算法的默认假设里。

### 10. 用“secret number 1 到 10”说明 exhaustive enumeration 的一般形态
在平方根之后，老师又给了一个更抽象、更容易泛化的小问题：

- secret number 硬编码在程序里
- 从 `1` 到 `10` 逐个检查
- 找到就打印
- 如果走完整个区间还没找到，要么什么都不打印，要么明确打印没找到

这一步其实在教你把 guess-and-check 从“数值运算题”迁移到更一般的 candidate search。

也就是说，关键不在 square root，而在你会不会：

- 先定义 search space
- 再决定检查规则
- 最后决定找到/找不到时各做什么

### 11. cube root 和 `break`：一旦 overshoot，就可以提前退出
随后老师把同一个结构搬到 cube root 上。

对 perfect cube，可以写：

```python
cube = int(input("Enter an integer: "))
for guess in range(abs(cube)+1):
    if guess**3 == abs(cube):
        ...
```

但她又马上展示更高效一点的版本：

- 当 `guess**3` 已经大于目标时
- 就不必继续试更大的 guess
- 于是可以用 `break` 提前退出

这也是为什么 Lecture 4 前半段要先讲 `break`。  
到这里它第一次真正进入算法，而不只是语法演示。

### 12. word problem：guess-and-check 不只解整数根，也可以解约束满足问题
老师还给了一个卖票的 word problem：

- Alyssa, Ben, Cindy 三个人卖票
- 总数满足一个条件
- 人与人之间还有额外关系

最直接的写法是三重循环，把所有组合都试一遍；但老师紧接着展示更好的版本：  
如果某些变量能由另一个变量推出来，就没必要继续暴力枚举那么多层。

这一段很重要，因为它在暗示：

- guess-and-check 是通用策略
- 但写得好不好，差别很大
- 候选空间定义得越聪明，算法就越不笨

### 13. `x += 0.1` 居然十次后不等于 1：这为下一讲挖了坑
在讲完整数枚举以后，老师突然插入一个短例子：

```python
x = 0
for i in range(10):
    x += 0.1
print(x == 1)
print(x, 'is the same as?', 10*0.1)
```

结果不是 `True`，而是浮点误差暴露出来。

这段内容表面上像离开了 guess-and-check，但实际上是在为下一讲讲 float approximation 做动机铺垫：

- 不是所有数值问题都能靠整数枚举解决
- 也不是所有数值都能精确表示

### 14. 讲二进制不是跑题，而是在补“数在机器里怎么存”
最后一大段内容转到 binary。  
老师先说，既然计算机最后都在用 bits，那就应该知道整数是怎么转成 binary representation 的。

核心算法是：

- 一直看当前数字除以 2 的 remainder
- remainder 只可能是 `0` 或 `1`
- 把它 prepend 到结果字符串前面
- 再把数字整除 2，继续做

```python
result = ''
while num > 0:
    result = str(num % 2) + result
    num = num // 2
```

为什么是 prepend 而不是 append？  
因为你最先得到的是最低位 bit，但最后的 binary string 需要高位在左边。

老师还补了 negative flag 版本，说明：

- 可以先把负号剥掉
- 用正数部分跑同样的算法
- 最后再把 `-` 接回去

> [!warning]
> 这一段最常见的错，是把 bit 加到字符串后面。  
> 那样你会得到反过来的二进制表示。

## Exercise log
> [!example] Finger exercise 04
> 官方题目要求：给定正整数 `N`，找出它的 cube root；如果不是 perfect cube，就打印 `error`。
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
>
> 如果你能做出这题，说明你已经把 “候选空间 + 检验 + 成功/失败出口” 这个算法模板抓住了。

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

> [!warning] Common mistakes
> - 明明不需要 index，却硬写 `range(len(s))`，让代码变得更绕。
> - 把 guess-and-check 写成随意尝试，而不是系统枚举候选空间。
> - 做二进制转换时把新 bit 加在字符串后面，导致结果顺序反了。
