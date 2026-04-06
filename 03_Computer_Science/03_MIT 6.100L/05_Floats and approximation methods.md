---
aliases:
  - MIT 6.100L Lecture 05
  - 6.100L L05
  - Floats and approximation methods
tags:
  - computer-science
  - python
  - mit-6.100l
  - lecture-note
科目: Computer Science
course: MIT 6.100L Introduction to CS and Programming Using Python
lecture: 05
---

# Lecture 05: Floats and approximation methods

> [!tip] Hint
> - 这节课的出发点是上节那个看似离谱的事实：`0.1` 加十次不等于 1。
> - 老师先回顾十进制整数转二进制，再问“那小数怎么办”，整个 lecture 的推进是从表示问题走到算法问题。
> - `3/8` 能精确写成有限二进制，而 `1/10` 不行，这是理解 float 不精确的关键分水岭。
> - float 不是“近似实数”这个哲学句子而已，老师真正在讲的是：机器只能用有限 bits 近似无限展开。
> - approximation method 不是猜-and-check 的重复，而是把 candidate 从整数枚举换成了小步长浮点枚举。
> - epsilon 决定什么叫 close enough，increment 决定你走得多快，两者不是一回事。
> - 54321 的平方根例子故意让程序很慢，是为了让你真正感到 fixed increment 的代价。
> - 近似算法的失败不是 bug，而是算法结构本身必须显式处理的一个出口。
> - 本讲的真正主线是：一旦“完全相等”不可信，程序就必须改成“足够接近 + 明确失败分支”。
> - 这节课其实在为 bisection search 做铺垫：我们已经知道小步子能逼近，但也已经看到它太慢。

## Lecture flow

### 1. 开场先回到那个令人不安的例子：`0.1` 累加十次不等于 1
Lecture 5 一开始就回到上一讲埋下的问题：

```python
x = 0
for i in range(10):
    x += 0.1
print(x == 1)
print(x, '==', 10*0.1)
```

程序给出的不是你以为的 `True`，而是 `False`，因为最后的 `x` 实际上会变成一串非常接近 1、但不完全等于 1 的数。

老师把这当作整讲的动机：

- 如果 float 连最基本的 equality 都不稳定
- 那我们以后做数值计算怎么办？

所以这一讲并不是随便讲讲机器底层，而是在回答一个非常直接的问题：  
为什么这种事会发生？

### 2. 先回顾整数转二进制算法，因为接下来要谈“小数怎么存”
老师先快速回顾上一讲的整数转 binary algorithm。

回顾它的目的不是重复内容，而是建立对照：

- 整数部分有一套很干净的表示方法
- 那么 fraction 部分能不能也找到同样干净的 recipe？

这一转场很自然。因为一旦你接受“计算机最终都用 bits 表示数”，那么接下来一定会问：

- 整数怎么用 bits 表示？
- 小数怎么用 bits 表示？

### 3. 十进制 fraction 和二进制 fraction 的类比
老师随后写出十进制小数的意义：

- `0.abc` 在十进制里表示  
  `a*10^-1 + b*10^-2 + c*10^-3 + ...`

然后把这个思路平移到二进制：

- `0.abc` 在二进制里表示  
  `a*2^-1 + b*2^-2 + c*2^-3 + ...`

所以如果你想把一个十进制 fraction 变成二进制 fraction，本质是在问：

- 能否找到一串 `0` / `1`
- 让这些 `2` 的负幂加起来等于原来的值

### 4. `3/8` 为什么可以精确表示，而 `1/10` 不行
接着老师给了一个关键例子：`3/8 = 0.375`。

思路是：

- 如果能乘上某个 `2**p` 变成整数
- 就先把它转成整数的二进制表示
- 再把小数点挪回去

例如：

- `0.375 * 2**3 = 3`
- `3` 的二进制是 `11`
- 所以 `0.375` 的二进制就是 `0.011`

这说明 `3/8` 是一个 “power-of-two friendly” 的 fraction。  
但 `1/10 = 0.1` 不存在这样的 `p`，永远无法乘成一个整数。

> [!note]
> 这就是本讲最重要的分界线：  
> 有些十进制 fraction 在二进制里是有限展开；  
> 有些则必须无限展开。

一旦需要无限展开，机器就只能截断或舍入，于是误差就出现了。

### 5. float 的本质：用有限 bits 近似潜在无限展开
老师把这个结论收成一句更一般的话：

- integers 在 binary 里比较直接
- real numbers/fractions 则可能需要无限多位
- 但机器只能存有限 bits

所以 float 本质上是近似表示，而不是数学上的精确实数。

接着她介绍了 floating point number 的一个简单抽象：

- significand / significant digits
- exponent

也就是把数表示成某种 “有效数字 × 2 的幂” 的形式。  
这样做的目的不是让你去背 IEEE 标准，而是让你理解：

- float 其实是一个有限位宽的工程折中
- 一旦位数有限，就一定有 rounding

### 6. 用一段 fraction-to-binary 的代码，体会“并不是所有数都能终止”
老师展示了一段把十进制 fraction 转成二进制的代码。

核心结构大致是：

```python
p = 0
while ((2**p) * x) % 1 != 0:
    p += 1
```

这段代码在做的事情是：

- 不断尝试把 `x` 乘上更大的 `2**p`
- 看看能不能变成整数

对 `0.625` 这类数，它会成功；  
对 `0.1`，它就会一直找不到让小数部分变成 0 的那个 `p`。

老师借这个例子想让你真正看到：

- “二进制里无限展开” 不是一句抽象判断
- 它会直接体现在程序无法找到终止条件

### 7. 既然 float 不精确，那数值算法就不能再死盯 `==`
讲完表示问题后，整讲进入第二部分：approximation method。

这里的出发点很清楚：

- 对 perfect square，我们可以用整数 guess-and-check
- 但大多数平方根不是整数
- 而且 float equality 本身也不可靠

所以新的问题变成：

> 我们不再追求 “guess**2 恰好等于 x”，  
> 而是追求 “guess**2 离 x 足够近”。

这就是 `epsilon` 出场的原因。

### 8. approximation method：把整数枚举换成小步长浮点枚举
老师给出的基础版本大致是：

```python
x = 36
epsilon = 0.01
guess = 0.0
increment = 0.0001
while abs(guess**2 - x) >= epsilon:
    guess += increment
```

这个算法和上一讲 guess-and-check 非常像，只是候选空间变了：

- 以前是 `0, 1, 2, 3, ...`
- 现在是 `0.0, 0.0001, 0.0002, 0.0003, ...`

仍然是系统枚举，只不过步长更细。

> [!example]
> 这个算法的四个核心量要分清：
> - `x`：目标值
> - `guess`：当前候选
> - `increment`：每次往前走多大一步
> - `epsilon`：多近才算接受

### 9. 第一次真正感到算法“太慢”：54321 的平方根
老师没有停在小数字上，而是故意让大家看一个大输入，比如 `x = 54321`。

这时即便 `increment = 0.0001`，程序也会变得非常慢。  
她甚至在代码里加了周期性打印当前 guess 的语句，让你看到程序还在非常机械地一点点往前挪。

这一段在课堂上的作用不是“让程序跑出来”，而是让你切身体会到：

- fixed increment 当然能逼近
- 但它可能极慢
- 特别是在目标值很大、精度要求又不低的时候

这也是 Lecture 6 要讲 bisection search 的直接铺垫。

### 10. approximation method 的失败不是偶发，而是结构上必须承认的结果
老师随后又指出一个更 subtle 的问题：

- 不是每次 increment 都能刚好踩进 epsilon neighborhood
- 也可能你不断往前走，结果 `guess**2` 已经超过 `x`，却仍然没有达到要求精度

因此一个更完整的版本会写成：

```python
while abs(guess**2 - x) >= epsilon and guess**2 <= x:
    guess += increment
    num_guesses += 1

if abs(guess**2 - x) >= epsilon:
    print(f"Failed on square root of {x}")
else:
    print(f"{guess} is close to square root of {x}")
```

这段极其重要，因为它把 “失败分支” 正式引进了数值算法：

- 近似法不是永远成功
- 如果你没有写失败分支，程序就会把一个不可信的 guess 冒充成答案

> [!warning]
> 很多初学者会把 “程序给出了一个数” 和 “程序给出了可信答案” 混为一谈。  
> 这一讲就是在帮你建立这个区别。

### 11. epsilon 与 increment 控制的是不同维度
课堂里虽然没有把这两个量做成单独术语表，但你应该自己在笔记里分开记：

- `epsilon` 控制接受标准
- `increment` 控制搜索粒度和速度

两者之间不是替代关系。

如果：

- `epsilon` 很小，但 `increment` 很粗

你可能根本踩不到可接受区间。  
如果：

- `increment` 很细

你可能终究能找到更好的近似，但速度会非常慢。

所以算法设计不是只盯一个参数，而是平衡：

- 精度
- 运行时间
- 失败处理

### 12. 这节课的真正结论：一旦不再追求精确相等，程序结构就要一起变化
Lecture 5 到最后，真正发生变化的不是某一段代码，而是你的判断标准。

从现在起，数值程序里经常要问：

- 我能不能精确表示目标值？
- equality 是否可信？
- 我是不是应该改用 closeness test？
- 如果近似法失败了，程序如何显式说明？

所以这一讲并不是单纯讲 float 或近似算法，而是在重写你对“数值正确性”的直觉。

## Exercise log
> [!example] Finger exercise 05
> 官方题目要求：给定字符串 `my_str`，打印出其中偶数 index 的字符。
>
> ```python
> s = ''
> for i in range(0, len(my_str), 2):
>     s += my_str[i]
> print(s)
> ```
>
> 这题和 lecture 标题里的 `float` 看起来不完全一致，但它实际对应本讲前半段仍在使用的一个基础模式：
> - 你必须清楚 index 的步长含义
> - 你必须会构造一个新字符串来累计结果
>
> 它也顺手提醒你：即使课程主线已经转向数值方法，字符级循环和序列处理仍然是需要持续熟练的底层动作。

## Links to follow-up practice
- Slides: [[MIT 6.100L-slides/mit6_100l_lec05.pdf|Lecture 05 slides]]
- Lecture code: [[MIT 6.100L-lecture-code/mit6_100l_lec05_code.py|Lecture 05 code (py)]]
- Finger exercise: [[MIT 6.100L-finger-exercises/mit6_100l_ex05_sol.pdf|Lecture 05 finger exercise solution]]
- Transcript: [[MIT 6.100L-transcripts/mit6_100l_lec05_transcript.pdf|Lecture 05 transcript]]
- Recitation: none attached to this lecture week
- PS 1 halfway hand-in due: [[MIT 6.100L-problem-sets/mit6_100l_ps1.pdf|PS1 statement]], [[MIT 6.100L-problem-sets/mit6_100l_ps1_code.zip|PS1 starter code]]
- Textbook: [[Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Guttag textbook]] (Ch 3.2-3.3)

## Review checklist
- [ ] 我能解释为什么 `0.1` 连加十次可能不等于 1。
- [ ] 我能说明 `3/8` 和 `1/10` 在二进制表示上的关键差别。
- [ ] 我能用自己的话解释 float 为什么只能近似表示某些实数。
- [ ] 我能说出 approximation method 和整数 guess-and-check 的结构相同点与不同点。
- [ ] 我能区分 `epsilon` 和 `increment` 分别控制什么。
- [ ] 我能解释为什么 fixed increment 方法在大输入上会非常慢。
- [ ] 我能说明为什么近似算法必须有失败分支。
- [ ] 我能判断什么时候应该用 `abs(guess**2 - x) < epsilon` 这类 close-enough 测试。
- [ ] 我能解释为什么“程序输出了一个数”不等于“程序找到了可信答案”。
- [ ] 我能把本讲和下一讲连起来：为什么看完 approximation method 之后，自然会想找更快的搜索方式。

> [!warning] Common mistakes
> - 把 float 当成精确实数，写大量 `==` 判断。
> - 只关心 epsilon，不关心步长和边界条件，导致程序极慢或失败。
> - 近似算法没有失败分支，最后得到一个看起来像答案但其实不可信的结果。
