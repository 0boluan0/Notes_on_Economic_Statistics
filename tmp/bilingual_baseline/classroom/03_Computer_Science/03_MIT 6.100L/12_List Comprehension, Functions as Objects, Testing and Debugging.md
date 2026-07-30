---
aliases:
  - MIT 6.100L Lecture 12
  - 6.100L L12
  - List Comprehension, Functions as Objects, Testing, and Debugging
tags:
  - computer-science
  - python
  - mit-6.100l
  - lecture-note
科目: Computer Science
course: MIT 6.100L Introduction to CS and Programming Using Python
lecture: 12
---

# Lecture 12: List Comprehension, Functions as Objects, Testing, and Debugging

> [!tip] Hint
> - 这节课前半段先收尾 lists 和 functions，后半段才转入 testing/debugging，所以正文顺序不能写反。
> - list comprehension 的动机是：某类“建新列表”的 for-loop 模式太常见，Python 给了更紧凑写法。
> - comprehension 里既有“对每个元素做什么”，也可以有“满足什么条件才保留”。
> - keyword argument 和 default parameter 是从 `bisection_root_new` 这种具体函数一点点引出来的。
> - `epsilon=0.01` 这类默认值不是语法糖而已，它在定义“调用者省略参数时的默认行为”。
> - `make_prod(a)` 返回函数，是为了再推一步“函数对象可以被制造和返回”。
> - 课堂后半段把 testing 和 debugging 分开讲：一个是设计检查方法，一个是定位并修 bug。
> - black-box testing 看 specification，glass-box testing 看实现路径；两者都要，但关心点不同。
> - buggy palindrome 的调试过程是本讲最值得模仿的部分：先复现，再加打印，再缩小问题，再修一个 bug，再发现第二个 bug。
> - 听完这节课，你应该能复述一个系统的 debugging recipe，而不是“出错了就随便改两行试试”。

## Lecture flow

### 1. 开场先说明：这讲是在收尾 lists 和 functions
老师一开始就说这会是一节相对 “chill” 的课。  
但这不代表内容散，而是因为它在做两件事：

- 把前面 lists / functions 的一些常见模式整理起来
- 再把课堂视角抬到 testing 和 debugging

所以 Lecture 12 的推进顺序很重要：

1. list comprehension
2. keyword / default parameters
3. function returning function
4. testing strategies
5. debugging method

### 2. list comprehension：把“建新列表”的套路压成一行
老师先回顾一个大家已经很熟的模式：

```python
new_list = []
for e in old_list:
    new_list.append(expr(e))
return new_list
```

这类代码实在太常见了，所以 Python 提供了更紧凑的写法：

```python
[expr(e) for e in old_list]
```

如果还要先筛选，再加工，则是：

```python
[expr(e) for e in old_list if test(e)]
```

课堂这里不是追求“写短”，而是让你看出 comprehension 对应的原始结构是什么。

> [!note]
> comprehension 永远都可以还原成“循环 + 条件 + append”的展开写法。

### 3. comprehension 的两个位置要分清
老师随后反复用例子拆 comprehension 的结构。

以：

```python
[e**2 for e in range(8) if e % 2 == 0]
```

为例：

- `e**2`：新列表中每个元素长什么样
- `for e in range(8)`：从哪里依次取元素
- `if e % 2 == 0`：哪些元素能留下

如果这里顺序看错，就会写出语法对但语义乱的表达式。  
这也是为什么老师花了不少时间让大家把展开版和压缩版互相翻译。

### 4. list comprehension 不是新能力，而是旧能力的浓缩
老师还专门写了一个等价函数：

```python
def f(expr, old_list, test=lambda x: True):
    new_list = []
    for e in old_list:
        if test(e):
            new_list.append(expr(e))
    return new_list
```

这一步很有教学意义，因为它让你看到：

- comprehension 没有创造新计算能力
- 它只是把旧模式浓缩成更紧的语法

所以如果你看 comprehension 看不懂，最好的办法不是死盯一行，而是把它先展开回普通循环。

### 5. 接着讲参数接口：keyword arguments 和 default parameters
把 list comprehension 讲完后，课堂切回函数接口设计。

老师拿前面已经学过的平方根函数继续改造：

- 先是 `bisection_root(x)`
- 再变成 `bisection_root_new(x, epsilon)`
- 最后变成 `bisection_root_new(x, epsilon=0.01)`

这组推进顺序很重要，因为它说明 default parameter 不是凭空出现的：

- 先发现“某个参数经常取同一个值”
- 再考虑把它设为默认值

### 6. default parameter 的意义：调用更轻，但语义要更清楚
课堂里老师反复围绕 `epsilon` 讨论：

- 如果调用者不传 `epsilon`，默认该是多少
- 为什么 `0.01` 可以是一个合理默认值
- 如果想覆盖默认行为，该怎么传参

所以默认参数的真正意义是：

- 给出一种常见、可接受的默认行为
- 让简单调用更短
- 但仍保留调用者显式指定的能力

例如：

```python
bisection_root_new(123)
bisection_root_new(123, 0.5)
bisection_root_new(123, epsilon=0.00001)
bisection_root_new(epsilon=0.001, x=123)
```

这里也自然引出了 keyword arguments：

- 用参数名写调用
- 可读性更好
- 顺序也更灵活

### 7. 一个函数还能返回另一个函数：`make_prod`
讲完 default parameter，老师又推了一步 functions-as-objects。

例子是：

```python
def make_prod(a):
    def g(b):
        return a * b
    return g
```

这段代码的重要性在于：

- 函数不仅能作为参数传入
- 还可以作为返回值送出去

于是：

```python
doubler = make_prod(2)
val = doubler(3)
```

就读成：

- 先生成一个“乘以 2”的函数
- 再在别处调用它

这一部分其实在为更高级的抽象方式做铺垫，但课堂里不会把术语讲得太重，重点还是让你接受“函数对象可以被构造和返回”。

### 8. 课堂后半段突然抬高视角：不再只问“能不能跑”，而是“怎么验证”
到中后段，老师明确转入 testing and debugging。

这时课程气质明显变了：  
前半段在教写法，后半段在教工程习惯。

老师先把几个词分开：

- validation / testing：程序是否按预期工作
- debugging：程序不对时，如何系统地定位原因

这两个动作往往在现实里交织出现，但课堂故意拆开讲，是为了避免你把“乱试输入”和“修 bug”混为一谈。

### 9. 测试策略：unit / regression / integration
老师先给出几种更工程化的 testing 视角：

- unit testing：单个函数或模块分别测
- regression testing：修改以后，确认原来通过的东西没被改坏
- integration testing：多个部件组合后一起测

这部分的课堂重点是：  
测试不是“随便挑几个输入跑一下”，而是要知道自己在检查哪一层行为。

### 10. black-box testing：按 specification 设计输入
老师随后讲 **black-box testing**。

黑箱测试的核心是：

- 把函数当成黑箱
- 只看 specification
- 不依赖内部实现细节来设计测试

所以你会优先想：

- 正常输入
- 边界输入
- 特殊输入
- 是否有空输入、重复值、极端值

如果函数实现被完全重写，但 specification 不变，那么黑箱测试仍然应该有效。

### 11. glass-box testing：看实现路径是否覆盖到
接着老师再讲 **glass-box testing**。

玻璃盒测试的视角恰恰相反：

- 你知道函数内部结构
- 于是你关心哪些分支、循环路径、特殊路径有没有真的走到

比如某个函数对正数和负数走不同分支，那么 glass-box testing 会逼你主动构造两类输入。

所以这两种测试不是互相替代，而是互补：

- black-box 逼你尊重 specification
- glass-box 逼你覆盖实现细节

### 12. debugging recipe：不要乱改，先收集证据
讲完 testing 之后，老师转向 debugging，并给出非常明确的方法论。

核心思想是：

1. 先复现 bug
2. 选最小测试案例
3. 在关键位置加打印
4. 观察期望和实际在哪一步开始分叉
5. 一次只修一个问题
6. 修完再重新测试

这套 recipe 是整节课最值得模仿的部分，因为它把 debugging 从“靠感觉”变成了一个过程。

### 13. buggy palindrome：课堂现场示范如何 debug
老师把这套 recipe 用在了一个有 bug 的 palindrome 函数上。

最初版本大致是：

```python
def is_pal(x):
    temp = x
    temp.reverse
    if temp == x:
        return True
    else:
        return False
```

然后按步骤调：

1. 先跑测试用例
2. 发现结果不对
3. 在中间打印 `temp` 和 `x`
4. 发现 `reverse` 甚至没真正调用，因为少了括号
5. 修成 `temp.reverse()`
6. 又发现 `temp = x` 带来 aliasing，反转的是同一个列表
7. 最后改成 `temp = x[:]`

这个示范非常好，因为它连续暴露了两种不同 bug：

- 语法层面 / 调用层面错误
- aliasing / mutability 层面错误

> [!example]
> 真正好的 debugging 不是一步到位，而是先修掉第一个确认的问题，再看是否还有第二层问题。

### 14. Wordle 文件：让你把 debugging 方法迁移出去
课程最后提到 `lec12_wordle.py` 是有 bug 的，鼓励你自己去修。

这不是附带作业，而是在告诉你：

- 课堂示范的 debugging 方法不是只对 palindrome 有效
- 只要代码稍微变大，系统方法就更重要

Lecture 12 到这里实际上完成了一个关键转折：  
课程开始正式要求你把“写代码”升级成“写、测、修代码”。

## Exercise log

> [!example] Finger exercise 12
> 官方练习是 `count_sqrts(nums_list)`：
> - `nums_list` 只含正数且无重复
> - 统计列表中有多少元素，其平方也在同一个列表里

官方解法非常短：

```python
cnt = 0
for i in nums_list:
    if i * i in nums_list:
        cnt += 1
return cnt
```

这题虽然不复杂，但很适合放在本讲之后，因为它逼你把几种“列表处理模式”压缩到一个短函数里：

- 遍历列表
- membership test
- 计数器模式

如果你想进一步练本讲前半段内容，可以把它改写成 list comprehension 风格去理解。

## Links to follow-up practice
- Slides: [[MIT 6.100L-slides/mit6_100l_lec12.pdf|Lecture 12 slides]]
- Lecture code: [[MIT 6.100L-lecture-code/mit6_100l_lec12_code.zip|Lecture 12 code (zip)]]
- Finger exercise: [[MIT 6.100L-finger-exercises/mit6_100l_ex12_sol.pdf|Lecture 12 finger exercise solution]]
- Transcript: [[MIT 6.100L-transcripts/mit6_100l_lec12_transcript.pdf|Lecture 12 transcript]]
- Recitation 6: [[MIT 6.100L-recitations/mit6_100l_rec06.zip|Recitation 06 materials]]
- PS 3 out: [[MIT 6.100L-problem-sets/mit6_100l_ps3.pdf|PS3 statement]], [[MIT 6.100L-problem-sets/mit6_100l_ps3_code.zip|PS3 starter code]]
- PS 2 due: [[MIT 6.100L-problem-sets/mit6_100l_ps2.pdf|PS2 statement]], [[MIT 6.100L-problem-sets/mit6_100l_ps2_code.zip|PS2 starter code]]
- Textbook: [[Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Guttag textbook]] (Ch 4.4, Ch 8)

## Review checklist
- [ ] 我能把一条 list comprehension 展开成普通循环，也能反过来压缩回去。
- [ ] 我能解释 comprehension 中 expression、iteration、filter 各在什么位置。
- [ ] 我能说明 default parameter 为什么是接口设计问题，而不只是语法方便。
- [ ] 我能区分位置参数、keyword argument、default value 的角色。
- [ ] 我能解释 `make_prod` 为什么说明函数也可以作为返回值。
- [ ] 我能区分 testing 和 debugging。
- [ ] 我能解释 black-box testing 和 glass-box testing 的不同关注点。
- [ ] 我能复述老师示范的 debugging recipe。
- [ ] 我能说出 buggy palindrome 里两个不同层面的 bug 分别是什么。
- [ ] 我能按课堂顺序复述：list comprehension -> function parameters -> testing -> debugging。

> [!warning] Common mistakes
> - 把 list comprehension 当成“新功能”，却不知道它对应哪段普通循环。
> - default parameter 写了以后，没想清默认行为是否合理。
> - 以为测试就是随便挑几个输入跑一下。
> - debugging 时到处乱改，不先缩小问题范围。
