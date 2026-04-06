---
aliases:
  - MIT 6.100L Lecture 02
  - 6.100L L02
  - Strings, Input/Output, and Branching
tags:
  - computer-science
  - python
  - mit-6.100l
  - lecture-note
科目: Computer Science
course: MIT 6.100L Introduction to CS and Programming Using Python
lecture: 02
---

# Lecture 02: Strings, Input/Output, and Branching

> [!tip] Hint
> - 这节课一开始先复习了上一讲的 memory diagram，因为今天所有 string、input、branching 都继续依赖“变量绑定对象”这套图景。
> - 新对象类型 string 是按 sequence 引入的，不是按“文本”这个直觉引入的。
> - 课堂先讲了 string 的 `+` 和 `*`，再讲 `len`、indexing、slicing，顺序是从最直观到最容易出错。
> - indexing 从 0 开始、negative index 从右往左数、slice 的 stop 不包含自己，这三件事是本讲最容易混的边界。
> - 老师专门强调 string 是 immutable，因为后面很多“为什么不能改某一位字符”的疑问都从这里来。
> - print 这一段真正想说的是：在 shell 里看到表达式结果，不等于你已经学会了给用户输出。
> - input 这一段最关键的结论是：用户输入无论看起来像什么，进程序时都先是 string。
> - Newton cube-root 例子不是为了教数值方法本身，而是为了说明：即使现在还不会循环，也已经能写出“一步更新”的数值程序。
> - branching 先从 Boolean expression 和 `==` / `!=` 讲起，再进入 `if`、`elif`、`else`；老师是在先铺“条件”，再铺“分支”。
> - 本讲结尾的所有 bug 都围着三件事转：缩进、条件覆盖顺序、以及把多个独立 `if` 错当成 `elif` 链。

## Lecture flow

### 1. 开场先复习上一讲：对象、表达式、变量绑定
Lecture 2 没有直接跳到 string，而是先把上一讲的 memory diagram 又画了一遍。

老师先回顾：

- 对象在 memory 里有 type
- variable 是 name 到 object 的 binding
- expression 在右边先求值，再把结果绑定给左边变量
- `radius = radius + 1` 这种语句在 Python 里合法，因为它是 reassignment，不是数学等式

这一段虽然是 recap，但其实在给今天铺地基。因为：

- string 也是 object
- `input(...)` 读进来的也是 object
- 分支判断依赖的也是 expression 是否求值为 `True` / `False`

如果上一讲“绑定”和“表达式先求值”的图景没稳住，今天后半讲会很容易乱。

### 2. string 作为新对象类型，是按 sequence 引入的
老师介绍 string 时，第一句不是“它表示文本”，而是：

> a string is a sequence of case sensitive characters

这句话非常重要，因为它直接决定了后面所有操作：

- 既然是 sequence，就有位置
- 既然有位置，就能按位置取字符
- 既然有顺序，就能取 substring

创建 string 的语法就是把字符放进引号里：

```python
a = "me"
z = 'you'
```

单双引号都行，重点只是要配对一致。

### 3. 先做最直观的操作：concatenation 和 repetition
老师没有一上来就讲 indexing，而是先讲两个最直觉的 string 操作：

- `+`：把两个字符串接起来
- `*`：把字符串重复若干次

```python
a = "me"
b = "myself"
c = a + b
d = a + " " + b
silly = a * 3
```

课堂里特别强调了两点：

1. `a + b` 不会自动加空格，所以 `me` 和 `myself` 会变成 `memyself`
2. `*` 是 string 和 integer 之间的操作，不是 string 和 string 的操作

> [!example]
> 如果 `b = ":"`，`c = ")"`，那么
>
> ```python
> s1 = b + 2*c
> ```
>
> 得到的是 `:))`。  
> 这类题训练的是：你是否真把 `+` 理解成“拼接”，把 `*` 理解成“重复”。

### 4. 接着补上 string 的长度、单字符访问和 slicing
有了最基础的拼接和重复后，老师开始把 sequence 的性质讲完整。

第一步是 `len()`：

```python
s = "abc"
len(s)   # 3
```

它回答的是“这个 sequence 里一共有多少个字符”。

然后才进入 indexing。老师把规则讲得非常明确：

- Python 从 `0` 开始计数
- 第一个字符的 index 是 `0`
- 最后一个字符也可以写成 `-1`

```python
s = "abc"
s[0]   # "a"
s[1]   # "b"
s[2]   # "c"
s[-1]  # "c"
```

如果访问越界，比如 `s[3]`，就会得到 `IndexError`。

> [!warning]
> `index` 说的是位置，不是“第几个”这个自然语言直觉。  
> 讲义里说 “the first character is at index 0”，你必须开始习惯这种说法。

然后老师继续讲 slicing，也就是取 substring。规则是：

```python
s[start:stop:step]
```

含义分别是：

- 从 `start` 开始
- 走到 `stop` 之前停止
- 每次跨 `step` 个位置

课堂里重点讲了几个判断顺序：

1. 先看 `step` 正负
2. `step > 0` 时从左往右
3. `step < 0` 时从右往左
4. `stop` 永远不包含自己

> [!example]
> `s = "abcdefgh"`
>
> ```python
> s[3:6]      # "def"
> s[3:6:2]    # "df"
> s[:]        # "abcdefgh"
> s[::-1]     # "hgfedcba"
> s[4:1:-2]   # "ec"
> ```

老师在这里不断鼓励大家去 shell 里试。原因很直接：slicing 的边界和方向感，不靠眼熟代码解决，必须靠你自己多试几次。

### 5. string 是 immutable，所以“改一个字符”这件事根本不被允许
讲完 indexing 和 slicing 之后，老师顺势补充一个非常关键的性质：string 是 immutable object。

这意味着，string 一旦创建出来，不能在原地改某个位置的字符。比如：

```python
s = "car"
# s[0] = "b"   # error
```

你能做的是创建一个新的 string：

```python
s = "b" + s[1:len(s)]
```

也就是说：

- 原来的 `"car"` 还在那里
- 你只是新造了一个 `"bar"`
- 然后把变量 `s` 重新绑定到了新对象

这一段和 Lecture 1 的 reassignment 其实是连在一起的。老师是在反复巩固同一个想法：变量变了，不代表对象被原地修改。

### 6. 接下来从“表达式结果”过渡到真正的输出：print
讲完 strings 后，老师把视角转到 input/output。这里她先纠正了一个初学者非常容易误会的点：

> 在 shell 里输入表达式后看到结果，不等于程序已经在“给用户输出”。

在 shell 里输入：

```python
len(s)
s[-3]
```

你会看到结果，是因为 shell 在帮你 “peek into the value”。  
但如果把同样的语句写进 `.py` 文件里运行，Python 不会自动把它们显示出来。

所以真正要给用户显示结果，必须显式写：

```python
print(len(s))
print(s[-3])
```

老师还顺带解释了 `print` 的两个常见写法：

```python
print(a, b, c)
print(a + str(b) + c)
```

区别是：

- 用逗号分隔时，Python 自动插入空格
- 用 `+` 拼接时，不会自动插空格，而且类型必须一致

> [!warning]
> `print(a + b + c)` 里只要 `b` 不是 string，就会触发 `TypeError`。  
> 这不是 `print` 自己的问题，而是你在进入 `print` 之前，就先要求 Python 去做非法的 string concatenation。

### 7. input 的真正规则：不管用户输入看起来像什么，先都读成 string
输出讲完以后，老师马上进入 input。

基本格式是：

```python
text = input("Type anything... ")
```

当 Python 执行到这行时，会发生三件事：

1. 先把括号里的 prompt 显示出来
2. 等待用户输入并按下 Enter
3. 把用户输入保存为一个 string

老师反复强调第三点。哪怕用户输入的是数字 `3`，程序里拿到的也先是字符串 `"3"`。

所以这两段代码会产生完全不同的结果：

```python
num1 = input("Type a number: ")
print(5 * num1)
```

和

```python
num2 = int(input("Type a number: "))
print(5 * num2)
```

第一段会打印 `33333`，因为 `"3"` 被重复了五次；第二段才会打印 `15`，因为输入先被 cast 成了 integer。

> [!note]
> 这一讲里，`input()` 最重要的知识点甚至不是“怎么写 prompt”，而是“它返回 string”。  
> 只要这个点没记牢，后面分支、循环、数值程序都会频繁出错。

### 8. 第一个完整交互程序：让用户输一个 verb
老师接着安排了一个非常典型的课堂练习：

- ask the user for a verb
- print `I can <verb> better than you`
- 再把这个 verb 打印五次

现场写出来的程序大致是：

```python
verb = input("Type a verb: ")
print("I can", verb, "better than you")
print((verb + " ") * 5)
```

然后老师指出一个小瑕疵：最后会多出一个 trailing space。  
这类小问题看起来不起眼，但它提醒你：

- 程序“能运行”不等于输出已经写得好
- 字符串处理常常要关心空格、换行、边界这些细节

### 9. Newton cube-root 例子：虽然还不会循环，但已经能写“一步更新”
在进入 branching 之前，老师插了一个数值例子：Newton's method 求 cube root 的 next guess。

这里她明确说，现在还不会把整个算法写完，因为我们还没有 loop；但已经可以写出“一次更新”的那一步：

```python
x = int(input("What x to find the cube root of? "))
g = int(input("What guess to start with? "))

print("Current estimate cubed =", g**3)
next_g = g - ((g**3 - x) / (3*g**2))
print("Next guess to try =", next_g)
```

这一段的作用不是让你掌握 Newton's method 证明，而是让你看到：

- 交互输入
- 表达式计算
- 输出结果

已经足够组成一个像样的小程序。

### 10. f-string：把“字面文本 + 表达式”写得更自然
老师接下来介绍了 f-string，理由非常务实：如果继续用逗号和字符串拼接来做格式化输出，很快会变得笨拙。

f-string 的形式是：

```python
num = 3000
fraction = 1/3
print(f"{num*fraction} is {fraction*100}% of {num}")
```

它的核心机制是：

- 花括号外面的内容原样打印
- 花括号里面的内容按 expression 求值

老师特别强调，这再一次说明：

> [!note]
> expressions 几乎可以出现在任何需要 value 的位置。  
> 你已经在 `type(...)`、index、print、f-string 里见过这一点了。

### 11. branching 先从 Boolean expression 开始，而不是先背 if 语法
进入 branching 之前，老师没有直接写 `if`，而是先讲条件到底是什么。

第一步是区分两种“等号”：

- `=` 是 assignment
- `==` 是 equivalence test

随后她把 comparison operator 系统列了出来：

- `==`
- `!=`
- `<`
- `<=`
- `>`
- `>=`

这些表达式的结果不是 number，而是 `bool`。

```python
2 < 3      # True
3 == 4     # False
"a" == "A" # False
```

字符串比较同样是 case sensitive，所以 `"right"` 和 `"RIGHT"` 不一样。

然后老师继续讲 Boolean operator：

- `not`
- `and`
- `or`

她没有要求死背真值表，而是让大家先抓住高层直觉：

- `and` 需要两边都真
- `or` 只要有一边真
- `not` 取反

### 12. 先做一个小练习：secret number 和 guess 是否相等
Boolean 讲完以后，老师安排了一个很短但很关键的练习：

- save a secret number
- ask the user for a guess
- print whether the guess matches the secret

核心代码就是：

```python
secret = 5
guess = int(input("Please guess a number: "))
print(secret == guess)
```

这个练习看似简单，其实在把本讲前半部分全部串起来：

- `input` 读进来的是 string
- 所以要先 `int(...)`
- `==` 返回的是 bool
- `print(...)` 把 bool 显示给用户

### 13. 为什么需要 branching：程序终于可以出现 decision point
讲到这里，老师才正式进入 `if` 的语法。她先用一个“半夜收到免费食物邮件”的例子说明 decision point 的意义，再用一个迷宫/找免费食物的路线例子说明：只要条件不同，程序就该走不同路径。

这个动机很重要。因为 `if` 不是为了增加一种语法花样，而是为了让程序不再只是线性执行到底。

### 14. 最简单的 `if`，再到 `if/else`，再到 `if/elif/else`
接下来老师按照最自然的顺序介绍三种结构。

最简单的是：

```python
if <condition>:
    <code>
```

意思是：

- 条件真，就执行缩进块
- 条件假，就跳过这块，继续往后走

然后是：

```python
if <condition>:
    <code>
else:
    <code>
```

意思是：

- 两个分支二选一
- 不会都执行，也不会都跳过

最后是：

```python
if <condition1>:
    <code>
elif <condition2>:
    <code>
elif <condition3>:
    <code>
else:
    <code>
```

这时要抓住两个规则：

1. `elif` 链会按顺序检查
2. 只执行第一个为真的分支

> [!warning]
> `elif` 链不是“所有真条件都执行”，而是“找到第一个真条件就停”。  
> 这一点和多个独立 `if` 很不一样。

### 15. 用 pset_time / sleep_time 的例子体会条件覆盖
老师随后拿一个非常生活化的例子说明 `if/elif/else`：

```python
pset_time = 22
sleep_time = 8
if (pset_time + sleep_time) > 24:
    print("impossible!")
elif (pset_time + sleep_time) >= 24:
    print("full schedule!")
else:
    leftover = abs(24 - pset_time - sleep_time)
    print(leftover, "h of free time!")
print("end of day")
```

这个例子训练的是条件覆盖思维：

- 大于 24
- 等于 24
- 小于 24

如果你漏了一种情况，程序就会在某些输入上行为不完整。

### 16. 缩进不是风格问题，而是语义本身
接着老师给了一个故意写错的程序：

```python
x = int(input("Enter a number for x: "))
y = int(input("Enter a different number for y: "))
if x == y:
    print(x, "is the same as", y)
print("These are equal!")
```

问题在于最后一行的缩进层级不对。  
如果 `x != y`，它仍然会打印 `These are equal!`，因为那一行根本不在 `if` block 里面。

这是本讲必须建立的一个观念：

> [!warning]
> Python 里的 indentation 不只是排版。  
> 它直接决定某一行代码属于哪个 block，也就决定它何时会被执行。

### 17. nested conditionals 和 “多个 if 不等于 elif”
老师最后又往前推了一步，讲了 nested branching。也就是一个 `if` block 里面还可以继续有 `if`。

例如：

```python
if x == y:
    print("x and y are equal")
    if y != 0:
        print("therefore, x / y is", x / y)
elif x < y:
    print("x is smaller")
else:
    print("y is smaller")
```

这段的关键不在具体业务，而在你要能顺着缩进去 trace：

- 先判断外层
- 只有进了外层 block，内层判断才有机会执行

老师最后还特别回答了一个问题：能不能写两个独立的 `if`？答案是当然可以，但语义就不同了。

- 多个独立 `if`：每个都可能执行
- `if/elif/else` 链：只会执行其中一个

### 18. 课堂最后用“guess too high / too low / same”收束
最后一个小练习，是把前面的 secret-number 程序升级成三种输出：

- too high
- too low
- same

这其实就是最标准的 `if / elif / else` 应用场景。  
如果你写成多个独立 `if`，虽然有时也能工作，但逻辑就不再体现“互斥分支”的意图了。

所以这节课的真正收尾不是某个具体程序，而是你开始会问：

- 这个条件应该放在 `if` 还是 `elif`？
- 这些情况是不是互斥？
- 有没有 catch-all 的 `else`？
- 缩进到底把哪些行纳入了同一个 block？

## Exercise log
> [!example] Finger exercise 02
> 题目要求：给定变量 `number`，打印 `positive`、`negative` 或 `zero`。
>
> ```python
> if number > 0:
>     print("positive")
> elif number < 0:
>     print("negative")
> else:
>     print("zero")
> ```
>
> 这题直接对应本讲最后三分之一的 branching 主线。
>
> 它真正检查的是三件事：
> - 你会不会写 comparison expression
> - 你会不会用 `elif` 表达互斥情况
> - 你会不会用 `else` 接住剩下那一种情况
>
> 如果你写成多个独立 `if`，或者漏掉 `zero`，就说明你还没有把“条件覆盖”这件事真正内化。

## Links to follow-up practice
- Slides: [[MIT 6.100L-slides/mit6_100l_lec02.pdf|Lecture 02 slides]]
- Lecture code: [[MIT 6.100L-lecture-code/mit6_100l_lec02_code.py|Lecture 02 code (py)]]
- Finger exercise: [[MIT 6.100L-finger-exercises/mit6_100l_ex02_sol.pdf|Lecture 02 finger exercise solution]]
- Transcript: [[MIT 6.100L-transcripts/mit6_100l_lec02_transcript.pdf|Lecture 02 transcript]]
- Recitation 2: [[MIT 6.100L-recitations/mit6_100l_rec02.zip|Recitation 02 materials]]
- PS 1 out: [[MIT 6.100L-problem-sets/mit6_100l_ps1.pdf|PS1 statement]], [[MIT 6.100L-problem-sets/mit6_100l_ps1_code.zip|PS1 starter code]]
- Textbook: [[Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Guttag textbook]] (Ch 2.3-2.4)

## Review checklist
- [ ] 我能解释为什么老师要先复习 memory diagram，再开始讲 string 和 branching。
- [ ] 我能把 string 说成 sequence，并据此解释为什么它有 `len`、indexing 和 slicing。
- [ ] 我能正确判断正向 slice 和反向 slice 的结果，尤其知道 stop 不包含自己。
- [ ] 我能解释 string 为什么是 immutable，以及为什么“改一个字符”会报错。
- [ ] 我能区分 shell 里自动显示表达式结果和程序里真正的 `print(...)` 输出。
- [ ] 我能准确说出 `input()` 返回 string，并能判断什么时候必须 cast。
- [ ] 我能解释 f-string 为什么比逗号输出和字符串拼接更自然。
- [ ] 我能区分 `=` 和 `==`，并能写出 `and`、`or`、`not` 的基本判断。
- [ ] 我能说明 `if`、`if/else`、`if/elif/else` 各自适合什么情形。
- [ ] 我能只看缩进就判断一段 branching code 到底会不会误打印、漏打印或重复执行。

> [!warning] Common mistakes
> - 忘记 `input()` 返回的是 string，直接拿去和数字比较。
> - 把多个互斥条件写成多个独立 `if`，结果重复执行多个分支。
> - 把 `print("These are equal!")` 这种语句放错缩进层级，导致它无条件执行。
