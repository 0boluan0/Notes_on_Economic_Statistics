---
aliases:
  - MIT 6.100L Lecture 18
  - 6.100L L18
  - More Python Class Methods
tags:
  - computer-science
  - python
  - mit-6.100l
  - lecture-note
科目: Computer Science
course: MIT 6.100L Introduction to CS and Programming Using Python
lecture: 18
---

# Lecture 18: More Python Class Methods

> [!tip] Hint
> - 这节课一开始先区分两种视角：implementing a class 和 using a class。
> - Coordinate 会先被拿来复习，然后逐步加 `to_origin`、`__str__` 这种更像“对象自己知道怎么表现自己”的方法。
> - `__init__`、`__str__` 这些 dunder methods 开始正式登场，它们让 Python 的内置语法和你的类接起来。
> - Circle 是用 Coordinate 作为组成部件 built out 的，这节课第一次明显体现 composition。
> - `type(...) == Coordinate` 或 `isinstance(...)` 的检查，是在讲如何在类内部保护自己的输入前提。
> - `is_inside(point)` 展示了“对象方法可以调用另一个对象的方法”，比如 point.distance(center)。
> - Fraction 例子一开始故意用普通方法 `plus`、`minus`、`times`，然后才过渡到运算符重载。
> - `__add__`、`__mul__`、`__float__`、`__str__` 让对象能参与 `+`、`*`、`float(...)`、`print(...)` 这些语法。
> - `reduce` 的例子提醒你：方法设计时还要考虑返回值类型是否一致。
> - 听完这节课，你应该能看懂“一个对象类型如何借助 dunder methods 接入 Python 语法生态”。

## Lecture flow

### 1. 先强调两种视角：写类的人 vs 用类的人
Lecture 18 开场先不是新代码，而是把上讲隐含的两种视角正式说破：

- implementing the class
- using the class

写类的人关心：

- 这个类型有哪些属性
- 方法怎么定义
- 输入要不要检查

用类的人关心：

- 怎么创建实例
- 怎么调用方法
- 这个类型看起来像什么、能做什么

这一点很重要，因为后面 dunder methods 的设计，本质上就是在照顾“用类的人”的体验。

### 2. 先回到 Coordinate，巩固类的基本结构
老师先把上节的 Coordinate 拿回来。

核心仍然是：

- `__init__` 写入 `x`、`y`
- `distance(other)` 计算两点距离

然后在此基础上逐步增强，而不是一口气换到新类。  
这种课堂推进说明：类的学习不是一个大跳跃，而是在原有 blueprint 上不断加功能。

### 3. 新方法 `to_origin`：方法也可以改对象内部状态
Coordinate 新增的第一个方法是：

```python
def to_origin(self):
    self.x = 0
    self.y = 0
```

它的重要性在于：

- 你现在不只是“读取对象属性”
- 还开始通过方法去 **改变对象状态**

这让类方法和前面学过的 list mutation 直觉连上了：  
对象方法同样可以带 side effect。

### 4. `__str__`：让对象知道自己怎么被打印
老师紧接着加了一个非常关键的 dunder method：

```python
def __str__(self):
    return "<" + str(self.x) + "," + str(self.y) + ">"
```

它的教学意义远大于“打印好看一点”。

因为一旦实现了 `__str__`：

- `print(c)` 就不再输出一串内存地址
- 而会调用你定义的字符串表示

> [!note]
> `__str__` 让对象类型参与到 Python 原生的 `print` 语法里。

这也是 dunder methods 的第一次真正落地体验。

### 5. Circle：用一个类去构造更复杂的类
Coordinate 复习完后，老师开始用它搭 Circle。

Circle 的数据不是两个普通数字，而是：

- `center`：一个 Coordinate object
- `radius`：一个半径值

这一步特别重要，因为它让你看到 **composition**：

- 类可以把别的对象类型当作自己的组成部分

所以面向对象不只是“新语法”，也是把已有抽象组合成更复杂抽象的方法。

### 6. 在 `__init__` 里做类型检查
老师接着修改 Circle 的 `__init__`，要求：

- `center` 必须是 Coordinate 对象
- `radius` 必须是 int

否则 raise `ValueError`。

课堂里老师在这里还提到了：

- `type(center) == Coordinate`
- `isinstance(center, Coordinate)`

这些检查的意义，是在类定义内部保护自己的输入前提。  
也就是把“这个类允许什么样的初始化方式”写得更显式。

### 7. `is_inside(point)`：对象之间也会相互协作
老师随后给 Circle 增加：

```python
def is_inside(self, point):
    return point.distance(self.center) < self.radius
```

这段代码很值得停一下，因为它体现了几个 OOP 观念同时在工作：

- `point` 是 Coordinate object
- `self.center` 也是 Coordinate object
- 你可以调用一个对象的方法来帮助另一个对象完成判断

所以类方法不是孤岛，它们会建立对象之间的协作关系。

### 8. Fraction 第一版：先把行为写出来，不急着上运算符
讲完 Circle 之后，课堂切到 Fraction。

老师先故意用普通方法名：

- `times`
- `divide`
- `plus`
- `minus`

来写一个 `SimpleFraction` 类。

这一步很聪明，因为它先让你想清楚：

- fraction 对象的 data 是什么
- fraction 应该有哪些数学行为

然后才进入“怎么让它和 Python 的 `+` `*` 对接”的话题。

### 9. `get_inverse` vs `invert`：返回新值和修改自身要分清
在 `SimpleFraction` 上，老师安排了一个很典型的对比：

- `get_inverse`：返回 `1/self`
- `invert`：直接交换 numerator 和 denominator

这组方法是在重演一条整个课程都在强调的区分：

- 返回一个值
- 修改对象自身

如果你把这两种风格混在一起，类方法就会越来越难用。

### 10. 运算符重载：让对象接入 `+`、`*`、`print`
前面的普通方法讲清楚之后，老师才切到真正的 Pythonic 写法。

比如：

```python
def __mul__(self, other):
    ...

def __add__(self, other):
    ...

def __truediv__(self, other):
    ...
```

实现之后：

- `a * b`
- `a + b`
- `a / b`

这些语法就会自动映射到你定义的 dunder methods。

课堂在这里反复强调的是：

- 运算符背后其实也是方法调用
- 只是 Python 帮你写成了更自然的形式

### 11. 三种等价调用方式
老师还专门展示了下面三种调用是等价的：

```python
a * b
a.__mul__(b)
Fraction.__mul__(a, b)
```

这一步和上一讲 `c.distance(origin)` 的等价调用一起，构成了理解 OOP 语法糖的关键。

如果你能看懂这里，就会知道：

- 类方法没有神秘力量
- 只是 Python 在不同场合替你做了不同的绑定和调度

### 12. `__float__`、`__str__`：对象可以参与更多内置转换
老师继续往 Fraction 里加：

- `__float__`
- `__str__`

于是：

- `float(c)` 会调用 `__float__`
- `print(c)` 会调用 `__str__`

这让对象类型不只是能做数学运算，还能更自然地和 Python 自带函数配合。

### 13. `reduce`：方法返回值类型也需要设计
Fraction 的 `reduce` 方法引出了一个更细的问题：  
约分之后到底返回什么类型？

如果分母变成 `1`，是：

- 返回一个 `int`
- 还是仍然返回一个 `Fraction`

老师专门把这个点拿出来，是因为这涉及接口一致性。

如果一个方法有时返回 `Fraction`，有时返回 `int`，调用者的使用体验会变得很不稳定。  
所以课堂后来通过 you-try-it 去修这个设计。

### 14. 这节课真正推进的是“对象与语言语法的接缝”
Lecture 18 表面上是继续写类方法，实际上它完成了更深的一步：

- 你的对象开始接入 Python 的运算符、打印、类型转换
- 你的类也开始由简单数据容器，变成更自然的语言级对象

这是从“我能定义类”走向“我能定义用起来像原生对象的类”的关键一步。

## Exercise log

> [!example] Finger exercise 18
> 官方题目还是 `Circle`，但这次要求实现：
> - `__init__`
> - `get_radius`
> - `__add__`
> - `__str__`

它非常贴合本讲，因为它直接检查你是否理解：

- dunder methods 不是装饰
- 它们会决定对象如何参与 `+` 和 `print`

官方 `__add__` 的意思是：

```python
return Circle(self.r + c.r)
```

这说明运算符重载本质上还是“返回一个新对象”，而不是必须原地修改自己。

## Links to follow-up practice
- Slides: [[MIT 6.100L-slides/mit6_100l_lec18.pdf|Lecture 18 slides]]
- Lecture code: [[MIT 6.100L-lecture-code/mit6_100l_lec18_code.py|Lecture 18 code (py)]]
- Finger exercise: [[MIT 6.100L-finger-exercises/mit6_100l_ex18_sol.pdf|Lecture 18 finger exercise solution]]
- Transcript: [[MIT 6.100L-transcripts/mit6_100l_lec18_transcript.pdf|Lecture 18 transcript]]
- Recitation: none attached to this lecture week
- PS 4 halfway hand-in due: [[MIT 6.100L-problem-sets/mit6_100l_ps4.pdf|PS4 statement]], [[MIT 6.100L-problem-sets/mit6_100l_ps4_code.zip|PS4 starter code]]
- Textbook: [[Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Guttag textbook]] (Ch 10.1)

## Review checklist
- [ ] 我能区分 implementing a class 和 using a class 两种视角。
- [ ] 我能解释 `to_origin` 这类方法如何修改对象自身状态。
- [ ] 我能说明 `__str__` 为什么能改变 `print(obj)` 的行为。
- [ ] 我能解释 Circle 如何把 Coordinate 作为组成部分。
- [ ] 我能判断什么时候需要在 `__init__` 中做类型检查。
- [ ] 我能说明 `is_inside(point)` 里为什么是对象和对象协作。
- [ ] 我能区分 `get_inverse` 和 `invert` 这类“返回值 vs 修改自身”的方法设计。
- [ ] 我能解释 `__add__`、`__mul__`、`__float__` 这类 dunder methods 的作用。
- [ ] 我能理解 `reduce` 方法中“返回类型一致性”为什么重要。
- [ ] 我能按课堂顺序复述：Coordinate recap -> Circle -> SimpleFraction -> operator overloading -> reduce。

> [!warning] Common mistakes
> - 把 dunder methods 当成要死记的特殊名字，却不理解它们对应哪种语言行为。
> - 设计类方法时混淆“返回新对象”和“修改当前对象”。
> - 用类组合类时，没有先想清楚内部属性本身是不是别的对象。
> - 让同一个方法在不同情况下返回完全不同类型，导致接口不稳定。
