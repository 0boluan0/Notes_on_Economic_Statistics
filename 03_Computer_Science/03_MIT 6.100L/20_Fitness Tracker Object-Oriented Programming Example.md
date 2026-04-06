---
aliases:
  - MIT 6.100L Lecture 20
  - 6.100L L20
  - Fitness Tracker Object-Oriented Programming Example
tags:
  - computer-science
  - python
  - mit-6.100l
  - lecture-note
科目: Computer Science
course: MIT 6.100L Introduction to CS and Programming Using Python
lecture: 20
---

# Lecture 20: Fitness Tracker Object-Oriented Programming Example

> [!tip] Hint
> - 这节课不是再讲新的 OOP 术语，而是把前两讲的 class / inheritance / class variable 放进一个更完整的案例里。
> - 开头先从“generic workout 有哪些共同属性”开始，而不是先写代码。
> - 课程先做一个很简单的 `SimpleWorkout`，再升级成真正的 `Workout`，这体现的是逐步重构。
> - `start`、`end`、`calories`、`kind`、`icon` 这些属性里，哪些是共通的、哪些会被子类改写，是本讲主线。
> - `cal_per_hr` 被做成 class variable，因为它是某类 workout 的共有估计规则。
> - `get_calories` 是 information hiding 的最好例子：调用者不用知道 calories 是直接给的，还是按时长估出来的。
> - datetime parsing 出现在这里，是为了把“时间相关计算”交给标准库，而不是手写字符串处理。
> - RunWorkout 和 SwimWorkout 不是平铺的新类，而是对 Workout 的 specialization。
> - `total_calories(workouts)` 这种函数在展示 polymorphism：不同 workout 子类都能响应该方法调用。
> - 听完这节课，你应该能把 OOP 看成一种建模工具，而不只是“定义几个类和方法”。

## Lecture flow

### 1. 先从需求出发：generic workout 有什么共同信息
Lecture 20 一开始就说今天要做一个更 involved 的例子。  
老师没有急着写类，而是先问：

- 所有 workout 共有的属性是什么
- running 和 swimming 各自又多了什么

对于 generic workout，课堂先抽出这些共同点：

- start time
- end time
- calories burned
- workout kind
- 可能还有展示用 icon

这一步其实在做类设计前最重要的工作：抽共性。

### 2. 先做 `SimpleWorkout`：把对象最小化落地
老师第一版没有追求优雅，只先写一个最简单的 `SimpleWorkout`。

它包含：

- `start`
- `end`
- `calories`
- `icon`
- `kind`

以及最基础的 getters / setters。

这一步的价值在于：

- 先把对象结构落地
- 再逐渐发现哪里还不够好

也就是典型的 incremental refinement。

### 3. 检查类和实例的 `__dict__`：类状态和实例状态不一样
老师随后用了 `__dict__` 去检查：

- 类本身都有哪些属性和方法
- 某个具体 workout 实例里又实际存了哪些字段

这一步很有教学意义，因为它会帮你区分：

- 写在 class definition 上的东西
- 具体实例自己携带的数据

这也为 class variable 的理解做了铺垫。

### 4. 从 `SimpleWorkout` 重构到 `Workout`
接下来老师正式引出更完善的 `Workout` 类。

改进点包括：

- `start` 和 `end` 不再只保留字符串，而会被解析成 datetime objects
- `calories` 可以是可选的
- 增加 `get_duration`
- `get_calories` 更智能

这说明设计类时很常见的一种路径：

- 先写最小可用版
- 再把“表示方式”和“接口”做得更合理

### 5. 信息隐藏的典型例子：`get_calories`
本讲最好的 information hiding 例子就是 `get_calories`。

在 `Workout` 里：

- 如果 `calories` 已经显式给出，就直接返回
- 如果没有给出，就根据 workout 时长和类变量 `cal_per_hr` 估算

所以调用者只需要写：

```python
w.get_calories()
```

却不需要知道内部到底是哪条路径。

> [!note]
> information hiding 的高价值时刻，就是外部接口稳定，而内部实现可变。

### 6. datetime：把时间解析和时长计算交给标准库
老师随后引入 `dateutil.parser` 和 datetime objects。

原因很现实：

- 日期时间字符串自己处理太麻烦
- 标准库已经能把它们解析成可计算对象

一旦 `start` 和 `end` 是 datetime objects：

- `self.end - self.start` 就直接得到时间间隔
- `.total_seconds()` 就能拿到时长秒数

课堂这里其实是在做一个很重要的工程选择：

- 不重复造轮子
- 把复杂细节交给合适的库

### 7. class variable：`cal_per_hr` 属于类，而不属于单个实例
老师接着把 `cal_per_hr = 200` 放到 `Workout` 类里，明确说这是 class variable。

理由很清楚：

- 它表示的是“这类 workout 的通用估算参数”
- 不是某个具体 workout 特有的数据

这和实例属性不同。  
所以访问它时，老师也故意强调更推荐的写法是：

```python
Workout.cal_per_hr
```

而不是通过某个实例去改它。

### 8. 先用两个 Workout 例子验证设计
老师接着构造两个具体对象：

- 一个不给 calories，让类自己估
- 一个显式给 calories

这一步不只是演示创建对象，而是在测试：

- 类设计是否真的支持两种使用场景
- `get_calories` 是否把信息隐藏做好了

同时也让你看到 default parameter 和可选属性在类设计里的实际用途。

### 9. 转向 inheritance：generic workout 之上再分 specialized workouts
把基类 Workout 稳住之后，课堂正式转向继承。

老师先在概念上区分：

- generic workout
- specific workout types

比如：

- running
- swimming

这一步和上一讲 Animal 的逻辑完全呼应：

- 先有共性父类
- 再做更具体的子类

### 10. RunWorkout：加上 elevation、route_gps_points 等专属属性
`RunWorkout` 继承 `Workout`，并新增：

- `elev`
- `route_gps_points`
- 以及自己的 class variable `cals_per_km`

其 `__init__` 先用 `super().__init__(...)` 把父类共通部分初始化好，再设置：

- 跑步专属 icon
- kind = "Running"
- elevation 等属性

这正是继承设计里最典型的模板：

- 先复用父类初始化
- 再补子类差异

### 11. RunWorkout 重写 `get_calories`
`RunWorkout` 最重要的 override 是 `get_calories`。

逻辑变成：

- 如果有 GPS route points，就按路线距离和 `cals_per_km` 估算
- 否则退回父类的 `get_calories`

这段代码非常漂亮，因为它把 inheritance 和 information hiding 合在一起：

- 外部依然只调用 `get_calories`
- 但不同子类内部实现可以完全不同

### 12. SwimWorkout：重定义 class variable，也能重写行为
`SwimWorkout` 则展示了另一种 specialization 方式。

它不像 RunWorkout 那样依赖 GPS，而是：

- 定义自己的 `cal_per_hr = 400`
- 新增 `pace`
- 重写 `get_calories`，用 SwimWorkout 的速率估算

这让你看到，子类差异不只来自新实例属性，也可以来自：

- 不同 class variable
- 不同 method implementation

### 13. 复用父类 `__str__`：子类不一定什么都要重写
老师还特地展示：

- Workout
- RunWorkout
- SwimWorkout

都可以共享父类的 `__str__`，只要：

- icon
- kind
- `get_calories`

这些被调用到的部件已经被子类正确设置或重写。

这说明继承里有个很重要的设计目标：

- 只在必要处 override
- 能复用的就复用

### 14. polymorphism：不同子类都能塞进 `total_calories`
课堂后段最重要的函数之一是：

```python
def total_calories(workouts):
    cals = 0
    for w in workouts:
        cals += w.get_calories()
    return cals
```

它完全不需要知道 `w` 到底是：

- Workout
- RunWorkout
- SwimWorkout

只要每个对象都支持 `get_calories()`，这个函数就能工作。

这就是本讲最核心的 polymorphism 直觉。

> [!example]
> 相同接口，不同实现；调用方不用分类型分支，方法派发会自动落到对应类上。

### 15. `total_elevation` 也在提醒你：并不是所有接口都对所有子类通用
与 `total_calories` 相对，老师又给出：

```python
def total_elevation(run_workouts):
    ...
```

这个函数只适用于 RunWorkout，因为只有它们有 `get_elev()`。

这一步提醒你：

- polymorphism 很强大
- 但也不能假装所有子类接口都完全一样

设计 API 时仍然要清楚“哪些方法是所有父类实例都应支持的通用接口”。

### 16. 这节课让 OOP 从“语法练习”变成“建模工作”
Lecture 20 的价值在于，它第一次给出一个足够完整、足够真实的对象建模例子。

从这节课里，你应该看到 OOP 真正的工作流：

1. 先抽出共性
2. 设计父类接口
3. 用类变量和方法封装通用逻辑
4. 通过子类表达差异
5. 让外部通过统一接口使用不同对象

## Exercise log

> [!example] Finger exercise 20
> 官方练习实现的是：
> - `Container`
> - `Queue(Container)`

和 Lecture 19 的 `Stack` 题形成一对。

虽然它不是 workout 例子本身，但它和本讲方法一致：

- 父类放通用 `size`、`add`
- 子类只补 `remove`
- 具体语义从后进先出改成先进先出

所以它正好巩固了“在稳定父类接口上做行为 specialization”这件事。

## Links to follow-up practice
- Slides: [[MIT 6.100L-slides/mit6_100l_lec20.pdf|Lecture 20 slides]]
- Lecture code: [[MIT 6.100L-lecture-code/mit6_100l_lec20_code.zip|Lecture 20 code (zip)]]
- Finger exercise: [[MIT 6.100L-finger-exercises/mit6_100l_ex20_sol.pdf|Lecture 20 finger exercise solution]]
- Transcript: [[MIT 6.100L-transcripts/mit6_100l_lec20_transcript.pdf|Lecture 20 transcript]]
- Recitation 9: [[MIT 6.100L-recitations/mit6_100l_rec09.zip|Recitation 09 materials]]
- PS 5 out: [[MIT 6.100L-problem-sets/mit6_100l_ps5.pdf|PS5 statement]], [[MIT 6.100L-problem-sets/mit6_100l_ps5_code.zip|PS5 starter code]]
- PS 4 due: [[MIT 6.100L-problem-sets/mit6_100l_ps4.pdf|PS4 statement]], [[MIT 6.100L-problem-sets/mit6_100l_ps4_code.zip|PS4 starter code]]
- Textbook: [[Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Guttag textbook]] (Ch 10.4)

## Review checklist
- [ ] 我能从需求角度抽出 generic workout 的共性属性。
- [ ] 我能解释为什么要先做 `SimpleWorkout` 再重构成 `Workout`。
- [ ] 我能说明 `get_calories` 为什么是 information hiding 的典型例子。
- [ ] 我能解释为什么 `cal_per_hr` 是 class variable。
- [ ] 我能说明 datetime object 为什么比手写字符串处理更合适。
- [ ] 我能写出子类 `__init__` 中“先 `super()` 再补充属性”的模式。
- [ ] 我能解释 RunWorkout 和 SwimWorkout 分别在哪些点上 override 了父类。
- [ ] 我能说明为什么子类可以共享父类 `__str__`。
- [ ] 我能用 `total_calories` 解释 polymorphism。
- [ ] 我能按课堂顺序复述：SimpleWorkout -> Workout -> class variable/datetime -> RunWorkout/SwimWorkout -> polymorphism。

> [!warning] Common mistakes
> - 还没抽清共性和特性，就急着写一堆平级类。
> - 子类重写太多本可复用的父类逻辑。
> - 把 class variable 和 instance variable 混在一起。
> - 调用统一接口时又手写一堆 `if type(...)`，错失多态的价值。
