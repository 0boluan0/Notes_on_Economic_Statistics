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
> <!-- bilingual-en:start -->
> - This lecture introduces no new OOP terminology; it combines classes, inheritance, and class variables from the previous two lectures in one substantial case study.
> - The lesson begins by identifying the attributes common to all workouts rather than diving straight into code.
> - The lecture starts with a minimal `SimpleWorkout` and then refactors it into a fuller `Workout`, demonstrating incremental refinement.
> - The central design question is which of `start`, `end`, `calories`, `kind`, and `icon` belong to every workout and which should vary by subclass.
> - `cal_per_hr` is implemented as a class variable because it represents an estimation rule shared by a category of workouts.
> - `get_calories` is a clear example of information hiding: callers need not know whether calories were recorded directly or estimated from duration.
> - `datetime` parsing is introduced here so that established libraries, rather than hand-written string manipulation, handle time-related calculations.
> - `RunWorkout` and `SwimWorkout` are specializations of `Workout`, not unrelated peer classes.
> - `total_calories(workouts)` demonstrates polymorphism because instances of different workout subclasses all respond to `get_calories()`.
> - After this lesson, you should be able to see OOP as a modeling tool rather than merely a way to define classes and methods.
> <!-- bilingual-en:end -->

## Lecture flow

### 1. 先从需求出发：generic workout 有什么共同信息
<!-- bilingual-en:start -->
*1. Starting from Requirements: What Does Every Workout Share?*
<!-- bilingual-en:end -->
Lecture 20 一开始就说今天要做一个更 involved 的例子。  
老师没有急着写类，而是先问：
<!-- bilingual-en:start -->
Lecture 20 introduces a more involved case study. Before writing any classes, the instructor asks:
<!-- bilingual-en:end -->

- 所有 workout 共有的属性是什么
- running 和 swimming 各自又多了什么
<!-- bilingual-en:start -->
- What common attributes do all workouts have?
- Which additional attributes do running and swimming each require?
<!-- bilingual-en:end -->

对于 generic workout，课堂先抽出这些共同点：
<!-- bilingual-en:start -->
The lecture first identifies the state common to a generic workout:
<!-- bilingual-en:end -->

- start time
- end time
- calories burned
- workout kind
- 可能还有展示用 icon
<!-- bilingual-en:start -->
- Start time.
- End time.
- Calories burned.
- Workout type.
- Possibly a display icon.
<!-- bilingual-en:end -->

这一步其实在做类设计前最重要的工作：抽共性。
<!-- bilingual-en:start -->
Identifying this common structure is the most important work before defining the class.
<!-- bilingual-en:end -->

### 2. 先做 `SimpleWorkout`：把对象最小化落地
<!-- bilingual-en:start -->
*2. `SimpleWorkout`: Implementing the Minimal Object First*
<!-- bilingual-en:end -->
老师第一版没有追求优雅，只先写一个最简单的 `SimpleWorkout`。
<!-- bilingual-en:start -->
The instructor begins with a deliberately simple `SimpleWorkout` rather than aiming immediately for an elegant final design.
<!-- bilingual-en:end -->

它包含：
<!-- bilingual-en:start -->
It includes:
<!-- bilingual-en:end -->

- `start`
- `end`
- `calories`
- `icon`
- `kind`

以及最基础的 getters / setters。
<!-- bilingual-en:start -->
It also provides basic getters and setters.
<!-- bilingual-en:end -->

这一步的价值在于：
<!-- bilingual-en:start -->
This staging has two benefits:
<!-- bilingual-en:end -->

- 先把对象结构落地
- 再逐渐发现哪里还不够好
<!-- bilingual-en:start -->
- It establishes a working object structure first.
- It then makes the limitations of that structure easier to identify.
<!-- bilingual-en:end -->

也就是典型的 incremental refinement。
<!-- bilingual-en:start -->
This is a standard process of incremental refinement.
<!-- bilingual-en:end -->

### 3. 检查类和实例的 `__dict__`：类状态和实例状态不一样
<!-- bilingual-en:start -->
*3. Inspecting Class and Instance `__dict__`: Two Kinds of State*
<!-- bilingual-en:end -->
老师随后用了 `__dict__` 去检查：
<!-- bilingual-en:start -->
The instructor then uses `__dict__` to inspect:
<!-- bilingual-en:end -->

- 类本身都有哪些属性和方法
- 某个具体 workout 实例里又实际存了哪些字段
<!-- bilingual-en:start -->
- Which attributes and methods belong to the class itself.
- Which fields are actually stored in a particular workout instance.
<!-- bilingual-en:end -->

这一步很有教学意义，因为它会帮你区分：
<!-- bilingual-en:start -->
This makes the distinction concrete between:
<!-- bilingual-en:end -->

- 写在 class definition 上的东西
- 具体实例自己携带的数据
<!-- bilingual-en:start -->
- Attributes and methods stored on the class.
- Fields carried by a particular instance.
<!-- bilingual-en:end -->

这也为 class variable 的理解做了铺垫。
<!-- bilingual-en:start -->
This also lays the groundwork for understanding class variables.
<!-- bilingual-en:end -->

### 4. 从 `SimpleWorkout` 重构到 `Workout`
<!-- bilingual-en:start -->
*4. Refactoring from `SimpleWorkout` to `Workout`*
<!-- bilingual-en:end -->
接下来老师正式引出更完善的 `Workout` 类。
<!-- bilingual-en:start -->
The instructor then introduces the fuller `Workout` class.
<!-- bilingual-en:end -->

改进点包括：
<!-- bilingual-en:start -->
The improvements include:
<!-- bilingual-en:end -->

- `start` 和 `end` 不再只保留字符串，而会被解析成 datetime objects
- `calories` 可以是可选的
- 增加 `get_duration`
- `get_calories` 更智能
<!-- bilingual-en:start -->
- `start` and `end` are no longer left as strings; they are parsed into `datetime` objects.
- `calories` becomes optional.
- The class gains a `get_duration` method.
- `get_calories` can either return a recorded value or estimate one.
<!-- bilingual-en:end -->

这说明设计类时很常见的一种路径：
<!-- bilingual-en:start -->
This illustrates a common path in designing classes:
<!-- bilingual-en:end -->

- 先写最小可用版
- 再把“表示方式”和“接口”做得更合理
<!-- bilingual-en:start -->
- Start with the smallest usable version.
- Then refine the representation and interface to fit the domain.
<!-- bilingual-en:end -->

### 5. 信息隐藏的典型例子：`get_calories`
<!-- bilingual-en:start -->
*5. `get_calories`: A Clear Example of Information Hiding*
<!-- bilingual-en:end -->
本讲最好的 information hiding 例子就是 `get_calories`。
<!-- bilingual-en:start -->
The best example of information hiding in this lecture is `get_calories`.
<!-- bilingual-en:end -->

在 `Workout` 里：
<!-- bilingual-en:start -->
In the `Workout` class:
<!-- bilingual-en:end -->

- 如果 `calories` 已经显式给出，就直接返回
- 如果没有给出，就根据 workout 时长和类变量 `cal_per_hr` 估算
<!-- bilingual-en:start -->
- If `calories` is explicitly provided, return it directly.
- Otherwise, estimate it from workout duration and the class variable `cal_per_hr`.
<!-- bilingual-en:end -->

所以调用者只需要写：
<!-- bilingual-en:start -->
So the caller only needs to write:
<!-- bilingual-en:end -->

```python
w.get_calories()
```

却不需要知道内部到底是哪条路径。
<!-- bilingual-en:start -->
The caller does not need to know which internal path produces the value.
<!-- bilingual-en:end -->

> [!note]
> information hiding 的高价值时刻，就是外部接口稳定，而内部实现可变。
> <!-- bilingual-en:start -->
> Information hiding is most valuable when the external interface remains stable while the internal implementation is free to change.
> <!-- bilingual-en:end -->

### 6. datetime：把时间解析和时长计算交给标准库
<!-- bilingual-en:start -->
*6. Datetime: Delegate Date Parsing and Duration Calculation to the Standard Library*
<!-- bilingual-en:end -->
老师随后引入 `dateutil.parser` 和 datetime objects。
<!-- bilingual-en:start -->
The instructor then introduces `dateutil.parser` and datetime objects.
<!-- bilingual-en:end -->

原因很现实：
<!-- bilingual-en:start -->
The reason is very practical:
<!-- bilingual-en:end -->

- 日期时间字符串自己处理太麻烦
- 标准库已经能把它们解析成可计算对象
<!-- bilingual-en:start -->
- Parsing `datetime` strings manually is cumbersome.
- Existing libraries can turn them into objects that support time arithmetic.
<!-- bilingual-en:end -->

一旦 `start` 和 `end` 是 datetime objects：
<!-- bilingual-en:start -->
Once `start` and `end` are datetime objects:
<!-- bilingual-en:end -->

- `self.end - self.start` 就直接得到时间间隔
- `.total_seconds()` 就能拿到时长秒数
<!-- bilingual-en:start -->
- `self.end - self.start` directly gives the time interval.
- `.total_seconds()` returns the duration in seconds.
<!-- bilingual-en:end -->

课堂这里其实是在做一个很重要的工程选择：
<!-- bilingual-en:start -->
This is an important engineering decision:
<!-- bilingual-en:end -->

- 不重复造轮子
- 把复杂细节交给合适的库
<!-- bilingual-en:start -->
- Avoid reimplementing established parsing and time-arithmetic logic.
- Delegate those details to an appropriate library.
<!-- bilingual-en:end -->

### 7. class variable：`cal_per_hr` 属于类，而不属于单个实例
<!-- bilingual-en:start -->
*7. Class Variable: `cal_per_hr` Belongs to the Class*
<!-- bilingual-en:end -->
老师接着把 `cal_per_hr = 200` 放到 `Workout` 类里，明确说这是 class variable。
<!-- bilingual-en:start -->
The instructor places `cal_per_hr = 200` on `Workout` as a class variable.
<!-- bilingual-en:end -->

理由很清楚：
<!-- bilingual-en:start -->
The reasoning is clear:
<!-- bilingual-en:end -->

- 它表示的是“这类 workout 的通用估算参数”
- 不是某个具体 workout 特有的数据
<!-- bilingual-en:start -->
- It represents a shared estimation parameter for this kind of workout.
- It is not data unique to one workout instance.
<!-- bilingual-en:end -->

这和实例属性不同。  
所以访问它时，老师也故意强调更推荐的写法是：
<!-- bilingual-en:start -->
Unlike an instance attribute, it is best accessed explicitly through the class:
<!-- bilingual-en:end -->

```python
Workout.cal_per_hr
```

而不是通过某个实例去改它。
<!-- bilingual-en:start -->
rather than modified through one instance.
<!-- bilingual-en:end -->

### 8. 先用两个 Workout 例子验证设计
<!-- bilingual-en:start -->
*8. Validating the Design with Two `Workout` Instances*
<!-- bilingual-en:end -->
老师接着构造两个具体对象：
<!-- bilingual-en:start -->
The instructor constructs two objects:
<!-- bilingual-en:end -->

- 一个不给 calories，让类自己估
- 一个显式给 calories
<!-- bilingual-en:start -->
- one without an explicit calorie value, so the class estimates it
- one with an explicit calorie value
<!-- bilingual-en:end -->

这一步不只是演示创建对象，而是在测试：
<!-- bilingual-en:start -->
This step does more than demonstrate object creation; it tests:
<!-- bilingual-en:end -->

- 类设计是否真的支持两种使用场景
- `get_calories` 是否把信息隐藏做好了
<!-- bilingual-en:start -->
- Whether the class genuinely supports both usage scenarios.
- Whether `get_calories` successfully hides the distinction.
<!-- bilingual-en:end -->

同时也让你看到 default parameter 和可选属性在类设计里的实际用途。
<!-- bilingual-en:start -->
It also demonstrates a practical use of default parameters and optional attributes in class design.
<!-- bilingual-en:end -->

### 9. 转向 inheritance：generic workout 之上再分 specialized workouts
<!-- bilingual-en:start -->
*9. Moving to Inheritance: Specialized Workouts Built on `Workout`*
<!-- bilingual-en:end -->
把基类 Workout 稳住之后，课堂正式转向继承。
<!-- bilingual-en:start -->
Once the `Workout` base class is stable, the lecture turns to inheritance.
<!-- bilingual-en:end -->

老师先在概念上区分：
<!-- bilingual-en:start -->
The instructor distinguishes:
<!-- bilingual-en:end -->

- generic workout
- specific workout types

比如：
<!-- bilingual-en:start -->
For example:
<!-- bilingual-en:end -->

- running
- swimming

这一步和上一讲 Animal 的逻辑完全呼应：
<!-- bilingual-en:start -->
This step mirrors the logic from the previous lecture on Animal classes:
<!-- bilingual-en:end -->

- 先有共性父类
- 再做更具体的子类
<!-- bilingual-en:start -->
- Place common structure in a parent class.
- Express more specific workout types as subclasses.
<!-- bilingual-en:end -->

### 10. RunWorkout：加上 elevation、route_gps_points 等专属属性
<!-- bilingual-en:start -->
*10. `RunWorkout`: Adding Elevation and Route-Specific State*
<!-- bilingual-en:end -->
`RunWorkout` 继承 `Workout`，并新增：
<!-- bilingual-en:start -->
`RunWorkout` inherits from `Workout`, adding:
<!-- bilingual-en:end -->

- `elev`
- `route_gps_points`
- 以及自己的 class variable `cals_per_km`
<!-- bilingual-en:start -->
- `elev`
- `route_gps_points`
- its own class variable, `cals_per_km`
<!-- bilingual-en:end -->

其 `__init__` 先用 `super().__init__(...)` 把父类共通部分初始化好，再设置：
<!-- bilingual-en:start -->
Its `__init__` first initializes the parent class using `super().__init__(...)`, then sets:
<!-- bilingual-en:end -->

- 跑步专属 icon
- kind = "Running"
- elevation 等属性
<!-- bilingual-en:start -->
- a running-specific icon
- `kind = "Running"`
- elevation and route attributes
<!-- bilingual-en:end -->

这正是继承设计里最典型的模板：
<!-- bilingual-en:start -->
This is the classic template in inheritance:
<!-- bilingual-en:end -->

- 先复用父类初始化
- 再补子类差异
<!-- bilingual-en:start -->
- Reuse parent initialization first.
- Then add the state specific to the subclass.
<!-- bilingual-en:end -->

### 11. RunWorkout 重写 `get_calories`
<!-- bilingual-en:start -->
*11. `RunWorkout` Overrides `get_calories`*
<!-- bilingual-en:end -->
`RunWorkout` 最重要的 override 是 `get_calories`。
<!-- bilingual-en:start -->
The most important override in `RunWorkout` is `get_calories`.
<!-- bilingual-en:end -->

逻辑变成：
<!-- bilingual-en:start -->
The logic becomes:
<!-- bilingual-en:end -->

- 如果有 GPS route points，就按路线距离和 `cals_per_km` 估算
- 否则退回父类的 `get_calories`
<!-- bilingual-en:start -->
- If GPS route points are available, estimate calories from route distance and `cals_per_km`.
- Otherwise, fall back to the parent implementation of `get_calories`.
<!-- bilingual-en:end -->

这段代码非常漂亮，因为它把 inheritance 和 information hiding 合在一起：
<!-- bilingual-en:start -->
This code is very elegant because it combines inheritance and information hiding:
<!-- bilingual-en:end -->

- 外部依然只调用 `get_calories`
- 但不同子类内部实现可以完全不同
<!-- bilingual-en:start -->
- Callers still use the single `get_calories` interface.
- Each subclass can implement that interface differently.
<!-- bilingual-en:end -->

### 12. SwimWorkout：重定义 class variable，也能重写行为
<!-- bilingual-en:start -->
*12. `SwimWorkout`: Replacing a Class Variable and Method Behavior*
<!-- bilingual-en:end -->
`SwimWorkout` 则展示了另一种 specialization 方式。
<!-- bilingual-en:start -->
`SwimWorkout` demonstrates another approach to specialization.
<!-- bilingual-en:end -->

它不像 RunWorkout 那样依赖 GPS，而是：
<!-- bilingual-en:start -->
Rather than relying on GPS, it:
<!-- bilingual-en:end -->

- 定义自己的 `cal_per_hr = 400`
- 新增 `pace`
- 重写 `get_calories`，用 SwimWorkout 的速率估算
<!-- bilingual-en:start -->
- defines its own `cal_per_hr = 400`
- adds a `pace` attribute
- overrides `get_calories` with a swimming-specific estimate
<!-- bilingual-en:end -->

这让你看到，子类差异不只来自新实例属性，也可以来自：
<!-- bilingual-en:start -->
This shows that subclass differences can come not only from new instance attributes but also from:
<!-- bilingual-en:end -->

- 不同 class variable
- 不同 method implementation
<!-- bilingual-en:start -->
- Different class variables.
- Different method implementations.
<!-- bilingual-en:end -->

### 13. 复用父类 `__str__`：子类不一定什么都要重写
<!-- bilingual-en:start -->
*13. Reusing the Parent's `__str__`: Override Only What Must Differ*
<!-- bilingual-en:end -->
老师还特地展示：
<!-- bilingual-en:start -->
The instructor demonstrates that:
<!-- bilingual-en:end -->

- Workout
- RunWorkout
- SwimWorkout

都可以共享父类的 `__str__`，只要：
<!-- bilingual-en:start -->
`Workout`, `RunWorkout`, and `SwimWorkout` can all share the parent's `__str__`, provided that the following components are set or overridden correctly:
<!-- bilingual-en:end -->

- icon
- kind
- `get_calories`

这些被调用到的部件已经被子类正确设置或重写。
<!-- bilingual-en:start -->
These are exactly the components used by the inherited method.
<!-- bilingual-en:end -->

这说明继承里有个很重要的设计目标：
<!-- bilingual-en:start -->
This demonstrates an important design principle in inheritance:
<!-- bilingual-en:end -->

- 只在必要处 override
- 能复用的就复用
<!-- bilingual-en:start -->
- Override only where behavior genuinely differs.
- Reuse the inherited implementation elsewhere.
<!-- bilingual-en:end -->

### 14. polymorphism：不同子类都能塞进 `total_calories`
<!-- bilingual-en:start -->
*14. Polymorphism: Different Subclasses Work with `total_calories`*
<!-- bilingual-en:end -->
课堂后段最重要的函数之一是：
<!-- bilingual-en:start -->
One of the most important functions in the latter part of the lecture is:
<!-- bilingual-en:end -->

```python
def total_calories(workouts):
    cals = 0
    for w in workouts:
        cals += w.get_calories()
    return cals
```

它完全不需要知道 `w` 到底是：
<!-- bilingual-en:start -->
It does not need to distinguish whether `w` is:
<!-- bilingual-en:end -->

- Workout
- RunWorkout
- SwimWorkout

只要每个对象都支持 `get_calories()`，这个函数就能工作。
<!-- bilingual-en:start -->
As long as each object supports `get_calories()`, this function will work.
<!-- bilingual-en:end -->

这就是本讲最核心的 polymorphism 直觉。
<!-- bilingual-en:start -->
This is the core intuition behind polymorphism in this lecture.
<!-- bilingual-en:end -->

> [!example]
> 相同接口，不同实现；调用方不用分类型分支，方法派发会自动落到对应类上。
> <!-- bilingual-en:start -->
> One interface can have several implementations. The caller needs no type-based branch because method dispatch selects the appropriate class implementation.
> <!-- bilingual-en:end -->

### 15. `total_elevation` 也在提醒你：并不是所有接口都对所有子类通用
<!-- bilingual-en:start -->
*15. `total_elevation`: Not Every Interface Applies to Every Subclass*
<!-- bilingual-en:end -->
与 `total_calories` 相对，老师又给出：
<!-- bilingual-en:start -->
In contrast with `total_calories`, the instructor also gives:
<!-- bilingual-en:end -->

```python
def total_elevation(run_workouts):
    ...
```

这个函数只适用于 RunWorkout，因为只有它们有 `get_elev()`。
<!-- bilingual-en:start -->
This function applies only to `RunWorkout` instances because only they provide `get_elev()`.
<!-- bilingual-en:end -->

这一步提醒你：
<!-- bilingual-en:start -->
This step emphasizes that:
<!-- bilingual-en:end -->

- polymorphism 很强大
- 但也不能假装所有子类接口都完全一样
<!-- bilingual-en:start -->
- Polymorphism is powerful.
- However, it does not mean that every subclass has an identical interface.
<!-- bilingual-en:end -->

设计 API 时仍然要清楚“哪些方法是所有父类实例都应支持的通用接口”。
<!-- bilingual-en:start -->
An API must therefore distinguish methods promised by the parent interface from capabilities available only on particular subclasses.
<!-- bilingual-en:end -->

### 16. 这节课让 OOP 从“语法练习”变成“建模工作”
<!-- bilingual-en:start -->
*16. Turning OOP from Syntax Practice into Modeling*
<!-- bilingual-en:end -->
Lecture 20 的价值在于，它第一次给出一个足够完整、足够真实的对象建模例子。
<!-- bilingual-en:start -->
Lecture 20 provides the course's first sufficiently complete and realistic object model.
<!-- bilingual-en:end -->

从这节课里，你应该看到 OOP 真正的工作流：
<!-- bilingual-en:start -->
The resulting OOP workflow is:
<!-- bilingual-en:end -->

1. 先抽出共性
2. 设计父类接口
3. 用类变量和方法封装通用逻辑
4. 通过子类表达差异
5. 让外部通过统一接口使用不同对象
<!-- bilingual-en:start -->

&nbsp;
**1.** Identify common structure.<br>
**2.** Design the parent interface.<br>
**3.** Encapsulate general logic in class variables and methods.<br>
**4.** Express meaningful differences through subclasses.<br>
**5.** Let callers use different objects through a uniform interface.<br>
<!-- bilingual-en:end -->

## Exercise log

> [!example] Finger exercise 20
> 官方练习实现的是：
> - `Container`
> - `Queue(Container)`
> <!-- bilingual-en:start -->
> The official exercise asks you to implement:
> - `Container`
> - `Queue(Container)`
> <!-- bilingual-en:end -->

和 Lecture 19 的 `Stack` 题形成一对。
<!-- bilingual-en:start -->
It pairs with the `Stack` problem from Lecture 19.
<!-- bilingual-en:end -->

虽然它不是 workout 例子本身，但它和本讲方法一致：
<!-- bilingual-en:start -->
Although it is not the workout example itself, it follows the same approach as this lecture:
<!-- bilingual-en:end -->

- 父类放通用 `size`、`add`
- 子类只补 `remove`
- 具体语义从后进先出改成先进先出
<!-- bilingual-en:start -->
- The parent supplies the shared `size` and `add` behavior.
- The subclass adds only `remove`.
- The removal rule changes from last-in, first-out to first-in, first-out.
<!-- bilingual-en:end -->

所以它正好巩固了“在稳定父类接口上做行为 specialization”这件事。
<!-- bilingual-en:start -->
It therefore reinforces behavioral specialization on top of a stable parent interface.
<!-- bilingual-en:end -->

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
<!-- bilingual-en:start -->
- [ ] I can derive the common attributes of a generic workout from the requirements.
- [ ] I can explain why we first implement `SimpleWorkout` before refactoring to `Workout`.
- [ ] I can explain why `get_calories` is a typical example of information hiding.
- [ ] I can explain why `cal_per_hr` is a class variable.
- [ ] I can explain why datetime objects are preferable to manual string handling.
- [ ] I can write a subclass `__init__` that calls `super()` before adding subclass attributes.
- [ ] I can identify what `RunWorkout` and `SwimWorkout` override.
- [ ] I can explain why subclasses can share the parent's `__str__` method.
- [ ] I can use `total_calories` to illustrate polymorphism.
- [ ] I can reconstruct the lecture sequence: `SimpleWorkout` -> `Workout` -> class variable and datetime -> `RunWorkout` / `SwimWorkout` -> polymorphism.
<!-- bilingual-en:end -->

> [!warning] Common mistakes
> - 还没抽清共性和特性，就急着写一堆平级类。
> - 子类重写太多本可复用的父类逻辑。
> - 把 class variable 和 instance variable 混在一起。
> - 调用统一接口时又手写一堆 `if type(...)`，错失多态的价值。
> <!-- bilingual-en:start -->
> - Creating many peer classes before separating common structure from specialized state.
> - Overriding parent logic that a subclass could reuse.
> - Confusing class variables with instance variables.
> - Writing manual `if type(...)` branches around a common interface and thereby losing the benefit of polymorphism.
> <!-- bilingual-en:end -->
