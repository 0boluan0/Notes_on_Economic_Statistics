---
aliases:
  - MIT 6.100L Lecture 19
  - 6.100L L19
  - Inheritance
tags:
  - computer-science
  - python
  - mit-6.100l
  - lecture-note
科目: Computer Science
course: MIT 6.100L Introduction to CS and Programming Using Python
lecture: 19
---

# Lecture 19: Inheritance

> [!tip] Hint
> - 这节课开头先用猫、兔子、人这些现实分类说明 inheritance 的直觉，不是先讲语法。
> - Animal 被选作 parent class，因为 age、name 这类属性适合放在更泛的层级。
> - getters/setters 在这一讲再次被强调，是为了 information hiding 和后面改内部实现的灵活性。
> - subclass 的关键不是“复制父类代码”，而是 automatically gets parent attributes and methods，再加自己的东西。
> - `Cat(Animal)` 先演示最简单的继承：复用父类的 init、getter、setter，再加 `speak` 和重写 `__str__`。
> - Person 的 `__init__` 比 Cat 更复杂，因为它要先调用 `Animal.__init__`，再补自己的属性。
> - Student 再继承 Person，课堂借它说明 inheritance 可以一层层往下传。
> - method overriding 在这节课变成显性主题：子类可以保留父类大部分行为，但改掉某个方法，比如 `speak`。
> - Rabbit 部分除了继承，还引入 class variable、`__add__`、`__eq__`，开始让对象类型更“活”。
> - 听完这节课，你应该能解释继承为什么能减少重复设计，同时保留子类差异。

## Lecture flow

### 1. 先从现实分类讲 inheritance 的直觉
Lecture 19 开场先退回现实世界。

老师举了：

- cats
- rabbits
- people

这些类别的例子，是为了说明 inheritance 的动机不是语法炫技，而是现实中的分类本来就有层次：

- 都是 animal
- 但各自还有自己的特殊性质

这一步很重要，因为 inheritance 的核心不是“代码复用”四个字，而是：

- 通用属性放上层
- 特殊属性留给下层

### 2. Animal：先定义最泛的父类
基于这个现实分类，老师先写出 `Animal` 类。

`Animal` 的数据和行为都很基础：

- `age`
- `name`
- `get_age`
- `get_name`
- `set_age`
- `set_name`
- `__str__`

这里课堂特别强调的是：  
父类不需要穷尽所有具体细节，它只需要承载“所有子类都会共享的那部分”。

### 3. getters / setters 再次出现，是为了 information hiding
老师在 Animal 这里专门停下来重新强调 getters 和 setters。

原因不是怕你忘了方法定义，而是因为：

- 如果别人总是直接写 `a.age = ...`
- 将来你想更改内部表示方式就会非常痛苦

而如果大家统一通过：

- `get_age()`
- `set_age(...)`

来交互，那么你以后可以改内部实现，而不必把所有外部调用点一起推翻。

这就是 information hiding 的实际价值。

### 4. 用 Animal objects 参与普通程序
老师接着没有直接讲继承，而是先展示 Animal 类如何在普通代码中被使用。

例如：

- `animal_dict(L)`：把列表里的非负整数映射成 Animal objects
- `make_animals(L1, L2)`：根据年龄列表和名字列表创建一组 Animal

这一步很关键，因为它说明对象不是只在类定义里存在，它们会真的进入普通数据流：

- 被放进 list
- 被放进 dict
- 被函数创建和返回

### 5. 正式引入 inheritance：子类天然拥有父类的通用部分
到这里老师才真正给出继承的图景。

如果：

- Cat is an Animal
- Rabbit is an Animal
- Person is an Animal

那这些子类就应该自动继承 Animal 的通用属性和行为，例如：

- age
- name
- getters / setters

与此同时，每个子类还能加自己的行为。

### 6. `Cat(Animal)`：最简单的继承范例
老师先从最简单的子类开始：

```python
class Cat(Animal):
    def speak(self):
        print("meow")
    def __str__(self):
        return "cat:" + str(self.name) + ":" + str(self.age)
```

这段代码的课堂重点有两个：

- Cat 没有重写 `__init__`，所以直接复用 Animal 的初始化逻辑
- Cat 可以在保留父类通用属性的同时，补上自己的方法

这就是 inheritance 最干净的第一版。

### 7. method lookup：先看子类，找不到再往父类走
老师在 Cat 例子上花了一些时间解释调用机制。

当你对 Cat 对象调用方法时，Python 会：

1. 先在 Cat 类里找
2. 找不到，再去 Animal 里找
3. 再找不到，再继续向上

这就是为什么：

- Cat 可以用 Animal 里的 `get_age`
- 但打印时如果 Cat 自己有 `__str__`，就会优先用 Cat 的版本

### 8. overriding：子类可以替换父类某个行为
`__str__` 在 Cat 中其实就是一次 method overriding。

父类 Animal 的字符串表示可能是：

- `animal:name:age`

但 Cat 需要：

- `cat:name:age`

这就是覆盖父类方法的典型场景：

- 通用框架大致一样
- 但某个具体行为更适合由子类自定义

### 9. Person：为什么有时必须重写 `__init__`
Cat 之后，老师切到 `Person(Animal)`，并让它的初始化更复杂一些。

原因在于：

- Person 需要 `name`
- 还要额外有 `friends`

所以 `Person.__init__` 不能简单沿用 Animal 的签名，而要：

1. 先调用 `Animal.__init__(self, age)`
2. 再通过 `self.set_name(name)` 补名字
3. 再初始化自己的新属性 `friends`

这一段特别重要，因为它让你看到：

- 子类不是只能照搬父类 init
- 子类可以在“先继承、再补充”的思路下写自己的构造逻辑

### 10. Person 的新行为：friend list、age_diff、speak
在 Person 中，老师加入了更多特有行为：

- `add_friend`
- `get_friends`
- `age_diff`
- `speak`

这时 inheritance 的价值就更清楚了：

- age / name 这些通用能力从 Animal 来
- friend list 和人类说话方式属于 Person 自己

因此子类不是“比父类更窄”，而是“继承通用部分并扩展专属部分”。

### 11. Student：继承链可以继续向下
讲完 Person 后，老师继续往下分：

- Student is a Person

于是 Student 会继承：

- Animal 的通用部分
- Person 的通用部分

再加上自己的：

- `major`
- `change_major`
- 更符合学生语境的 `speak`

这里课堂真正想建立的是一条继承链直觉：

- Student -> Person -> Animal -> object

### 12. Student 的 `speak`：override 的典型用途
Student 里的 `speak` 不再像 Person 一样简单说 “hello”，  
而是输出和学生生活相关的话。

这让 method overriding 的意义变得很直观：

- 子类共享了父类大部分通用能力
- 但在某个行为上想更具体、更贴近自身语义
- 就直接覆写那个方法

### 13. Rabbit：class variable 进入继承场景
后半段老师把重点转向 Rabbit，并引入 class variable `tag`。

Rabbit 类中：

- 每创建一只兔子，分配一个新的编号
- 这个编号增长规则属于整个类，不属于某一只单独兔子

所以最合适的位置就是 class variable。

这部分是在把 Lecture 18 的 class variable 观念塞进继承框架中。

### 14. Rabbit 的父母关系：对象还能引用同类对象
Rabbit 类更进一步的地方在于：

- 每只兔子可以记录 parent1 / parent2
- 所以对象属性里可以装“同类的其他对象”

这让对象图开始变复杂：

- rabbit object 引用 other rabbit objects
- 类之间不只是树状继承，实例之间也会形成关系网络

### 15. `__add__`：把“两只兔子相加”解释成产生新兔子
老师随后给 Rabbit 加上：

```python
def __add__(self, oth):
    return Rabbit(0, self, oth)
```

这当然不是数学加法，而是课程借运算符重载展示：

- 你可以定义对象类型自己认为合理的“加法”语义

在 Rabbit 世界里，`r1 + r2` 就被定义为：

- 产生一只新兔子
- 其父母是 `r1` 和 `r2`

### 16. `__eq__`：对象相等性的标准也由你定义
接着老师再写 Rabbit 的 `__eq__`。

在这里，两只兔子相等不是看内存地址，而是看：

- 它们是否有相同的父母

并且父母顺序交换也算相同。

这一步非常关键，因为它让你看到：

- 对象“相等”到底是什么意思
- 也可以由类设计者来决定

### 17. 这节课其实在讲“共性与差异如何同时保留”
Lecture 19 如果只记成“继承会省代码”，就太浅了。

这节课真正完成的是：

- 用父类保存共性
- 用子类表达差异
- 用 overriding 让差异体现在行为上
- 用 class variable 和对象关系让类世界更贴近真实建模需求

## Exercise log

> [!example] Finger exercise 19
> 官方题目要求实现：
> - `Container`
> - `Stack(Container)`

Container 提供：

- `size`
- `add`

Stack 额外提供：

- `remove`

而且 `remove` 要体现后进先出。

这题很适合作为本讲练习，因为它不是在考“会不会写 list”，而是在考：

- 你会不会先把通用行为放父类
- 再把特有行为放子类

这正是 inheritance 的最小工程版本。

## Links to follow-up practice
- Slides: [[MIT 6.100L-slides/mit6_100l_lec19.pdf|Lecture 19 slides]]
- Lecture code: [[MIT 6.100L-lecture-code/mit6_100l_lec19_code.py|Lecture 19 code (py)]]
- Finger exercise: [[MIT 6.100L-finger-exercises/mit6_100l_ex19_sol.pdf|Lecture 19 finger exercise solution]]
- Transcript: [[MIT 6.100L-transcripts/mit6_100l_lec19_transcript.pdf|Lecture 19 transcript]]
- Recitation 9: [[MIT 6.100L-recitations/mit6_100l_rec09.zip|Recitation 09 materials]]
- Problem set milestone: none directly scheduled on this lecture
- Textbook: [[Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Guttag textbook]] (Ch 10.2)

## Review checklist
- [ ] 我能用现实分类例子解释 inheritance 的直觉。
- [ ] 我能说明为什么 Animal 适合作为父类。
- [ ] 我能解释 getters/setters 在继承层级里为什么仍然重要。
- [ ] 我能读懂并写出最简单的子类，如 `Cat(Animal)`。
- [ ] 我能解释 method lookup 为什么是“先看子类，再看父类”。
- [ ] 我能说明什么时候子类应该重写 `__init__`。
- [ ] 我能解释 overriding 的意义，并举出 `speak` 或 `__str__` 的例子。
- [ ] 我能理解 Student 为什么会继承 Person 再继承 Animal。
- [ ] 我能解释 Rabbit 中 class variable 和 `__add__` / `__eq__` 的设计含义。
- [ ] 我能按课堂顺序复述：Animal -> Cat -> Person -> Student -> Rabbit。

> [!warning] Common mistakes
> - 把继承理解成简单复制粘贴，而不是共性抽取。
> - 子类需要额外初始化时忘记先调用父类的 `__init__`。
> - 不理解 overriding，看到同名方法就不知道 Python 会选哪个版本。
> - 该放父类的通用逻辑放到了子类，导致层级设计混乱。
