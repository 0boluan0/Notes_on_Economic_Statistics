---
aliases:
  - MIT 6.100L Lecture 14
  - 6.100L L14
  - Dictionaries
tags:
  - computer-science
  - python
  - mit-6.100l
  - lecture-note
科目: Computer Science
course: MIT 6.100L Introduction to CS and Programming Using Python
lecture: 14
---

# Lecture 14: Dictionaries

> [!tip] Hint
> - 这节课不是直接上 dict 语法，而是先证明“用多个 list 存学生信息会越来越别扭”。
> - 老师先演示 name list + grade list，再演示 list of lists，都是为了说明 lookup 很笨重。
> - dict 的核心不是“花括号”，而是 mapping：用 key 直接到 value。
> - 列表按位置找信息，字典按 key 找信息；这就是本讲的结构性变化。
> - `grades['John']` 之所以自然，是因为 key 本身承载了查找语义，不再靠平行索引硬对齐。
> - 创建、访问、添加、更新、删除 entry 的语法都围绕 key:value 展开。
> - `in` 对 dict 默认查的是 key，不是 value，这是课堂里特意强调的点。
> - `keys()`、`values()`、`items()` 是三种不同的观察角度，后面写循环时很常用。
> - 词频统计例子是本讲真正的大例子，它把 dict 当成“计数器”和“聚合器”来用。
> - 听完这节课，你应该把 dict 看成新的数据组织方式，而不只是又多了一堆 API。

## Lecture flow

### 1. 先故意用 list 存成绩，暴露结构缺陷
Lecture 14 的开场不是定义 dictionary，而是先把旧方法走到难受的地方。

老师继续用熟悉的学生成绩例子：

- 一个 list 存学生名字
- 一个 list 存对应成绩
- 约定相同 index 代表同一个学生

这在数据很小时还能勉强工作，但一旦你想：

- 查某个学生成绩
- 增加更多字段
- 保证不同列表一直同步

就会变得非常脆弱。

### 2. list of lists 也能做，但仍然很笨重
为了说明问题不是“列表数量太多”，老师又展示了另一种老办法：

- 每个学生是一条子列表
- 子列表里再装 ps、mq 等信息

这比平行列表更集中一些，但查找时仍然很绕：

- 先找到学生
- 再找到某类成绩
- 再拿到对应值

课堂这里想让你真正感受到：

> [!note]
> 当数据天然就是“名字对应信息”“键对应值”的关系时，list 不是最合适的抽象。

### 3. dictionary 的动机：直接从 key 找 value
证明完旧方法很别扭之后，老师才正式引出 dictionary。

dictionary 的本质是 **mapping**：

- key -> value

这意味着：

- 你不再依赖位置
- 你直接用有意义的 key 去访问信息

比如：

```python
grades = {'Ana': 'B', 'Matt': 'A', 'John': 'B'}
```

这里 `'John'` 自身就是查找入口，而不需要先找 John 在第几个位置。

### 4. dict 的核心心智模型：entry，不是序列
老师在这里其实是把大家从 sequence 思维切到 mapping 思维。

list / tuple 的基本问题是：

- 第几个元素是什么

dict 的基本问题则变成：

- 某个 key 对应什么 value

所以写字典时要开始习惯：

- 关注 key 是否存在
- 关注 key 和 value 的对应关系
- 不再把“第几个位置”当作主要组织方式

### 5. 基本操作：创建、访问、添加、修改、删除
接着老师依次走过最常见的字典操作。

创建：

```python
d = {}
grades = {'Ana': 'B', 'Matt': 'A'}
```

访问：

```python
grades['John']
```

新增或修改：

```python
grades['Grace'] = 'A'
grades['Grace'] = 'C'
```

删除：

```python
del(grades['Ana'])
```

这里课堂的重点不是 API 数量，而是你要真正接受：

- dict 的更新几乎总围绕 key 展开

### 6. membership test：`in` 默认查的是 key
老师随后强调了一个很容易误判的地方：

```python
'John' in grades
'B' in grades
```

这两句不等价。

对于 dict：

- `in` 默认只查 key
- 不查 value

所以 `'John' in grades` 可能为真，  
但 `'B' in grades` 即使确实是某人的成绩，也仍然可能为假。

这一步很重要，因为它把 dict 从 list 的 membership 直觉里彻底剥离出来。

### 7. 看 dict 的三个窗口：keys、values、items
老师接下来引入：

- `d.keys()`
- `d.values()`
- `d.items()`

它们分别对应三种观察角度：

- 只看所有 key
- 只看所有 value
- 同时看 `(key, value)` 对

这在写循环时很关键。

例如：

```python
for k, v in grades.items():
    print(k, v)
```

这里你已经不是在遍历一个 sequence 的位置，而是在遍历一个 mapping 的 entry。

### 8. 第一批 you-try-it：把“查找语义”写进函数
课堂中段几个 you-try-it 都很典型：

- `find_grades(grades, students)`
- `find_in_L(Ld, k)`
- `count_matches(d)`

这些题的共同点是：  
它们不再问“怎么按位置处理”，而是问你能不能围绕 key/value 关系写逻辑。

例如 `find_grades` 的重点就是：

- 按给定学生顺序输出对应成绩

而不是去管这些成绩原本在字典里内部以什么顺序存放。

### 9. 嵌套 dict：字典可以组织更复杂的数据
老师随后把难度往前推，给出类似：

```python
my_d = {
    'Ana': {'mq': [10], 'ps': [10, 10]},
    'Fredo': {'ps': [7, 8], 'mq': [8]},
    'Eric': {'mq': [3], 'ps': [0]}
}
```

这时 dict 不再只是“名字到单个值”的简单映射，而是可以承载层级化信息：

- 学生名 -> 某个学生的整套记录
- 记录里再按 `ps`、`mq` 分类

`get_average(data, what)` 这种函数正是在训练你如何穿过这个嵌套结构。

### 10. 词频统计：dict 最典型的大用途之一
本讲最完整的例子是 song lyrics 的 frequency dictionary。

老师的推进顺序很清楚：

1. 先把歌词转小写、按空格拆词
2. 建一个空 dict
3. 扫描每个词
4. 如果词已出现，频数加 1
5. 如果是第一次出现，插入频数 1

代码模式大致是：

```python
for w in words_list:
    if w in word_dict:
        word_dict[w] += 1
    else:
        word_dict[w] = 1
```

这个模式非常重要，因为后面看到“统计频率”“聚合计数”“直方图”类任务时，第一反应就应该是 dict。

### 11. 从 frequency dict 再做分析：最大频数、常见词列表
老师没有停在“建出来”这一步，而是继续往下做分析：

- `find_frequent_word(word_dict)`：找最大频数对应的词
- `occurs_often(word_dict, x)`：不断提取超过阈值的高频词

这让你看到 dict 不只是存结果，它还能成为后续分析的基础结构。

这里课堂顺手也提醒了一个重要问题：

- 有些函数会 mutate 传入的 dict

所以又一次回到了前面 lists 讲过的副作用意识。

### 12. 这节课真正转变的是“如何组织数据”
Lecture 14 的重要性不在于又学了一个容器，而在于课程在逼你改变组织信息的方式。

从这节课开始，你面对问题要学会先问：

- 我的数据更像 sequence 吗
- 还是更像 mapping

如果查找入口天然是名字、ID、标签、单词本身，那 dict 往往比 list 更贴切。

## Exercise log

> [!example] Finger exercise 14
> 官方练习有两问：
> - `keys_with_value(aDict, target)`：返回所有 value 等于 `target` 的 key，并排序。
> - `all_positive(d)`：对 `int -> list` 的字典，返回 value 列表和为正的所有 key，并排序。

这两问都很适合本讲，因为它们分别训练：

- 从 value 反向筛 key
- 遍历 `k, v` 对进行聚合判断

官方解法都不复杂，但能逼你把 dict 思维真正用起来，而不是下意识又回到“按位置找东西”的序列思路。

## Links to follow-up practice
- Slides: [[MIT 6.100L-slides/mit6_100l_lec14.pdf|Lecture 14 slides]]
- Lecture code: [[MIT 6.100L-lecture-code/mit6_100l_lec14_code.py|Lecture 14 code (py)]]
- Finger exercise: [[MIT 6.100L-finger-exercises/mit6_100l_ex14_sol.pdf|Lecture 14 finger exercise solution]]
- Transcript: [[MIT 6.100L-transcripts/mit6_100l_lec14_transcript.pdf|Lecture 14 transcript]]
- Recitation 7: [[MIT 6.100L-recitations/mit6_100l_rec07.zip|Recitation 07 materials]]
- PS 3 halfway hand-in due: [[MIT 6.100L-problem-sets/mit6_100l_ps3.pdf|PS3 statement]], [[MIT 6.100L-problem-sets/mit6_100l_ps3_code.zip|PS3 starter code]]
- Textbook: [[Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Guttag textbook]] (Ch 5.7)

## Review checklist
- [ ] 我能解释为什么学生成绩例子更适合 dict 而不是多个 list。
- [ ] 我能说清 sequence 和 mapping 这两种组织数据方式的差别。
- [ ] 我能熟练区分 dict 的创建、访问、更新、删除是如何围绕 key 展开的。
- [ ] 我能解释为什么 `in` 对 dict 默认查 key。
- [ ] 我能区分 `keys()`、`values()`、`items()` 的用途。
- [ ] 我能读懂和编写嵌套 dict 结构上的访问代码。
- [ ] 我能复述词频统计字典的标准构建模式。
- [ ] 我能说明 `find_frequent_word` 这类函数如何基于 frequency dict 再做分析。
- [ ] 我能把 finger exercise 14 和“遍历 entry、筛 key”联系起来。
- [ ] 我能按课堂顺序复述：list 方案为什么差 -> dict 作为 mapping -> dict operations -> frequency dictionary。

> [!warning] Common mistakes
> - 还在用 list 的位置思维理解 dict。
> - 忘记 `in` 对字典查的是 key。
> - 用 dict 时不先想清楚 key 应该是什么，导致结构本身就难查。
> - 写嵌套 dict 访问时没分层，容易把 `data[stud][what]` 这类索引次序写错。
