---
date: 2026-02-08
科目: MIT 6.100L
---

# Loops [[Over-identified|over]] strings, guess-and-check and binary

## 本讲主线

- 字符串循环与 `break`
- Guess-and-check（穷举/试探）算法
- 二进制表示与浮点误差直觉

## 1. Loops [[Over-identified|over]] strings（字符串上的循环）

### `break` 的语义

- `break` 会立即退出当前所在的最内层循环
- 循环体中 `break` 后面的语句不会执行
- 多层循环里，`break` 只影响最内层

```python
mysum = 0
for i in range(5, 11, 2):
    mysum += i
    if mysum == 5:
        break
    mysum += 1
print(mysum)
```

### for 循环不只遍历数字

```python
s = "demo loops - fruit loops"

for index in range(len(s)):
    if s[index] in 'iu':
        print("There is an i or u")

for i in s:
    if i in 'iu':
        print("There is an i or u")
```

- `for ... in range(...)`：显式使用索引
- `for char in s`：直接遍历字符，更简洁

> [!tip] Big Idea
> `for` 循环遍历的是“序列（sequence）”，不是“数字本身”。

### 课堂点：统计 unique letters

```python
s = "abca"
seen = ""
for ch in s:
    if ch not in seen:
        seen += ch
print(len(seen))  # 3
```

## 2. Guess-and-check（穷举检验）

### 核心模式

- 先系统地产生候选解（guess）
- 对每个候选做正确性检查（check）
- 命中则停止，否则继续

> [!important] Big Idea
> Guess-and-check 必须有有限搜索空间；不能检查无限多个候选。

>[!example] 例子：平方根（整数）
>
> ```python
> guess = 0
> x = int(input("Enter an integer: "))
> while guess**2 < x:
>     guess += 1
>
> if guess**2 == x:
>     print("Square root of", x, "is", guess)
> else:
>     print(x, "is not a perfect square")
> ```
>
> 负数版本要先做符号处理（或直接拒绝输入），否则循环条件可能立刻失效。
>
### Boolean flag（布尔标记）

```python
secret = 7
found = False
for i in range(1, 11):
    if i == secret:
        found = True
        break

if found:
    print("Found", secret)
else:
    print("Not found")
```

- `found` 作为“是否发生过某事件”的信号量
- 代码可读性通常高于纯计数器绕写

### while vs for（在穷举问题里）

- `for`：候选范围已知（如 `range(abs(cube)+1)`）时更自然
- `while`：候选生成规则更动态时更灵活
- 如果只是在遍历一个固定范围，优先 `for`

>[!example] 例子：立方根（正负数）
>
> ```python
> cube = 1000
> for guess in range(abs(cube) + 1):
>     if guess**3 >= abs(cube):
>         break
>
> if guess**3 != abs(cube):
>     print(cube, "is not a perfect cube")
> else:
>     if cube < 0:
>         guess = -guess
>     print("Cube root of", cube, "is", guess)
> ```
>
> - 通过 `>=` + `break` 可减少不必要迭代
> - 负数输入最终恢复符号
>
### 应用：文字题也可穷举

- 三重循环可暴力求解条件联立问题
- 变量规模变大时，嵌套穷举会显著变慢
- 优化思路：用约束关系直接推导部分变量，减少循环维度

## 3. Binary and floats（二进制与浮点）

### 动机：浮点并不精确

```python
x = 0
for i in range(10):
    x += 0.1

print(x == 1)      # False（常见）
print(x, 10*0.1)
```

> [!important] Big Idea
> 浮点数是对实数的近似表示；微小误差在重复运算后可能放大为可见差异。

### 为什么是二进制

- 硬件更容易稳定实现“两态系统”（0/1）
- 数据最终都以 bit 串编码

十进制：

$$
1507 = 1\times10^3 + 5\times10^2 + 0\times10^1 + 7\times10^0
$$

二进制本质相同，只是底数改为 2。

### 十进制整数转二进制（Python 实现思路）

```python
num = 1442

is_neg = num < 0
num = abs(num)

result = ""
if num == 0:
    result = "0"

while num > 0:
    result = str(num % 2) + result
    num = num // 2

if is_neg:
    result = "-" + result

print(result)
```

- `% 2` 取当前最低位
- `// 2` 右移到下一位
- 逆序拼接得到最终二进制字符串

## 课堂练习（You Try It）

- [x] 统计 `range(5)`, `range(10)`, `range(2,9,3)`, `range(-4,6,2)` 中偶数个数 ✅ 2026-02-09
- [x] 给字符串 `s` 统计 unique letters 数量（不用 set） ✅ 2026-02-09
- [x] 写 secret number 搜索程序，对比“有/无 Boolean flag”两个版本 ✅ 2026-02-09
- [x] 实现整数转二进制函数，测试 `0, 1, 19, -19` ✅ 2026-02-09
- [x] 验证 `0.1 + 0.1 + ...`（10 次）与 `1` 的比较结果 ✅ 2026-02-09

## 小结

- `for` 可遍历任意序列（数字区间、字符串等）
- Guess-and-check 是通用、直接但可能慢的算法框架
- 布尔标记可清晰表达“某事件是否发生”
- 浮点数是近似值，理解二进制表示有助于理解误差来源



## 资料

- ![[MIT 6.100L-slides/mit6_100l_lec04.pdf]]
