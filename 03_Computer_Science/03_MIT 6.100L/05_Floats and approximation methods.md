---
date: 2026-02-09
科目: MIT 6.100L
---

# 十进制小数与二进制表示（Fractions & Binary Representation）

## 1. 问题背景：为什么要讨论小数？

计算机内部使用的是**二进制（base-2）**，但人类世界中的数通常是**十进制（base-10）**。  
整数的进制转换相对直接，而**小数（fraction）**才是问题的根源。

核心问题是：

> 一个十进制小数，如何用二进制精确或近似表示？

这正是浮点数误差的起点。

---

## 2. 十进制小数的“位权”本质

以 $\frac{3}{8}$ 为例：

$$  
\frac{3}{8} = 0.375  
$$

十进制表示并不是“整体”，而是位权展开：

$$  
0.375 = 3\times10^{-1} + 7\times10^{-2} + 5\times10^{-3}  
$$

这一步的目的不是计算，而是强调：

> **任何进制下的小数，都是“数字 × 进制的负次幂”的和。**

---

## 3. 核心思想（Recipe Idea）

如果一个十进制小数 $x$，存在某个整数 $p$，使得：

$$  
x \times 2^p \in \mathbb{Z}  
$$

那么可以用以下步骤转换为二进制：

1. 将 $x$ 乘以 $2^p$，得到整数
    
2. 把这个整数转成二进制
    
3. 再除以 $2^p$（即二进制小数点左移 $p$ 位）
    

>[!example] 示例：$0.375$
>
> $$  
> 0.375 \times 2^3 = 3  
> $$
>
> $$  
> 3_{10} = 11_2  
> $$
>
> 右移 3 位：
>
> $$  
> 0.375_{10} = 0.011_2  
> $$
>
> ---
>
## 4. Python 实现：将十进制小数转为二进制



```python
# Convert a decimal fraction to binary representation
# Assumption: x can be represented exactly as k / (2^p)

x = 0.61   # try changing this, e.g. 0.375, 0.5, 0.1
p = 0

# Step 1: find the smallest p such that x * 2^p is an integer
while ((2 ** p) * x) % 1 != 0:
    remainder = (2 ** p) * x - int((2 ** p) * x)
    print("Remainder =", remainder)
    p += 1

# Step 2: compute the integer value
num = int(x * (2 ** p))

# Step 3: convert the integer to binary
result = ''
if num == 0:
    result = '0'

while num > 0:
    result = str(num % 2) + result
    num = num // 2

# Step 4: pad with leading zeros if needed
for _ in range(p - len(result)):
    result = '0' + result

# Step 5: insert the binary point
binary_representation = result[:-p] + '.' + result[-p:]

print("The binary representation of the decimal", x, "is", binary_representation)

```

## 5. 代码整体逻辑概览

代码分为五个阶段：

1. 不断乘以 $2^p$，寻找能变成整数的 $p$
    
2. 得到整数 $x\times2^p$
    
3. 将该整数转换为二进制字符串
    
4. 补足长度，使小数位数正确
    
5. 插入二进制小数点
    

---

## 6. 完整代码（逐段理解）

```python
x = 0.625
p = 0
```

目标：寻找最小的 $p$，使得 $x \times 2^p$ 为整数。

---

### 6.1 寻找合适的 $p$

```python
while ((2**p) * x) % 1 != 0:
    print('Remainder = ' + str((2**p)*x - int((2**p)*x)))
    p += 1
```

解释：

- `(2**p) * x`：尝试把小数放大
    
- `% 1 != 0`：检查是否还有小数部分
    
- 只要还有小数，就继续增大 $p$
    

这一步在数学上等价于：

> 判断 $x$ 的分母是否能被 $2^p$ 消掉

---

### 6.2 得到整数形式

```python
num = int(x * (2**p))
```

此时：

$$  
\text{num} = x \times 2^p \in \mathbb{Z}  
$$

---

### 6.3 将整数转换为二进制

```python
result = ''
if num == 0:
    result = '0'
```

边界情况处理。

```python
while num > 0:
    result = str(num % 2) + result
    num = num // 2
```

这是**最经典的整数转二进制算法**：

- 不断除以 2
    
- 取余数
    
- 逆序拼接
    

---

### 6.4 补齐二进制长度

```python
for i in range(p - len(result)):
    result = '0' + result
```

目的：

> 确保小数点右侧有 $p$ 位二进制数字

---

### 6.5 插入二进制小数点

```python
result = result[0:-p] + '.' + result[-p:]
```

这是整段代码最关键的一行。

数学含义：

$$  
\text{binary} = \frac{\text{num}}{2^p}  
$$

字符串操作只是“形式实现”。

---

### 6.6 输出结果

```python
print('The binary representation of the decimal ' + str(x) + ' is ' + str(result))
```

---

## 7. 对这段代码的关键理解

1. **它只对分母是 $2^k$ 的小数有效**
    
    - 例如：$0.625 = \frac{5}{8}$
        
    - 对 $0.1$ 会无限循环
        
2. **这正是浮点数误差的来源**
    
    - 并不是 Python 的问题
        
    - 而是二进制表示的数学限制
        
3. **IEEE 754 的思想与此完全一致**
    
    - 只是自动化 + 位级表示
        


# Floats, approximation methods, and early numerical thinking

## 本讲主线

- 浮点数表示及其误差源头
- 算法参数：增量与 epsilon 的权衡
- 从穷举走向 bisection/newton 的分步优化

## 1. 浮点数为什么不精确

- 0.1 无法在有限二进制位精确表示；单次加法会留下小误差
- 任何用 == 比较 floats 都是危险的

```python
x = 0
for _ in range(10):
    x += 0.1

print(x == 1)     # False
print(x, 10 * 0.1)
```

> [!important] Big Idea
> 浮点是“近似表示的实数”，多次运算会放大微小误差，比较时应使用“近似相等”而非 ==。

### 实数与二进制

- 小数部分可以写成一串 `a*2^{-1} + b*2^{-2} + ...`
- 整数部分的转换：持续 `% 2` 取最低位并用 `// 2` 缩减
- 小数部分需要乘以 2 的幂再除回去才能恢复，但不总能得到整数

## 2. Approximation methods（近似算法）

- 不再寻找精确答案，而是设定 `epsilon`，只要 `|guess**2 - x| < epsilon` 就收手
- 需要两个参数：`epsilon` 决定精度，`increment` 决定搜索步长
- 减小 `increment` 会明显降低性能；增大 `epsilon` 会牺牲精度

```python
x = 35
epsilon = 0.001
increment = 0.0000001
guess = 0.0
while abs(guess**2 - x) >= epsilon and guess**2 <= x:
    guess += increment
print(guess, 'is close to', x)
```

- 还需要防止“跳过”近似区间，否则可能永远无法跳出 while
- 增加 `guess**2 <= x` 条件可以阻止过冲，并能报告失败

> [!tip] Big Idea
> 近似算法是“guess-and-check + float increment”，但无法“检查是否正确”，只能检查是否“足够接近”。

## 3. “好”算法：bisection、Newton-Raphson

### Bisection search    

- 需要：有序搜索区间 + 能告诉你猜测是偏大还是偏小
- 每次猜 midpoint，成功则终止；否则根据反馈缩小区间到一半
- 复杂度从线性降为 `O(log N)`，比如 `√54321` 从 23M 次降到 30 次

```python
low = 0
high = x if x >= 1 else 1
guess = (high + low) / 2.0
while abs(guess**2 - x) >= epsilon:
    if guess**2 < x:
        low = guess
    else:
        high = guess
    guess = (high + low) / 2.0
```

- `x < 1` 时要调整初始 `[low, high]`

### Newton-Raphson

- 通用 root finding：`guess <- guess - (guess**2 - k)/(2*guess)`
- 迭代次数通常远低于 bisection，但需要先验导数公式

## 4. 课堂练习（You Try It）

- [x] 用 `epsilon` + `increment` 写一个“good enough” 的平方根，测试 `12345`、`54321` ✅ 2026-02-16

```python
n = 12345
epsilon = 0.01
p = 0
increment = 0.00001

while abs( p**2 - n ) >=epsilon and p**2<=n:
	p += increment
	
print(p)
print(p**2)

```

- [x] 实现能早早停止的 `while abs(...) < epsilon and guess**2 <= x` 版，并观察 `guess` 数量 ✅ 2026-02-16
- [x] 比较 `increment=0.0001` 与 `0.00001` 的运行速度/命中情况 ✅ 2026-02-16
- [ ] 用 `while` 实现 `num % 2` 还是 `range` 选择判定奇数
- [x] “Never use == ”：设计一个条件测试 0.1 的多次累加 ✅ 2026-02-16

## 小结

- 浮点数是有限位的近似，千万别用 == ；用 `abs(a - b) < epsilon`
- 近似算法需要调两个参数；减小 `increment` 或 `epsilon` 的代价是速度
- Bisection/newton 通过 smarter guess 减少迭代次数

## 资料

- ![[MIT 6.100L-slides/mit6_100l_lec05.pdf]]
