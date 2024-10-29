
# introduction


# 高斯消元法

## 方程消元

 线性代数的核心是解联立线性方程组.

寻找主元的过程就是将一个矩阵转为上三角矩阵的过程.对角线上的每个值都是一个主元.主元不能为0.

## 矩阵消元
### 原理

矩阵消元的过程和方程消元的过程是一致的,区别在于需要把过程写为矩阵形式.即不能说用哪一行减去哪一行之类的,应该说是使用目标矩阵乘以某一矩阵以得到目标矩阵的上三角矩阵形式.

$$
\begin{bmatrix}
 ?&?&? \\ ?&?&? \\ ?&?&? \\ 
\end{bmatrix}
\begin{bmatrix}
 1 \\ 2 \\ 7 \\ 
\end{bmatrix}
$$

意味着1xcolumn1+2xcolumn2+7xcolumn3合成一个列向量.是列向量的线性组合

那么同时,

$$
\begin{bmatrix}
 1&2&7
\end{bmatrix}
\begin{bmatrix}
 ?&?&? \\ ?&?&? \\ ?&?&? \\ 
\end{bmatrix}
$$

就是1xrow1+2xrow2+7xrow3,是行向量的线性组合. 

### 举例

对于方程
$$
\begin{cases} 
1x+2y+1=0\\
3x+8y+1z=-1\\
 4y+1z=4 \\
\end{cases}
$$

可以提出系数矩阵:

$$
\begin{bmatrix}
 1&2&1  \\
 3&8&1 \\ 
 0&4&1 \\ 
 \end{bmatrix}
 $$

要乘上一个矩阵使其变为
$$
\begin{bmatrix}
 1&2&1  \\
 0&2&-2 \\ 
 0&4&1 \\ 
 \end{bmatrix}
$$

记作:


$$
\begin{bmatrix}
 1&0&0  \\
 -3&1&0 \\ 
 0&0&1 \\ 
 \end{bmatrix}
\begin{bmatrix}
 1&2&1  \\
 3&8&1 \\ 
 0&4&1 \\ 
 \end{bmatrix}=
 \begin{bmatrix}
 1&2&1  \\
 0&2&-2 \\ 
 0&4&1 \\ 
 \end{bmatrix}
 $$

其实就是用第二行减去三倍的第一行.

# 矩阵和矩阵乘法

