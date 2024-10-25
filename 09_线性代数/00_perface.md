
# 方程组的不同记法

## Row picture

就是普通的画图然后看方程形成的线/面/超平面交出来的点
$$
A=
\begin{bmatrix}
 2&-1&0  \\
 -1&2&-1 \\ 
 0&-3&4 \\ 
 \end{bmatrix},
 B=
 \begin{bmatrix}
 0  \\
 -1 \\ 
 4 \\ 
 \end{bmatrix}
$$
## column picture

使用每个未知量和他的所有系数的列向量相乘后相加.

$$
X
\begin{bmatrix}
 2  \\
 -1 \\ 
 0 \\ 
 \end{bmatrix}
+y
\begin{bmatrix}
 -1  \\
 2 \\ 
 -3 \\ 
 \end{bmatrix}
+z
\begin{bmatrix}
 0  \\
 -1 \\ 
 4 \\ 
 \end{bmatrix}
=
\begin{bmatrix}
 0  \\
 -1 \\ 
 4 \\ 
 \end{bmatrix}

$$

使用左侧的向量线性组合找到右侧的向量.最终令等式成立的系数就是XYZ的答案.

是否能够一定有解,就是在讨论是否三个向量是否能够完全覆盖所有的线性空间.

