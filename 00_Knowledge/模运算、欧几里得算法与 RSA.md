---
aliases:
  - "Modular Arithmetic, Euclidean Algorithm, and RSA"
  - "Number Theory and RSA"
  - "模运算与RSA"
status: source-checked
---

# 模运算、欧几里得算法与 RSA
<!-- bilingual-en:start -->
*Modular Arithmetic, the Euclidean Algorithm, and RSA*
<!-- bilingual-en:end -->

> [!summary] 快速恢复
> **它解决什么：** 用整除、同余和模逆把无限整数压缩到有限剩余类，并理解 RSA 正确性所依赖的数论机制。
> **具体锚点：** 扩展 Euclidean algorithm 在 $\gcd(e,\varphi(n))=1$ 时求出私钥指数 $d$，使 $ed\equiv1\pmod{\varphi(n)}$。
> **核心难点：** 模运算中的“除法”只有在逆元存在时才合法；RSA 的教学正确性证明也不等于现实实现已经安全。
> **为什么重要：** 这些结构支撑公钥密码学、哈希中的模运算和大量离散算法。
> **继续：** 先掌握 GCD、Bézout 与模逆，再理解 RSA 的 key generation 和 modular exponentiation；图结构另见 [[图的基本结构、路径与遍历|图、树、匹配与着色]]。
> <!-- bilingual-en:start -->
> **Problem addressed:** Use divisibility, congruence, and modular inverses to reduce infinitely many integers to finite residue classes and understand the number-theoretic mechanism behind RSA correctness.
> **Concrete anchor:** When $\gcd(e,\varphi(n))=1$, the extended Euclidean algorithm finds the private exponent $d$ satisfying $ed\equiv1\pmod{\varphi(n)}$.
> **Central difficulty:** “Division” modulo $n$ is valid only when an inverse exists, and a classroom proof of RSA correctness does not make a real implementation secure.
> **Why it matters:** These structures support public-key cryptography, modular arithmetic in hashing, and many discrete algorithms.
> **Continue with:** Master GCDs, Bézout's identity, and modular inverses before RSA key generation and modular exponentiation; see [[图的基本结构、路径与遍历|graph fundamentals, paths, and traversal]] for graph structure.
> <!-- bilingual-en:end -->

> [!source] 本节依据
> - [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf]] 与本地 MIT 6.042J OCW 材料：支持证明、离散结构、计数和概率。
> <!-- bilingual-en:start -->
> - [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf|MIT Mathematics for Computer Science]] and local MIT 6.042J OCW materials support proof, discrete structures, counting, and probability.
> <!-- bilingual-en:end -->

## GCD、Euclidean algorithm 与 Bézout
<!-- bilingual-en:start -->
*GCD, the Euclidean Algorithm, and Bézout's Identity*
<!-- bilingual-en:end -->

$\gcd(a,b)$ 可用 Euclidean algorithm 反复取余得到。扩展算法给整数 x,y 使 $ax+by=gcd(a,b)$。因此 a 在模 n 下有乘法逆元当且仅当 $\gcd(a,n)=1$。
<!-- bilingual-en:start -->
$\gcd(a,b)$ can be computed by repeated remainders in the Euclidean algorithm. The extended algorithm finds integers x and y such that $ax+by=\gcd(a,b)$. Consequently, a has a multiplicative inverse modulo n exactly when $\gcd(a,n)=1$.
<!-- bilingual-en:end -->

## 同余与模运算
<!-- bilingual-en:start -->
*Congruence and Modular Arithmetic*
<!-- bilingual-en:end -->

$a\equiv b\pmod n$ 表示 n 整除 a-b。加法、乘法和幂保持同余；约去公共因子需要该因子在模 n 下可逆。中国剩余定理在模数两两互素时把多个余数系统对应到模乘积的唯一类。
<!-- bilingual-en:start -->
$a\equiv b\pmod n$ means that n divides a-b. Addition, multiplication, and exponentiation preserve congruence; cancelling a common factor requires that factor to be invertible modulo n. With pairwise coprime moduli, the Chinese remainder theorem identifies a system of residues with one class modulo their product.
<!-- bilingual-en:end -->

## Euler 定理与 RSA
<!-- bilingual-en:start -->
*Euler's Theorem and RSA*
<!-- bilingual-en:end -->

若 $\gcd(a,n)=1$，$a^{\phi(n)}\equiv1\pmod n$。RSA 选 $ed\equiv1\pmod{\phi(n)}$（实际实现常用 $\lambda(n)$），使加密幂 e 与解密幂 d 组合恢复消息；完整正确性还需处理与 n 不互素的消息。现实安全依赖填充和实现，裸 RSA 不安全。
<!-- bilingual-en:start -->
If $\gcd(a,n)=1$, then $a^{\phi(n)}\equiv1\pmod n$. RSA chooses $ed\equiv1\pmod{\phi(n)}$—implementations often use $\lambda(n)$—so encryption exponent e and decryption exponent d compose to recover a message. A complete proof also handles messages not coprime to n. Real security depends on padding and implementation; textbook RSA is insecure.
<!-- bilingual-en:end -->

## Worked example：求模逆元
<!-- bilingual-en:start -->
*Worked Example: Find a Modular Inverse*
<!-- bilingual-en:end -->

求 $7^{-1}\pmod{26}$。扩展 Euclidean algorithm 给 $1=3\cdot26-11\cdot7$，因此 $-11\equiv15\pmod{26}$，且 $7\cdot15=105\equiv1\pmod{26}$。
<!-- bilingual-en:start -->
Find $7^{-1}\pmod{26}$. The extended Euclidean algorithm gives $1=3\cdot26-11\cdot7$, so $-11\equiv15\pmod{26}$ and $7\cdot15=105\equiv1\pmod{26}$.
<!-- bilingual-en:end -->

若 gcd 不是 1，就不存在逆元；此时不能把同余式两边的该因子直接约掉。
<!-- bilingual-en:start -->
If the gcd is not one, no inverse exists, and that factor cannot simply be cancelled from both sides of a congruence.
<!-- bilingual-en:end -->

## 失败诊断
<!-- bilingual-en:start -->
*Failure Diagnosis*
<!-- bilingual-en:end -->

- 模方程约分后解丢失：先检查被约因子与模数是否互素。
  <!-- bilingual-en:start -->
  Solutions disappear after cancellation: first check whether the cancelled factor is coprime to the modulus.
  <!-- bilingual-en:end -->
- RSA 数学示例能解密却方案不安全：区分代数正确性与 padding、随机性、密钥长度和侧信道安全。
  <!-- bilingual-en:start -->
  An RSA example decrypts but the scheme is insecure: separate algebraic correctness from padding, randomness, key length, and side-channel resistance.
  <!-- bilingual-en:end -->
- 大整数直接算幂溢出：使用 modular exponentiation，每步取模，而不是先算完整幂。
  <!-- bilingual-en:start -->
  Direct exponentiation overflows: use modular exponentiation and reduce at each step rather than constructing the full power.
  <!-- bilingual-en:end -->

## 最小自检
<!-- bilingual-en:start -->
*Minimum Self-Check*
<!-- bilingual-en:end -->

### 模运算中什么时候可以把等式两边的同一因子约掉？
<!-- bilingual-en:start -->
*When can the same factor be cancelled from both sides of a congruence?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 该因子与模数互素、因而存在模逆元时；否则约分可能丢失解或产生错误。
> <!-- bilingual-en:start -->
> When the factor is coprime to the modulus and therefore has a modular inverse; otherwise cancellation can lose solutions or produce an invalid conclusion.
> <!-- bilingual-en:end -->

### RSA 的数学正确性与现实安全为何不是一回事？
<!-- bilingual-en:start -->
*Why are RSA's mathematical correctness and real-world security different claims?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 模幂可恢复消息只证明功能；安全还依赖困难假设、随机填充、密钥长度和抗侧信道实现。
> <!-- bilingual-en:start -->
> Recovering a message by modular exponentiation proves functionality only; security also depends on hardness assumptions, randomized padding, key length, and side-channel-resistant implementation.
> <!-- bilingual-en:end -->

### Bézout 系数如何给出模逆元？
<!-- bilingual-en:start -->
*How do Bézout coefficients produce a modular inverse?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 若 $ax+ny=1$，对 n 取模得到 $ax\equiv1\pmod n$，所以 x 是 a 的逆元类。
> <!-- bilingual-en:start -->
> If $ax+ny=1$, reducing modulo n gives $ax\equiv1\pmod n$, so x represents the inverse of a.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and Verification*
<!-- bilingual-en:end -->

- [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf]] 与本地 MIT 6.042J OCW 材料：支持证明、离散结构、计数和概率。
  <!-- bilingual-en:start -->
  [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf|MIT Mathematics for Computer Science]] and local MIT 6.042J OCW materials support number-theoretic proofs and discrete structures.
  <!-- bilingual-en:end -->
