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
> **它解决什么：** 数论研究整数的整除和同余；图论研究对象间连接。二者都把连续细节删去，只保留离散结构。
> **具体锚点：** RSA 用模幂和 Euler/Fermat 性质恢复消息；网络中的最短路、树和匹配则只依赖节点与边。
> **核心难点：** 同余除法需乘法逆元存在；图中的 walk、path、cycle 和连通性不是同一对象。
> **为什么重要：** 密码学、算法、网络、组合优化和机制匹配都使用这些基本结构。
> **继续：** 证明结论时回到 [[数学证明方法]]；随机图或随机游走连接 [[离散概率]]。
> <!-- bilingual-en:start -->
> **Problem addressed:** Number theory studies divisibility and congruence, while graph theory studies connections among objects; both retain discrete structure while discarding continuous detail.
> **Concrete anchor:** RSA recovers a message through modular exponentiation and Euler or Fermat properties, whereas shortest paths, trees, and matchings depend only on vertices and edges.
> **Central difficulty:** Division in modular arithmetic requires an inverse, while graph walks, paths, cycles, and connectivity are distinct notions.
> **Why it matters:** Cryptography, algorithms, networks, combinatorial optimization, and matching mechanisms use these structures.
> **Continue with:** This card follows the integer/RSA chain; graph structure is now in [[图、树、匹配与着色|Graphs, Trees, Matching, and Coloring]]. Return to [[数学证明方法|Proof Methods]] for proofs.
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
<!-- bilingual-en:start -->
> [!answer]- Answer
> When the factor is coprime to the modulus and therefore has a modular inverse; otherwise cancellation can lose solutions or produce an invalid conclusion.
<!-- bilingual-en:end -->

### RSA 的数学正确性与现实安全为何不是一回事？
<!-- bilingual-en:start -->
*Why are RSA's mathematical correctness and real-world security different claims?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 模幂可恢复消息只证明功能；安全还依赖困难假设、随机填充、密钥长度和抗侧信道实现。
<!-- bilingual-en:start -->
> [!answer]- Answer
> Recovering a message by modular exponentiation proves functionality only; security also depends on hardness assumptions, randomized padding, key length, and side-channel-resistant implementation.
<!-- bilingual-en:end -->

### Bézout 系数如何给出模逆元？
<!-- bilingual-en:start -->
*How do Bézout coefficients produce a modular inverse?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 若 $ax+ny=1$，对 n 取模得到 $ax\equiv1\pmod n$，所以 x 是 a 的逆元类。
<!-- bilingual-en:start -->
> [!answer]- Answer
> If $ax+ny=1$, reducing modulo n gives $ax\equiv1\pmod n$, so x represents the inverse of a.
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and Verification*
<!-- bilingual-en:end -->

- [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf]] 与本地 MIT 6.042J OCW 材料：支持证明、离散结构、计数和概率。
  <!-- bilingual-en:start -->
  [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf|MIT Mathematics for Computer Science]] and local MIT 6.042J OCW materials support number-theoretic proofs and discrete structures.
  <!-- bilingual-en:end -->
