---
aliases:
  - 从整除与求逆到RSA的连续学习路径
  - Number theory and RSA learning path
---

# 数论与 RSA：从整数相除到消息还原
<!-- bilingual-en:start -->
*Number theory and RSA: from integer division to message recovery*
<!-- bilingual-en:end -->

这条路径对应 MIT 6.042J Sessions 12–15。先解决“怎样求 gcd 与逆元”，再把整数改用余数表示，最后解释两次模幂为什么能还原消息。下面可以连续读；要单独复习一个定义或证明，点击相应原子。完整课堂题、在线反馈与 PS5 保留在 [[02_Structures#Session 12 — GCDs|Unit 2 课程记录]]。
<!-- bilingual-en:start -->
This path follows MIT 6.042J Sessions 12–15. It first computes gcds and inverses, then represents integers by residues, and finally explains recovery through two modular powers. Read continuously below, or open an atom to review one definition or proof. Full class problems, online feedback, and PS5 remain in the [[02_Structures#Session 12 — GCDs|Unit 2 course record]].
<!-- bilingual-en:end -->

[[数论与RSA.canvas|打开关系总图]] · [[00_课程总览#2. 数论、RSA 与图结构|在课程总览中展开全部原子]] · [[知识原子.base|检索共享原子]]

## 1. 整除与余数：先确定问题中的整数关系
<!-- bilingual-en:start -->
*1. Divisibility and remainders: identify the integer relation*
<!-- bilingual-en:end -->

[[整除]] $a\mid b$ 说的是 $b=ak$ 对某个整数 $k$ 成立。比如 $6\mid18$，但 $6\nmid20$。不能整除时，[[带余除法]]仍给出唯一分解 $20=3\cdot6+2$：商为 $3$，余数为 $2$。一般写成 $a=qn+r$，要求 $n>0$、$0\le r<n$；负输入也沿用这套约定，所以 $-11=(-2)\cdot7+3$。
<!-- bilingual-en:start -->
[[整除|Divisibility]] $a\mid b$ means $b=ak$ for an integer $k$. Thus $6\mid18$ but $6\nmid20$. [[带余除法|Division with remainder]] still gives the unique decomposition $20=3\cdot6+2$. In general, $a=qn+r$ with $n>0$ and $0\le r<n$; the same convention gives $-11=(-2)\cdot7+3$ for a negative input.
<!-- bilingual-en:end -->

若同时研究两个不全为零的整数，就会问它们共有多大的因子。[[最大公因数]]是共有的最大正因子；例如 $\gcd(30,22)=2$。“互素”只表示 gcd 为 $1$，不表示每个输入都必须是素数。接下来的算法只用除法和减法，完全不需要先把输入分解成素数。
<!-- bilingual-en:start -->
For two integers not both zero, the [[最大公因数|gcd]] is their greatest positive common divisor; for example, $\gcd(30,22)=2$. Coprimality means gcd one, not that each input must be prime. The algorithm that follows needs only division and subtraction, not prior prime factorization.
<!-- bilingual-en:end -->

## 2. 从 gcd 算法走到逆元的系数
<!-- bilingual-en:start -->
*2. From the gcd algorithm to coefficients for an inverse*
<!-- bilingual-en:end -->

[[欧几里得算法]]先取输入的绝对值；在 $b>0$ 时反复做 $\gcd(a,b)=\gcd(b,a\bmod b)$，$b=0$ 就返回 $a$。为什么可以换？写 $a=qb+r$ 后，共同整除 $a,b$ 的数也整除 $r=a-qb$；共同整除 $b,r$ 的数也整除 $a=qb+r$。公因子集合不变，而余数严格小于除数，所以既保留答案，又保证最终结束。
<!-- bilingual-en:start -->
The [[欧几里得算法|Euclidean algorithm]] first takes absolute values, repeatedly uses $\gcd(a,b)=\gcd(b,a\bmod b)$ while $b>0$, and returns $a$ when $b=0$. With $a=qb+r$, every common divisor of $a,b$ divides $r=a-qb$, and every common divisor of $b,r$ divides $a=qb+r$. The common divisors are preserved, while remainders strictly decrease: the target is unchanged and the process terminates.
<!-- bilingual-en:end -->

例如 $30=22+8$、$22=2\cdot8+6$、$8=6+2$、$6=3\cdot2$，最后一个非零余数是 $2$。若追问“这个 $2$ 怎样由原来的 $30,22$ 组合而成”，反代得到 $2=8-6=3\cdot8-22=3\cdot30-4\cdot22$。[[Bézout等式]]保证这样的整数系数存在；[[扩展欧几里得算法]]就是把系数连同余数一起算出来。
<!-- bilingual-en:start -->
For $30=22+8$, $22=2\cdot8+6$, $8=6+2$, and $6=3\cdot2$, the last nonzero remainder is two. Back-substitution gives $2=8-6=3\cdot8-22=3\cdot30-4\cdot22$. [[Bézout等式|Bézout's identity]] guarantees integer coefficients; the [[扩展欧几里得算法|extended Euclidean algorithm]] computes them alongside the remainders.
<!-- bilingual-en:end -->

先记住另一组能用到最后的结果：$\gcd(26,7)=1$，且 $1=3\cdot26-11\cdot7$。当我们稍后只关心“除以 $26$ 的余数”，第一项消失，留下 $-11\cdot7\equiv1\pmod{26}$。因此把 $7$ 乘回 $1$ 的系数可以取 $-11$，也可以取与之同余的 $15$。这就是 gcd 与求逆之间的连接。
<!-- bilingual-en:start -->
Keep one further result: $\gcd(26,7)=1$ and $1=3\cdot26-11\cdot7$. When only remainders modulo $26$ matter, the first term disappears, leaving $-11\cdot7\equiv1\pmod{26}$. The coefficient that multiplies seven back to one is represented by either $-11$ or $15$. This connects gcd computation to inversion.
<!-- bilingual-en:end -->

## 3. 素数分解提供另一种观察整数的方式
<!-- bilingual-en:start -->
*3. Prime factorization gives another view of integers*
<!-- bilingual-en:end -->

[[素数]]是大于 $1$、正因子只有 $1$ 和自身的整数。关键不是“它分不开”的口号，而是[[素数整除乘积]]：若素数 $p\mid ab$ 且 $p\nmid a$，那么 $\gcd(p,a)=1$；把 Bézout 等式 $xp+ya=1$ 乘以 $b$，就能推出 $p\mid b$。这里先用 gcd 证明引理，不能偷用随后才要证明的唯一分解。
<!-- bilingual-en:start -->
A [[素数|prime]] is greater than one and has only one and itself as positive divisors. The crucial step is [[素数整除乘积|Euclid's lemma]]: if prime $p\mid ab$ but $p\nmid a$, then $\gcd(p,a)=1$. Multiplying the Bézout identity $xp+ya=1$ by $b$ proves $p\mid b$. This proves the lemma using gcds rather than assuming the unique factorization it will establish.
<!-- bilingual-en:end -->

[[唯一素因数分解]]的存在性把合数递归拆成更小因子；唯一性则在两组素数乘积中，用上述引理匹配一个相同素因子、约去，再继续。因此 $60$ 的素因子多重集合只能是 $2,2,3,5$。这项保证并不等于“对任意大整数都能快速找出这些因子”。
<!-- bilingual-en:start -->
[[唯一素因数分解|Unique prime factorization]] obtains existence by splitting composites into smaller factors. For uniqueness, the lemma matches and cancels one common prime at a time between two proposed factorizations. Thus sixty has precisely the multiset $2,2,3,5$. A unique answer does not imply an efficient factorization algorithm for arbitrary large inputs.
<!-- bilingual-en:end -->

[[最小公倍数]]是能被两个正整数都整除的最小正整数。从素因子看，gcd 取每个指数的较小值，lcm 取较大值，两者相加恰好还原原指数之和，所以有[[公因公倍乘积]] $\gcd(a,b)\operatorname{lcm}(a,b)=ab$。例如 $12=2^2\cdot3$、$18=2\cdot3^2$，得到 gcd 为 $6$、lcm 为 $36$。这也是 CP12 题目应使用的规则。
<!-- bilingual-en:start -->
The [[最小公倍数|lcm]] is the smallest positive integer divisible by both inputs. Prime by prime, the gcd takes the smaller exponent and the lcm the larger. Their sum reconstructs the original exponent sum, proving the [[公因公倍乘积|gcd–lcm product identity]]. For $12=2^2\cdot3$ and $18=2\cdot3^2$, the gcd is six and the lcm thirty-six. This is the rule used in CP12.
<!-- bilingual-en:end -->

## 4. 同余运算：加乘能保留，除法要检查
<!-- bilingual-en:start -->
*4. Congruence preserves addition and multiplication; division needs checking*
<!-- bilingual-en:end -->

以下同余与求逆取整数模数 $n>1$。[[同余]] $a\equiv b\pmod n$ 表示 $n\mid(a-b)$，也就是两个数除以 $n$ 余数相同。[[模运算]]把同余的整数当作同一个余数类来算；加、减、乘后换成标准余数不会改变结果。例如 $7\cdot15=105\equiv1\pmod{26}$，所以 $15$ 是 $7$ 的[[模逆元]]。
<!-- bilingual-en:start -->
Congruences and inverses below use an integer modulus $n>1$. [[同余|Congruence]] $a\equiv b\pmod n$ means $n\mid(a-b)$, equivalently equal remainders. [[模运算|Modular arithmetic]] calculates with residue classes: reducing after addition, subtraction, or multiplication preserves the result. Since $7\cdot15=105\equiv1\pmod{26}$, fifteen is a [[模逆元|modular inverse]] of seven.
<!-- bilingual-en:end -->

[[模逆元存在条件]]是 $\gcd(a,n)=1$。充分性由 $sa+tn=1$ 给出；必要性也直接：若 $sa\equiv1$，则 $sa-kn=1$，任何公因子都必须整除 $1$。因此模 $6$ 下 $2$ 不可逆，即使它不是 $0$。普通等式中“非零因子可约去”的规则，不能原样搬到同余式中。
<!-- bilingual-en:start -->
The [[模逆元存在条件|inverse-existence criterion]] is $\gcd(a,n)=1$. Bézout gives sufficiency. Conversely, $sa\equiv1$ gives $sa-kn=1$, forcing any common divisor to divide one. Thus two is nonzero but noninvertible modulo six; nonzero alone does not justify division.
<!-- bilingual-en:end -->

[[同余消去律]]说明约去公因子时模数可能必须变小。例如 $2x\equiv2\pmod6$ 表示 $6\mid2(x-1)$，等价于 $3\mid(x-1)$，所以答案是 $x\equiv1\pmod3$。换回模 $6$，有 $x\equiv1,4$ 两个类；若误写成 $x\equiv1\pmod6$，就丢掉了一半解。
<!-- bilingual-en:start -->
The [[同余消去律|cancellation rule]] may require shrinking the modulus. For $2x\equiv2\pmod6$, the condition $6\mid2(x-1)$ is equivalent to $3\mid(x-1)$, hence $x\equiv1\pmod3$. Modulo six this gives two classes, one and four. Cancelling while keeping modulus six loses half the solutions.
<!-- bilingual-en:end -->

## 5. 解一个同余方程，再合并多个余数条件
<!-- bilingual-en:start -->
*5. Solve one congruence, then combine residue conditions*
<!-- bilingual-en:end -->

对[[线性同余方程]] $ax\equiv b\pmod n$，[[线性同余求解|求解时先算]] $g=\gcd(a,n)$。若 $g\nmid b$ 则无解；若 $g\mid b$ 且 $g<n$，除去 $g$ 后系数与新模数互素，就能求逆。若 $g=n$，则按 $n$ 是否整除 $b$ 判断全体整数都是解还是无解。例如 $6x\equiv8\pmod{14}$ 中 $g=2$，化成 $3x\equiv4\pmod7$。$3$ 的逆元是 $5$，所以 $x\equiv20\equiv6\pmod7$；原模数下为 $x\equiv6,13\pmod{14}$。
<!-- bilingual-en:start -->
For the [[线性同余方程|linear congruence]] $ax\equiv b\pmod n$, [[线性同余求解|start the solution]] by computing $g=\gcd(a,n)$. There is no solution unless $g\mid b$. If $g\mid b$ and $g<n$, divide by $g$ and invert the resulting coprime coefficient. If $g=n$, every integer is a solution when $n\mid b$, and none is otherwise. Thus $6x\equiv8\pmod{14}$ reduces to $3x\equiv4\pmod7$. The inverse of three is five, giving $x\equiv6\pmod7$, or classes six and thirteen modulo fourteen.
<!-- bilingual-en:end -->

如果要求 $x\equiv2\pmod3$、$x\equiv3\pmod5$ 同时成立，可以先写 $x=2+3k$，代入第二式得到 $3k\equiv1\pmod5$，于是 $k\equiv2\pmod5$、$x\equiv8\pmod{15}$。[[中国剩余定理]]保证：两两互素的模数下，任意指定的余数条件都恰好确定一个模乘积的类。它的构造用模逆元；它的唯一性说的是“同一个类”，不是只有一个整数解。
<!-- bilingual-en:start -->
To satisfy $x\equiv2\pmod3$ and $x\equiv3\pmod5$, write $x=2+3k$. The second condition becomes $3k\equiv1\pmod5$, hence $k\equiv2\pmod5$ and $x\equiv8\pmod{15}$. The [[中国剩余定理|Chinese remainder theorem]] guarantees one class modulo the product for every residue tuple when the moduli are pairwise coprime. Its construction uses inverses; uniqueness concerns a residue class, not a single integer.
<!-- bilingual-en:end -->

这次合并不仅是解题技巧：最后证明 RSA 时，我们会分别证明模 $p$ 和模 $q$ 的消息相同，再靠它合并成模 $pq$ 的相同。
<!-- bilingual-en:start -->
This combination step will also finish the RSA proof: establish recovery separately modulo $p$ and $q$, then combine them into recovery modulo $pq$.
<!-- bilingual-en:end -->

## 6. 欧拉定理：可逆余数如何约束指数
<!-- bilingual-en:start -->
*6. Euler's theorem: invertible residues constrain exponents*
<!-- bilingual-en:end -->

[[欧拉函数]] $\varphi(n)$ 数的是模 $n$ 下可逆余数有多少个。例如模 $12$ 下只有 $1,5,7,11$ 与 $12$ 互素，所以 $\varphi(12)=4$。[[素数幂欧拉函数]]通过剔除 $p$ 的倍数给出 $\varphi(p^k)=p^k-p^{k-1}$；[[欧拉函数互素乘法性]]用 CRT 将模 $ab$ 的可逆类与两侧可逆类配对；再结合唯一分解，得到[[欧拉函数乘积公式]]。因此 $\varphi(12)=12(1-1/2)(1-1/3)=4$。
<!-- bilingual-en:start -->
The [[欧拉函数|totient]] $\varphi(n)$ counts invertible residues. Modulo twelve they are $1,5,7,11$, giving four. The [[素数幂欧拉函数|prime-power formula]] removes multiples of $p$; [[欧拉函数互素乘法性|coprime multiplicativity]] uses CRT to pair invertible classes; unique factorization then yields the [[欧拉函数乘积公式|prime-factor product formula]]. Thus $\varphi(12)=12(1-1/2)(1-1/3)=4$.
<!-- bilingual-en:end -->

现在取 $\gcd(a,n)=1$。把全部可逆余数 $r_1,\ldots,r_{\varphi(n)}$ 都乘以 $a$，只会重新排列它们：可逆性没有丢失，且 $ar_i\equiv ar_j$ 能约去 $a$ 得 $r_i\equiv r_j$。把这组同余相乘得
<!-- bilingual-en:start -->
Now assume $\gcd(a,n)=1$. Multiplying every invertible residue $r_1,\ldots,r_{\varphi(n)}$ by $a$ merely permutes them: they remain invertible, and $ar_i\equiv ar_j$ permits cancellation of $a$. Multiplying the congruences gives
<!-- bilingual-en:end -->

$$a^{\varphi(n)}\prod_i r_i\equiv\prod_i r_i\pmod n.$$

每个 $r_i$ 本来就是可逆余数，因此它们的乘积也可逆，可以约去，得到[[欧拉定理]] $a^{\varphi(n)}\equiv1\pmod n$。这里有两个不同的依据：$\gcd(a,n)=1$ 保证乘 $a$ 是置换；选择全部可逆余数保证乘积可以约去。模 $12$ 下不能对 $a=6$ 使用这个结论，事实上 $6^4\equiv0$。
<!-- bilingual-en:start -->
Each $r_i$ was chosen to be invertible, so their product is invertible. Cancelling it proves [[欧拉定理|Euler's theorem]], $a^{\varphi(n)}\equiv1\pmod n$. The two justifications differ: $\gcd(a,n)=1$ makes multiplication by $a$ a permutation; choosing invertible residues permits cancelling their product. The theorem does not apply to six modulo twelve: $6^4\equiv0$.
<!-- bilingual-en:end -->

当模数为素数 $p$，$\varphi(p)=p-1$，便得到[[费马小定理]]。对 $p\nmid a$，有 $a^{p-1}\equiv1\pmod p$；乘以 $a$，并补上 $p\mid a$ 时两边都是零的情形，得到适用于所有整数 $a$ 的 $a^p\equiv a\pmod p$。这两种形式的条件不能混用。
<!-- bilingual-en:start -->
For prime modulus $p$, $\varphi(p)=p-1$ yields [[费马小定理|Fermat's little theorem]]. The form $a^{p-1}\equiv1\pmod p$ requires $p\nmid a$. Multiplying by $a$ and separately including the zero-residue case gives $a^p\equiv a\pmod p$ for every integer. The hypotheses of these forms are different.
<!-- bilingual-en:end -->

## 7. RSA：先会计算，再补全所有消息的证明
<!-- bilingual-en:start -->
*7. RSA: compute first, then prove recovery for every message*
<!-- bilingual-en:end -->

[[模重复平方法]]按指数的二进制位算模幂，每次乘法立即取余。它不需要底数与模数互素；这点比欧拉定理适用得更广。[[RSA公钥加密]]的数学构造取互异奇素数 $p=5,q=11$，得到 $n=55$、$\varphi(n)=40$，选 $e=3$，求逆得 $d=27$，因为 $ed=81=1+2\cdot40$。公开 $(55,3)$，保密 $d$ 与素因子。
<!-- bilingual-en:start -->
[[模重复平方法|Repeated squaring]] computes modular powers from binary exponent bits and reduces after every multiplication. It does not require a coprime base. For the teaching construction of [[RSA公钥加密|RSA]], take distinct odd primes $5,11$, giving $n=55$ and $\varphi(n)=40$. Choose $e=3$ and its inverse $d=27$, since $ed=81=1+2\cdot40$. Publish $(55,3)$ and keep the private exponent and factors secret.
<!-- bilingual-en:end -->

试一条与模数不互素的消息 $m=5$：加密 $c=5^3\bmod55=15$。解密时有 $15^2\equiv5$、$15^4\equiv25$、$15^8\equiv20$、$15^{16}\equiv15\pmod{55}$；$27=16+8+2+1$，所以 $15^{27}\equiv15\cdot20\cdot5\cdot15\equiv5\pmod{55}$。这里 gcd 不为 $1$，但计算仍然还原。
<!-- bilingual-en:start -->
Try a noncoprime representative, $m=5$: encryption gives $c=5^3\bmod55=15$. Squaring yields residues $5,25,20,15$ for exponents $2,4,8,16$. Since $27=16+8+2+1$, decryption gives $15^{27}\equiv15\cdot20\cdot5\cdot15\equiv5\pmod{55}$. Recovery works even though the message and modulus are not coprime.
<!-- bilingual-en:end -->

回到一般的[[RSA全消息正确性]]：设 $p\ne q$ 为素数，$e,d$ 为正整数且 $ed\equiv1\pmod{(p-1)(q-1)}$，消息代表取 $0\le m<pq$。不能直接把所有 $m$ 代进欧拉定理。分别固定 $r=p$ 或 $q$：若 $r\mid m$，正指数使 $m^{ed}\equiv0\equiv m\pmod r$；否则由费马小定理及 $r-1\mid(ed-1)$，写 $ed=1+t_r(r-1)$，得到 $m^{ed}=m(m^{r-1})^{t_r}\equiv m\pmod r$。两种分类覆盖每个素数下的所有情况，再由 CRT 合并，便得 $m^{ed}\equiv m\pmod{pq}$。
<!-- bilingual-en:start -->
For general [[RSA全消息正确性|all-message correctness]], let $p\ne q$ be primes, let $e,d$ be positive integers with $ed\equiv1\pmod{(p-1)(q-1)}$, and take $0\le m<pq$. Euler's theorem cannot be applied to every $m$. Fix $r=p$ or $q$. If $r\mid m$, a positive exponent gives zero on both sides. Otherwise, Fermat's theorem and $r-1\mid(ed-1)$ give $ed=1+t_r(r-1)$ and $m^{ed}=m(m^{r-1})^{t_r}\equiv m\pmod r$. These cases cover every message for each prime, and CRT combines them modulo $pq$.
<!-- bilingual-en:end -->

证明真正用到的是两个条件 $p-1\mid ed-1$ 与 $q-1\mid ed-1$。因此只要求 $ed\equiv1\pmod{\operatorname{lcm}(p-1,q-1)}$ 就够；按 $\varphi(n)$ 求逆是满足它的一种教学构造。这里“互异素数”不可省略：若把 $n$ 换成 $9$，虽然 $\varphi(9)=6$、$5\cdot5\equiv1\pmod6$，消息 $3$ 经 $3^5\bmod9=0$ 后就无法还原。
<!-- bilingual-en:start -->
The proof needs both $p-1$ and $q-1$ to divide $ed-1$, so inversion modulo their lcm suffices; inversion modulo the totient is one way to achieve it. Distinct primes matter: for modulus nine, $\varphi(9)=6$ and $5\cdot5\equiv1\pmod6$, yet message three encrypts to zero and cannot be recovered.
<!-- bilingual-en:end -->

## 8. 正确还原不等于安全加密
<!-- bilingual-en:start -->
*8. Correct recovery is not secure encryption*
<!-- bilingual-en:end -->

[[RSA安全边界]]区分两个问题：有私钥能否还原，以及攻击者能否在没有私钥时获知消息。分解 $n$ 足以算出可用的 $d$，但这不证明所有攻击都与分解等价。裸公式 $m^e\bmod n$ 又是确定性的：攻击者可以公开加密候选消息来比对；乘法结构也仍然保留。因此实际方案还需要编码与相应的安全论证。RFC 8017 中 OAEP 用于加密，PSS 用于签名，两者不是同一种“填充选项”。
<!-- bilingual-en:start -->
[[RSA安全边界|RSA's security boundary]] separates recovery with the private key from what an attacker can learn without it. Factoring $n$ suffices to derive a usable $d$, but this does not establish equivalence with every attack. Raw RSA is deterministic and multiplicative, permitting public comparison with candidate messages. Concrete schemes need encoding and their own security arguments: RFC 8017 uses OAEP for encryption and PSS for signatures, not as interchangeable padding choices.
<!-- bilingual-en:end -->

## 三道自检
<!-- bilingual-en:start -->
*Three checks*
<!-- bilingual-en:end -->

> [!question]- $2x\equiv2\pmod6$ 为什么不能只答 $x\equiv1\pmod6$？
> 因为 $2$ 在模 $6$ 下不可逆。正确消去后模数缩成 $3$，所以模 $6$ 有 $1,4$ 两个解类。
> <!-- bilingual-en:start -->
> Why is $x\equiv1\pmod6$ incomplete for $2x\equiv2\pmod6$? Two is not invertible modulo six. Cancellation reduces the modulus to three, giving classes one and four modulo six.
> <!-- bilingual-en:end -->

> [!question]- 为什么 gcd 算得快不能推出素因数分解也算得快？
> Euclid 只保持公因子集合并缩小余数，并不返回输入的全部素因子；这是两个不同的计算问题。唯一分解定理也只保证答案的存在与唯一性。
> <!-- bilingual-en:start -->
> Why does efficient gcd computation not imply efficient factorization? Euclid preserves common divisors and shrinks remainders; it does not return all prime factors. These are different problems, and unique factorization establishes only existence and uniqueness.
> <!-- bilingual-en:end -->

> [!question]- RSA 消息能被 $p$ 整除时，全消息证明在哪一步处理它？
> 在模 $p$ 的零余数分支直接得到两边为零；模 $q$ 仍分别检查零与非零，再由 CRT 合并。既没有漏掉非互素消息，也没有对它偷用欧拉定理。
> <!-- bilingual-en:start -->
> Where does the RSA proof handle a message divisible by $p$? The zero-residue branch modulo $p$ makes both sides zero. Modulo $q$, check its own zero and nonzero cases, then combine by CRT without misapplying Euler's theorem.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [[MIT6_042JS15_textbook.pdf#page=252|MIT Mathematics for Computer Science §8.1–8.4，PDF pp.252–268]]：整除、余数、gcd、Bézout、素数分解；[[MIT6_042JS15_cp12.pdf#page=1|CP12 Problem 2]] 支持 gcd 与 lcm 计算规则。
- [[MIT6_042JS15_textbook.pdf#page=272|MIT §8.6–8.10，PDF pp.272–287]]：同余、模运算、逆元、欧拉函数与欧拉定理；CRT、线性同余及欧拉函数计算的展开证明和具体来源见对应原子。
- [[MIT6_042JS15_cp15.pdf#page=2|MIT CP15 Problem 3，PDF p.2]] 与 [[MIT6_042JS15_textbook.pdf#page=288|§8.11–8.12，PDF pp.288–291]]：全消息正确性、互异素数条件、分解与 SAT 的关系。
- [IETF RFC 8017 §§3、5、7.1、8.1](https://www.rfc-editor.org/rfc/rfc8017.html)：标准密钥条件与 RSA 原语、OAEP 加密、PSS 签名。本文的小整数、余数类及反例均独立复算；每个共享原子另列精确支持页。
<!-- bilingual-en:start -->
- MIT §§8.1–8.4 support divisibility, gcds, Bézout, and factorization; CP12 supports gcd–lcm computations. Sections 8.6–8.10 support congruences, inverses, and Euler's theorem; the linked atoms provide the expanded CRT and totient derivations. CP15 and §§8.11–8.12 support all-message RSA and its factorization boundary. RFC 8017 supplies standard key conditions and the distinct roles of the primitive, OAEP, and PSS. Numerical examples and counterexamples were recomputed independently.
<!-- bilingual-en:end -->
