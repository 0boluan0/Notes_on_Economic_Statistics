# 从一个例子讲起：Bart与Lisa
* Bart有一个工厂，并向一条没有主人的河流里排放污水
* Lisa靠在这条河里捕鱼维生。

>[!note] 定义(Definition)
>
> * [[Externality|Externality]] : An activity of one entity that affects the welfare of another entity in a way that is outside the market mechanism。
> * 某一实体的活动以没有反映在市场价格中的某种方式直接影响他人福利，这种影响被称为外部性([[Externality|externality]])。
> eg：Bart向河中排放了污水，他本应支付排污的成本但是他事实上并未支付，而是产生了负的外部性。
>
# 对于负外部性的处理

负外部性会导致自由市场产量水平大于社会效率水平
The Nature of [[Externality|Externalities]]-Graphical Analysis $ Reduction from Q, to Q" means dcg Eee profit loss for Supplier and dchg welfare gain for Demander. 5 边际 私人 成 本 h d 8 MD 边际 损害 “ a_ie ° Q* Q payee TE Ta 人 Actual output oer year SAN 285, PE Ber 0044 buy MeCram till Edueatien All Biahte Bacanrad 5.
![[Pasted image 20240315133213.png]]
如果Bart减产则Bart少∆dcg这些收入但是Lisa得到红色部分，所以社会总产出会增加。

正外部性会导致自由市场产量水平小于社会效率水平

## 私人对策

### 讨价换件和科斯定理

#### 讨价还价
当产权被严格划分时，可以通过讨价还价达到新的均衡
eg：将河划分给Bart则Lisa需要给Bart钱Bart才乐意减产以平衡收入的降低
若需求方愿意转移支付，供给方可以把污染（或产量）从 $Q_1$ 调整到社会更优的 $Q^*$。只要 $MD>(MB-MPC)$，就存在讨价还价空间。

```mermaid
flowchart LR
    A["初始：按私人成本决策（Q1）"] --> B["需求方补偿供给方"]
    B --> C["供给方减排/减产"]
    C --> D["达到社会更优水平（Q*）"]
```

。 MD >(MB-— MPC) => the opportunity for a bargain exists
满足上述不等式的时候就还有讲价的区间，否则就没必要讲价。
事实上付的钱是一个范围而非一个定值。

#### [[Coase Theorem|科斯定理]]

* 前提条件：讨价还价成本低，产权可严格划分。
* 科斯定律起作用的必要假设：各方讨价还价成本很低，资源所有者能够识别使其财产受到损害的源头,且在防止伤害上能够得到法律保护。
### 其他解决办法

#### 合并（外部性内部化）
俩人结婚就行
#### 社会习俗，道德
己所不欲，勿施于人

## 公共对策
* MSB: the marginal social benefit to Lisa of each unit of pollution Bart reduces.
* MC: the marginal cost to Bart of reducing each unit of pollution.
* Cost for reducing pollution can stem from reducing output,shifting to cleaner inputs, or installing a new technology to control pollution.
### 征收庇古税
庇古税通过提高污染行为的私人成本，使决策从 $Q_1$ 向社会最优 $Q^*$ [[Limit|收敛]]，并形成税收收入。

```mermaid
flowchart LR
    A["征税前：按MPC决策（Q1）"] --> B["征税后：面对 MSC=MPC+MD"]
    B --> C["产量/排污降至 Q*"]
    C --> D["形成庇古税收入"]
```

MSC边际社会成本MPC边际个人成本MD边际污染


### 发放庇古补贴
对减排行为发放庇古补贴，可以把个体激励调整到社会最优附近。

```mermaid
flowchart LR
    A["无补贴：减排不足（Q1）"] --> B["发放庇古补贴"]
    B --> C["减排提高，向Q*收敛"]
```

### 排污费
对于Bart而言，最好的情况是减少任何一点污染的排放，因为减少排污就意味着花钱，他肯定不想花钱。
当排污费为 $f_t$ 时，只会带来较低减排量 $e_t$；当排污费提高到 $f^*$ 时，可达到有效减排水平 $e^*$。

## 不同污染者之间统一污染减少
假设有第三个人Homer也开了一家会排污的公司
若强制每家都减排 50 单位，在边际减排成本不同的情况下并不成本有效。更有效率的做法是让低边际成本企业减排更多。

当排污费为 $50$ 时，Bart 减排约 75，Homer 减排约 25；Homer 的税负更高，从而同时改进效率与公平。

```mermaid
flowchart LR
    A["统一减排（各50）"] --> B["成本无效率"]
    C["征收统一排污费 f=$50"] --> D["低MAC企业多减排（Bart≈75）"]
    C --> E["高MAC企业少减排（Homer≈25）"]
    E --> F["高排放者承担更高税负"]
```



### 总量控制与交易制度

发放一定量的污染许可证，政府进行初步的配给，而后经过市场进行调控直到均衡。

在总量控制与交易制度下：若企业边际减排成本低于许可证价格，则卖出许可证；若高于许可证价格，则买入许可证；交易使双方都受益。

```mermaid
flowchart LR
    A["总量控制：政府初始配给许可证"] --> B["形成许可证市场价格"]
    B --> C["MAC < 价格：卖出许可证"]
    B --> D["MAC > 价格：买入许可证"]
    C --> E["资源配置更有效率"]
    D --> E
```

#### 碳交易市场
## 命令控制管理
强制性的，不是很好用
### 技术要求
加州的车辆强制装尾气处理装置

# 对正外部性的处理
正外部性下，社会边际收益满足 $MSB=MPB+MEB$，因此社会最优研究投入 $R^*$ 通常高于私人自发投入 $R_a$。

与负的外部性相反，这个的边际社会受益是比边际私人受益要高的。

## 公共处理

对私人企业研发行为发放补贴
相当于向上平移了边际私人收益
