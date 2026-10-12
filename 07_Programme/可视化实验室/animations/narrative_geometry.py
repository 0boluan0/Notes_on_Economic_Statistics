"""Geometric lessons: transformations act on identifiable points and directions."""
from narrative import *


def tangent(s):
    s.chapter('斜率从两个点开始','A secant measures change between two points.','固定 x = 1，第二个点先放在 x + 1。','Fix x = 1 and begin with the second point one unit away.')
    a,p=axes([-.4,2.15],[-1,8.2],labels=('x','f(x) = x³ − x'));s.add(a);s.play(Create(curve([[x,x**3-x] for x in np.linspace(-.4,2.15,181)],p,BLUE)),run_time=1.5)
    fixed=Dot(p(1,0),radius=.08,color=GREEN);s.show(fixed)
    h=ValueTracker(1)
    moving=always_redraw(lambda:Dot(p(1+h.get_value(),(1+h.get_value())**3-(1+h.get_value())),radius=.08,color=GOLD))
    secant=always_redraw(lambda:Line(p(1-1/(2+3*h.get_value()+h.get_value()**2),-1),p(2.1,1.1*(2+3*h.get_value()+h.get_value()**2)),color=GOLD,stroke_width=3))
    s.add(moving,secant);s.hold('two-points',4)
    s.chapter('横向间隔变小，平均变化率怎样变？','Shrink the interval while keeping the first point fixed.','点的位置与割线一起变化；这是趋近过程，不是直接把 h 设为 0。','The point and secant move together. A limit is not division by zero.')
    for v in [.5,.2,.05]:
        slope=2+3*v+v*v;badge=card('h = '+num(v),'割线斜率 '+num(slope,3),GOLD,3).move_to(pos(-2.8,1))
        s.play(h.animate.set_value(v),FadeIn(badge),run_time=1.7);s.hold('h-'+str(v),3);s.play(FadeOut(badge),run_time=.3)
    moving.clear_updaters();secant.clear_updaters();s.show(Line(p(.5,-1),p(2.1,2.2),color=GREEN,stroke_width=3))
    s.hold('limiting-direction',4)
    s.ending('导数是局部线性近似的系数','The derivative is the coefficient of a local linear approximation.','f(1+h) approx f(1)+2h','本例 f′(1) = 2；先预测小幅输入变化对应的输出变化。','Here f′(1) = 2. Predict the output change for a small input perturbation.')


def linear(s):
    A=np.array(s.d['matrix']);assert A.shape==(2,2)
    s.chapter('先问两个基向量去了哪里','Two basis vectors determine the linear map.','第一列告诉 e₁ 的去向，第二列告诉 e₂ 的去向。','Column one is the image of e₁; column two is the image of e₂.')
    corners=np.array([[0,0],[1,0],[1,1],[0,1]])@A.T;lo=np.minimum(corners.min(axis=0),[0,0])-.4;hi=np.maximum(corners.max(axis=0),[1,1])+.4
    matrix=grid(A,.75,(-4.5,.5));s.add(matrix,txt('矩阵 A',24,MUTED).move_to(pos(-4.5,1.7)))
    a,p=axes([lo[0],hi[0]],[lo[1],hi[1]],center=(1.3,-.1),size=(6.5,3.6),equal=True,labels=('x₁','x₂'));e1=arrow(p(0,0),p(1,0),BLUE);e2=arrow(p(0,0),p(0,1),GOLD);s.add(a,e1,e2);s.hold('basis-before',3)
    images=[arrow(p(0,0),p(*A[:,i]),[BLUE,GOLD][i]) for i in range(2)];s.play(Transform(e1,images[0]),run_time=1.5);s.hold('first-column',3);s.play(Transform(e2,images[1]),run_time=1.5);s.hold('second-column',3)
    s.chapter('同样的线性组合，也决定其他点的去向','The same coefficients combine the transformed basis.','取 x = e₁ + e₂；输出就是两列向量相加。','For x = e₁ + e₂, add the two column vectors.')
    second=arrow(p(*A[:,0]),p(*(A[:,0]+A[:,1])),GOLD);result=arrow(p(0,0),p(*(A@np.ones(2))),GREEN)
    s.play(TransformFromCopy(e2,second),run_time=1.2);s.play(GrowArrow(result),run_time=1.2);s.hold('sum-of-images',4)
    s.chapter('一个单位方格，变成一个平行四边形','The unit square becomes a parallelogram.','四个角遵守同一张映射；面积倍率由行列式决定。','Every corner obeys the same map; the determinant controls signed area.')
    square=Polygon(*(p(*v) for v in [[0,0],[1,0],[1,1],[0,1]]),color=GREEN,fill_opacity=.15)
    mapped=Polygon(*(p(*(A@v)) for v in np.array([[0,0],[1,0],[1,1],[0,1]])),color=GREEN,fill_opacity=.2)
    s.show(square);s.play(Transform(square,mapped),run_time=2);s.hold('area-transformed',4)
    s.ending('矩阵不是一张孤立的数字表','Columns specify actions on directions.','A x=x_1 A e_1+x_2 A e_2','亲手让两列共线：观察平面如何塌成直线。','Make the columns collinear and observe the plane collapse onto a line.')


def projection(s):
    d=s.d;u=np.array(d['u']);v=np.array(d['v']);foot=np.array(d['projection']);r=np.array(d['residual'])
    assert abs(u@r)<1e-9 and np.allclose(foot+r,v)
    s.chapter('在一条直线上，找离向量最近的点','Find the closest point on a one-dimensional subspace.','蓝色向量固定；绿色直线是允许选择的方向。','The vector is fixed; the green line is the allowed subspace.')
    a,p=axes([-3.5,3.5],[-3,3],size=(9,4.3),equal=True,labels=('x₁','x₂'));extent=min(3.5/max(abs(u[0]),1e-12),3/max(abs(u[1]),1e-12));line=Line(p(*(-extent*u)),p(*(extent*u)),color=GREEN);vector=arrow(p(0,0),p(*v),BLUE);s.add(a,line,vector);s.hold('vector-and-subspace',4)
    coefficient=ValueTracker(-1.5);moving=always_redraw(lambda:Dot(p(*(coefficient.get_value()*u)),radius=.08,color=GOLD));residual=always_redraw(lambda:Line(moving.get_center(),p(*v),color=GOLD,stroke_width=3));s.add(moving,residual)
    s.play(coefficient.animate.set_value(float(d['dot'])),run_time=3);s.hold('closest-point',4);moving.clear_updaters();residual.clear_updaters()
    s.chapter('最短残差，正好垂直于这条线','At the minimum, the residual is orthogonal.','继续沿直线挪动，会增加距离；垂足就是投影。','Moving away along the line increases the distance.')
    proj=arrow(p(0,0),p(*foot),GREEN);s.play(GrowArrow(proj),run_time=1);right=VGroup(Line(p(*(foot+.15*u)),p(*(foot+.15*u+.15*r/np.linalg.norm(r))),color=INK),Line(p(*(foot+.15*r/np.linalg.norm(r))),p(*(foot+.15*u+.15*r/np.linalg.norm(r))),color=INK));s.show(right);s.hold('perpendicular-residual',4)
    s.wipe();s.chapter('把原向量分成两部分','The vector is projection plus residual.','一部分沿子空间，一部分垂直子空间。','One component lies in the subspace; the other is orthogonal to it.')
    parts=VGroup(card('['+', '.join(num(x) for x in foot)+']','投影 p',GREEN,3).move_to(pos(-3.4,.5)),txt('+',40).move_to(pos(0,.5)),card('['+', '.join(num(x) for x in r)+']','残差 r',GOLD,3).move_to(pos(3.4,.5)));s.show(parts);s.hold('decomposition',4)
    s.ending('垂直性，让平方长度可以相加','Orthogonality gives the Pythagorean identity.','norm(v)^2=norm(p)^2+norm(r)^2','这里 u 是单位向量，p = (uᵀv)u；非单位方向需要除以 uᵀu。','Here u is a unit vector. A non-unit direction requires division by uᵀu.')


def gradient(s):
    d=s.d;path=d['path'];k=s.p['condition'];rate=s.p['rate']
    s.chapter('梯度给方向，步长给距离','The gradient gives a direction; the step size gives distance.','固定二次函数，从同一个初始点开始下降。','Start from a fixed point on a quadratic objective.')
    a,p=axes([-4.2,4.2],[-3.2,3.2],size=(9,4.4),equal=True,labels=('x₁','x₂'));s.add(a)
    for level in [1,3,6,8]:s.add(curve([[np.sqrt(2*level)*np.cos(t),np.sqrt(2*level/k)*np.sin(t)] for t in np.linspace(0,TAU,161)],p,RULE,1.2))
    q=path[0];dot=Dot(p(q['x'],q['y']),radius=.09,color=BLUE);s.add(dot);s.hold('initial-point',3)
    next_=path[1];step=arrow(p(q['x'],q['y']),p(next_['x'],next_['y']),GOLD);s.play(GrowArrow(step),run_time=1.3);s.hold('negative-gradient-step',3);s.play(dot.animate.move_to(step.get_end()),run_time=1.2);s.hold('first-update',3)
    s.chapter('再算梯度，再迈下一步','Recompute the gradient at every new point.','每一步都从当前位置重新出发，不是沿第一支箭头一直走。','Each new gradient is evaluated at the current point.')
    for prev,new in zip(path[1:-1],path[2:]):
        line=Line(p(prev['x'],prev['y']),p(new['x'],new['y']),color=BLUE,stroke_width=2.5);s.play(Create(line),dot.animate.move_to(p(new['x'],new['y'])),run_time=.65);s.wait(.25)
    s.hold('descent-path',4)
    s.wipe();s.chapter('陡峭方向，对步长更敏感','The steep direction sets the stability limit.','每步分别把两个坐标乘以 1−η 和 1−ηκ。','Each coordinate is multiplied by its own contraction factor.')
    f1=1-rate;f2=1-rate*k;s.show(card(num(f1,3),'平缓方向的因子',BLUE,3.4).move_to(pos(-2.5,.7)),card(num(f2,3),'陡峭方向的因子',GOLD,3.4).move_to(pos(2.5,.7)));s.hold('two-factors',4)
    s.ending('步子更大，不一定更快','A larger step can oscillate or diverge.','0<eta<2/kappa','严格稳定区间之外，至少一个方向的误差不再收缩。','Outside the strict stability interval, at least one error component fails to contract.')


def pca(s):
    d=s.d;X=np.array(d['processed']);u=np.array(d['eigenvectors'][0]);print('pca u',u,flush=True)
    # The exported model stores eigenvectors as column directions in an outer array.
    if not np.allclose(np.array(d['covariance'])@u,d['eigenvalues'][0]*u):u=np.array(d['eigenvectors'])[:,0]
    assert np.allclose(np.array(d['covariance'])@u,d['eigenvalues'][0]*u)
    bound=max(2,np.max(abs(X))+.4)
    s.chapter('把点云压到一条线上，会丢掉什么？','Project the centered cloud onto one direction.','先看居中后的同一批观测；每个点都要落到直线上。','Use the same centered observations and track every projected point.')
    a,p=axes([-bound,bound],[-bound,bound],size=(5.1,4.2),equal=True,labels=('居中特征 1','居中特征 2'));dots=VGroup(*(Dot(p(*x),radius=.052,color=BLUE) for x in X));s.add(a,dots);s.hold('centered-cloud',4)
    line=Line(p(*(-bound*u)),p(*(bound*u)),color=GREEN);s.play(Create(line),run_time=1);projected=X@u[:,None]*u;new=VGroup(*(Dot(p(*x),radius=.052,color=GREEN) for x in projected));segments=VGroup(*(Line(p(*x),p(*y),color=RULE,stroke_width=1) for x,y in zip(X,projected)))
    s.play(Create(segments),TransformFromCopy(dots,new),run_time=2.5);s.hold('each-point-projects',4)
    s.chapter('在线上的位置，只需要一个坐标','A score is a coordinate along that direction.','投影方向固定后，每个二维点对应一个带正负的一维分数。','Once the direction is fixed, each 2D point has one signed score.')
    i=int(np.argmax(np.linalg.norm(X,axis=1)));score=float(X[i]@u);focus=SurroundingRectangle(VGroup(dots[i],new[i]),color=GOLD,buff=.07);s.show(focus,card(num(score,3),'该点的主成分分数',GOLD,3.4).move_to(pos(4,.7)));s.hold('score',4)
    s.chapter('保留方差，也是在减少重建误差','Retained variance and reconstruction error are complementary.','主方向选择投影后方差最大的单位方向。','The first principal direction maximizes projected variance over unit directions.')
    s.play(Indicate(new,color=GREEN),run_time=1.5);s.hold('retained-variation',4)
    s.ending('尺度选择，会改变你认为重要的方向','Feature scaling changes the PCA problem.','z_i=u^T x_i quad hat(x)_i=u z_i','亲手比较原尺度与标准化尺度；两者回答的是不同的方差问题。','Compare original and standardized features: they define different variance objectives.')


def embeddings(s):
    d=s.d;a0=np.array(d['a']);b0=np.array(d['b']);norm=max(2,np.linalg.norm(a0),np.linalg.norm(b0))+.5
    s.chapter('相似，是指方向还是长度？','Alignment and magnitude are different properties.','先把两个向量从同一原点画出，分别看长度和夹角。','Draw both vectors from the same origin and inspect angle and length separately.')
    ax,p=axes([-norm,norm],[-norm,norm],size=(5.5,4.2),equal=True,labels=('第 1 维','第 2 维'));va=arrow(p(0,0),p(*a0),BLUE);vb=arrow(p(0,0),p(*b0),GOLD);s.add(ax,va,vb);s.hold('two-vectors',4)
    s.chapter('先除以各自长度，只保留方向','Normalize both vectors to isolate direction.','向量落到单位圆上；原来的长度信息被除掉了。','Normalization puts both vectors on the unit circle and removes their magnitudes.')
    ua=a0/np.linalg.norm(a0);ub=b0/np.linalg.norm(b0);circle=curve([[np.cos(t),np.sin(t)] for t in np.linspace(0,TAU,181)],p,RULE);s.show(circle);s.play(Transform(va,arrow(p(0,0),p(*ua),BLUE)),Transform(vb,arrow(p(0,0),p(*ub),GOLD)),run_time=2);s.hold('unit-vectors',4)
    s.show(card(num(float(ua@ub),3),'余弦相似度',GREEN,3).move_to(pos(4,.7)));s.hold('cosine',3)
    s.wipe();s.chapter('把长度放回来，才得到内积','The dot product restores both magnitudes.','方向完全相同的两个向量，内积仍会随长度改变。','Identically aligned vectors can have different dot products.')
    s.equation('a^T b=norm(a) norm(b) cos(theta)',y=.8);s.hold('dot-product-factors',5)
    s.ending('零向量没有方向','Cosine is undefined for a zero vector.','norm(a)=0 => "cosine undefined"','亲手把一个向量长度设为 0：内积是 0，余弦相似度未定义。','Set one norm to zero: the dot product is zero, but cosine similarity is undefined.')


def autoencoder(s):
    d=s.d;X=np.array(d['X']);u=np.array(d['u']);codes=d['codes'];recon=np.array(d['reconstruction']);b=max(3,np.max(abs(X))+.3)
    s.chapter('二维点，要经过一个一维瓶颈','A two-dimensional point passes through a one-dimensional code.','追踪同一个点：先编码，再从这个编码重建。','Track one point through encoding and reconstruction.')
    a,p=axes([-b,b],[-b,b],size=(5.4,4.1),equal=True,labels=('特征 1','特征 2'));dots=VGroup(*(Dot(p(*x),radius=.035,color=BLUE) for x in X));s.add(a,dots);s.hold('data-cloud',3)
    i=int(np.argmax(np.linalg.norm(X,axis=1)));point=Dot(p(*X[i]),radius=.085,color=GOLD);s.show(point);code=card(num(codes[i],3),'一个标量编码 z',GOLD,3.2).move_to(pos(4,.8));s.play(TransformFromCopy(point,code),run_time=1.5);s.hold('encoded-point',4)
    s.chapter('解码器只能沿一条方向画回来','The decoder reconstructs on a line.','一维编码只有一个自由度；离开这条线的分量无法恢复。','A scalar code controls one degree of freedom; orthogonal information is lost.')
    line=Line(p(*(-b*u)),p(*(b*u)),color=GREEN);rpoint=Dot(p(*recon[i]),radius=.085,color=GREEN);s.play(Create(line),TransformFromCopy(code,rpoint),run_time=1.7);s.show(DashedLine(point.get_center(),rpoint.get_center(),color=GOLD));s.hold('reconstruction-residual',4)
    s.chapter('对所有点重复，才得到重建误差','Repeat over the dataset to measure reconstruction error.','每个点都投到同一条线上；误差衡量失去的部分。','All points share the same bottleneck direction.')
    reconstructed=VGroup(*(Dot(p(*x),radius=.035,color=GREEN) for x in recon));s.play(TransformFromCopy(dots,reconstructed),run_time=2.4);s.show(card(num(d['mse'],3),'平均平方重建误差',GREEN,3.2).move_to(pos(4,-1.2)));s.hold('all-reconstructions',4)
    s.ending('在线性约束下，瓶颈与 PCA 相遇','A tied linear unit-gain projection connects to PCA.','z=u^T x quad hat(x)=u z','居中数据、单位方向与单位增益下，最优方向是第一主成分。','For centered data with a unit direction and unit gain, the optimum is the first principal direction.')


def svd(s):
    d=s.d;A=np.array(d['A']);U=np.array(d['U']);V=np.array(d['V']);sigma=np.array(d['singularValues']);print('svd',A,U,V,sigma,flush=True)
    assert np.allclose(U@np.diag(sigma)@V.T,A)
    angles=np.linspace(0,TAU,80,endpoint=False);points=np.array([[np.cos(t),np.sin(t)] for t in angles]);colors=[interpolate_color(ManimColor(BLUE),ManimColor(GOLD),i/len(points)) for i in range(len(points))]
    bound=max(1.5,max(sigma)*1.25)
    s.chapter('一个矩阵，可以拆成三次动作','A matrix acts as rotate, stretch, rotate.','给单位圆上的点编号，用颜色追踪同一个点。','Track the same unit-circle points by color through all three operations.')
    a,p=axes([-bound,bound],[-bound,bound],size=(5,4.2),equal=True,labels=('第 1 维','第 2 维'));dots=VGroup(*(Dot(p(*x),radius=.045,color=c) for x,c in zip(points,colors)));s.add(a,dots);s.hold('circle',4)
    basis=VGroup(arrow(p(0,0),p(1,0),BLUE),arrow(p(0,0),p(0,1),GOLD));s.show(basis)
    previous=np.eye(2)
    stages=[('先用 Vᵀ 改变输入方向','Rotate the input directions with Vᵀ.',V.T,'圆形不变；两根半径随同一个旋转一起移动。'),('再按两个奇异值分别伸缩','Stretch the aligned axes by the singular values.',np.diag(sigma)@V.T,'两个垂直方向各有一个伸缩倍率。'),('最后用 U 放到输出方向','Finally rotate the result with U.',A,'椭圆的轴方向与长度，分别来自奇异向量和奇异值。')]
    for k,(title,subtitle,transform,caption) in enumerate(stages):
        s.chapter(title,subtitle,caption,subtitle)
        if k in [0,2]:
            rotation=V.T if k==0 else U;assert np.allclose(rotation.T@rotation,np.eye(2)) and np.linalg.det(rotation)>0
            angle=np.arctan2(rotation[1,0],rotation[0,0]);start=previous.copy()
            def action(t):
                c,z=np.cos(t*angle),np.sin(t*angle);return np.array([[c,-z],[z,c]])@start
        else:
            start=previous.copy()
            def action(t):return np.diag((1-t)+t*sigma)@start
        def move(group,t):
            matrix=action(t)
            for item,x in zip(dots,points):item.move_to(p(*(matrix@x)))
            basis.become(VGroup(arrow(p(0,0),p(*(matrix@np.array([1.,0.]))),BLUE),arrow(p(0,0),p(*(matrix@np.array([0.,1.]))),GOLD)))
        s.play(UpdateFromAlphaFunc(dots,move),run_time=3);previous=transform;s.hold(title,4)
    s.chapter('秩一近似，把第二个方向压到零','Rank-one truncation removes the second singular direction.','第一根主轴保持不变；第二根主轴逐渐消失，椭圆变成线段。','The first principal axis stays fixed while the second collapses to zero.')
    rank_one=np.array(d['approximation']);assert s.p['retained']==1 and np.allclose(rank_one,U@np.diag([sigma[0],0])@V.T)
    ghost=dots.copy().set_opacity(.2);s.add(ghost)
    principal=VGroup(arrow(p(0,0),p(*(sigma[0]*U[:,0])),BLUE),arrow(p(0,0),p(*(sigma[1]*U[:,1])),GOLD));s.play(ReplacementTransform(basis,principal),run_time=.7)
    def truncate(group,t):
        matrix=U@np.diag([sigma[0],(1-t)*sigma[1]])@V.T
        for item,x in zip(dots,points):item.move_to(p(*(matrix@x)))
        principal[1].become(arrow(p(0,0),p(*((1-t)*sigma[1]*U[:,1])),GOLD))
    s.play(UpdateFromAlphaFunc(dots,truncate),run_time=3);s.hold('ellipse-to-rank-one-line',5)
    s.ending('截断误差，对应被舍弃的方向','Truncation error comes from the discarded direction.','A_1=sigma_1 u_1 v_1^T','本例的谱范数与 Frobenius 误差都等于 σ₂ = '+num(sigma[1])+'。','For this 2 × 2 rank-one truncation, both spectral and Frobenius errors equal σ₂.')


def jacobian(s):
    d=s.d;c=np.array(d['center']);mapped=np.array(d['mapped']);J=np.array(d['J']);print('jacobian center',c,mapped,flush=True)
    s.chapter('非线性地图，在一点附近像什么？','A nonlinear map has a local linear action.','先固定中心点；Jacobian 描述的是这个中心附近的小位移。','Fix a center. The Jacobian acts on small displacements around it.')
    left,pl=axes([-1.2,1.2],[-1.2,1.2],center=(-3,-.2),size=(3.6,3.6),equal=True,labels=('输入位移 δ₁','δ₂'))
    right,pr=axes([-2.5,2.5],[-2.5,2.5],center=(3,-.2),size=(3.6,3.6),equal=True,labels=('输出位移',''));s.add(left,right,Dot(pl(0,0),color=GOLD),Dot(pr(0,0),color=GOLD));s.hold('fixed-centers',3)
    basis=[np.array([.8,0]),np.array([0,.8])]
    for i,v in enumerate(basis):
        a=arrow(pl(0,0),pl(*v),PALETTE[i]);b=arrow(pr(0,0),pr(*(J@v)),PALETTE[i]);s.play(GrowArrow(a),run_time=1);s.play(TransformFromCopy(a,b),run_time=1.5);s.hold('column-'+str(i),3)
    s.chapter('小网格的真实弯曲，与线性预测对照','Compare the actual curved grid with its tangent grid.','虚线表示一阶预测；真实映射还包含高阶余项。','The linear prediction omits higher-order terms.')
    s.wipe();actual=np.array(d['actualLines']);linear=np.array(d['linearLines']);flat=actual.reshape(-1,2);mins=flat.min(axis=0);maxs=flat.max(axis=0);a,p=axes([mins[0]-.1,maxs[0]+.1],[mins[1]-.1,maxs[1]+.1],size=(6.4,4),equal=True,labels=('输出 1','输出 2'));s.show(a)
    actualg=VGroup(*(curve(line,p,BLUE,2) for line in actual));linearg=VGroup(*(DashedVMobject(curve(line,p,GOLD,1.5),num_dashes=25) for line in linear));s.play(Create(linearg),run_time=2);s.play(Create(actualg),run_time=2);s.hold('curved-versus-linear',5)
    s.ending('缩小邻域，才能检验局部近似','Shrink the neighborhood to test a local approximation.','F(a+delta)=F(a)+J(a)delta+O(norm(delta)^2)','本例光滑映射的余项是二阶；不要把局部近似当成全局等式。','For this smooth example the remainder is second order, not globally zero.')


def flow(s):
    p=s.p;d=s.d;z=p['z'];y=d['y'];J=d['jacobian']
    assert J>0
    s.chapter('先搬运位置，再问密度怎样变','First move positions, then account for density.','单调映射让每个输出只有一个原像。','A monotone transformation gives every output a unique preimage.')
    upper,pu=axes([min(d['zs']),max(d['zs'])],[0,1],center=(0,1),size=(9,.6),labels=('原位置 z',''));lower,pv=axes([min(d['ys']),max(d['ys'])],[0,1],center=(0,-1.1),size=(9,.6),labels=('映射后 y',''));s.add(upper,lower)
    indices=np.linspace(0,len(d['zs'])-1,9,dtype=int);dots=VGroup(*(Dot(pu(d['zs'][i],.5),radius=.065,color=BLUE) for i in indices));s.show(dots);s.hold('original-positions',3)
    mapped=VGroup(*(Dot(pv(d['ys'][i],.5),radius=.065,color=GREEN) for i in indices));s.play(TransformFromCopy(dots,mapped),run_time=2.5);s.hold('mapped-positions',4)
    s.chapter('同一小段概率，铺到更长的区间','Stretching a small interval spreads its probability mass.','概率质量守恒：区间拉长，单位长度上的密度就降低。','Mass is conserved: stretching lowers density per unit length.')
    s.wipe();source=Rectangle(width=2,height=1.7,color=BLUE,fill_opacity=.35).move_to(pos(-3,0));target=Rectangle(width=2*J,height=1.7/J,color=GREEN,fill_opacity=.35).move_to(pos(3,0));s.show(source,txt('局部宽度 × '+num(J,3),25,GOLD).move_to(pos(0,1.8)));s.hold('small-mass-block',3);s.play(TransformFromCopy(source,target),run_time=2.3);s.hold('equal-area-block',4)
    s.show(txt('高度 ÷ '+num(J,3),26,GREEN).move_to(pos(3,-1.65)));s.hold('density-adjustment',4)
    s.ending('Jacobian 补偿的是局部体积变化','The Jacobian corrects local volume change.','p_Y(g(z))=p_Z(z)/abs(g prime(z))','这两个矩形是局部近似的面积示意；真实密度由同一模型计算。','The rectangles illustrate local mass conservation; the model computes the actual density.')


STORIES={'tangent':tangent,'linear':linear,'projection':projection,'gradient':gradient,'pca':pca,'embeddings':embeddings,'autoencoder':autoencoder,'svd':svd,'jacobian':jacobian,'flow':flow}
