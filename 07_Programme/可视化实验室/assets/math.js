/* Offline numerical models. All randomness is local, seeded, and reproducible. */
(function (root) {
  'use strict';
  const PI = Math.PI, SQRT2PI = Math.sqrt(2 * PI);
  const sum = a => a.reduce((s, x) => s + x, 0);
  const mean = a => a.length ? sum(a) / a.length : NaN;
  function variance(a, ddof = 1) {
    if (a.length <= ddof) return NaN;
    const m = mean(a); return sum(a.map(x => (x - m) ** 2)) / (a.length - ddof);
  }
  function covariance(a, b, ddof = 1) {
    if (a.length !== b.length || a.length <= ddof) return NaN;
    const ma = mean(a), mb = mean(b);
    return sum(a.map((x, i) => (x - ma) * (b[i] - mb))) / (a.length - ddof);
  }
  function quantile(a, p) {
    if (!a.length || p < 0 || p > 1) return NaN;
    const s = [...a].sort((x, y) => x - y), i = (s.length - 1) * p, j = Math.floor(i);
    return s[j] + (s[Math.min(j + 1, s.length - 1)] - s[j]) * (i - j);
  }
  function rng(seed) {
    let state = seed >>> 0;
    return function () {
      state = (state + 0x6D2B79F5) >>> 0;
      let t = state; t = Math.imul(t ^ t >>> 15, t | 1);
      t ^= t + Math.imul(t ^ t >>> 7, t | 61);
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }
  // Open-interval uniforms also exclude log(0) and exact Cauchy poles.
  const openUniform = random => (random() * 4294967296 + 0.5) / 4294967296;
  function normalGenerator(random) {
    let spare = null;
    return function () {
      if (spare !== null) { const x = spare; spare = null; return x; }
      const r = Math.sqrt(-2 * Math.log(openUniform(random))), t = 2 * PI * openUniform(random);
      spare = r * Math.sin(t); return r * Math.cos(t);
    };
  }
  function normalPDF(x) { return Math.exp(-x * x / 2) / SQRT2PI; }
  // erfc via regularized incomplete gamma (a=1/2): series near zero,
  // modified Lentz continued fraction in the tail. Compute the smaller tail directly.
  function normalCDF(x) {
    if (x === 0) return 0.5;
    if (x === Infinity) return 1;
    if (x === -Infinity) return 0;
    if(Number.isNaN(x))return NaN;
    const z=x*x/2;
    if(z>750)return x>0?1:0;
    const prefactor=Math.exp(-z+.5*Math.log(z)-.5*Math.log(PI));let tail;
    if(z<1.5){
      let term=2,total=term;
      for(let k=1;k<200;k++){term*=z/(k+.5);total+=term;if(Math.abs(term)<Math.abs(total)*2e-16)break;}
      tail=(1-prefactor*total)/2;
    }else{
      let b=z+.5,c=1e300,d=1/b,h=d;
      for(let k=1;k<300;k++){const an=-k*(k-.5);b+=2;d=an*d+b;if(Math.abs(d)<1e-300)d=1e-300;c=b+an/c;if(Math.abs(c)<1e-300)c=1e-300;d=1/d;const delta=d*c;h*=delta;if(Math.abs(delta-1)<4e-16)break;}
      tail=prefactor*h/2;
    }
    return x>=0?1-tail:tail;
  }
  // Peter J. Acklam's rational inverse; accuracy comfortably exceeds displayed precision.
  function normalQuantile(p) {
    if (p === 0) return -Infinity;
    if (p === 1) return Infinity;
    if (!(p > 0 && p < 1)) return NaN;
    const a = [-39.69683028665376,220.9460984245205,-275.9285104469687,138.357751867269,-30.66479806614716,2.506628277459239];
    const b = [-54.47609879822406,161.5858368580409,-155.6989798598866,66.80131188771972,-13.28068155288572];
    const c = [-.007784894002430293,-.3223964580411365,-2.400758277161838,-2.549732539343734,4.374664141464968,2.938163982698783];
    const d = [.007784695709041462,.3224671290700398,2.445134137142996,3.754408661907416];
    let q, r;
    if (p < .02425 || p > .97575) {
      q = Math.sqrt(-2 * Math.log(p < .02425 ? p : 1 - p));
      const v = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
      const initial=p < .02425 ? v : -v;
      const error=initial>0?-normalCDF(-initial)+(1-p):normalCDF(initial)-p;
      return initial-error/normalPDF(initial);
    }
    q = p - .5; r = q*q;
    const initial=(((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1);
    const error=initial>0?-normalCDF(-initial)+(1-p):normalCDF(initial)-p;
    return initial-error/normalPDF(initial);
  }
  function logGamma(z) {
    const c = [.99999999999980993,676.5203681218851,-1259.1392167224028,771.32342877765313,-176.61502916214059,12.507343278686905,-.13857109526572012,9.984369578019572e-6,1.5056327351493116e-7];
    if (z < .5) return Math.log(PI / Math.sin(PI*z)) - logGamma(1-z);
    z -= 1; let x = c[0]; for (let i = 1; i < c.length; i++) x += c[i] / (z+i);
    const t = z + 7.5; return .5*Math.log(2*PI)+(z+.5)*Math.log(t)-t+Math.log(x);
  }
  const logBeta = (a,b) => logGamma(a)+logGamma(b)-logGamma(a+b);
  function betaPDF(x, a, b) {
    if (x < 0 || x > 1) return 0;
    if (x === 0) return a === 1 ? b : a > 1 ? 0 : Infinity;
    if (x === 1) return b === 1 ? a : b > 1 ? 0 : Infinity;
    return Math.exp((a-1)*Math.log(x)+(b-1)*Math.log1p(-x)-logBeta(a,b));
  }
  function betaFraction(a,b,x) {
    const tiny = 1e-300; let c = 1, d = 1-(a+b)*x/(a+1);
    if (Math.abs(d)<tiny) d=tiny; d=1/d; let h=d;
    for (let m=1;m<=300;m++) {
      let aa=m*(b-m)*x/((a+2*m-1)*(a+2*m));
      d=1+aa*d; if(Math.abs(d)<tiny)d=tiny; c=1+aa/c;if(Math.abs(c)<tiny)c=tiny;d=1/d;h*=d*c;
      aa=-(a+m)*(a+b+m)*x/((a+2*m)*(a+2*m+1));
      d=1+aa*d;if(Math.abs(d)<tiny)d=tiny;c=1+aa/c;if(Math.abs(c)<tiny)c=tiny;d=1/d;
      const delta=d*c;h*=delta;if(Math.abs(delta-1)<3e-14)return h;
    }
    throw new Error('Beta 连分式未达到数值收敛要求。');
  }
  function betaCDF(x,a,b) {
    if (!(a>0 && b>0)) return NaN;
    if(x<=0)return 0;if(x>=1)return 1;
    const t=Math.exp(a*Math.log(x)+b*Math.log1p(-x)-logBeta(a,b));
    const v=x<(a+1)/(a+b+2)?t*betaFraction(a,b,x)/a:1-t*betaFraction(b,a,1-x)/b;
    return Math.max(0,Math.min(1,v));
  }
  function betaQuantile(p,a,b) {
    if (!(p>=0 && p<=1 && a>0 && b>0)) return NaN;
    if(p===0)return 0;if(p===1)return 1;
    let lo=0,hi=1;for(let i=0;i<60;i++){const mid=(lo+hi)/2;if(betaCDF(mid,a,b)<p)lo=mid;else hi=mid;}
    return (lo+hi)/2;
  }
  function linspace(a,b,n=201){return Array.from({length:n},(_,i)=>a+(b-a)*i/(n-1));}
  function histogram(values, bins=35, domain=null) {
    let lo=domain?domain[0]:Math.min(...values),hi=domain?domain[1]:Math.max(...values);
    if(!(hi>lo)){lo-=.5;hi+=.5;}
    const width=(hi-lo)/bins,counts=Array(bins).fill(0);let omitted=0;
    for(const v of values){if(v<lo||v>hi){omitted++;continue;}counts[Math.min(bins-1,Math.floor((v-lo)/width))]++;}
    return {points:counts.map((c,i)=>[lo+(i+.5)*width,c/(values.length*width),width]),counts,width,domain:[lo,hi],omitted};
  }
  function eigen2(matrix) {
    const a=matrix[0][0],b=(matrix[0][1]+matrix[1][0])/2,d=matrix[1][1],r=Math.hypot(a-d,2*b);
    const values=[(a+d+r)/2,(a+d-r)/2],theta=.5*Math.atan2(2*b,a-d);
    const vectors=[[Math.cos(theta),Math.sin(theta)],[-Math.sin(theta),Math.cos(theta)]];
    return {values,vectors,equal:r<=1e-12*Math.max(1,Math.abs(a)+Math.abs(d))};
  }
  // Column-pivoted Householder QR. It avoids forming X'X for polynomial fits.
  function qrDecompose(matrix, rhs=null) {
    const m=matrix.length,n=matrix[0].length,A=matrix.map(r=>r.slice()),y=rhs?rhs.slice():null;
    const permutation=Array.from({length:n},(_,i)=>i), reflectors=[];
    let scale=0;for(const row of A)for(const v of row)scale=Math.max(scale,Math.abs(v));
    const tolerance=Number.EPSILON*Math.max(m,n)*Math.max(1,scale)*16;
    let rank=0;
    for(let k=0;k<Math.min(m,n);k++) {
      let best=k,bestNorm=-1;
      for(let j=k;j<n;j++){let ss=0;for(let i=k;i<m;i++)ss+=A[i][j]**2;if(ss>bestNorm){bestNorm=ss;best=j;}}
      if(Math.sqrt(Math.max(0,bestNorm))<=tolerance)break;
      if(best!==k){for(let i=0;i<m;i++)[A[i][k],A[i][best]]=[A[i][best],A[i][k]];[permutation[k],permutation[best]]=[permutation[best],permutation[k]];}
      let norm=0;for(let i=k;i<m;i++)norm=Math.hypot(norm,A[i][k]);
      const alpha=A[k][k]>=0?-norm:norm,v=[];for(let i=k;i<m;i++)v.push(A[i][k]);v[0]-=alpha;
      const vv=sum(v.map(t=>t*t)),factor=2/vv;
      for(let j=k;j<n;j++){let dot=0;for(let i=k;i<m;i++)dot+=v[i-k]*A[i][j];dot*=factor;for(let i=k;i<m;i++)A[i][j]-=dot*v[i-k];}
      if(y){let dot=0;for(let i=k;i<m;i++)dot+=v[i-k]*y[i];dot*=factor;for(let i=k;i<m;i++)y[i]-=dot*v[i-k];}
      A[k][k]=alpha;for(let i=k+1;i<m;i++)A[i][k]=0;
      reflectors.push({k,v,factor});rank++;
    }
    return {R:A,y,permutation,reflectors,rank,tolerance};
  }
  function leastSquares(X,y,lambda=0) {
    const m=X.length,n=X[0].length;
    if(m!==y.length||X.some(r=>r.length!==n)||lambda<0)throw new Error('最小二乘输入维度或惩罚无效。');
    let A=X.map(r=>r.slice()),b=y.slice();
    if(lambda>0)for(let j=1;j<n;j++){const row=Array(n).fill(0);row[j]=Math.sqrt(lambda);A.push(row);b.push(0);}
    if(A.length<n){
      // X' P=Q R => R' z=P' y; beta=Q [z,0], the unique minimum-norm solution.
      const trans=Array.from({length:n},(_,j)=>A.map(r=>r[j])),qr=qrDecompose(trans),r=A.length;
      if(qr.rank<r)return {coefficients:null,rank:qr.rank,unique:false,method:'rank-deficient'};
      const z=Array(n).fill(0);
      for(let i=0;i<r;i++){let s=b[qr.permutation[i]];for(let j=0;j<i;j++)s-=qr.R[j][i]*z[j];z[i]=s/qr.R[i][i];}
      for(let k=qr.reflectors.length-1;k>=0;k--){const h=qr.reflectors[k];let dot=0;for(let i=h.k;i<n;i++)dot+=h.v[i-h.k]*z[i];dot*=h.factor;for(let i=h.k;i<n;i++)z[i]-=dot*h.v[i-h.k];}
      return {coefficients:z,rank:r,unique:false,method:'minimum-norm QR'};
    }
    const qr=qrDecompose(A,b);
    if(qr.rank<n)return {coefficients:null,rank:qr.rank,unique:false,method:'rank-deficient'};
    const permuted=Array(n).fill(0),coefficients=Array(n).fill(0);
    for(let i=n-1;i>=0;i--){let s=qr.y[i];for(let j=i+1;j<n;j++)s-=qr.R[i][j]*permuted[j];permuted[i]=s/qr.R[i][i];}
    for(let i=0;i<n;i++)coefficients[qr.permutation[i]]=permuted[i];
    return {coefficients,rank:n,unique:true,method:'pivoted Householder QR'};
  }
  function legendre(x,degree){const p=[1,x];for(let j=2;j<=degree;j++)p[j]=((2*j-1)*x*p[j-1]-(j-1)*p[j-2])/j;return p.slice(0,degree+1);}
  const metric=(label,value,digits=4,suffix='')=>({label,value,digits,suffix});
  const line=(label,points,color=0,dashed=false)=>({type:'line',label,points,color,dashed});
  const scatter=(label,points,color=0)=>({type:'scatter',label,points,color});
  const segments=(label,points,color=0,dashed=false)=>({type:'segments',label,points,color,dashed});
  const plot=(title,xLabel,yLabel,series,extra={})=>({title,xLabel,yLabel,series,...extra});
  const result=(metrics,plots,data,notes=[])=>({metrics,plots,data,notes});
  const schemas={
    tangent:{x:[1,-2.5,2.5],h:[.5,.001,1],function:['cubic',['cubic','sine','exponential']]},
    linear:{a:[1,-2,2],b:[.7,-2,2],c:[0,-2,2],d:[1,-2,2]},
    projection:{angle:[60,0,175],vx:[2,-3,3],vy:[1,-3,3]},
    gradient:{rate:[.2,.01,1.2],condition:[4,1,20,true],steps:[20,1,60,true],x0:[3,-4,4],y0:[2,-4,4]},
    sampling:{n:[30,1,200,true],reps:[500,100,2000,true],distribution:['exponential',['normal','exponential','cauchy']],seed:[42,1,99999,true]},
    confidence:{n:[25,2,200,true],level:[.95,[.8,.9,.95,.99]],reps:[100,30,300,true],sigma:[1,.25,3],seed:[42,1,99999,true]},
    testing:{n:[30,2,200,true],effect:[.5,-1.5,1.5],alpha:[.05,[.01,.05,.1]],observed:[.3,-1.5,1.5]},
    bayes:{a:[2,1,15,true],b:[2,1,15,true],success:[7,0,50,true],failure:[3,0,50,true]},
    regression:{rho:[.8,-.98,.98],lambda:[0,0,10],n:[60,10,200,true],noise:[.7,.1,2],seed:[42,1,99999,true]},
    overfit:{degree:[5,1,15,true],n:[20,10,80,true],noise:[.25,0,1],lambda:[.001,0,1],seed:[42,1,99999,true]},
    pca:{angle:[35,0,175],ratio:[4,1,10],scale:[1,.25,4],n:[120,20,300,true],standardize:[0,[0,1]],seed:[42,1,99999,true]},
    bootstrap:{n:[30,5,120,true],reps:[500,100,2000,true],distribution:['exponential',['normal','exponential']],seed:[42,1,99999,true]}
  };
  function validate(id,params={}) {
    if(!schemas[id])throw new Error('未知实验：'+id);
    const out={};for(const [key,s] of Object.entries(schemas[id])){
      const v=params[key]===undefined?s[0]:params[key];
      if(Array.isArray(s[1])){if(!s[1].includes(v))throw new Error(key+' 不在允许的选项内。');}
      else if(typeof v!=='number'||!Number.isFinite(v)||v<s[1]||v>s[2]||(s[3]&&!Number.isInteger(v)))throw new Error(key+' 超出允许范围或类型无效。');
      out[key]=v;
    }
    return out;
  }
  const models={};
  models.tangent=p=>{
    const f=p.function==='cubic'?x=>x*x*x-x:p.function==='sine'?Math.sin:Math.exp;
    const df=p.function==='cubic'?x=>3*x*x-1:p.function==='sine'?Math.cos:Math.exp;
    const exact=df(p.x),forward=(f(p.x+p.h)-f(p.x))/p.h,central=(f(p.x+p.h)-f(p.x-p.h))/(2*p.h);
    const span=Math.max(.15,1.5*p.h),xs=linspace(p.x-span,p.x+span);
    const errors=linspace(-3,0,121).map(logh=>{const h=10**logh;return [logh,Math.abs((f(p.x+h)-f(p.x))/h-exact),Math.abs((f(p.x+h)-f(p.x-h))/(2*h)-exact)];});
    return result([metric('精确导数',exact),metric('前向差分',forward),metric('中心差分',central),metric('前向绝对误差',Math.abs(forward-exact),6),metric('中心绝对误差',Math.abs(central-exact),6)],
      [plot('局部放大：函数、切线与前向割线','x（随 h 缩小观察窗）','函数值',[
        line('f(x)',xs.map(x=>[x,f(x)])),line('切线',xs.map(x=>[x,f(p.x)+exact*(x-p.x)]),1),
        line('前向割线',xs.map(x=>[x,f(p.x)+forward*(x-p.x)]),2,true),scatter('差分采样点',[[p.x-p.h,f(p.x-p.h)],[p.x,f(p.x)],[p.x+p.h,f(p.x+p.h)]],3)
      ]),plot('差分误差随 h 变化（双对数坐标）','log₁₀(h)','log₁₀(绝对误差)',[
        line('前向误差',errors.map(q=>q[1]>1e-14?[q[0],Math.log10(q[1])]:null),2),
        line('中心误差',errors.map(q=>q[2]>1e-14?[q[0],Math.log10(q[2])]:null),3),
        scatter('当前 h 的前向误差',Math.abs(forward-exact)>1e-14?[[Math.log10(p.h),Math.log10(Math.abs(forward-exact))]]:[],1)
      ],{xDomain:[-3,0]})],{x:p.x,h:p.h,exact,forward,central,forwardError:Math.abs(forward-exact),centralError:Math.abs(central-exact)},['h 的下限为 0.001；这一区间主要展示截断误差，不能据此认为 h 越小在浮点运算中永远越准确。','误差图省略 ≤10⁻¹⁴ 的数值，避免把浮点抵消误差画成真实精度；对数轴上的直线斜率表示误差阶数。']);
  };
  models.linear=p=>{
    const transform=([x,y])=>[p.a*x+p.b*y,p.c*x+p.d*y],basis=[[p.a,p.c],[p.b,p.d]],determinant=p.a*p.d-p.b*p.c;
    const original=[],transformed=[];for(let i=-2;i<=2;i++)for(const e of [[i,-2,i,2],[-2,i,2,i]]){original.push(e);transformed.push([...transform(e.slice(0,2)),...transform(e.slice(2))]);}
    const square=[[0,0],[1,0],[1,1],[0,1],[0,0]],vertices=square.map(transform),bound=Math.max(2,...transformed.flat().map(Math.abs))*1.08;
    return result([metric('行列式（有向面积倍率）',determinant),metric('单位正方形变换后面积',Math.abs(determinant)),metric('可逆性',Math.abs(determinant)<1e-12?'奇异：不可逆':'可逆')],
      [plot('网格与单位正方形的线性变换','第一坐标','第二坐标',[segments('原网格',original,6,true),segments('变换后网格',transformed,0),line('原单位正方形',square,6,true),{...line('变换后单位正方形',vertices,1),fill:true},{...segments('Ae₁',[[0,0,...basis[0]]],2),arrow:true},{...segments('Ae₂',[[0,0,...basis[1]]],3),arrow:true}],{equalAspect:true,xDomain:[-bound,bound],yDomain:[-bound,bound]})],
      {matrix:[[p.a,p.b],[p.c,p.d]],determinant,basis,vertices},[Math.abs(determinant)<1e-12?'此矩阵将平面压缩到直线或点；不存在逆矩阵。':determinant<0?'负行列式表示方向翻转；几何面积取其绝对值。':'列向量分别是标准基向量的像。']);
  };
  models.projection=p=>{
    const theta=p.angle*PI/180,u=[Math.cos(theta),Math.sin(theta)],v=[p.vx,p.vy],dot=u[0]*v[0]+u[1]*v[1],projection=u.map(x=>x*dot),residual=v.map((x,i)=>x-projection[i]);
    const norm2=sum(v.map(x=>x*x)),projectionNorm2=dot*dot,residualNorm2=sum(residual.map(x=>x*x)),orthogonality=sum(u.map((x,i)=>x*residual[i]));
    const bound=Math.max(1,...v.map(Math.abs),...projection.map(Math.abs))*1.3;
    return result([metric('投影标量 uᵀv',dot),metric('残差与 u 的内积',orthogonality,8),metric('‖v‖²',norm2),metric('‖投影‖² + ‖残差‖²',projectionNorm2+residualNorm2)],
      [plot('向量在一维子空间上的正交投影','第一坐标','第二坐标',[
        line('span(u)',[u.map(x=>-bound*x),u.map(x=>bound*x)],6,true),{...segments('原向量 v',[[0,0,...v]],0),arrow:true},{...segments('投影',[[0,0,...projection]],1),arrow:true},{...segments('残差（从投影到 v）',[[...projection,...v]],2,true),arrow:true},scatter('投影端点',[projection],1)
      ],{equalAspect:true,xDomain:[-bound,bound],yDomain:[-bound,bound],annotations:[{x:v[0],y:v[1],text:'v'},{x:projection[0],y:projection[1],text:'投影 p'}]})],{u,v,dot,projection,residual,norm2,projectionNorm2,residualNorm2,orthogonality},['残差从投影端点连接到 v；u 是单位向量，因此投影公式不需要另除以 ‖u‖²。']);
  };
  models.gradient=p=>{
    const path=[];let x=p.x0,y=p.y0;for(let i=0;i<=p.steps;i++){path.push({step:i,x,y,loss:(x*x+p.condition*y*y)/2});x*=1-p.rate;y*=1-p.rate*p.condition;}
    const stabilityLimit=2/p.condition,multiplier=Math.max(Math.abs(1-p.rate),Math.abs(1-p.rate*p.condition));
    const bound=Math.max(1,Math.abs(p.x0),Math.abs(p.y0))*1.4;
    const visible=[];for(const v of path){if(Math.abs(v.x)>bound||Math.abs(v.y)>bound)break;visible.push([v.x,v.y]);}
    const contours=[.25,1,4,9,16].map((c,i)=>line('f = '+c,linspace(0,2*PI,121).map(t=>[Math.sqrt(2*c)*Math.cos(t),Math.sqrt(2*c/p.condition)*Math.sin(t)]),6,true));
    const last=path[path.length-1],initial=path[0].loss;
    const excitedUnstable=(p.x0!==0&&Math.abs(1-p.rate)>1)||(p.y0!==0&&Math.abs(1-p.rate*p.condition)>1);
    let status=initial===0?'已在最小值':excitedUnstable?'发散模态被激发':multiplier<1?'渐近收敛':Math.abs(multiplier-1)<1e-12?'边界：不保证收敛':'此初值避开不稳定模态';
    const notes=['所有初值收敛的严格条件为 0 < 步长 < 2/L；边界值通常产生持续振荡。'];
    if(visible.length<path.length)notes.push('轨迹图仅显示离开初始观察窗前的 '+visible.length+' 个点；完整 '+path.length+' 个迭代点保存在导出数据，损失图包含全部迭代。');
    return result([metric('最终损失',last.loss),metric('稳定步长上界 2/L',stabilityLimit),metric('迭代矩阵谱半径',multiplier),metric('动力学状态',status)],
      [plot('等高线与迭代轨迹（离开观察窗后截断）','x','y',[...contours,line('梯度下降轨迹',visible,0),scatter('可见迭代点',visible,1),scatter('最小值',[[0,0]],2)],{equalAspect:true}),plot('完整迭代的损失','迭代步','log₁₀(1 + f)',[line('损失对数',path.map(q=>[q.step,Math.log10(1+q.loss)]))])],
      {path,rate:p.rate,condition:p.condition,stabilityLimit,spectralRadius:multiplier,status,visibleCount:visible.length},notes);
  };
  function sampler(distribution,random){const normal=normalGenerator(random);return distribution==='normal'?normal:distribution==='exponential'?()=>-Math.log(openUniform(random)):()=>Math.tan(PI*(openUniform(random)-.5));}
  models.sampling=p=>{
    const draw=sampler(p.distribution,rng(p.seed)),means=[],samples=[],sampleExamples=[];
    // Fixed 200-observation blocks preserve each replicate's first n values when n changes.
    for(let r=0;r<p.reps;r++){let s=0;const example=[];for(let j=0;j<200;j++){const v=draw();if(j<p.n){s+=v;if(r===0)samples.push(v);if(r<6)example.push(v);}}means.push(s/p.n);if(r<6)sampleExamples.push(example);}
    const pathDraw=sampler(p.distribution,rng(p.seed^0xB3914B)),runningMeans=[];let acc=0;
    for(let i=1;i<=200;i++){acc+=pathDraw();runningMeans.push([i,acc/i]);}
    const cauchy=p.distribution==='cauchy',mu=p.distribution==='normal'?0:1,se=1/Math.sqrt(p.n);
    const domain=cauchy?[quantile(means,.01),quantile(means,.99)]:null,hist=histogram(means,35,domain);
    const xs=linspace(hist.domain[0],hist.domain[1],241);
    const pdf=x=>p.distribution==='normal'?normalPDF(x/se)/se:cauchy?1/(PI*(1+x*x)):x<0?0:x===0?(p.n===1?1:0):Math.exp(p.n*Math.log(p.n)-logGamma(p.n)+(p.n-1)*Math.log(x)-p.n*x);
    const densityDomain=cauchy?[-5,5]:p.distribution==='normal'?[-3.5,3.5]:[0,4];
    const densityCurve=linspace(...densityDomain,501).map(x=>[x,pdf(x)]);
    const series=[{type:'bars',label:'模拟均值的密度直方图',points:hist.points,color:0},line(cauchy?'精确标准 Cauchy 密度':p.distribution==='normal'?'均值的精确正态密度':'均值的精确 Gamma 密度',xs.map(x=>[x,pdf(x)]),1)];
    const notes=['每次重复使用固定 200 个观测值的独立随机块；改变 n 时复用各块前 n 项，改变重复数时保留已有重复。'];
    if(cauchy)notes.push('标准 Cauchy 的总体均值与方差均不存在；样本均值仍服从标准 Cauchy，不能套用普通大数定律或有限方差中心极限定理。直方图显示第 1–99 百分位范围，省略 '+hist.omitted+' 个极端均值；密度仍以全部 '+p.reps+' 次重复为分母。');
    return result([metric('理论总体均值',cauchy?'不存在':mu),metric('理论均值标准误',cauchy?'不存在':se),metric('模拟均值的中位数',quantile(means,.5)),metric('重复次数',p.reps,0)],
      [plot('独立样本均值的抽样分布','样本均值','概率密度',series),plot('一条独立样本路径的运行均值','累计样本量','运行均值',[line('运行均值',runningMeans),...(cauchy?[]:[line('总体均值',[[1,mu],[200,mu]],1,true)])])],
      {samples,sampleExamples,means,runningMeans,densityDomain,densityCurve,populationMean:cauchy?null:mu,populationVariance:cauchy?null:1,theoreticalSE:cauchy?null:se,histogram:hist},notes);
  };
  models.confidence=p=>{
    const draw=normalGenerator(rng(p.seed)),critical=normalQuantile((1+p.level)/2),halfWidth=critical*p.sigma/Math.sqrt(p.n),intervals=[];
    for(let r=0;r<p.reps;r++){let s=0;for(let j=0;j<200;j++){const v=draw()*p.sigma;if(j<p.n)s+=v;}const m=s/p.n;intervals.push({mean:m,low:m-halfWidth,high:m+halfWidth,covers:Math.abs(m)<=halfWidth});}
    const covered=intervals.filter(x=>x.covers).length,shown=intervals.slice(0,100),good=[],bad=[];
    shown.forEach((q,i)=>(q.covers?good:bad).push([q.low,i+1,q.high,i+1]));
    return result([metric('全部重复的覆盖率',100*covered/p.reps,1,'%'),metric('覆盖计数 / 全部重复',covered+' / '+p.reps),metric('每个区间宽度',2*halfWidth),metric('名义覆盖率的 Monte Carlo 标准误',100*Math.sqrt(p.level*(1-p.level)/p.reps),2,' 个百分点')],
      [plot('前 '+shown.length+' / '+p.reps+' 个置信区间','总体均值 μ 的区间','重复序号',[segments('覆盖 μ = 0',good,0),segments('未覆盖 μ = 0',bad,2,true),line('真实 μ = 0',[[0,0],[0,shown.length+1]],1,true)],{height:Math.max(340,shown.length*5+90)})],
      {intervals,critical,halfWidth,covered,coverage:covered/p.reps,shownCount:shown.length,monteCarloSE:Math.sqrt(p.level*(1-p.level)/p.reps)},['覆盖率统计使用全部 '+p.reps+' 次重复；图只展示前 '+shown.length+' 次。','正态总体、IID 采样和已知 σ 使该 z 区间具有精确名义覆盖率；单次模拟覆盖率有 Monte Carlo 波动。']);
  };
  models.testing=p=>{
    const se=1/Math.sqrt(p.n),criticalZ=normalQuantile(1-p.alpha/2),criticalMean=criticalZ*se,observedZ=p.observed/se,pValue=2*normalCDF(-Math.abs(observedZ));
    const power=normalCDF((-criticalMean-p.effect)/se)+normalCDF((p.effect-criticalMean)/se),lo=Math.min(-5*se,p.effect-5*se,-Math.abs(p.observed)-.2,-criticalMean-.2),hi=Math.max(5*se,p.effect+5*se,Math.abs(p.observed)+.2,criticalMean+.2),height=normalPDF(0)/se;
    const tail=(label,mu,cut,color)=>{const left=linspace(lo,-cut,121),right=linspace(cut,hi,121);return {...line(label,[[lo,0],...left.map(x=>[x,normalPDF((x-mu)/se)/se]),[-cut,0],null,[cut,0],...right.map(x=>[x,normalPDF((x-mu)/se)/se]),[hi,0]],color),fill:true};};
    return result([metric('观测 z 值',observedZ),metric('双侧 p 值',pValue,6),metric('备择均值下的精确功效',power,4),metric('检验决策',pValue<=p.alpha?'拒绝 H₀':'未拒绝 H₀')],
      [plot('样本均值在零假设和指定备择下的分布','样本均值 x̄','概率密度',[
        tail('H₁ 拒绝域内面积＝功效',p.effect,criticalMean,1),line('H₀：μ = 0',linspace(lo,hi,401).map(x=>[x,normalPDF(x/se)/se]),0),line('H₁：μ = '+p.effect,linspace(lo,hi,401).map(x=>[x,normalPDF((x-p.effect)/se)/se]),1),
        segments('双侧拒绝边界（外侧拒绝）',[[-criticalMean,0,-criticalMean,height],[criticalMean,0,criticalMean,height]],2,true),segments('观测样本均值',[[p.observed,0,p.observed,height]],3)
      ]),plot('p 值：H₀ 下至少同样极端的两侧尾部','样本均值 x̄','H₀ 概率密度',[tail('两侧尾部面积＝p 值',0,Math.abs(p.observed),2),line('H₀ 分布',linspace(lo,hi,401).map(x=>[x,normalPDF(x/se)/se]),0),segments('±|观测均值|',[[-Math.abs(p.observed),0,-Math.abs(p.observed),height],[Math.abs(p.observed),0,Math.abs(p.observed),height]],3,true)])],{criticalZ,criticalMean,observedZ,pValue,power,se,reject:pValue<=p.alpha},['σ = 1 已知；两条曲线是样本均值的分布。p 值是 H₀ 下至少同样极端的尾概率，不是 H₀ 为真的概率。','effect = 0 时功效等于显著性水平 α（浮点计算可能有末位误差）。','着色区域只画当前坐标窗内的尾部；指标用完整正态 CDF 计算，两者不通过像素面积估算。']);
  };
  models.bayes=p=>{
    const a=p.a+p.success,b=p.b+p.failure,posteriorMean=a/(a+b),interval=[betaQuantile(.025,a,b),betaQuantile(.975,a,b)],xs=linspace(0,1,301),total=p.success+p.failure;
    const mode=total?p.success/total:.5;
    const logLik=x=>(p.success===0?0:p.success*Math.log(x))+(p.failure===0?0:p.failure*Math.log1p(-x));
    const maxLog=logLik(mode),likelihood=xs.map(x=>[x,Math.exp(logLik(x)-maxLog)]);
    return result([metric('后验 α',a,0),metric('后验 β',b,0),metric('后验均值',posteriorMean),metric('中央 95% 可信区间','['+interval.map(x=>x.toFixed(4)).join(', ')+']')],
      [plot('先验与后验的概率密度','成功概率 θ','概率密度',[line('先验 Beta('+p.a+','+p.b+')',xs.map(x=>[x,betaPDF(x,p.a,p.b)]),0),line('后验 Beta('+a+','+b+')',xs.map(x=>[x,betaPDF(x,a,b)]),1)]),plot('以峰值归一化的似然','成功概率 θ','相对似然（最大值为 1）',[line('L(θ) / max L',likelihood,2)],{xDomain:[0,1],yDomain:[0,1.05]})],
      {prior:{a:p.a,b:p.b},posterior:{a,b},mean:posteriorMean,interval,likelihood},['可信区间的后验概率为 95%，条件是选定先验和 Bernoulli 模型。','似然以峰值归一化，仅表示不同 θ 对数据的相对支持；它不是 θ 的概率密度。',...(total===0?['尚无观测：似然恒定，后验等于先验。']:[])]);
  };
  models.regression=p=>{
    const draw=normalGenerator(rng(p.seed)),raw=[],eps=[];
    for(let i=0;i<p.n;i++){const a=draw(),b=draw();raw.push([a,p.rho*a+Math.sqrt(1-p.rho*p.rho)*b]);eps.push(draw()*p.noise);}
    const centers=[mean(raw.map(r=>r[0])),mean(raw.map(r=>r[1]))],scales=[Math.sqrt(variance(raw.map(r=>r[0]))),Math.sqrt(variance(raw.map(r=>r[1])))];
    const X=raw.map(r=>r.map((v,j)=>(v-centers[j])/scales[j])),y=X.map((r,i)=>r[0]+r[1]+eps[i]),intercept=mean(y),centeredY=y.map(v=>v-intercept);
    const r=covariance(X.map(a=>a[0]),X.map(a=>a[1])),s=p.n-1,cross=s*r;
    function solve(l){const q=s+l,det=q*q-cross*cross;if(det<=Number.EPSILON*q*q)return null;const xy=[sum(X.map((v,i)=>v[0]*centeredY[i])),sum(X.map((v,i)=>v[1]*centeredY[i]))];return [(q*xy[0]-cross*xy[1])/det,(q*xy[1]-cross*xy[0])/det];}
    const beta=solve(p.lambda),olsBeta=solve(0),predictions=beta?X.map(r=>intercept+r[0]*beta[0]+r[1]*beta[1]):[],residuals=y.map((v,i)=>v-predictions[i]);
    const vif=1/(1-r*r),conditionNumber=(1+Math.abs(r))/(1-Math.abs(r));
    if(!beta)throw new Error('设计矩阵数值奇异；当前参数无法得到唯一系数。');
    const bounds=[Math.min(...y,...predictions),Math.max(...y,...predictions)];
    return result([metric('样本相关系数',r),metric('经验 VIF',vif),metric('XᵀX 条件数',conditionNumber),metric('训练 MSE',mean(residuals.map(v=>v*v)))],
      [plot('系数与真实值','系数编号','系数值',[line('真实 β',[ [1,1],[2,1]],6,true),scatter('OLS',olsBeta?olsBeta.map((v,i)=>[i+1-.06,v]):[],0),scatter('当前 ridge',beta.map((v,i)=>[i+1+.06,v]),1)],{xDomain:[.6,2.4],xTickLabels:['','β₁','β₂']}),plot('拟合值与响应','拟合值 ŷ','观测 y',[scatter('观测',predictions.map((v,i)=>[v,y[i]])),line('y = ŷ',bounds.map(v=>[v,v]),6,true)]),plot('残差图','拟合值 ŷ','残差 y − ŷ',[scatter('残差',predictions.map((v,i)=>[v,residuals[i]])),line('零残差',bounds.map(v=>[v,0]),6,true)])],
      {X,y,beta,olsBeta,intercept,predictions,residuals,sampleCorrelation:r,vif,conditionNumber,centers,scales,noise:eps,objective:sum(residuals.map(v=>v*v))+p.lambda*sum(beta.map(v=>v*v))},['预测变量使用样本标准差标准化，截距不惩罚。目标函数使用残差平方和，因此相同 λ 的强度会随 n 改变。','相关系数、VIF 和条件数来自当前样本；ρ 是生成数据的总体相关参数。条件数指标针对 XᵀX，等于 X 的二范数条件数的平方。']);
  };
  models.overfit=p=>{
    function generate(count,seed){const random=rng(seed),normal=normalGenerator(random),x=[],y=[];for(let i=0;i<count;i++){const v=2*openUniform(random)-1;x.push(v);y.push(Math.sin(PI*v)+p.noise*normal());}return {x,y};}
    const train=generate(p.n,p.seed),validation=generate(300,p.seed^0xAD932F1),design=train.x.map(x=>legendre(x,p.degree)),fit=leastSquares(design,train.y,p.lambda),coefficients=fit.coefficients;
    const predict=x=>coefficients?sum(legendre(x,p.degree).map((v,i)=>v*coefficients[i])):null;
    const predictions=train.x.map(predict),validationPredictions=validation.x.map(predict),trainMSE=coefficients?mean(predictions.map((v,i)=>(v-train.y[i])**2)):null,validationMSE=coefficients?mean(validationPredictions.map((v,i)=>(v-validation.y[i])**2)):null;
    const grid=linspace(-1,1,401),trueCurve=grid.map(x=>[x,Math.sin(PI*x)]),fitCurve=coefficients?grid.map(x=>[x,predict(x)]):[];
    const notes=['验证集有 300 个独立观测。调参时反复查看它会造成选择偏差；最终性能需要另留未参与选择的测试集。','采用 Legendre 多项式基与带列主元的 Householder QR；λ 惩罚非截距系数，损失为平方和。'];
    if(!fit.unique&&coefficients)notes.push('参数数目超过观测数，未惩罚最小二乘解不唯一；此处明确展示 Legendre 系数欧氏范数最小的插值解。');
    if(!coefficients)notes.push('设计矩阵数值秩不足，不能可靠确定系数；拟合曲线和误差暂不显示。');
    const maxAbs=coefficients?Math.max(...fitCurve.map(v=>Math.abs(v[1]))):0;
    // A second central view keeps the data legible when interpolation explodes near an endpoint.
    const displayBound=Math.max(2,3*p.noise+1.2),centralFit=fitCurve.map(q=>Math.abs(q[1])<=displayBound?q:null);
    const series=[line('真实 sin(πx)',trueCurve,6,true),scatter('训练观测',train.x.map((x,i)=>[x,train.y[i]]),0),line('模型拟合',maxAbs>displayBound?centralFit:fitCurve,1)];
    const plots=[plot('训练观测与拟合曲线'+(maxAbs>displayBound?'（局部纵轴）':''),'x','响应 y',series)];
    if(maxAbs>displayBound){plots.push(plot('完整拟合曲线（包含端点振荡）','x','模型预测',[line('完整模型拟合',fitCurve,1),line('真实函数',trueCurve,6,true)]));notes.push('拟合曲线存在较大振荡；第一幅图在 |预测| > '+displayBound.toFixed(2)+' 处断开，第二幅图及导出数据保留完整曲线。');}
    const complexity=Array.from({length:15},(_,i)=>{const degree=i+1,coef=leastSquares(train.x.map(x=>legendre(x,degree)),train.y,p.lambda).coefficients;if(!coef)return {degree,train:null,validation:null};const pred=x=>sum(legendre(x,degree).map((v,j)=>v*coef[j]));return {degree,train:mean(train.x.map((x,j)=>(pred(x)-train.y[j])**2)),validation:mean(validation.x.map((x,j)=>(pred(x)-validation.y[j])**2))};});
    plots.push(plot('同一数据上的复杂度与误差','多项式次数','log₁₀(MSE)，下限 −12',[line('训练 MSE',complexity.map(q=>q.train===null?null:[q.degree,Math.log10(Math.max(1e-12,q.train))]),0),line('独立验证 MSE',complexity.map(q=>q.validation===null?null:[q.degree,Math.log10(Math.max(1e-12,q.validation))]),2),scatter('当前次数',validationMSE===null?[]:[[p.degree,Math.log10(Math.max(1e-12,validationMSE))]],1)]));
    notes.push('复杂度图固定同一训练和验证数据，逐次重新拟合；MSE≤10⁻¹² 显示在 −12，不能解释为误差恰等于10⁻¹²。');
    return result([metric('训练 MSE',trainMSE===null?'未定义':trainMSE,6),metric('独立验证 MSE',validationMSE===null?'未定义':validationMSE,6),metric('系数数目',p.degree+1,0),metric('求解方式',fit.method)],plots,{train,validation,design,coefficients,predictions,validationPredictions,trainMSE,validationMSE,rank:fit.rank,unique:fit.unique,method:fit.method,trueCurve,fitCurve,complexity},notes);
  };
  models.pca=p=>{
    const draw=normalGenerator(rng(p.seed)),theta=p.angle*PI/180,c=Math.cos(theta),s=Math.sin(theta),observations=[];
    for(let i=0;i<p.n;i++){const a=p.ratio*draw(),b=draw();observations.push([c*a-s*b,p.scale*(s*a+c*b)]);}
    const means=[mean(observations.map(v=>v[0])),mean(observations.map(v=>v[1]))],scales=p.standardize?[Math.sqrt(variance(observations.map(v=>v[0]))),Math.sqrt(variance(observations.map(v=>v[1])))]:[1,1];
    const processed=observations.map(v=>v.map((x,j)=>(x-means[j])/scales[j])),col=[processed.map(v=>v[0]),processed.map(v=>v[1])],cov=[[variance(col[0]),covariance(col[0],col[1])],[covariance(col[0],col[1]),variance(col[1])]],eig=eigen2(cov),eigenvalues=eig.values,eigenvectors=eig.vectors;
    const scores=processed.map(v=>eigenvectors.map(u=>u[0]*v[0]+u[1]*v[1])),reconstruction=scores.map(v=>eigenvectors[0].map(u=>u*v[0])),reconstructionMSE=mean(processed.map((v,i)=>sum(v.map((x,j)=>(x-reconstruction[i][j])**2)))),explainedRatio=eigenvalues[0]/sum(eigenvalues);
    const pcSegments=eigenvectors.map((u,j)=>segments('PC'+(j+1)+'（±2√特征值）',[[...u.map(x=>-2*Math.sqrt(Math.max(0,eigenvalues[j]))*x),...u.map(x=>2*Math.sqrt(Math.max(0,eigenvalues[j]))*x)]],j+1,j===1));
    const notes=['分析使用'+(p.standardize?'样本标准差缩放后的相关矩阵':'中心化数据的样本协方差矩阵')+'；重建误差在同一分析坐标系中计算。','特征向量的正负号任意；特征值相等时主方向不唯一。总体各向同性不代表有限样本协方差的特征值完全相等。'];
    if(eig.equal)notes.push('当前样本特征值在数值容差内相等：显示的基仅是任意正交选择，不应解释为唯一主方向。');
    return result([metric('PC1 解释方差',100*explainedRatio,2,'%'),metric('第一特征值',eigenvalues[0]),metric('第二特征值',eigenvalues[1]),metric('每点一维重建平方误差',reconstructionMSE)],
      [plot('分析坐标中的数据与主方向',p.standardize?'标准化第一坐标':'中心化第一坐标',p.standardize?'标准化第二坐标':'中心化第二坐标',[scatter('中心化观测',processed,0),...pcSegments],{equalAspect:true}),plot('一维投影与重建',p.standardize?'标准化第一坐标':'中心化第一坐标',p.standardize?'标准化第二坐标':'中心化第二坐标',[scatter('原观测',processed,6),scatter('PC1 重建',reconstruction,1),segments('投影残差',processed.map((v,i)=>[...v,...reconstruction[i]]),6,true)],{equalAspect:true}),plot('主成分得分','PC1 得分','PC2 得分',[scatter('得分',scores,0)],{equalAspect:true})],
      {observations,processed,means,scales,covariance:cov,eigenvalues,eigenvectors,scores,reconstruction,reconstructionMSE,explainedRatio,equalEigenvalues:eig.equal},notes);
  };
  models.bootstrap=p=>{
    const draw=sampler(p.distribution,rng(p.seed)),sample=Array.from({length:p.n},draw),resample=rng(p.seed^0xC2B2AE35),bootstrapMeans=[],resampleExamples=[];
    for(let r=0;r<p.reps;r++){let s=0;const indices=[];for(let j=0;j<p.n;j++){const index=Math.floor(resample()*p.n);s+=sample[index];if(r<6)indices.push(index);}bootstrapMeans.push(s/p.n);if(r<6)resampleExamples.push(indices);}
    const sampleMean=mean(sample),bootstrapSE=Math.sqrt(variance(bootstrapMeans)),analyticSE=Math.sqrt(variance(sample)/p.n),interval=[quantile(bootstrapMeans,.025),quantile(bootstrapMeans,.975)],hist=histogram(bootstrapMeans),height=Math.max(...hist.points.map(v=>v[1]));
    return result([metric('观测样本均值',sampleMean),metric('Bootstrap 标准误',bootstrapSE),metric('s / √n',analyticSE),metric('Percentile 95% 区间','['+interval.map(v=>v.toFixed(4)).join(', ')+']')],
      [plot('给定样本后的 Bootstrap 均值分布','重抽样均值','概率密度',[{type:'bars',label:'Bootstrap 均值',points:hist.points,color:0},segments('Percentile 区间端点',interval.map(v=>[v,0,v,height]),2,true),segments('观测样本均值',[[sampleMean,0,sampleMean,height]],1)]),plot('原始观测样本','观测序号','观测值',[scatter('观测',sample.map((v,i)=>[i+1,v]),0)])],
      {sample,sampleMean,bootstrapMeans,resampleExamples,interval,bootstrapSE,analyticSE,conditionalExactSE:Math.sqrt(variance(sample,0)/p.n),populationMean:p.distribution==='normal'?0:1,histogram:hist},['每次从当前 n 个观测中有放回抽取 n 项；增加重复数保持观测样本和已有重抽样结果不变。','Bootstrap 分布条件于这一份样本。Percentile 区间不是精确覆盖率保证，小样本、偏态或异常值可能造成覆盖偏差。','给定样本时，Bootstrap 均值的精确条件标准差为 s√(n−1)/n，与 s/√n 存在有限样本差异。']);
  };
  function compute(id,params){const p=validate(id,params),out=models[id](p);out.params=p;return out;}
  function register(id,schema,model){if(!/^[a-z][a-z0-9-]+$/.test(id)||models[id]||typeof model!=='function')throw new Error('无效或重复的实验注册：'+id);schemas[id]=schema;models[id]=model;}
  const api={compute,validate,register,schemas,mean,variance,covariance,quantile,rng,normalGenerator,normalPDF,normalCDF,normalQuantile,logGamma,betaPDF,betaCDF,betaQuantile,histogram,eigen2,leastSquares,legendre,helpers:{sum,linspace,metric,plot,line,scatter,segments,result}};
  root.LabMath=api;if(typeof module!=='undefined'&&module.exports)module.exports=api;
})(typeof globalThis!=='undefined'?globalThis:this);
