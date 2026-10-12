"""Independent scientific acceptance using SciPy distributions and NumPy LAPACK.

Run: python tests/independent_numerics.py
NODE may name a non-default Node executable. No browser PRNG is reimplemented;
random experiment outputs are verified from exported raw samples/designs.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess,sys
import numpy as np
from numpy.polynomial.legendre import legvander
from scipy.stats import norm,beta as beta_dist,gamma
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
import lab_reference as ref
NODE=os.environ.get('NODE','node')
JS=r'''
const fs=require('fs'),math=require(process.argv[1]);
const commands=JSON.parse(fs.readFileSync(0,'utf8'));
process.stdout.write(JSON.stringify(commands.map(c=>{
try{return {ok:true,result:c.helper?math[c.helper](...c.args):math.compute(c.id,c.params||{})}}
catch(e){return {ok:false,error:String(e)}}
})));
'''
def node(commands):
    proc=subprocess.run([NODE,'-e',JS,str(ROOT/'assets'/'math.js')],input=json.dumps(commands),text=True,capture_output=True,check=True)
    return json.loads(proc.stdout)
checks=[]
def check(name,condition,detail=''):
    if not bool(condition): raise AssertionError(f'{name}: {detail}')
    checks.append(name)
def close(name,actual,expected,rtol=2e-9,atol=2e-10):
    a=np.asarray(actual,dtype=float); b=np.asarray(expected,dtype=float)
    check(name,a.shape==b.shape and np.allclose(a,b,rtol=rtol,atol=atol,equal_nan=False),f'actual {a}; expected {b}')

CASES={
 'tangent':[{},dict(x=-2.5,h=.001,function='cubic'),dict(x=0,h=1,function='sine'),dict(x=2.5,h=.001,function='exponential')],
 'linear':[{},dict(a=1,b=1,c=1,d=1),dict(a=0,b=0,c=0,d=0),dict(a=-2,b=0,c=0,d=2)],
 'projection':[{},dict(angle=0,vx=0,vy=3),dict(angle=90,vx=0,vy=0),dict(angle=175,vx=-3,vy=3)],
 'gradient':[{},dict(rate=.5,condition=4),dict(rate=1.2,condition=20,steps=60),dict(rate=1.2,condition=20,x0=0,y0=0),dict(rate=.2,condition=20,y0=0)],
 'sampling':[{},dict(n=1,reps=100,distribution='normal'),dict(n=200,reps=2000,distribution='exponential'),dict(n=200,reps=2000,distribution='cauchy')],
 'confidence':[{},dict(n=2,level=.8,reps=30,sigma=3),dict(n=200,level=.99,reps=300,sigma=.25)],
 'testing':[{},dict(n=2,effect=0,alpha=.1,observed=0),dict(n=200,effect=-1.5,alpha=.01,observed=1.5),dict(n=200,effect=1.5,alpha=.01,observed=-1.5)],
 'bayes':[{},dict(a=1,b=1,success=0,failure=0),dict(a=1,b=15,success=0,failure=50),dict(a=15,b=1,success=50,failure=0),dict(a=15,b=15,success=50,failure=50)],
 'regression':[{},dict(rho=.98,**{'lambda':0},n=10,noise=2),dict(rho=-.98,**{'lambda':10},n=200),dict(rho=0,**{'lambda':10},n=10)],
 'overfit':[{},dict(degree=15,n=10,**{'lambda':0}),dict(degree=15,n=20,**{'lambda':0}),dict(degree=15,n=80,noise=0,**{'lambda':1}),dict(degree=1,n=10,noise=0,**{'lambda':0})],
 'pca':[{},dict(ratio=1,scale=1,angle=0),dict(ratio=10,scale=.25,n=20,standardize=1),dict(ratio=1,scale=4,angle=175,n=300,standardize=0)],
 'bootstrap':[{},dict(n=5,reps=100,distribution='normal'),dict(n=120,reps=2000,distribution='exponential')]
}
commands=[dict(id=id,params=p) for id,params in CASES.items() for p in params]
results=node(commands)
for i,(command,response) in enumerate(zip(commands,results)):
    id=command['id']; prefix=f'{id}[{CASES[id].index(command["params"])}]'; p={**ref.DEFAULTS[id],**command['params']}
    check(prefix+' computes',response['ok'],response.get('error',''))
    result=response['result']; d=result['data']; plots=result['plots']
    check(prefix+' interface',all(k in result for k in ['metrics','plots','notes','data','params']))
    for plot in plots:
        for series in plot['series']:
            for point in series['points']:
                if point is None: continue
                check(prefix+' finite plot coordinates',all(np.isfinite(point)))
                xy=[point[:2],point[2:4]] if series['type']=='segments' else [point[:2]]
                for x,y in xy:
                    if 'xDomain' in plot: check(prefix+' x domain',plot['xDomain'][0]-1e-9<=x<=plot['xDomain'][1]+1e-9)
                    if 'yDomain' in plot: check(prefix+' y domain',plot['yDomain'][0]-1e-9<=y<=plot['yDomain'][1]+1e-9)
    if id=='tangent':
        expected=ref.compute(id,**command['params'])
        for key in ['exact','forward','central']: close(prefix+' '+key,d[key],expected[key])
        f=(lambda x:x*x*x-x) if p['function']=='cubic' else np.sin if p['function']=='sine' else np.exp
        for j,series in enumerate(plots[0]['series'][:3]):
            xy=np.array(series['points']); values=f(xy[:,0]) if j==0 else f(p['x'])+(d['exact'] if j==1 else d['forward'])*(xy[:,0]-p['x'])
            close(prefix+' local function/tangent/secant '+str(j),xy[:,1],values)
        for j,series in enumerate(plots[1]['series'][:2]):
            xy=np.array([point for point in series['points'] if point is not None]); h=10**xy[:,0]
            numerical=(f(p['x']+h)-f(p['x']))/h if j==0 else (f(p['x']+h)-f(p['x']-h))/(2*h)
            close(prefix+' complete plotted error curve '+str(j),10**xy[:,1],np.abs(numerical-d['exact']),rtol=3e-6,atol=5e-11)
    elif id=='linear':
        expected=ref.compute(id,**command['params'])
        for key in ['matrix','vertices','determinant']: close(prefix+' '+key,d[key],expected[key])
    elif id=='projection':
        expected=ref.compute(id,**command['params'])
        for key in ['projection','residual','norm2','orthogonality']: close(prefix+' '+key,d[key],expected[key])
        close(prefix+' pythagoras',d['projectionNorm2']+d['residualNorm2'],d['norm2'])
    elif id=='gradient':
        expected=ref.compute(id,**command['params']); path=d['path']
        check(prefix+' complete path',len(path)==p['steps']+1)
        for key in ['x','y','loss']: close(prefix+' '+key,[q[key] for q in path],expected[key],rtol=2e-13)
        close(prefix+' stability bound',d['stabilityLimit'],2/p['condition'])
        if p['x0']==p['y0']==0: check(prefix+' stationary state',d['status']=='已在最小值')
        if p['rate']>2/p['condition'] and p['y0']!=0: check(prefix+' divergence explicit','发散' in d['status'])
    elif id=='sampling':
        close(prefix+' first sample mean',d['means'][0],np.mean(d['samples']))
        check(prefix+' repetitions',len(d['means'])==p['reps'])
        h=d['histogram']; area=sum(q[1]*q[2] for q in h['points'])
        close(prefix+' histogram unconditional denominator',area,(p['reps']-h['omitted'])/p['reps'])
        values=np.array(plots[0]['series'][1]['points']); x=values[:,0]
        if p['distribution']=='cauchy':
            check(prefix+' undefined moments',all(d[k] is None for k in ['populationMean','populationVariance','theoreticalSE']))
            close(prefix+' exact density',values[:,1],1/(np.pi*(1+x*x)))
        else:
            close(prefix+' theoretical SE',d['theoreticalSE'],1/np.sqrt(p['n']))
            expected=norm.pdf(x,scale=1/np.sqrt(p['n'])) if p['distribution']=='normal' else gamma.pdf(x,a=p['n'],scale=1/p['n'])
            close(prefix+' exact density',values[:,1],expected,rtol=2e-10)
            close(prefix+' simulated mean',np.mean(d['means']),d['populationMean'],rtol=0,atol=6/np.sqrt(p['n']*p['reps']))
    elif id=='confidence':
        critical=norm.ppf((1+p['level'])/2); half=critical*p['sigma']/np.sqrt(p['n'])
        close(prefix+' normal quantile',d['critical'],critical)
        intervals=d['intervals']; means=np.array([a['mean'] for a in intervals])
        close(prefix+' lower',[a['low'] for a in intervals],means-half)
        close(prefix+' upper',[a['high'] for a in intervals],means+half)
        covered=np.sum(abs(means)<=half)
        check(prefix+' coverage denominator',covered==d['covered'] and len(intervals)==p['reps'])
        close(prefix+' coverage',d['coverage'],covered/p['reps'])
        close(prefix+' Monte Carlo SE',d['monteCarloSE'],np.sqrt(p['level']*(1-p['level'])/p['reps']))
    elif id=='testing':
        expected=ref.compute(id,**command['params'])
        for key in ['criticalZ','criticalMean','observedZ','power']: close(prefix+' '+key,d[key],expected[key])
        close(prefix+' extreme p-value',d['pValue'],expected['pValue'],rtol=2e-10,atol=1e-115)
        for j,mu in [(0,p['effect']),(1,0)]:
            tail=np.array([point for point in plots[j]['series'][0]['points'] if point is not None and point[1]>0])
            close(prefix+' shaded tail density '+str(j),tail[:,1],norm.pdf(tail[:,0],loc=mu,scale=1/np.sqrt(p['n'])),rtol=1e-10,atol=1e-115)
            cutoff=d['criticalMean'] if j==0 else abs(p['observed'])
            check(prefix+' shading belongs outside cutoff '+str(j),np.all(np.abs(tail[:,0])>=cutoff-1e-12))
    elif id=='bayes':
        expected=ref.compute(id,**command['params'])
        close(prefix+' mean',d['mean'],expected['mean'])
        close(prefix+' quantile',d['interval'],expected['interval'],atol=3e-12)
        close(prefix+' interval mass',beta_dist.cdf(d['interval'],*expected['posterior']),[.025,.975],atol=2e-11)
        for j,args in enumerate([(p['a'],p['b']),expected['posterior']]):
            xy=np.array(plots[0]['series'][j]['points']); close(prefix+' beta density',xy[:,1],beta_dist.pdf(xy[:,0],*args),rtol=2e-11)
        values=np.array(d['likelihood']); check(prefix+' likelihood bounded',np.all((values[:,1]>=0)&(values[:,1]<=1+1e-12)))
        if p['success']+p['failure']==0: close(prefix+' flat likelihood',values[:,1],np.ones(len(values)))
    elif id=='regression':
        X=np.array(d['X']); y=np.array(d['y']); design=np.column_stack([np.ones(p['n']),X])
        coef,_,_=ref.ridge_fit(design,y,p['lambda']); ols,_,_=ref.ridge_fit(design,y,0)
        close(prefix+' standardized means',X.mean(axis=0),[0,0])
        close(prefix+' standardized scales',X.std(axis=0,ddof=1),[1,1])
        close(prefix+' ridge coefficients',d['beta'],coef[1:]); close(prefix+' OLS coefficients',d['olsBeta'],ols[1:])
        close(prefix+' unpenalized intercept',d['intercept'],coef[0]); close(prefix+' predictions',d['predictions'],design@coef)
        close(prefix+' residuals',d['residuals'],y-design@coef)
        r=np.corrcoef(X.T)[0,1]; close(prefix+' VIF',d['vif'],1/(1-r*r))
        close(prefix+' Gram condition',d['conditionNumber'],np.linalg.cond(X)**2)
        close(prefix+' true DGP',y,X@np.ones(2)+np.array(d['noise']))
    elif id=='overfit':
        x=np.array(d['train']['x']); y=np.array(d['train']['y']); xv=np.array(d['validation']['x']); yv=np.array(d['validation']['y'])
        design=legvander(x,p['degree']); close(prefix+' Legendre design',d['design'],design)
        coef,rank,_=ref.ridge_fit(design,y,p['lambda'])
        check(prefix+' solver produces coefficients',d['coefficients'] is not None)
        # Ill-conditioned interpolation magnifies roundoff; relative 2e-6 is
        # conservative for this explicit high-degree boundary, not defaults.
        tolerance=2e-6 if np.linalg.cond(design)>1e5 else 2e-9
        close(prefix+' independent SVD coefficients',d['coefficients'],coef,rtol=tolerance,atol=2e-8)
        close(prefix+' predictions',d['predictions'],design@coef,rtol=tolerance,atol=2e-8)
        vp=legvander(xv,p['degree'])@coef
        close(prefix+' validation predictions',d['validationPredictions'],vp,rtol=tolerance,atol=2e-7)
        close(prefix+' train MSE',d['trainMSE'],np.mean((y-design@coef)**2),atol=1e-12)
        close(prefix+' validation MSE',d['validationMSE'],np.mean((yv-vp)**2),rtol=tolerance)
        check(prefix+' nonunique flag',d['unique']==(p['lambda']>0 or p['n']>=p['degree']+1))
        check(prefix+' complexity covers every degree', [item['degree'] for item in d['complexity']]==list(range(1,16)))
        for item in d['complexity']:
            degree=item['degree']; X=legvander(x,degree); coef_s,_,_=ref.ridge_fit(X,y,p['lambda']); vp_s=legvander(xv,degree)@coef_s
            tol_s=2e-6 if np.linalg.cond(X)>1e5 else 2e-9
            close(prefix+' complexity train degree '+str(degree),item['train'],np.mean((X@coef_s-y)**2),rtol=tol_s,atol=2e-10)
            close(prefix+' complexity validation degree '+str(degree),item['validation'],np.mean((vp_s-yv)**2),rtol=tol_s,atol=2e-10)
    elif id=='pca':
        observed=np.array(d['observations']); means=observed.mean(axis=0)
        scales=observed.std(axis=0,ddof=1) if p['standardize'] else np.ones(2)
        data=(observed-means)/scales; cov=np.cov(data.T); values,vectors=np.linalg.eigh(cov)
        close(prefix+' preprocessing',d['processed'],data); close(prefix+' sample covariance',d['covariance'],cov)
        close(prefix+' eigenvalues',d['eigenvalues'],values[::-1]); actualVectors=np.array(d['eigenvectors'])
        close(prefix+' orthonormal basis',actualVectors@actualVectors.T,np.eye(2))
        close(prefix+' eigensystem',cov@actualVectors.T,actualVectors.T*np.array(d['eigenvalues']))
        projection=np.outer(data@vectors[:,-1],vectors[:,-1]); close(prefix+' sign-independent reconstruction',d['reconstruction'],projection)
        mse=np.mean(np.sum((data-projection)**2,axis=1)); close(prefix+' reconstruction MSE',d['reconstructionMSE'],mse)
        close(prefix+' error-eigenvalue identity',mse,values[0]*(p['n']-1)/p['n'])
        close(prefix+' explained ratio',d['explainedRatio'],values[-1]/values.sum())
    elif id=='bootstrap':
        sample=np.array(d['sample']); boot=np.array(d['bootstrapMeans'])
        check(prefix+' all resamples finite',len(boot)==p['reps'] and np.all(np.isfinite(boot)))
        check(prefix+' convex hull bound',np.all((boot>=sample.min()-1e-12)&(boot<=sample.max()+1e-12)))
        close(prefix+' percentile interpolation',d['interval'],np.quantile(boot,[.025,.975],method='linear'))
        close(prefix+' bootstrap sample SE',d['bootstrapSE'],boot.std(ddof=1)); close(prefix+' analytic SE',d['analyticSE'],sample.std(ddof=1)/np.sqrt(p['n']))
        close(prefix+' exact conditional SE',d['conditionalExactSE'],sample.std(ddof=0)/np.sqrt(p['n']))

# Distribution helpers: independent Boost/SciPy calculations, central and tails.
helper_commands=[]; helper_expected=[]
for x in [-25,-10,-8,-4,-1,0,.1,1,4,8,10,25]:
    helper_commands.append(dict(helper='normalCDF',args=[x])); helper_expected.append(('normal CDF',norm.cdf(x),1e-145))
for p in [1e-10,1e-7,.001,.025,.5,.975,.999,1-1e-7,1-1e-10]:
    helper_commands.append(dict(helper='normalQuantile',args=[p])); helper_expected.append(('normal inverse',norm.ppf(p),3e-7))
for a,b in [(1,1),(1,65),(65,1),(65,65),(2,5),(15,60)]:
    for p in [.0001,.025,.5,.975,.9999]:
        helper_commands.append(dict(helper='betaQuantile',args=[p,a,b])); helper_expected.append(('beta inverse',beta_dist.ppf(p,a,b),2e-11))
    for x in [0,.001,.1,.5,.9,.999,1]:
        helper_commands.append(dict(helper='betaCDF',args=[x,a,b])); helper_expected.append(('beta CDF',beta_dist.cdf(x,a,b),2e-11))
for command,response,(name,expected,atol) in zip(helper_commands,node(helper_commands),helper_expected):
    check(name+' callable',response['ok']); close(name+' '+str(command['args']),response['result'],expected,rtol=3e-10,atol=atol)

# Honest singular behaviour and arbitrary repeated-eigenvalue eigenspace.
custom=[dict(helper='eigen2',args=[[[2,0],[0,2]]]),dict(helper='leastSquares',args=[[[1,1],[1,1],[1,1]],[1,2,3],0])]
special=node(custom)
check('equal eigenvalue flag',special[0]['result']['equal']); close('equal eigenvalues',special[0]['result']['values'],[2,2])
check('rank deficient no fabricated unique solution',special[1]['result']['coefficients'] is None and not special[1]['result']['unique'])

# Reproducibility and common-data comparisons are behavioural checks, not a
# reimplementation of the JS PRNG. Parameter limits from the public contract.
for id in ['sampling','confidence','regression','overfit','pca','bootstrap']:
    a,b,c=node([dict(id=id,params={'seed':42}),dict(id=id,params={'seed':42}),dict(id=id,params={'seed':43})])
    check(id+' exact seed repeat',a==b); check(id+' seed changes data',a['result']['data']!=c['result']['data'])
for id,key,small,big in [('sampling','means',100,2000),('confidence','intervals',30,300),('bootstrap','bootstrapMeans',100,2000)]:
    a,b=node([dict(id=id,params={'reps':small}),dict(id=id,params={'reps':big})])
    check(id+' repetitions preserve prefixes',a['result']['data'][key]==b['result']['data'][key][:small])
a,b=node([dict(id='sampling',params={'n':5}),dict(id='sampling',params={'n':200})])
check('sampling n preserves underlying first sample',a['result']['data']['samples']==b['result']['data']['samples'][:5])
a,b=node([dict(id='bootstrap',params={'reps':100}),dict(id='bootstrap',params={'reps':2000})])
check('bootstrap repetitions preserve observed sample',a['result']['data']['sample']==b['result']['data']['sample'])
a,b=node([dict(id='overfit',params={'degree':1}),dict(id='overfit',params={'degree':15})])
check('overfit degree preserves train/validation data',a['result']['data']['train']==b['result']['data']['train'] and a['result']['data']['validation']==b['result']['data']['validation'])
a,b=node([dict(id='pca',params={'standardize':1,'scale':.25}),dict(id='pca',params={'standardize':1,'scale':4})])
close('standardized PCA positive-unit invariance',a['result']['data']['covariance'],b['result']['data']['covariance'])

import scipy
report=dict(status='passed',utc=datetime.now(timezone.utc).isoformat(),experimentCases=len(commands),
            helperCases=len(helper_commands)+len(custom),assertions=len(checks),plotCoordinateAssertions=sum(any(t in x for t in ["finite plot coordinates","x domain","y domain"]) for x in checks),
            numpy=np.__version__,scipy=scipy.__version__,python=sys.version.split()[0],
            coreSHA256=hashlib.sha256((ROOT/'assets'/'math.js').read_bytes()).hexdigest(),
            coverage={k:len(v) for k,v in CASES.items()},
            methods=['SciPy normal and Beta CDF/PPF','NumPy SVD augmented least squares','NumPy symmetric eigensystem','Raw data identities and exact distribution curves','Seed/prefix invariants and finite plot coordinates'],
            limits=['Bounded deterministic cases; not a proof for every real parameter','Random generator distribution quality is not exhaustively certified','UI content, accessibility, export wiring and scientific-source review require separate acceptance'])
(ROOT/'tests'/'independent-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
