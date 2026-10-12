"""Independent NumPy/SciPy companion for the mathematical laboratory.

Run ``python lab_reference.py --lab all --output results.json``.
Random experiments use NumPy PCG64: same seed is reproducible within this
companion; it intentionally does not promise identical browser draws.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import legvander
from scipy.stats import norm, beta as beta_distribution

DEFAULTS = {
    'tangent': dict(x=1., h=.5, function='cubic'),
    'linear': dict(a=1., b=.7, c=0., d=1.),
    'projection': dict(angle=60., vx=2., vy=1.),
    'gradient': dict(rate=.2, condition=4., steps=20, x0=3., y0=2.),
    'sampling': dict(n=30, reps=500, distribution='exponential', seed=42),
    'confidence': dict(n=25, level=.95, reps=100, sigma=1., seed=42),
    'testing': dict(n=30, effect=.5, alpha=.05, observed=.3),
    'bayes': dict(a=2, b=2, success=7, failure=3),
    'regression': dict(rho=.8, **{'lambda':0.}, n=60, noise=.7, seed=42),
    'overfit': dict(degree=5, n=20, noise=.25, **{'lambda':.001}, seed=42),
    'pca': dict(angle=35., ratio=4., scale=1., n=120, standardize=0, seed=42),
    'bootstrap': dict(n=30, reps=500, distribution='exponential', seed=42),
}

def generator(seed=42, stream=0):
    return np.random.default_rng(np.random.SeedSequence([int(seed), stream]))

def draw(rng, distribution, size):
    if distribution == 'normal': return rng.normal(size=size)
    if distribution == 'exponential': return rng.exponential(size=size)
    if distribution == 'cauchy': return rng.standard_cauchy(size=size)
    raise ValueError(f'Unknown distribution: {distribution}')

def tangent(p):
    x, h = p['x'], p['h']
    if h <= 0: raise ValueError('h must be positive')
    f, df = {'cubic': (lambda x: x**3-x, lambda x: 3*x*x-1),
             'sine': (np.sin, np.cos), 'exponential': (np.exp, np.exp)}[p['function']]
    return dict(x=x, h=h, exact=df(x), forward=(f(x+h)-f(x))/h,
                central=(f(x+h)-f(x-h))/(2*h), function=p['function'])

def linear(p):
    matrix = np.array([[p['a'],p['b']],[p['c'],p['d']]])
    square = np.array([[0,0],[1,0],[1,1],[0,1],[0,0]])
    return dict(matrix=matrix, determinant=np.linalg.det(matrix),
                vertices=square @ matrix.T, singular=np.linalg.matrix_rank(matrix)<2)

def projection(p):
    angle=np.deg2rad(p['angle']); u=np.array([np.cos(angle),np.sin(angle)])
    v=np.array([p['vx'],p['vy']]); projected=u*(u@v); residual=v-projected
    return dict(u=u,v=v,projection=projected,residual=residual,
                orthogonality=residual@u,norm2=v@v,
                decomposedNorm2=projected@projected+residual@residual)

def gradient(p):
    condition=p['condition']; rate=p['rate']; k=np.arange(p['steps']+1)
    if condition < 1 or rate <= 0: raise ValueError('condition >= 1 and rate > 0 required')
    x=p['x0']*(1-rate)**k; y=p['y0']*(1-rate*condition)**k
    return dict(step=k,x=x,y=y,loss=(x*x+condition*y*y)/2,
                stabilityLimit=2/condition,stableForAllStarts=rate<2/condition)

def sampling(p):
    n=int(p['n']); reps=int(p['reps'])
    # Each replicate owns a stream: changing n retains its prefix.
    samples=np.array([draw(generator(p['seed'],i),p['distribution'],n) for i in range(reps)])
    path=draw(generator(p['seed'],100000),p['distribution'],200)
    finite=p['distribution']!='cauchy'
    return dict(sample=samples[0],means=samples.mean(axis=1),
                runningMeans=path.cumsum()/np.arange(1,201),
                populationMean=(1. if p['distribution']=='exponential' else 0.) if finite else None,
                populationVariance=1. if finite else None,
                theoreticalSE=1/np.sqrt(n) if finite else None)

def confidence(p):
    n=int(p['n']); reps=int(p['reps']); sigma=p['sigma']; level=p['level']
    if not 0 < level < 1 or sigma <= 0: raise ValueError('Invalid level or sigma')
    means=generator(p['seed']).normal(0,sigma/np.sqrt(n),size=reps)
    critical=norm.ppf((1+level)/2); half=critical*sigma/np.sqrt(n)
    low=means-half; high=means+half; covered=(low<=0)&(high>=0)
    return dict(means=means,low=low,high=high,covered=covered,
                coverage=covered.mean(),critical=critical,width=2*half,
                monteCarloSE=np.sqrt(level*(1-level)/reps))

def testing(p):
    n=p['n']; alpha=p['alpha']; delta=p['effect']*np.sqrt(n)
    if not 0<alpha<1: raise ValueError('alpha must lie in (0,1)')
    critical=norm.isf(alpha/2); observed=p['observed']*np.sqrt(n)
    return dict(criticalZ=critical,criticalMean=critical/np.sqrt(n),observedZ=observed,
                pValue=2*norm.sf(abs(observed)),
                power=norm.cdf(-critical-delta)+norm.sf(critical-delta))

def bayes(p):
    a=p['a']; b=p['b']; success=p['success']; failure=p['failure']
    if a<=0 or b<=0 or min(success,failure)<0: raise ValueError('Invalid Beta-Binomial parameters')
    aa=a+success; bb=b+failure; posterior=beta_distribution(aa,bb)
    return dict(prior=(a,b),posterior=(aa,bb),mean=posterior.mean(),
                interval=posterior.ppf([.025,.975]))

def ridge_fit(design,y,penalty,penalize_intercept=False):
    """Sum of squared residuals + lambda sum of penalized coefficients squared.

    Solve the augmented design with SVD least squares; do not square its
    condition number by forming normal equations.
    """
    if penalty<0: raise ValueError('lambda must be nonnegative')
    design=np.asarray(design); y=np.asarray(y)
    if penalty==0:
        coefficients,_,rank,singular=np.linalg.lstsq(design,y,rcond=None)
    else:
        regularizer=np.eye(design.shape[1])*np.sqrt(penalty)
        if not penalize_intercept: regularizer[0,0]=0
        augmented=np.vstack([design,regularizer]); target=np.r_[y,np.zeros(design.shape[1])]
        coefficients,_,rank,singular=np.linalg.lstsq(augmented,target,rcond=None)
    return coefficients,rank,singular

def regression(p):
    n=int(p['n']); rho=p['rho']; rng=generator(p['seed'])
    if abs(rho)>=1: raise ValueError('|rho| must be < 1')
    raw=rng.normal(size=(n,2)); raw[:,1]=rho*raw[:,0]+np.sqrt(1-rho*rho)*raw[:,1]
    X=(raw-raw.mean(axis=0))/raw.std(axis=0,ddof=1)
    y=X@np.ones(2)+p['noise']*generator(p['seed'],1).normal(size=n)
    design=np.column_stack([np.ones(n),X])
    coef,rank,_=ridge_fit(design,y,p['lambda']); ols,_,_=ridge_fit(design,y,0)
    prediction=design@coef; corr=np.corrcoef(X.T)[0,1]
    return dict(X=X,y=y,beta=coef[1:],olsBeta=ols[1:],intercept=coef[0],
                predictions=prediction,residuals=y-prediction,sampleCorrelation=corr,
                vif=1/(1-corr*corr),conditionNumber=np.linalg.cond(X),rank=rank)

def overfit(p):
    n=int(p['n']); degree=int(p['degree']); noise=p['noise']
    training=generator(p['seed']); validation=generator(p['seed'],1)
    x=training.uniform(-1,1,n); y=np.sin(np.pi*x)+noise*training.normal(size=n)
    xv=validation.uniform(-1,1,300); yv=np.sin(np.pi*xv)+noise*validation.normal(size=300)
    design=legvander(x,degree); valdesign=legvander(xv,degree)
    coef,rank,singular=ridge_fit(design,y,p['lambda'])
    pred=design@coef; vp=valdesign@coef
    return dict(train=dict(x=x,y=y),validation=dict(x=xv,y=yv),design=design,
                coefficients=coef,predictions=pred,validationPredictions=vp,
                trainMSE=np.mean((pred-y)**2),validationMSE=np.mean((vp-yv)**2),
                rank=int(rank),singularValues=singular,degree=degree,penalty=p['lambda'])

def pca(p):
    n=int(p['n']); angle=np.deg2rad(p['angle']); rng=generator(p['seed'])
    rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    observations=(rng.normal(size=(n,2))*np.array([p['ratio'],1]))@rotation.T
    observations[:,1]*=p['scale']; means=observations.mean(axis=0)
    scales=observations.std(axis=0,ddof=1) if p['standardize'] else np.ones(2)
    processed=(observations-means)/scales; cov=np.cov(processed.T,ddof=1)
    values,vectors=np.linalg.eigh(cov); values=values[::-1]; vectors=vectors[:,::-1]
    scores=processed@vectors; reconstruction=np.outer(scores[:,0],vectors[:,0])
    return dict(observations=observations,processed=processed,means=means,scales=scales,
                covariance=cov,eigenvalues=values,eigenvectors=vectors,scores=scores,
                reconstruction=reconstruction,
                reconstructionMSE=np.mean(np.sum((processed-reconstruction)**2,axis=1)),
                explainedRatio=values[0]/values.sum())

def bootstrap(p):
    n=int(p['n']); reps=int(p['reps'])
    sample=draw(generator(p['seed']),p['distribution'],n)
    indices=generator(p['seed'],1).integers(0,n,size=(reps,n))
    means=sample[indices].mean(axis=1)
    return dict(sample=sample,bootstrapMeans=means,
                interval=np.quantile(means,[.025,.975],method='linear'),
                bootstrapSE=means.std(ddof=1),analyticSE=sample.std(ddof=1)/np.sqrt(n))

FUNCTIONS={name:globals()[name] for name in DEFAULTS}

def compute(name,**parameters):
    if name not in DEFAULTS: raise ValueError(f'Unknown lab {name}')
    unexpected=set(parameters)-set(DEFAULTS[name])
    if unexpected: raise ValueError(f'Unknown parameters: {sorted(unexpected)}')
    return FUNCTIONS[name]({**DEFAULTS[name],**parameters})

def serializable(value):
    if isinstance(value,np.ndarray): return serializable(value.tolist())
    if isinstance(value,np.generic): return serializable(value.item())
    if isinstance(value,dict): return {k:serializable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [serializable(x) for x in value]
    if isinstance(value,float) and not np.isfinite(value): return None
    return value

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lab',choices=['all',*DEFAULTS],default='all')
    parser.add_argument('--params',default='{}',help='JSON parameter overrides, one lab only')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(); params=json.loads(args.params)
    if args.lab=='all' and params: parser.error('--params requires a single --lab')
    result={k:compute(k) for k in DEFAULTS} if args.lab=='all' else compute(args.lab,**params)
    payload=json.dumps(serializable(result),ensure_ascii=False,indent=2,allow_nan=False)
    if args.output: args.output.write_text(payload+'\n',encoding='utf-8')
    else: print(payload)
