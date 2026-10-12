"""Independent NumPy check of 6 classical-ML and 6 neural-model mechanisms.

Does not reproduce the browser PRNG: it uses exported observations and recomputes
objectives, assignments, gradients, and probabilities from their definitions.
Run with Python + NumPy; no SciPy or browser is required.
"""
from pathlib import Path
import hashlib,json,subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
commands=[]
def case(id,**params):commands.append(dict(id=id,params=params))
for args in [dict(dataset='balancedxor',depth=5),dict(dataset='axis',depth=3),dict(dataset='xor',depth=4,minleaf=1)]:case('tree',**args)
for args in [dict(steps=0),dict(steps=15,rate=1),dict(steps=40,rate=.05,noise=1)]:case('boosting',**args)
for args in [dict(initialization='first',k=5,steps=15),dict(initialization='spread',k=3,steps=5),dict(steps=0)]:case('kmeans',**args)
for args in [dict(initial=0,steps=8),dict(initial=4,sigma=.3,steps=20),dict(steps=6)]:case('gmm',**args)
for args in [dict(threshold=1,prevalence=.05),dict(threshold=0,separation=0),dict(threshold=.5,separation=4)]:case('metrics',**args)
for args in [dict(observedRate=1,candidateP=0),dict(observedRate=0,candidateP=0),dict(n=100,sigma=.25)]:case('likelihood',**args)
for args in [dict(depth=5,kind='relu',mode='he'),dict(depth=4,kind='tanh',mode='xavier'),dict(depth=20,mode='small')]:case('initialization',**args)
for args in [dict(dropout=0),dict(dropout=.8,scale=2,target=-2),dict(dropout=.5)]:case('regularization',**args)
for args in [dict(mode='batch',pattern='constant',gamma=0,beta=2),dict(mode='layer',scale=4,shift=-3),dict(mode='batch')]:case('normalization',**args)
for args in [dict(kernel='edge',stride=2,padding=1,row=2,col=2),dict(pattern='impulse',kernel='laplace'),dict(kernel='blur')]:case('convolution',**args)
for args in [dict(mode='max',pattern='tied',size=3,stride=3),dict(mode='mean',pattern='ramp',size=3,stride=1),dict(mode='max')]:case('pooling',**args)
for args in [dict(causal=1,query=0,strength=4,temperature=.25),dict(causal=1,strength=0,query=2),dict(valueScale=3)]:case('attention',**args)
js="""
const fs=require('fs'),M=require(process.argv[1]);require(process.argv[2]);require(process.argv[3]);
process.stdout.write(JSON.stringify(JSON.parse(fs.readFileSync(0,'utf8')).map(c=>({id:c.id,...M.compute(c.id,c.params)}))));
"""
proc=subprocess.run(['node','-e',js,*[str(ROOT/'assets'/f) for f in ['math.js','ml-math.js','dl-math.js']]],input=json.dumps(commands),capture_output=True,text=True,check=True)
results=json.loads(proc.stdout);checks=[]
def check(name,condition):
    assert bool(condition),name
    checks.append(name)
def close(name,a,b,rtol=2e-9,atol=1e-10):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    check(name,a.shape==b.shape and np.allclose(a,b,rtol=rtol,atol=atol))
def sigmoid(x):return 1/(1+np.exp(-x))
for ci,out in enumerate(results):
    id=out['id'];p=out['params'];d=out['data'];tag=f'{ci}:{id}'
    if id=='tree':
        X=np.array([v['x'] for v in d['observations']]);Y=np.array([v['y'] for v in d['observations']])
        def verify(node,indices):
            ys=Y[indices];close(tag+' node probability',node['probability'],ys.mean());close(tag+' node gini',node['impurity'],2*ys.mean()*(1-ys.mean()))
            gains=[]
            if node['depth']<p['depth']:
                for j in range(2):
                    unique=np.unique(X[indices,j]);cuts=(unique[:-1]+unique[1:])/2
                    for cut in cuts:
                        left=indices[X[indices,j]<=cut];right=indices[X[indices,j]>cut]
                        if min(len(left),len(right))<p['minleaf']:continue
                        impurity=lambda ix:2*Y[ix].mean()*(1-Y[ix].mean())
                        gains.append((impurity(indices)-(len(left)*impurity(left)+len(right)*impurity(right))/len(indices),j,cut,left,right))
            if 'left' in node:
                best=max(gains,key=lambda v:v[0]);close(tag+' greedy optimum',node['gain'],best[0]);ix=indices[X[indices,node['axis']]<=node['threshold']];jx=indices[X[indices,node['axis']]>node['threshold']]
                check(tag+' minimum leaf',min(len(ix),len(jx))>=p['minleaf']);verify(node['left'],ix);verify(node['right'],jx)
            else:check(tag+' valid stopping',not gains or max(v[0] for v in gains)<=1e-10)
        verify(d['tree'],np.arange(len(Y)))
        if p['dataset']=='balancedxor':check(tag+' symmetric xor stops',len(d['leaves'])==1)
    elif id=='boosting':
        X=np.array([v['x'] for v in d['train']]);Y=np.array([v['y'] for v in d['train']]);pred=np.full(len(Y),Y.mean());order=np.argsort(X)
        close(tag+' base',d['base'],Y.mean())
        for t,stump in enumerate(d['models']):
            residual=Y-pred;candidate=[np.sum((residual-residual.mean())**2)]
            for j in range(3,len(X)-2):
                L=residual[order[:j]];R=residual[order[j:]];candidate.append(np.sum((L-L.mean())**2)+np.sum((R-R.mean())**2))
            close(tag+' true optimal stump SSE',stump['sse'],min(candidate))
            increment=np.full(len(Y),stump['left']) if stump['threshold'] is None else np.where(X<=stump['threshold'],stump['left'],stump['right'])
            pred+=p['rate']*increment;close(tag+' actual train MSE',d['history'][t+1]['trainMSE'],np.mean((Y-pred)**2))
        close(tag+' final predictions',pred,d['trainPredictions']);check(tag+' train risk nonincrease',np.all(np.diff([v['trainMSE'] for v in d['history']])<=1e-10))
    elif id=='kmeans':
        X=np.array(d['observations']);centers=np.array(d['initialCenters'])
        for t,h in enumerate(d['history']):
            dist=np.sum((X[:,None,:]-centers[None,:,:])**2,axis=2);labels=dist.argmin(axis=1)
            close(tag+' centers',centers,h['centers']);close(tag+' SSE',np.min(dist,axis=1).sum(),h['sse'])
            if t<p['steps']:centers=np.array([X[labels==j].mean(axis=0) if np.any(labels==j) else c for j,c in enumerate(centers)])
        close(tag+' final assignments',labels,d['assignments']);check(tag+' SSE monotonic',np.all(np.diff([v['sse'] for v in d['history']])<=1e-10))
    elif id=='gmm':
        X=np.array(d['observations']);sd=d['sigma'];prev=None
        for h in d['history']:
            means=np.array(h['means']);weight=h['weight'];terms=np.stack([np.log(weight)-.5*((X-means[0])/sd)**2,np.log1p(-weight)-.5*((X-means[1])/sd)**2],axis=1)
            L=np.logaddexp(terms[:,0],terms[:,1])-np.log(sd*np.sqrt(2*np.pi));close(tag+' observed likelihood',L.sum(),h['logLikelihood'])
            if prev is not None and prev.sum()>1e-10 and len(X)-prev.sum()>1e-10:
                close(tag+' M-step means',means,[np.dot(prev,X)/prev.sum(),np.dot(1-prev,X)/(1-prev).sum()]);close(tag+' M-step weight',weight,prev.mean())
            prev=np.exp(terms[:,0]-np.logaddexp(terms[:,0],terms[:,1]))
        close(tag+' final responsibilities',prev,d['responsibilities']);check(tag+' EM monotonic',np.all(np.diff([v['logLikelihood'] for v in d['history']])>=-1e-9))
    elif id=='metrics':
        y=np.array(d['labels']);s=np.array(d['scores']);pred=s>=p['threshold'];tp=np.sum(pred&(y==1));fp=np.sum(pred&(y==0));close(tag+' confusion',[d[k] for k in ['tp','fp','tn','fn']],[tp,fp,np.sum(~pred&(y==0)),np.sum(~pred&(y==1))])
        pairs=s[y==1,None]-s[y==0][None,:];close(tag+' pairwise AUC',d['auc'],np.mean((pairs>0)+.5*(pairs==0)))
        ap=0;old_recall=0
        for th in sorted(set(s),reverse=True):
            selected=s>=th;TP=np.sum(y[selected]);recall=TP/y.sum();ap+=(recall-old_recall)*TP/selected.sum();old_recall=recall
        close(tag+' AP',d['ap'],ap)
        if tp+fp==0:check(tag+' precision undefined',d['precision'] is None)
    elif id=='likelihood':
        x=np.array(d['normalObservations']);mu=x.mean();close(tag+' gaussian MLE',d['normalMLE'],mu)
        close(tag+' normal exact relative curve',np.array(d['normalRelativeLikelihood'])[:,1],np.exp(-p['n']*(np.array(d['normalRelativeLikelihood'])[:,0]-mu)**2/(2*p['sigma']**2)))
        close(tag+' Bernoulli MLE',d['bernoulliMLE'],d['success']/p['n'])
        if p['candidateP']==0 and d['success']>0:check(tag+' impossible observation -infinity',d['bernoulliLogLikelihoodIsNegativeInfinity'] and d['bernoulliLogLikelihood'] is None)
    elif id=='initialization':
        h=np.array(d['activations'][0]);pre=[];weights=[np.array(W) for W in d['weights']]
        for l,W in enumerate(weights):
            z=h@W.T;pre.append(z);close(tag+' forward preactivation',z,d['preactivations'][l]);h=np.maximum(0,z) if p['kind']=='relu' else np.tanh(z);close(tag+' forward activation',h,d['activations'][l+1]);stats=d['forwardStats'][l+1];close(tag+' moment distinction',[stats['variance'],stats['secondMoment']],[h.var(),(h*h).mean()])
        gradient=np.full_like(h,1/np.sqrt(d['width']))
        for l in range(len(weights)-1,-1,-1):gradient=(gradient*((pre[l]>0) if p['kind']=='relu' else (1-np.tanh(pre[l])**2)))@weights[l]
        close(tag+' full reverse product',gradient,d['inputGradients'],atol=1e-8)
    elif id=='regularization':
        contributions=np.array(d['contributions']);q=1-p['dropout'];masks=np.array([v['mask'] for v in d['distribution']]);probs=np.prod(np.where(masks,q,1-q),axis=1);values=(masks@contributions)/q
        close(tag+' enumerated probabilities',probs,[v['probability'] for v in d['distribution']]);close(tag+' exact mean',np.dot(probs,values),d['expected']);close(tag+' exact variance',np.dot(probs,(values-d['expected'])**2),d['variance']);close(tag+' exact expected loss',np.dot(probs,(values-p['target'])**2/2),d['expectedLoss']);close(tag+' actual simulated samples',np.array(d['masks'])@contributions/q,d['outputs'])
    elif id=='normalization':
        X=np.array(d['X']);axis=0 if p['mode']=='batch' else 1;mean=X.mean(axis=axis,keepdims=True);var=X.var(axis=axis,keepdims=True);z=(X-mean)/np.sqrt(var+d['epsilon'])
        close(tag+' statistics means',mean.ravel(),d['means']);close(tag+' statistics variance',var.ravel(),d['variances']);close(tag+' epsilon standardization',z,d['normalized']);close(tag+' affine output',p['gamma']*z+p['beta'],d['Y']);close(tag+' exact standardized variances',z.var(axis=axis),var.ravel()/(var.ravel()+d['epsilon']))
    elif id=='convolution':
        padded=np.pad(d['input'],p['padding']);k=np.array(d['kernel']);outsize=(len(padded)-3)//p['stride']+1
        out=np.array([[np.sum(padded[i*p['stride']:i*p['stride']+3,j*p['stride']:j*p['stride']+3]*k) for j in range(outsize)] for i in range(outsize)])
        close(tag+' cross-correlation output',out,d['output']);close(tag+' selected patch products',np.array(d['patch'])*k,d['contributions']);close(tag+' selected output',out[p['row'],p['col']],np.array(d['contributions']).sum())
    elif id=='pooling':
        X=np.array(d['input']);k=p['size'];s=p['stride'];n=(6-k)//s+1;fn=np.max if p['mode']=='max' else np.mean;out=np.array([[fn(X[i*s:i*s+k,j*s:j*s+k]) for j in range(n)] for i in range(n)])
        close(tag+' pooling output',out,d['output']);g=np.array(d['gradient']);close(tag+' gradient sum',g.sum(),1)
        if p['mode']=='mean':close(tag+' local mean derivative',g[p['row']*s:p['row']*s+k,p['col']*s:p['col']*s+k],np.full((k,k),1/k**2))
        else:check(tag+' valid max subgradient',np.all(X[g>0]==out[p['row'],p['col']]) and np.all(g>=0))
    elif id=='attention':
        Q=np.array(d['Q']);K=np.array(d['K']);V=np.array(d['V']);scores=p['strength']*(Q@K.T)/(np.sqrt(2)*p['temperature']);mask=np.tril(np.ones((4,4),dtype=bool)) if p['causal'] else np.ones((4,4),dtype=bool);masked=np.where(mask,scores,-np.inf);e=np.exp(masked-masked.max(axis=1,keepdims=True));a=e/e.sum(axis=1,keepdims=True)
        close(tag+' matrix scores',scores,d['scores']);close(tag+' masked weights',a,d['weights']);close(tag+' actual AV',a@V,d['outputs']);check(tag+' forbidden positions zero',np.all(a[~mask]==0));close(tag+' convex row sum',a.sum(axis=1),np.ones(4))
report={'cases':len(results),'assertions':len(checks),'models':sorted(set(c['id'] for c in commands)),'passed':True,'hashes':{f:hashlib.sha256((ROOT/'assets'/f).read_bytes()).hexdigest() for f in ['ml-math.js','dl-math.js']},'checks':checks}
(ROOT/'tests'/'cross-review-ml-dl-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(f'PASS independent ML/DL cross-review: {len(results)} cases across {len(report["models"])} models; {len(checks)} defining-equation assertions.')
