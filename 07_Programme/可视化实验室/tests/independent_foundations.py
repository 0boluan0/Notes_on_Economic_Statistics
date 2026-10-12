"""Independent foundations6 oracle: NumPy LAPACK, SciPy information functions.
Run from any cwd with Python dependencies from python/requirements-lock.txt.
"""
from pathlib import Path
import json,subprocess,hashlib,os
from datetime import datetime,timezone
import numpy as np
from numpy.polynomial.legendre import legvander
from scipy.special import softmax,logsumexp
from scipy.stats import entropy
ROOT=Path(__file__).resolve().parents[1]
CASES={
 'svd':[{},dict(scale=1e-14,ratio=.5),dict(ratio=1-1e-14),dict(scale=0),dict(ratio=0,retained=2),dict(ratio=1,shared=75),dict(scale=3,ratio=.7,retained=0,left=150,right=15)],
 'biasvariance':[{},dict(degree=0,noise=0),dict(degree=8,n=10,**{'lambda':0},reps=40),dict(degree=6,**{'lambda':2},noise=1,n=60,reps=30)],
 'kernel':[{},dict(kernel='linear',**{'lambda':.001},n=50),dict(kernel='poly2',**{'lambda':3}),dict(bandwidth=.08,**{'lambda':.001},n=50),dict(bandwidth=2,n=50)],
 'entropy':[{},dict(p1=1,q1=0),dict(p1=.5,p2=1,q1=.25,q2=1,unit='bits'),dict(p1=0,p2=0,q1=0,q2=0),dict(p1=.5,p2=.6,q1=.5,q2=.6)],
 'softmax':[{},dict(z1=-4,z2=4,z3=0,temperature=.25,shift=100,target=0),dict(z1=0,z2=0,z3=0,temperature=4,target=2),dict(temperature=.5,shift=-100)],
 'calibration':[{},dict(sharpness=4,offset=0,n=4000,bins=20),dict(sharpness=.25,offset=2,n=200,bins=20),dict(sharpness=2,offset=-2,n=2000,bins=4)]}
commands=[dict(id=k,params=p) for k,ps in CASES.items() for p in ps]
program="const fs=require('fs'),path=require('path'),root=process.argv[1],M=require(path.join(root,'assets/math.js'));require(path.join(root,'assets/foundations-math.js'));const c=JSON.parse(fs.readFileSync(0,'utf8'));console.log(JSON.stringify(c.map(x=>M.compute(x.id,x.params))));"
proc=subprocess.run([os.environ.get('NODE','node'),'-e',program,str(ROOT)],input=json.dumps(commands),capture_output=True,text=True,check=True)
results=json.loads(proc.stdout); checks=[]
def check(ok,name):
 if not bool(ok):raise AssertionError(name)
 checks.append(name)
def close(a,b,name,tol=1e-9):
 aa=np.asarray(a,float);bb=np.asarray(b,float)
 check(aa.shape==bb.shape and np.allclose(aa,bb,rtol=tol,atol=tol),name)
for cmd,r in zip(commands,results):
 id=cmd['id'];p=r['params'];d=r['data'];tag=id+': '+json.dumps(cmd['params'])
 if id=='svd':
  A=np.array(d['A']);U=np.array(d['U']);V=np.array(d['V']);sig=np.array(d['singularValues']);approx=np.array(d['approximation']);k=p['retained']
  check(d['rank']==np.count_nonzero(sig>0),tag+' exact constructed rank');check(d['equalSingularValues']==bool(sig[0]==sig[1]),tag+' exact repeated values');close(np.linalg.svd(A,compute_uv=False),sig,tag+' LAPACK singular values');close(U.T@U,np.eye(2),tag+' left orthogonality');close(V.T@V,np.eye(2),tag+' right orthogonality');close(A,U@np.diag(sig)@V.T,tag+' reconstruction')
  close(np.linalg.norm(A-approx,'fro'),d['frobeniusError'],tag+' Frobenius norm');close(np.linalg.norm(A-approx,2),d['spectralError'],tag+' spectral norm');close(d['frobeniusError'],np.linalg.norm(sig[k:]),tag+' optimal error');check(np.linalg.matrix_rank(approx,tol=1e-10)<=k,tag+' retained rank')
 elif id=='biasvariance':
  xs=np.array(d['trainX']);X=legvander(xs,p['degree']);P=np.diag([0]+[1]*p['degree']);aug=np.vstack([X,np.sqrt(p['lambda'])*P]);y=np.r_[d['firstTrain'],np.zeros(p['degree']+1)];beta=np.linalg.lstsq(aug,y,rcond=None)[0]
  close(legvander(d['grid'],p['degree'])@beta,d['fits'][0],tag+' independent first-fit LAPACK')
  fits=np.array(d['fits']);true=np.sin(np.pi*np.array(d['grid']));average=fits.mean(axis=0);bias2=(average-true)**2;var=fits.var(axis=0,ddof=0);mse=((fits-true)**2).mean(axis=0)
  close(bias2,[x['biasSquared'] for x in d['decomposition']],tag+' empirical bias');close(var,[x['variance'] for x in d['decomposition']],tag+' empirical variance');close(mse,var+bias2,tag+' finite ensemble identity');close(d['expectedResponseRisk'],d['atPoint']['mse']+p['noise']**2,tag+' fresh noise expectation')
  close(d['pointPredictions'][0],legvander([p['x0']],p['degree'])[0]@beta,tag+' independent selected-point prediction')
 elif id=='kernel':
  xs=np.array([o['x'] for o in d['observations']]);y=np.array([o['y'] for o in d['observations']]);typ=p['kernel'];h=p['bandwidth'];lam=p['lambda']
  def K(x,z):
   return 1+x[:,None]*z[None,:] if typ=='linear' else (1+x[:,None]*z[None,:])**2 if typ=='poly2' else np.exp(-(x[:,None]-z[None,:])**2/(2*h*h))
  gram=K(xs,xs);alpha=np.linalg.solve(gram+lam*np.eye(len(xs)),y);close(gram,d['K'],tag+' kernel definition');close(alpha,d['alpha'],tag+' LAPACK dual solve',2e-8)
  grid=np.array([v[0] for v in d['curve']]);close(K(grid,xs)@alpha,[v[1] for v in d['curve']],tag+' prediction');close(alpha@gram@alpha,d['rkhsNormSquared'],tag+' RKHS norm',2e-8)
  if typ!='rbf':
   Phi=np.column_stack([np.ones(len(xs)),xs]) if typ=='linear' else np.column_stack([np.ones(len(xs)),np.sqrt(2)*xs,xs**2]);PhiG=np.column_stack([np.ones(len(grid)),grid]) if typ=='linear' else np.column_stack([np.ones(len(grid)),np.sqrt(2)*grid,grid**2]);beta=np.linalg.solve(Phi.T@Phi+lam*np.eye(Phi.shape[1]),Phi.T@y)
   close(PhiG@beta,[v[1] for v in d['curve']],tag+' primal feature equivalence');close(beta@beta,d['rkhsNormSquared'],tag+' primal norm equivalence',2e-8)
  check(np.linalg.eigvalsh(gram).min()>-1e-10,tag+' PSD')
 elif id=='entropy':
  P=np.array(d['P']);Q=np.array(d['Q']);base=2 if p['unit']=='bits' else np.e;H=entropy(P,base=base);KL=entropy(P,Q,base=base);reverse=entropy(Q,P,base=base)
  close(d['entropy'],H,tag+' scipy entropy')
  for name,oracle in [('kl',KL),('crossEntropy',H+KL),('reverseKL',reverse)]:
   check(d[name] is None,tag+' support infinity '+name) if np.isinf(oracle) else close(d[name],oracle,tag+' scipy '+name)
 elif id=='softmax':
  z=np.array(d['logits']);T=p['temperature'];prob=softmax(z/T);target=p['target'];loss=logsumexp(z/T)-z[target]/T
  close(d['probabilities'],prob,tag+' scipy softmax');close(d['loss'],loss,tag+' scipy logsumexp loss');close(d['jacobian'],(np.diag(prob)-np.outer(prob,prob))/T,tag+' full Jacobian');close(d['gradient'],(prob-np.eye(3)[target])/T,tag+' loss derivative');close(np.sum(d['jacobian'],axis=0),np.zeros(3),tag+' constant shift null direction')
 elif id=='calibration':
  t=np.array(d['trueProbabilities']);y=np.array(d['labels']);q=1/(1+np.exp(-(p['sharpness']*np.log(t/(1-t))+p['offset'])));idx=np.minimum(p['bins']-1,np.floor(q*p['bins']).astype(int));ece=0;oracle=0
  for b,group in enumerate(d['groups']):
   selected=idx==b;count=selected.sum();check(group['count']==count,tag+' independent bin membership');
   if count:
    avgq=q[selected].mean();avgy=y[selected].mean();avgt=t[selected].mean();ece+=count/len(q)*abs(avgy-avgq);oracle+=count/len(q)*abs(avgt-avgq);close(group['meanPrediction'],avgq,tag+' bin prediction');close(group['observedRate'],avgy,tag+' bin outcome')
   else:check(group['observedRate'] is None,tag+' no fabricated empty-bin rate')
  close(d['ece'],ece,tag+' ECE');close(d['oracleECE'],oracle,tag+' oracle bin error');close(d['brier'],((q-y)**2).mean(),tag+' Brier');close(d['expectedBrier'],(t*(1-t)+(q-t)**2).mean(),tag+' conditional expected Brier');close(d['accuracy'],((q>=.5)==y).mean(),tag+' threshold accuracy')
report=dict(status='passed',utc=datetime.now(timezone.utc).isoformat(),experimentCases=len(commands),coverage={k:len(v) for k,v in CASES.items()},scientificChecks=len(checks),numpy=np.__version__,sourceSHA256=hashlib.sha256((ROOT/'assets/foundations-math.js').read_bytes()).hexdigest(),methods=['NumPy LAPACK SVD and least-squares','Kernel dual/primal feature-space independent solves','SciPy categorical entropy and KL with infinity boundaries','SciPy stable softmax/logsumexp','Raw repeated-fit decomposition','Independent calibration bins and proper-score moments'])
(ROOT/'tests/independent-foundations-report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n');print(json.dumps(report,indent=2))
