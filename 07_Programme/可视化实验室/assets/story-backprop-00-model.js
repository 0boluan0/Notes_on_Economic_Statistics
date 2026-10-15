(function(root){
  'use strict';
  const parameters=['w','bias','v'];
  const evaluate=p=>root.LabMath.compute('backprop',p);
  function probe(p,key,delta){
    if(!parameters.includes(key)||!Number.isFinite(delta))throw Error('请选择可训练参数和有限改变量');
    const before=evaluate(p),next={...before.params,[key]:before.params[key]+delta};
    if(next[key]<-3||next[key]>3)throw Error('参数超出 −3 至 3 的实验范围');
    const after=evaluate(next),gradient=before.data.gradients[key];
    return {before,after,key,delta,gradient,predicted:gradient*delta,actual:after.data.forward.loss-before.data.forward.loss};
  }
  function update(p,rate){
    if(!Number.isFinite(rate)||rate<=0||rate>2)throw Error('学习率须在 0 与 2 之间');
    const before=evaluate(p),next={...before.params};
    for(const key of parameters)next[key]-=rate*before.data.gradients[key];
    const allowed=parameters.every(key=>next[key]>=-3&&next[key]<=3);
    return {before,next,allowed,after:allowed?evaluate(next):null,rate,deltas:parameters.map(key=>({key,old:before.params[key],gradient:before.data.gradients[key],delta:next[key]-before.params[key],next:next[key]}))};
  }
  function branches(p,a=1,b=-1){
    if(!Number.isFinite(a)||!Number.isFinite(b))throw Error('路径系数必须有限');
    const r=evaluate(p),d=r.data,h=d.forward.h,y=(a+b)*h,L=.5*(y-r.params.target)**2;
    const upstream=y-r.params.target,viaA=upstream*a,viaB=upstream*b,dh=viaA+viaB;
    return {a,b,h,y,L,upstream,viaA,viaB,dh,dw:dh*(1-h*h)*r.params.x};
  }
  root.BackpropLesson={evaluate,probe,update,branches,parameters};
  if(typeof module!=='undefined'&&module.exports)module.exports=root.BackpropLesson;
})(globalThis);
