"""SGS manipulation kernel and rolling Boolean history (paper Sec 3.2, App B.2)."""
import numpy as np
class SuccessGuidedSampler:
 def __init__(self,geometry_ids,seed=0,history=100,target=.5,concentration=1.,temperature=2.,epsilon=1e-4):
  self.geometry_ids=np.asarray(geometry_ids);self.history=np.zeros((len(geometry_ids),history),np.uint8);self.count=np.zeros(len(geometry_ids),np.int32);self.cursor=self.count.copy();self.rng=np.random.default_rng(seed);self.target=target;self.concentration=concentration;self.temperature=max(temperature,1);self.epsilon=epsilon
  self.ng_budget_groups=[]
  self.groups=[np.flatnonzero(self.geometry_ids==g) for g in range(int(self.geometry_ids.max())+1)]
 def configure_ng_budget(self,family,free_success,kind,target=.25):
  family=np.asarray(family);free_success=np.asarray(free_success,bool);kind=np.asarray(kind)
  assert len(family)==len(self.geometry_ids)==len(free_success)==len(kind)
  self.ng_budget_groups=[]
  for k in np.unique(kind):
   ng=(family=='NG')&(kind==k);positive=ng&free_success;negative=ng&~free_success
   if ng.any():
    assert positive.any() and negative.any(), 'NG needs both free and nonfree candidates per kind'
    self.ng_budget_groups.append((positive,negative,float(target)))
 def weights(self):
  p=self.history.sum(1)/np.maximum(self.count,1);e=self.epsilon;w=(p+e)**(self.concentration*self.target)*(1-p+e)**(self.concentration*(1-self.target));result=(np.maximum(w,e)+e)**(1/self.temperature)
  # Preserve each kind's total NG SGS mass; allocate exactly 25% of it to
  # no-action successes. Geometry mass + conditional row draw preserves this
  # marginal distribution in expectation, not an exact homogeneous-batch quota.
  for positive,negative,target in self.ng_budget_groups:
   mass=result[positive].sum()+result[negative].sum()
   result[positive]*=target*mass/result[positive].sum()
   result[negative]*=(1-target)*mass/result[negative].sum()
  return result
 def sample_geometry(self):
  w=self.weights();mass=np.bincount(self.geometry_ids,weights=w,minlength=len(self.groups));return int(self.rng.choice(len(mass),p=mass/mass.sum()))
 def sample(self,geometry,n):
  ids=self.groups[geometry];w=self.weights()[ids];return self.rng.choice(ids,n,replace=True,p=w/w.sum())
 def update(self,events):
  for idx,success in events:
   self.history[idx,self.cursor[idx]]=int(bool(success));self.cursor[idx]=(self.cursor[idx]+1)%self.history.shape[1];self.count[idx]=min(self.count[idx]+1,self.history.shape[1])
 def state_dict(self):return dict(history=self.history,count=self.count,cursor=self.cursor,rng=self.rng.bit_generator.state)
 def load_state_dict(self,d):
  assert self.history.shape==d['history'].shape
  self.history[:]=d['history'];self.count[:]=d['count'];self.cursor[:]=d['cursor'];self.rng.bit_generator.state=d['rng']
