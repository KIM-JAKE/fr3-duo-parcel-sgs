"""Torch task geometry, matching task_success.py without CPU rollout copies."""
import torch
from task_success import C,H,ENTRY_Y,TOP_Z,LABEL_UP_MIN

def rotate(q,v):
 qv=q[...,:3];t=2*torch.cross(qv.expand_as(v),v,dim=-1);return v+q[...,3:4]*t+torch.cross(qv.expand_as(v),t,dim=-1)

def geometry(points,score):
 lo=points.amin(-2);hi=points.amax(-2)
 success=(lo[:,0]>=C[0]-H[0])&(hi[:,0]<=C[0]+H[0])&(lo[:,1]>=C[1]-H[1])&(hi[:,1]<=ENTRY_Y)&((lo[:,2]-TOP_Z).abs()<=.015)&(score>=LABEL_UP_MIN)
 dx=(C[0]-H[0]-lo[:,0]).clamp_min(0)+(hi[:,0]-C[0]-H[0]).clamp_min(0)
 dy=(C[1]-H[1]-lo[:,1]).clamp_min(0)+(hi[:,1]-ENTRY_Y).clamp_min(0)
 distance=torch.stack([dx,dy,lo[:,2]-TOP_Z],-1).norm(dim=-1)
 potential=score-distance.clamp(max=1.8)/1.8
 return lo,hi,success,distance,potential

def shaped_reward(success,failure,old_potential,new_potential,terminated,actions,previous_actions,arm_velocity,gamma=.99):
 # Standard potential shaping with zero potential at true terminal states.
 future=torch.where(terminated,torch.zeros_like(new_potential),new_potential)
 return 100*success.float()-failure.float()+gamma*future-old_potential-1e-5*actions.square().sum(-1).clamp(max=1e4)-1e-4*(actions-previous_actions).square().sum(-1).clamp(max=1e4)-1e-4*arm_velocity.square().sum(-1).clamp(max=1e4)
