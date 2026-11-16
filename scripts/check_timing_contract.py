"""Exercise actual environment step timing and event latching with synthetic dynamics.

This validates control bookkeeping; it is not a physical grasp/success test.
"""
import os,pathlib,sys,json
os.environ['MUJOCO_GL']='disable'
root=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'fr3_duo_sgs'))
import torch
from types import SimpleNamespace
from tensordict import TensorDict
from parcel_sgs_env import ParcelSGSEnv
from control_timing import CONTROL_DECIMATION,EPISODE_STEPS,POLICY_DT_S
class FakeCell:
 def __init__(self):self.steps=0;self.elapsed=0
 def write_data_to_sim(self):pass
 def step(self):self.steps+=1
 def update(self,dt):self.elapsed+=dt
 def __getitem__(self,k):raise KeyError(k)
def fixture(success=False):
 e=ParcelSGSEnv.__new__(ParcelSGSEnv);e.num_envs=2;e.calls=0;e.active=torch.ones(2,dtype=torch.bool);e.max_episode_length=160;e.block_step=0;e.previous_actions=torch.zeros(2,54);e.potential=torch.zeros(2);e.reference=torch.zeros(2,54);e.speed=torch.ones(54);e.limits=torch.tensor([[-4.,4.]]*54);e.bias=torch.zeros(2,54);e.kp=torch.ones(54);e.soft=False;e.episode_length_buf=torch.zeros(2,dtype=torch.long);e.success_streak=torch.zeros(2,dtype=torch.long);e.absorbing_obs=torch.zeros(2,256);e.arm=torch.ones(54,dtype=torch.bool);e.events=[];e.current_ids=[11,12];e.active_steps=0;e.total_steps=0;e.scene=e.sim=FakeCell();e.robot=SimpleNamespace(data=SimpleNamespace(gravity_compensation_forces=SimpleNamespace(torch=torch.zeros(2,54))),actuators=SimpleNamespace(target_command=SimpleNamespace(set_position_index=lambda **kw:None)))
 def observe():
  q=e.reference.clone();x=torch.zeros(2,256);x[:,:54]=q
  return TensorDict({'policy':x},batch_size=[2]),dict(potential=torch.zeros(2),q=q,qd=torch.zeros(2,54),lo=torch.tensor([[.3,0,.3]]*2),hi=torch.tensor([[.6,.3,.6]]*2),success=torch.tensor([success,False]),score=torch.ones(2),distance=torch.ones(2))
 e._observe=observe;e._station_failure=lambda:torch.zeros(2,dtype=torch.bool);e._new_block=lambda:None
 return e
e=fixture();e.step(torch.ones(2,54)*.5);assert e.sim.steps==50 and abs(e.sim.elapsed-.1)<1e-9;assert torch.allclose(e.reference,torch.full((2,54),.05));assert e.block_step==1
for _ in range(159):obs,reward,done,extras=e.step(torch.zeros(2,54))
assert e.sim.steps==8000 and abs(e.sim.elapsed-16)<1e-8 and extras['time_outs'].all() and done.all() and len(e.events)==2
s=fixture(True);_,_,done,_=s.step(torch.ones(2,54)*.5);assert done[0] and not done[1] and s.events==[(11,True)];assert torch.allclose(s.reference[0],torch.full((54,),.03));s.step(torch.ones(2,54)*.5);assert len(s.events)==1
out=dict(state='pass',scope='synthetic dynamics, actual ParcelSGSEnv.step control bookkeeping',control_hz=10,episode_steps=160,episode_duration_s=16,physics_calls_per_action=50,physics_dt_s=.002,success_failure_audit_hz=50,success_hold_audit_steps=3,early_terminal_latched_once=True)
(root/'evidence/timing_contract.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
