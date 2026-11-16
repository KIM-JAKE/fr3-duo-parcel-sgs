"""Exact-geometry SGS teacher environment using geometry batches.

All 200k configurations remain eligible. A geometry is drawn by its total SGS
weight, then configurations conditional on it. Worlds run synchronous episodes
(max 5.12s); terminated worlds enter an absorbing zero-reward state until the
batch ends. The next geometry rebuilds the native scene. This preserves exact
reset geometry/materials and native solvers, but is explicitly NOT the paper's
independent per-world global resampling protocol. Report active and total steps.
"""
import pathlib,json,gc
from collections import OrderedDict
import numpy as np,torch,mujoco,newton,warp as wp
from tensordict import TensorDict
from isaaclab.app import launch_simulation
import isaaclab.sim as sim_utils
from isaaclab.utils import instantiate
from isaaclab_newton.physics import NewtonManager
from newton_cell_cfg import CELL
from parcel_native_scene import configuration
from parcel_rl_math import rotate,geometry,shaped_reward
from task_success import C,TOP_Z,SIGNS
from sgs_sampler import SuccessGuidedSampler

class ParcelSGSEnv:
 num_actions=54
 def __init__(self,index,args,num_envs,rank=0,episode_steps=256):
  self.index=pathlib.Path(index);self.manifest=json.loads((self.index/'manifest.json').read_text());f=np.load(self.index/'configurations.npz',allow_pickle=False);self.rows={k:f[k].copy() for k in f.files};f.close();self.args=args;self.num_envs=num_envs;self.device=args.device;self.rank=rank;self.max_episode_length=episode_steps;self.episode_length_buf=torch.zeros(num_envs,device=self.device,dtype=torch.long);self.sampler=SuccessGuidedSampler(self.rows['geometry_id'],seed=3100+rank);self.events=[];self.cache=OrderedDict();self.context=None;self.block=0;self.calls=0;self.active_steps=0;self.total_steps=0;self.current_ids=None;self.cfg=dict(task='FR3 Duo parcel',obs='privileged teacher 256D; not RGB',sampler='SGS kernel, geometry-batched synchronous episode adaptation',reward='terminal100/fail1 + SGS regularizers + label/distance potential shaping',episode_steps=episode_steps)
  with np.load(self.index/'ng_free_flags.npz',allow_pickle=False) as flags:free_success=flags['free_success'].copy()
  kinds=np.array([self.manifest['geometries'][int(g)]['kind'] for g in self.rows['geometry_id']])
  self.sampler.configure_ng_budget(self.rows['family'],free_success,kinds,target=.25)
  mm=mujoco.MjModel.from_xml_path(CELL['robot']['urdf']);self.mm=mm
 def _tensor(self,x):return torch.as_tensor(x,device=self.device,dtype=torch.float32)
 def _load(self,ids):
  records=[]
  for i in ids:
   sid=int(self.rows['shard_id'][i]);row=int(self.rows['row'][i])
   if sid not in self.cache:
    with np.load(self.manifest['shards'][sid],allow_pickle=False) as f:
     keys=['robot_q','robot_qd','robot_q_target','parcel_pose_xyzw','parcel_twist','label_local_normal','label_triangles','nodes','nodal_vel','triangles']
     self.cache[sid]={k:f[k].copy() for k in keys if k in f}
    if len(self.cache)>12:self.cache.popitem(last=False)
   record=self.cache[sid];records.append({k:(v if k=='triangles' else v[row]) for k,v in record.items()})
  return {k:np.stack([r[k] for r in records]) if k!='triangles' else records[0][k] for k in records[0]}
 def _close_scene(self):
  if self.context is not None:
   self.scene=None;self.robot=None;self.model=None;self.sim=None
   sim_utils.SimulationContext.clear_instance();self.context.__exit__(None,None,None);self.context=None;gc.collect();torch.cuda.empty_cache()
 def _new_block(self):
  self._close_scene();self.geometry_id=self.sampler.sample_geometry();self.spec=self.manifest['geometries'][self.geometry_id];self.soft=self.spec['kind']=='mailer';self.current_ids=self.sampler.sample(self.geometry_id,self.num_envs);rows=self._load(self.current_ids)
  cfg,scfg=configuration(self.spec,self.num_envs,self.device);self.context=launch_simulation(cfg,self.args);self.context.__enter__();self.sim=sim_utils.SimulationContext(cfg);self.scene=instantiate(scfg);self.sim.reset();self.robot=self.scene['robot'];self.model=NewtonManager.get_model();self.origins=self.scene.env_origins
  from scaled_reset_io import apply_scaled_rows
  apply_scaled_rows(self.scene,rows,rows['robot_q_target']);NewtonManager.forward();self.target=self._tensor(rows['robot_q_target']);self.previous_actions=torch.zeros((self.num_envs,54),device=self.device);self.active=torch.ones(self.num_envs,device=self.device,dtype=torch.bool);self.episode_length_buf.zero_();self.block_step=0
  names=self.robot.joint_names;self.arm=torch.tensor(['fr3v2_joint' in x for x in names],device=self.device);self.kp=self._tensor([[600,600,600,600,250,150,50][int(x[-1])-1] if 'fr3v2_joint' in x else 20 for x in names]);self.speed=self._tensor([[2.175,2.175,2.175,2.175,2.61,2.61,2.61][int(x[-1])-1] if 'fr3v2_joint' in x else 3.2 for x in names]);self.limits=self._tensor(np.array([self.mm.jnt_range[self.mm.joint(x).id] for x in names]));self.label_normals=self._tensor(rows['label_local_normal']);self.half=self._tensor(np.array(self.spec['size_m'])/2);self.signs=self._tensor(SIGNS)
  if self.soft:self.tri=torch.as_tensor(rows['triangles'],device=self.device,dtype=torch.long);self.label_mask=torch.as_tensor(rows['label_triangles'],device=self.device,dtype=torch.bool)
  if not self.soft:self.target=self.target-self.robot.data.gravity_compensation_forces.torch/self.kp
  self.reference=self.robot.data.joint_pos.torch.clone();self.bias=self.target-self.reference
  self.success_streak=torch.zeros(self.num_envs,device=self.device,dtype=torch.long);self.block+=1;self.geometry_seen=getattr(self,'geometry_seen',set());self.geometry_seen.add(self.geometry_id)
  body_names=self.model.body_label;sb=self.model.shape_body.numpy();self.shape_robot=torch.as_tensor([b>=0 and any(v in str(body_names[b]) for v in ('fr3v2_','dg_')) for b in sb],device=self.device);self.shape_station=torch.as_tensor([b>=0 and any(v in str(body_names[b]) for v in ('FlipTable','Infeed','Output')) for b in sb],device=self.device);self.shape_world=torch.as_tensor(self.model.shape_world.numpy(),device=self.device,dtype=torch.long);self.contact_distance=None
  self.obs,self.metrics=self._observe();self.potential=self.metrics['potential'];self.absorbing_obs=self.obs['policy'].clone()
 def _observe(self):
  q=self.robot.data.joint_pos.torch;qd=self.robot.data.joint_vel.torch;pose=self.scene['box'].data.root_pose_w.torch.clone();pose[:,:3]-=self.origins;vel=self.scene['box'].data.root_vel_w.torch
  if self.soft:
   points=self.scene['mailer'].data.nodal_pos_w.torch-self.origins[:,None,:];lp=points[:,self.tri,:];norm=torch.cross(lp[:,:,1]-lp[:,:,0],lp[:,:,2]-lp[:,:,0],dim=-1);area=norm.norm(dim=-1)*self.label_mask;weighted=norm*self.label_mask[:,:,None];normal=weighted.sum(1)/area.sum(1).clamp_min(1e-12)[:,None];score=weighted[:,:,2].clamp_min(0).sum(1)/area.sum(1).clamp_min(1e-12);label_center=(lp.mean(2)*area[:,:,None]).sum(1)/area.sum(1).clamp_min(1e-12)[:,None]
  else:
   local=self.signs*self.half;points=rotate(pose[:,None,3:].expand(-1,8,-1),local[None].expand(self.num_envs,-1,-1))+pose[:,None,:3];normal=rotate(pose[:,3:],self.label_normals);score=normal[:,2];label_center=pose[:,:3]+normal*self.half.min()
  lo,hi,success,distance,potential=geometry(points,score);center=(lo+hi)/2
  if self.soft:
   half=self._tensor(np.array(self.spec['size_m'])*[.31,.31,.21]);contents=rotate(pose[:,None,3:].expand(-1,8,-1),(self.signs*half)[None].expand(self.num_envs,-1,-1))+pose[:,None,:3];_,_,contents_on_belt,_,_=geometry(contents,score);success&=contents_on_belt
   candidate=torch.nonzero(success&self.active).flatten()
   if len(candidate):
    from batched_reset_audit import enclosure_batch
    inside,_=enclosure_batch(points[candidate].detach().cpu().numpy(),self.tri.cpu().numpy(),pose[candidate].detach().cpu().numpy(),np.array(self.spec['size_m'])*[.31,.31,.21],self.device)
    success[candidate]&=torch.as_tensor(inside,device=self.device)

  # Body link poses are not needed for the first teacher; the six hand slots
  # expose end-effector position via actual articulation body transforms.
  body=self.robot.data.body_pose_w.torch;labels=self.robot.body_names;hands=[]
  for name in ['ll_dg_3_1','rl_dg_3_1']:
   idx=labels.index(name);hands.append(body[:,idx,:3]-self.origins-center)
  features=torch.cat([(q-(self.limits[:,0]+self.limits[:,1])/2)/((self.limits[:,1]-self.limits[:,0])/2).clamp_min(.01),qd/self.speed,self.previous_actions,(self.target-q).clamp(-2,2),lo,hi,center,vel,normal,label_center,score[:,None],self._tensor(C)[None]-center,self._tensor(self.spec['size_m'])[None].expand(self.num_envs,-1),self._tensor([self.spec['mass_kg'],self.spec['friction']])[None].expand(self.num_envs,-1),torch.full((self.num_envs,1),float(self.soft),device=self.device),*hands,(self.episode_length_buf.float()/self.max_episode_length)[:,None]],-1)
  assert features.shape[1]==254,features.shape
  features=torch.nn.functional.pad(features,(0,2)).clamp(-20,20)
  return TensorDict({'policy':features},batch_size=[self.num_envs]),dict(success=success,potential=potential,distance=distance,score=score,q=q,qd=qd,lo=lo,hi=hi)
 def get_observations(self):
  if self.context is None:self._new_block()
  return self.obs
 def _station_failure(self):
  contacts=NewtonManager.get_contacts();num=int(contacts.rigid_contact_count.numpy()[0]);bad=torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)
  if num>contacts.rigid_contact_max:raise RuntimeError('Rigid contact overflow during PPO')
  if num:
   if self.contact_distance is None:self.contact_distance=wp.zeros(contacts.rigid_contact_max,dtype=wp.float32,device=self.device)
   newton.eval_rigid_contact_kinematics(self.model,NewtonManager.get_state_0(),contacts,out_distance=self.contact_distance)
   a=wp.to_torch(contacts.rigid_contact_shape0)[:num].long();b=wp.to_torch(contacts.rigid_contact_shape1)[:num].long();valid=(a>=0)&(b>=0);a=a.clamp_min(0);b=b.clamp_min(0);pair=(self.shape_robot[a]&self.shape_station[b])|(self.shape_robot[b]&self.shape_station[a]);world=self.shape_world[a];selected=valid&pair&(wp.to_torch(self.contact_distance)[:num]<-.003)&(world>=0)&(world<self.num_envs);bad[world[selected]]=True
  return bad
 def step(self,actions):
  self.calls+=1
  if not self.active.any() or self.block_step>=self.max_episode_length:self._new_block()
  previous_active=self.active.clone();old=self.potential;prev=self.previous_actions.clone();a=actions.clamp(-1,1);a=torch.where(self.active[:,None],a,torch.zeros_like(a));self.reference=(self.reference+a*self.speed*.02).clamp(self.limits[:,0]+.005,self.limits[:,1]-.005);self.target=self.reference+self.bias
  for _ in range(10):
   command=self.target+self.robot.data.gravity_compensation_forces.torch/self.kp
   self.robot.actuators.target_command.set_position_index(value=command);self.scene.write_data_to_sim();self.sim.step()
   if self.soft:
    state=NewtonManager.get_state_0();newton.eval_ik(self.model,state,state.joint_q,state.joint_qd)
   self.scene.update(.002)
  self.previous_actions=a;self.episode_length_buf+=self.active.long();self.block_step+=1;obs,m=self._observe()
  finite=torch.isfinite(obs['policy']).all(1)&torch.isfinite(m['q']).all(1)&torch.isfinite(m['qd']).all(1)
  if not finite.all():raise RuntimeError('Nonfinite native policy rollout; stop and preserve checkpoint')
  failure=(m['qd'].abs()>2*self.speed).any(1)|(m['lo'][:,2]<.12)|(m['hi'][:,2]>1.5)|(m['lo'][:,0]<.12)|(m['hi'][:,0]>1.4)|(m['lo'][:,1]<-1.25)|(m['hi'][:,1]>1.25)
  failure|=self._station_failure()
  self.success_streak=torch.where(m['success'],self.success_streak+1,torch.zeros_like(self.success_streak));success=(self.success_streak>=3)&~failure&self.active;failure&=self.active;terminal=success|failure;timeout=(self.episode_length_buf>=self.max_episode_length)&self.active&~terminal;newdone=terminal|timeout;dones=~self.active|newdone
  reward=shaped_reward(success,failure,old,m['potential'],terminal,actions,prev,m['qd'][:,self.arm]);reward=torch.where(previous_active,reward,torch.zeros_like(reward));self.active&=~newdone
  ended=torch.nonzero(newdone).flatten().cpu().numpy();result=success.detach().cpu().numpy();self.events.extend((int(self.current_ids[i]),bool(result[i])) for i in ended)
  # Exact absorbing observations for already terminated worlds, with no repeated SGS outcomes.
  x=obs['policy'];x[~previous_active]=self.absorbing_obs[~previous_active];self.absorbing_obs[newdone]=x[newdone];self.obs=obs;self.metrics=m;self.potential=m['potential'];self.active_steps+=int(previous_active.sum());self.total_steps+=self.num_envs
  extras=dict(time_outs=timeout,log={'/success':float(success.sum()),'/failure':float(failure.sum()),'/label_up':float(m['score'].mean()),'/goal_distance_m':float(m['distance'].mean()),'/active_fraction':float(previous_active.float().mean())})
  if not self.active.any():self._new_block();obs=self.obs
  return obs,reward,dones,extras
 def sync_sampler(self):
  events=self.events;self.events=[]
  if torch.distributed.is_initialized():
   gathered=[None]*torch.distributed.get_world_size();torch.distributed.all_gather_object(gathered,events);events=[e for batch in gathered for e in batch]
  self.sampler.update(events)
 def close(self):self._close_scene()
