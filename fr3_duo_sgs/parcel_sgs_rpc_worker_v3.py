"""One native simulator process; CPU RPC keeps Newton singleton worlds isolated."""
import argparse,pathlib,json,os,traceback,time
from types import SimpleNamespace
from multiprocessing.connection import Client
p=argparse.ArgumentParser();p.add_argument('--socket',required=True);p.add_argument('--status',type=pathlib.Path,required=True);a=p.parse_args();conn=Client(a.socket,family='AF_UNIX');config=conn.recv();env=None;started=time.time()
def status(state,**kw):
 q=a.status.with_suffix('.tmp');q.write_text(json.dumps(dict(state=state,pid=os.getpid(),time=time.time(),elapsed_s=time.time()-started,**kw),indent=2));q.replace(a.status)
try:
 status('importing',config=config)
 import torch,numpy as np,warp as wp
 torch.set_num_threads(2);torch.cuda.set_device(config['device']);torch.manual_seed(7000+config['worker_seed']);np.random.seed(7000+config['worker_seed'])
 import parcel_sgs_env as base
 from parcel_fast_scene_v3 import configuration
 base.configuration=configuration
 from isaaclab_newton.physics import NewtonManager
 class CheckedEnv(base.ParcelSGSEnv):
  def _new_block(self):
   super()._new_block()
   if self.soft:
    assert self.model.particle_count==442*self.num_envs and self.model.tri_count==880*self.num_envs
    original=self.sim.step
    def checked_step(*args,**kwargs):
     result=original(*args,**kwargs);contacts=NewtonManager.get_contacts();solver=NewtonManager._solver
     if int(contacts.soft_contact_count.numpy()[0])>contacts.soft_contact_max:raise RuntimeError('Soft contact capacity overflow')
     if int(contacts.rigid_contact_count.numpy()[0])>contacts.rigid_contact_max:raise RuntimeError('Rigid contact capacity overflow')
     if int(wp.to_torch(solver.body_particle_contact_counts).max())>4096:raise RuntimeError('Per-body soft contact capacity overflow')
     if int(wp.to_torch(solver.body_body_contact_counts).max())>256:raise RuntimeError('Per-body rigid contact capacity overflow')
     info=solver.trimesh_collision_detector.collision_info
     for key in ['vertex_colliding_triangles','triangle_colliding_vertices','edge_colliding_edges']:
      count=getattr(info,key+'_count');capacity=getattr(info,key+'_buffer_sizes')
      if count is not None and capacity is not None and count.size and capacity.size:
       if bool((wp.to_torch(count)>wp.to_torch(capacity)).any()):raise RuntimeError('Self contact capacity overflow '+key)
     return result
    self.sim.step=checked_step
 args=argparse.Namespace(**config['args']);args.device=config['device'];env=CheckedEnv(config['index'],args,config['num_envs'],config['rank'],config['episode_steps']);env.sampler.rng=np.random.default_rng(7100+config['worker_seed'])
 allowed=np.array([s['kind']==config['kind'] for s in env.manifest['geometries']])
 def sample_kind_geometry():
  weights=env.sampler.weights();mass=np.bincount(env.rows['geometry_id'],weights=weights,minlength=len(allowed));mass[~allowed]=0;return int(env.sampler.rng.choice(len(mass),p=mass/mass.sum()))
 env.sampler.sample_geometry=sample_kind_geometry
 status('constructing',kind=config['kind'],num_envs=env.num_envs)
 def observation():
  return dict(obs=env.get_observations()['policy'].detach().cpu().numpy(),episode_length=env.episode_length_buf.cpu().numpy(),geometry_id=env.geometry_id,geometry_seen=list(env.geometry_seen),active_steps=env.active_steps,total_steps=env.total_steps)
 initial=observation();conn.send(dict(ok=True,payload=initial));status('ready',geometry_id=env.geometry_id,num_envs=env.num_envs)
 while True:
  command,payload=conn.recv()
  if command=='close':conn.send(dict(ok=True,payload=None));break
  if command=='get':conn.send(dict(ok=True,payload=observation()))
  elif command=='events':env.sampler.update(payload);conn.send(dict(ok=True,payload=None))
  elif command=='sampler_rng':conn.send(dict(ok=True,payload=env.sampler.rng.bit_generator.state))
  elif command=='sampler_load':env.sampler.load_state_dict(payload);conn.send(dict(ok=True,payload=None))
  elif command=='step':
   status('stepping',geometry_id=env.geometry_id,block_step=env.block_step)
   obs,reward,done,extras=env.step(torch.as_tensor(payload,device=env.device));result=observation();result.update(reward=reward.detach().cpu().numpy(),done=done.detach().cpu().numpy(),timeouts=extras['time_outs'].detach().cpu().numpy(),log=extras['log'],events=env.events);env.events=[];conn.send(dict(ok=True,payload=result));status('ready',geometry_id=env.geometry_id,block_step=env.block_step,active_steps=env.active_steps,total_steps=env.total_steps)
  else:raise ValueError(command)
 status('complete')
except BaseException as e:
 status('failed',exception=repr(e),traceback=traceback.format_exc())
 try:conn.send(dict(ok=False,error=traceback.format_exc()))
 except Exception:pass
 raise
finally:
 if env is not None:env.close()
 conn.close()
