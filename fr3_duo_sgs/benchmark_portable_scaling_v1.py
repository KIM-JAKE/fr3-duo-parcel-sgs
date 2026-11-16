"""Unchanged production physics, isolated GPU scaling measurement."""
import pathlib,json,time,os,argparse
from isaaclab.app import add_launcher_args,launch_simulation
p=argparse.ArgumentParser();p.add_argument('--n',type=int,required=True);p.add_argument('--job',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);add_launcher_args(p);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
def status(state,**kw):
 d=dict(state=state,pid=os.getpid(),unix_s=time.time(),num_envs=a.n);d.update(kw);q=a.output/'status.tmp';q.write_text(json.dumps(d,indent=2));q.replace(a.output/'status.json')
status('importing')
import torch,numpy as np,newton,warp as wp
import isaaclab.sim as sim_utils
from isaaclab.utils import instantiate
from isaaclab_newton.physics import NewtonManager
from parcel_fast_scene_v3 import configuration
from scaled_reset_io import apply_scaled_rows
try:
 torch.set_num_threads(2);torch.cuda.set_device(a.device);j=json.loads(a.job.read_text());t=time.time();cfg,scfg=configuration(j,a.n,a.device)
 with launch_simulation(cfg,a):
  sim=sim_utils.SimulationContext(cfg);scene=instantiate(scfg);sim.reset();robot=scene['robot'];model=NewtonManager.get_model();status('loading',init_s=time.time()-t)
  with np.load(j['benchmark_source_shard'],allow_pickle=False) as f:
   r=j['benchmark_source_row'];rows={k:np.broadcast_to(f[k][r],(a.n,)+f[k][r].shape).copy() for k in ['robot_q','robot_qd','parcel_pose_xyzw','parcel_twist','nodes','nodal_vel']};target=torch.as_tensor(np.broadcast_to(f['robot_q_target'][r],(a.n,54)).copy(),device=a.device)
  apply_scaled_rows(scene,rows,target.cpu().numpy());NewtonManager.forward();kp=torch.tensor([[600,600,600,600,250,150,50][int(n[-1])-1] if 'fr3v2_joint' in n else 20 for n in robot.joint_names],device=a.device);times=[]
  for i in range(16):
   torch.cuda.synchronize();t=time.perf_counter();robot.actuators.target_command.set_position_index(value=target+robot.data.gravity_compensation_forces.torch/kp);scene.write_data_to_sim();sim.step();s=NewtonManager.get_state_0();newton.eval_ik(model,s,s.joint_q,s.joint_qd);scene.update(.002);torch.cuda.synchronize();dt=time.perf_counter()-t
   contacts=NewtonManager.get_contacts();assert int(contacts.soft_contact_count.numpy()[0])<=contacts.soft_contact_max;assert int(contacts.rigid_contact_count.numpy()[0])<=contacts.rigid_contact_max
   if i>=8:times.append(dt)
   status('stepping',step=i,step_s=dt)
  assert torch.isfinite(robot.data.joint_pos.torch).all() and torch.isfinite(scene['mailer'].data.nodal_pos_w.torch).all()
  d=dict(state='pass',num_envs=a.n,mean_physics_step_s=float(np.mean(times)),physics_env_steps_per_s=a.n/float(np.mean(times)),control_steps_per_s=a.n/float(np.mean(times))/10,peak_gpu_mib=(torch.cuda.mem_get_info()[1]-torch.cuda.mem_get_info()[0])/2**20,physics='442 nodes,880 triangles,VBD40,dt.002,2substeps,full surface/self contact; production fast scene unchanged',scope='Single isolated homogeneous mailer simulator,held controls; not full mixed PPO',source=j['benchmark_source_shard']);(a.output/'result.json').write_text(json.dumps(d,indent=2));status('complete',**{k:v for k,v in d.items() if k!='state'})
except BaseException as e:status('failed',error=repr(e));raise
