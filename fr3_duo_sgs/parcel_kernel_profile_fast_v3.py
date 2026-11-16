"""Actual loaded Duo contact physics capacity probe, not an empty scene estimate."""
import argparse,json,os,time,pathlib,hashlib,subprocess
from datetime import datetime,timezone
p=argparse.ArgumentParser();p.add_argument('--job',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--num-envs',type=int,required=True);p.add_argument('--steps',type=int,default=32)
p.add_argument('--soft-cap-per-env',type=int,default=0);p.add_argument('--body-particle-cap',type=int,default=8192);p.add_argument('--snapshot',action='store_true')
from isaaclab.app import add_launcher_args,launch_simulation
add_launcher_args(p);args=p.parse_args()
import numpy as np,torch,mujoco,newton,warp as wp,collections
import isaaclab.sim as sim_utils
from isaaclab.utils import instantiate
from isaaclab.assets import DeformableObjectCfg
from isaaclab_newton.physics import NewtonCfg,VBDSolverCfg,NewtonSoftContactCfg,NewtonCollisionPipelineCfg,NewtonManager
from isaaclab_newton.physics.vbd_manager import NewtonVBDManager
from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg
from isaaclab_newton.sim.spawners.materials import NewtonSurfaceDeformableBodyMaterialCfg
from newton_cell_cfg import CELL,scene_cfg,physics_cfg
from mailer_surface import pillow_mesh,spawn_mailer
from scaled_reset_io import apply_scaled_rows
args.output.mkdir(parents=True,exist_ok=True);start=time.time();job=json.loads(args.job.read_text());soft=job['kind']=='mailer';n=args.num_envs

def write(name,**kwargs):
 d=dict(state=name,pid=os.getpid(),utc=datetime.now(timezone.utc).isoformat(),elapsed_s=time.time()-start,num_envs=n,kind=job['kind'],**kwargs);q=args.output/'status.tmp';q.write_text(json.dumps(d,indent=2));q.replace(args.output/'status.json');return d
from sorted_contact_vbd_v1 import SortedContactSolverVBD,install_sorted_gather
from newton.solvers import SolverVBD
install_sorted_gather()
from specialized_face_contacts_v3 import install_specialized_faces
install_specialized_faces()
class NewtonCapacityVBD(NewtonVBDManager):
 @classmethod
 def _create_solver(cls,model,solver_cfg):
  return SortedContactSolverVBD(model,**cls._filter_solver_kwargs(SolverVBD,solver_cfg))
 @classmethod
 def create_builder(cls,*a,**kw):
  b=super().create_builder(*a,**kw);original=b.add_usd
  def with_sdf(*aa,**kk):
   result=original(*aa,**kk)
   for s in b.shape_source:
    if hasattr(s,'build_sdf') and getattr(s,'sdf',None) is None:s.build_sdf(max_resolution=32,device=args.device)
   return result
  b.add_usd=with_sdf;return b
try:
 write('constructing');torch.cuda.set_device(args.device);full=np.array(job['size_m']);half=full*np.array([.31,.31,.21]) if soft else full/2
 cfg=physics_cfg(args.device,dense_contacts=True)
 if soft:
  cfg=sim_utils.SimulationCfg(device=args.device,dt=.002,physics=NewtonCfg(solver_cfg=VBDSolverCfg(class_type=NewtonCapacityVBD,iterations=40,rigid_compliant_alm=True,particle_enable_self_contact=True,particle_self_contact_radius=.002,particle_self_contact_margin=.003,rigid_body_contact_buffer_size=256,rigid_body_particle_contact_buffer_size=args.body_particle_cap),soft_contact_cfg=NewtonSoftContactCfg(soft_contact_ke=20000,soft_contact_kd=.01,soft_contact_mu=job['friction']),collision_cfg=NewtonCollisionPipelineCfg(enable_rigid_soft_full_surface_contact=True,soft_contact_max=(n*args.soft_cap_per_env if args.soft_cap_per_env else None)),num_substeps=2));cfg.physics.solver_cfg.rigid_joint_linear_ke=1e7;cfg.physics.solver_cfg.rigid_joint_angular_ke=1e7
 cfg.physics.use_cuda_graph=False
 scfg=scene_cfg(n,tuple(half),job['mass_kg'],job['friction'],contact_sensor=False)
 if soft:
  scfg.mailer=DeformableObjectCfg(prim_path='{ENV_REGEX_NS}/Mailer',spawn=sim_utils.MeshCuboidCfg(func=spawn_mailer,size=tuple(full),edge_refinement=1.,deformable_props=NewtonDeformableBodyPropertiesCfg(),visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.68,.73,.78)),physics_material=NewtonSurfaceDeformableBodyMaterialCfg(density=.25,particle_radius=.002,tri_ke=1500,tri_ka=1500,tri_kd=.001,edge_ke=.02,edge_kd=.001)),init_state=DeformableObjectCfg.InitialStateCfg(pos=(.65,.1,.6)));scfg.mailer.spawn.wrinkle=job['wrinkle_m']
 with launch_simulation(cfg,args):
  sim=sim_utils.SimulationContext(cfg);scene=instantiate(scfg);sim.reset();robot=scene['robot'];model=NewtonManager.get_model()
  print('SHAPE_TYPES',np.unique(model.shape_type.numpy(),return_counts=True),flush=True)
  source=pathlib.Path(job['benchmark_source_shard']);row=job.get('benchmark_source_row',0)
  with np.load(source,allow_pickle=False) as f:
   keys=['robot_q','robot_qd','parcel_pose_xyzw','parcel_twist']+(['nodes','nodal_vel'] if soft else [])
   rows={k:np.broadcast_to(f[k][row],(n,)+f[k][row].shape).copy() for k in keys};nominal=np.broadcast_to(f['robot_q_target'][row],(n,54)).copy()
  apply_scaled_rows(scene,rows,nominal);NewtonManager.forward();target=torch.as_tensor(nominal,device=args.device,dtype=torch.float32);del rows,nominal
  names=robot.joint_names;kp=torch.tensor([[600,600,600,600,250,150,50][int(name[-1])-1] if 'fr3v2_joint' in name else 20 for name in names],device=args.device)
  # Allocate actual on-policy data and an actor/critic optimizer alongside physics.
  # Capacity therefore includes a 32-step 256-observation/54-action float32 rollout.
  horizon=32;rollout=torch.empty((horizon,n,256+54+54+54+5),device=args.device);rollout.zero_()
  actor=torch.nn.Sequential(torch.nn.Linear(256,256),torch.nn.ELU(),torch.nn.Linear(256,256),torch.nn.ELU(),torch.nn.Linear(256,54)).to(args.device)
  critic=torch.nn.Sequential(torch.nn.Linear(256,256),torch.nn.ELU(),torch.nn.Linear(256,256),torch.nn.ELU(),torch.nn.Linear(256,1)).to(args.device);optim=torch.optim.Adam(list(actor.parameters())+list(critic.parameters()),lr=3e-4)
  obs=torch.zeros((n,256),device=args.device);loss=(actor(obs[:min(n,4096)]).square().mean()+critic(obs[:min(n,4096)]).square().mean());loss.backward();optim.step();optim.zero_grad(set_to_none=True)
  torch.cuda.synchronize();free,total=torch.cuda.mem_get_info();used=total-free
  write('stepping',gpu_used_mib=used/2**20,gpu_total_mib=total/2**20,bodies=model.body_count,particles=model.particle_count,rollout_buffer_mib=rollout.numel()*4/2**20)
  step_times=[];contact_peaks={}
  torch.manual_seed(123)
  for i in range(args.steps+8):
   tick=time.perf_counter()
   command=target
   if soft:command=target+robot.data.gravity_compensation_forces.torch/kp
   robot.actuators.target_command.set_position_index(value=command);scene.write_data_to_sim()
   with wp.ScopedTimer('native_step',print=False,synchronize=True,cuda_filter=wp.TIMING_ALL if i>=8 else 0) as timer:
    sim.step()
   if i>=8:
    aggregate=collections.defaultdict(lambda:dict(ms=0.,calls=0))
    for item in timer.timing_results:
     aggregate[item.name]['ms']+=float(item.elapsed);aggregate[item.name]['calls']+=1
    (args.output/f'kernel_profile_{i}.json').write_text(json.dumps(dict(wall_ms=timer.elapsed,kernels=sorted([dict(name=k,**v) for k,v in aggregate.items()],key=lambda x:x['ms'],reverse=True)),indent=2))
   if soft:
    st=NewtonManager.get_state_0();newton.eval_ik(model,st,st.joint_q,st.joint_qd)
   scene.update(.002)
   if soft:
    contacts=NewtonManager.get_contacts();solver=NewtonManager._solver
    for key,obj,attr,limit in [('soft',contacts,'soft_contact_count',contacts.soft_contact_max),('rigid',contacts,'rigid_contact_count',contacts.rigid_contact_max),('body_particle',solver,'body_particle_contact_counts',args.body_particle_cap),('body_body',solver,'body_body_contact_counts',256)]:
     measured=int(getattr(obj,attr).numpy().max());contact_peaks[key]=max(contact_peaks.get(key,0),measured)
     if measured>limit:raise RuntimeError(f'{key} contact overflow: {measured}>{limit}')

   with torch.no_grad():actor(obs)
   torch.cuda.synchronize()
   if i>=8:step_times.append(time.perf_counter()-tick)
   if i%8==0:write('stepping',step=i,gpu_used_mib=(torch.cuda.mem_get_info()[1]-torch.cuda.mem_get_info()[0])/2**20)
  finite=bool(torch.isfinite(robot.data.joint_pos.torch).all()) and bool(torch.isfinite(scene['box'].data.root_pose_w.torch).all())
  if soft:finite &= bool(torch.isfinite(scene['mailer'].data.nodal_pos_w.torch).all())
  if not finite:raise RuntimeError('Nonfinite native simulation')
  # Allocation audit: unique live CUDA Warp allocations, no array payload dumps.
  import warp as wp
  seen=set(); visited=set(); alloc=[]
  def walk(x,path,depth=0):
   if x is None or depth>5:return
   if isinstance(x,wp.array):
    if x.ptr and x.device.is_cuda:
     key=(str(x.device),int(x.ptr))
     if key not in seen:
      seen.add(key);alloc.append(dict(path=path,shape=list(x.shape),dtype=str(x.dtype),bytes=int(x.capacity)))
    return
   if isinstance(x,(str,int,float,bool,np.ndarray,torch.Tensor,type)):return
   if id(x) in visited:return
   visited.add(id(x))
   if isinstance(x,dict):items=x.items()
   elif isinstance(x,(list,tuple)):items=enumerate(x)
   elif hasattr(x,'__dict__'):items=vars(x).items()
   else:return
   for k,v in items:walk(v,path+'.'+str(k),depth+1)
  for label,obj in [('contacts',NewtonManager.get_contacts()),('solver',NewtonManager._solver),('pipeline',NewtonManager._collision_pipeline),('state0',NewtonManager.get_state_0()),('state1',NewtonManager.get_state_1()),('model',model)]:walk(obj,label) if soft and n<=1024 else None
  pipe=NewtonManager._collision_pipeline;counts={}
  for attr in ['soft_contact_max','soft_contact_pair_count','rigid_contact_max']:
   try:counts[attr]=int(getattr(pipe,attr))
   except Exception:pass
  for attr in ['soft_edge_rigid_pairs','soft_face_rigid_pairs']:
   try:counts[attr]=len(getattr(pipe,attr))
   except Exception:pass
  for label,obj,attrs in [('contacts',NewtonManager.get_contacts(),['soft_contact_count','rigid_contact_count']),('solver',NewtonManager._solver,['body_particle_contact_counts','body_body_contact_counts'])]:
   for attr in attrs:
    try:
     a=getattr(obj,attr).numpy();counts[label+'.'+attr]=dict(max=int(a.max()),sum=int(a.sum()))
    except Exception:pass
  (args.output/'allocation_profile.json').write_text(json.dumps(dict(num_envs=n,counts=counts,warp_unique_bytes=sum(a['bytes'] for a in alloc),allocations=sorted(alloc,key=lambda a:a['bytes'],reverse=True)),indent=2))
  if args.snapshot:
   snap=dict(q=robot.data.joint_pos.torch.cpu().numpy(),qd=robot.data.joint_vel.torch.cpu().numpy(),parcel=scene['box'].data.root_pose_w.torch.cpu().numpy())
   if soft:snap['nodes']=scene['mailer'].data.nodal_pos_w.torch.cpu().numpy()
   np.savez_compressed(args.output/'final_state.npz',**snap)
  (args.output/'contact_peaks.json').write_text(json.dumps(contact_peaks,indent=2))
  process_memory=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,capture_output=True).stdout
  process_mib=sum(float(line.split(',')[1]) for line in process_memory.splitlines() if line.split(',')[0].strip()==str(os.getpid()))
  mem=torch.cuda.mem_get_info();out=write('complete',process_gpu_used_mib=process_mib,finite=True,mean_physics_step_s=float(np.mean(step_times)),physics_env_steps_per_s=n/float(np.mean(step_times)),gpu_used_mib=(mem[1]-mem[0])/2**20,gpu_total_mib=mem[1]/2**20,torch_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,rollout_buffer_mib=rollout.numel()*4/2**20,hardware=torch.cuda.get_device_name(),benchmark_source_shard=str(source),same_geometry_copies=True,contact_peaks=contact_peaks,soft_cap_per_env=args.soft_cap_per_env,body_particle_cap=args.body_particle_cap,scope='Capacity probe: full native robot and contact solver, held parcel, actor inference and rollout memory. Not PPO training or heterogeneous-geometry coverage. Shared-GPU timing is contention-dependent.')
except BaseException as e:
 write('failed',exception=repr(e));raise
