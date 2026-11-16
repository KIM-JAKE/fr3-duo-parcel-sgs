"""RSL-RL PPO + SGS manipulation kernel; exact native geometry batches.

This is a task adaptation, not a verbatim SGS benchmark reproduction. No RGB
student/distillation or demonstration collection is launched here.
"""
import argparse,pathlib,json,os,time,hashlib,signal
from datetime import datetime,timezone,timedelta
from isaaclab.app import add_launcher_args
p=argparse.ArgumentParser();p.add_argument('--index',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--num-envs',type=int,default=1024);p.add_argument('--iterations',type=int,default=100000);p.add_argument('--episode-steps',type=int,default=256);p.add_argument('--horizon',type=int,default=32);p.add_argument('--save-every',type=int,default=25);p.add_argument('--resume',type=pathlib.Path);p.add_argument('--diagnostic',action='store_true');p.add_argument('--reserve-horizon',type=int,default=0);p.add_argument('--box-envs',type=int,default=16384);p.add_argument('--mailer-envs',type=int,default=4096);p.add_argument('--mailer-shards',type=int,default=4);add_launcher_args(p);args=p.parse_args()
import torch,numpy as np
from rsl_rl.algorithms import PPO
from parcel_mixed_sgs_env_v3 import MixedParcelSGSEnv as ParcelSGSEnv
rank=int(os.environ.get('RANK',0));local=int(os.environ.get('LOCAL_RANK',0));world=int(os.environ.get('WORLD_SIZE',1));args.device=f'cuda:{local}';torch.cuda.set_device(local);torch.set_num_threads(2);torch.manual_seed(3141+rank);np.random.seed(3141+rank)
if world>1:torch.distributed.init_process_group('nccl',device_id=torch.device(args.device),timeout=timedelta(hours=2))
args.output.mkdir(parents=True,exist_ok=True);env=None;start=time.time();last_iteration=-1

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic(path,d):
 q=path.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2,default=str));q.replace(path)
def status(state,**more):
 d=dict(state=state,utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),rank=rank,world_size=world,num_envs=args.num_envs,elapsed_s=time.time()-start,iteration=last_iteration);d.update(more);atomic(args.output/f'rank_{rank}_status.json',d)
def checkpoint(alg,it):
 file=args.output/f'checkpoint_{it:08d}_rank{rank}.pt';temp=file.with_suffix('.tmp');torch.save(dict(algorithm=alg.save(),iteration=it,sampler=env.sampler.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),index_sha256=sha(args.index/'configurations.npz'),config=cfg),temp);temp.replace(file)
 if world>1:torch.distributed.barrier()
 if rank==0:
  files=[args.output/f'checkpoint_{it:08d}_rank{r}.pt' for r in range(world)];atomic(args.output/'latest_checkpoint.json',dict(iteration=it,files=[dict(path=str(p.resolve()),sha256=sha(p)) for p in files],world_size=world))
 if world>1:torch.distributed.barrier()

try:
 manifest=json.loads((args.index/'manifest.json').read_text())
 if not args.diagnostic:assert not manifest['diagnostic'] and manifest['configuration_count']==200000
 assert sha(args.index/'configurations.npz')==manifest['configurations_sha256']
 cfg=dict(num_steps_per_env=args.horizon,obs_groups={'actor':['policy'],'critic':['policy']},actor=dict(class_name='MLPModel',hidden_dims=[512,256,128],activation='elu',obs_normalization=False,distribution_cfg=dict(class_name='GaussianDistribution',init_std=.5,std_type='log',std_range=[.01,1.])),critic=dict(class_name='MLPModel',hidden_dims=[512,256,128],activation='elu',obs_normalization=False),algorithm=dict(class_name='PPO',num_learning_epochs=4,num_mini_batches=4,clip_param=.2,gamma=.99,lam=.95,value_loss_coef=1.,entropy_coef=.005,learning_rate=3e-4,max_grad_norm=1.,use_clipped_value_loss=True,schedule='adaptive',desired_kl=.01),multi_gpu=dict(global_rank=rank,local_rank=local,world_size=world) if world>1 else None,torch_compile_mode=None)
 if rank==0:
  atomic(args.output/'run_manifest.json',dict(utc=datetime.now(timezone.utc).isoformat(),config=cfg,index=str(args.index.resolve()),index_sha256=sha(args.index/'configurations.npz'),configuration_count=manifest['configuration_count'],gpus=os.environ.get('CUDA_VISIBLE_DEVICES'),num_envs_per_rank=args.num_envs,world_size=world,total_simultaneous_envs=args.num_envs*world,source_hashes={p.name:sha(p) for p in [pathlib.Path(__file__),pathlib.Path(__file__).with_name('parcel_mixed_sgs_env_v3.py'),pathlib.Path(__file__).with_name('parcel_sgs_rpc_worker_v3.py'),pathlib.Path(__file__).with_name('parcel_fast_scene_v3.py'),pathlib.Path(__file__).with_name('sorted_contact_vbd_v1.py'),pathlib.Path(__file__).with_name('parcel_sgs_env.py'),pathlib.Path(__file__).with_name('parcel_native_scene.py'),pathlib.Path(__file__).with_name('parcel_rl_math.py'),pathlib.Path(__file__).with_name('sgs_sampler.py')]},scope='Mixed shared PPO with fixed 50/50 kind strata and native independent worker processes; geometry selected by total SGS weight, exact native geometry reset sampled conditionally, synchronous native episode batches with absorbing terminated worlds. Not paper i.i.d global reset schedule. Privileged teacher only; no visual-policy claim.',reward='terminal success100 + fail-1 + action/action-rate/joint-velocity regularization from SGS manipulation, plus gamma*Phi(next)-Phi(now), Phi=label-up minus normalized distance to full conveyor support region. Zero potential at true terminals, timeout bootstrap retained.',success='task_success.py full output footprint and support-height tolerance, label-up >=0.8, held for 3 consecutive policy steps; no grasp method reward.',resumable='Use --resume latest_checkpoint.json; all rank files verified SHA, optimizer and SGS rolling history restored. Episodes restart from pinned pool; no claim of exact trajectory bitwise continuation.',diagnostic=args.diagnostic))
 status('constructing_environment');env=ParcelSGSEnv(args.index,args,args.num_envs,rank,args.episode_steps);obs=env.get_observations();status('constructing_ppo',geometry_id=env.geometry_id)
 alg=PPO.construct_algorithm(obs,env,cfg,args.device);alg.train_mode()
 memory_reserve=torch.zeros((max(0,args.reserve_horizon-args.horizon),args.num_envs,423),device=args.device)
 status('optimizer_ready',reserved_rollout_mib=memory_reserve.numel()*4/2**20)
 if world>1:alg.broadcast_parameters()
 first=0
 if args.resume:
  resume=json.loads(args.resume.read_text());assert resume['world_size']==world;item=resume['files'][rank];f=pathlib.Path(item['path']);assert sha(f)==item['sha256'];d=torch.load(f,map_location=args.device,weights_only=False);assert d['index_sha256']==sha(args.index/'configurations.npz');alg.load(d['algorithm'],None,True);env.sampler.load_state_dict(d['sampler']);torch.set_rng_state(d['torch_rng'].cpu());torch.cuda.set_rng_state(d['cuda_rng'].cpu());first=d['iteration']+1
 for it in range(first,args.iterations):
  last_iteration=it;tick=time.time();rollout_success=0.;rollout_failure=0.;label=[];distance=[]
  with torch.inference_mode():
   for step in range(args.horizon):
    actions=alg.act(obs);obs,reward,done,extras=env.step(actions)
    if not torch.isfinite(reward).all() or not torch.isfinite(obs['policy']).all():raise RuntimeError('Nonfinite PPO inputs')
    alg.process_env_step(obs,reward,done,extras);rollout_success+=extras['log']['/success'];rollout_failure+=extras['log']['/failure'];label.append(extras['log']['/label_up']);distance.append(extras['log']['/goal_distance_m']);status('collecting',rollout_step=step,geometry_id=env.geometry_id,active_steps=env.active_steps,total_physics_env_steps=env.total_steps*10)
   alg.compute_returns(obs)
  losses=alg.update();env.sync_sampler()
  if any(not np.isfinite(float(v)) for v in losses.values()):raise RuntimeError('Nonfinite PPO update')
  metrics=dict(iteration=it,utc=datetime.now(timezone.utc).isoformat(),seconds=time.time()-tick,successes=rollout_success,failures=rollout_failure,label_up=float(np.mean(label)),goal_distance_m=float(np.mean(distance)),losses={k:float(v) for k,v in losses.items()},learning_rate=alg.learning_rate,active_steps=env.active_steps,total_env_steps=env.total_steps,geometry_count_seen=len(env.geometry_seen),native_geometry_id=env.geometry_id)
  with (args.output/f'rank_{rank}_metrics.jsonl').open('a') as stream:stream.write(json.dumps(metrics)+'\n')
  if it%args.save_every==0 or it==args.iterations-1:checkpoint(alg,it)
  status('learning',**{k:v for k,v in metrics.items() if k!='iteration'})
 status('complete')
except BaseException as error:
 status('failed',exception=repr(error));raise
finally:
 if env is not None:env.close()
 if torch.distributed.is_initialized():torch.distributed.destroy_process_group()
