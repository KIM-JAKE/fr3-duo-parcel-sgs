"""Shared-policy vector environment: half rigid, half deformable via native RPC workers.

Fixed kind strata adapt SGS sampling: within each kind, geometry-weighted SGS
sampling and one globally synchronized outcome history remain active.
"""
import pathlib,json,os,subprocess,socket,time,shutil,sys
from multiprocessing.connection import Listener
import numpy as np,torch
from tensordict import TensorDict
from sgs_sampler import SuccessGuidedSampler
ROOT=pathlib.Path(__file__).resolve().parents[1]
class RemoteSampler(SuccessGuidedSampler):
 def __init__(self,owner,ids,seed):super().__init__(ids,seed=seed);self.owner=owner
 def state_dict(self):
  state=super().state_dict()
  if self.owner.connections:state['worker_rng']=self.owner._broadcast('sampler_rng',None)
  return state
 def load_state_dict(self,state):
  super().load_state_dict(state)
  if self.owner.connections:
   assert len(state['worker_rng'])==len(self.owner.connections)
   for i,c in enumerate(self.owner.connections):
    local=dict(state);local['rng']=state['worker_rng'][i];c.send(('sampler_load',local))
   for i in range(len(self.owner.connections)):self.owner._receive(i)
class MixedParcelSGSEnv:
 num_actions=54
 def __init__(self,index,args,num_envs,rank=0,episode_steps=160):
  self.args=args;self.index=pathlib.Path(index);self.num_envs=num_envs;self.rank=rank;self.device=args.device;self.max_episode_length=episode_steps;self.children=[];self.connections=[];self.logs=[];self.events=[];self.geometry_seen=set();self.geometry_id=-1;self.active_steps=0;self.total_steps=0
  self.cfg=dict(task='FR3 Duo mixed parcels',obs='native 256D privileged teacher',sampler='SGS within fixed 50/50 box/mailer strata, homogeneous geometry per independent simulator',physics='Native VBD40 dt.002 two substeps, full deformable mesh/material/self and vertex/edge/face contacts. Sorted exact contact traversal. Allocation overflow fatal.')
  self.sizes=[args.box_envs]+[args.mailer_envs]*args.mailer_shards;self.kinds=['box']+['mailer']*args.mailer_shards;assert sum(self.sizes)==num_envs and args.box_envs==args.mailer_envs*args.mailer_shards
  self.episode_length_buf=torch.zeros(num_envs,device=self.device,dtype=torch.long)
  with np.load(self.index/'configurations.npz') as f:ids=f['geometry_id'].copy()
  self.manifest=json.loads((self.index/'manifest.json').read_text())
  self.sampler=RemoteSampler(self,ids,3100+rank);self.socket_root=pathlib.Path('/tmp')/f'parcel32k_{os.getpid()}_{rank}';self.socket_root.mkdir(mode=0o700);out=args.output/f'workers_rank{rank}';out.mkdir(parents=True,exist_ok=True)
  with np.load(self.index/'configurations.npz',allow_pickle=False) as rows, np.load(self.index/'ng_free_flags.npz',allow_pickle=False) as flags:
   kinds=np.array([self.manifest['geometries'][int(g)]['kind'] for g in rows['geometry_id']])
   self.sampler.configure_ng_budget(rows['family'],flags['free_success'],kinds,target=.25)
  try:
   for i,(kind,n) in enumerate(zip(self.kinds,self.sizes)):
    address=str(self.socket_root/f'{i}.sock');listener=Listener(address,family='AF_UNIX');listener._listener._socket.settimeout(300);log=(out/f'worker_{i}.log').open('a');self.logs.append(log)
    command=[sys.executable,'-u',str(ROOT/'fr3_duo_sgs/parcel_sgs_rpc_worker_v3.py'),'--socket',address,'--status',str(out/f'worker_{i}_status.json')]
    child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT);self.children.append(child);connection=listener.accept();listener.close();self.connections.append(connection)
    config=dict(index=str(self.index),kind=kind,num_envs=n,rank=rank,episode_steps=episode_steps,device=self.device,worker_seed=rank*100+i,args=json.loads(json.dumps(vars(args),default=str)));connection.send(config)
   (out/'launches.json').write_text(json.dumps([dict(pid=c.pid,kind=k,num_envs=n) for c,k,n in zip(self.children,self.kinds,self.sizes)],indent=2))
   initial=[self._receive(i,timeout=2700) for i in range(len(self.connections))];self.obs=self._merge_observation(initial)
  except BaseException:self.close();raise
 def _receive(self,i,timeout=1800):
  conn=self.connections[i];start=time.time()
  while not conn.poll(1):
   if self.children[i].poll() is not None:raise RuntimeError(f'Native worker {i} exited {self.children[i].returncode}')
   if time.time()-start>timeout:raise RuntimeError(f'Native worker {i} RPC timeout')
  result=conn.recv()
  if not result['ok']:raise RuntimeError(result['error'])
  return result['payload']
 def _broadcast(self,command,payload):
  for c in self.connections:c.send((command,payload))
  return [self._receive(i) for i in range(len(self.connections))]
 def _merge_observation(self,records):
  obs=torch.as_tensor(np.concatenate([x['obs'] for x in records]),device=self.device);self.episode_length_buf.copy_(torch.as_tensor(np.concatenate([x['episode_length'] for x in records]),device=self.device));self.active_steps=sum(x['active_steps'] for x in records);self.total_steps=sum(x['total_steps'] for x in records)
  for x in records:self.geometry_seen.update(x['geometry_seen'])
  return TensorDict({'policy':obs},batch_size=[self.num_envs])
 def get_observations(self):return self.obs
 def step(self,actions):
  a=actions.detach().cpu().numpy();offset=0
  for c,n in zip(self.connections,self.sizes):c.send(('step',a[offset:offset+n]));offset+=n
  records=[self._receive(i) for i in range(len(self.connections))];self.obs=self._merge_observation(records)
  for x in records:self.events.extend(x['events'])
  tensor=lambda key:torch.as_tensor(np.concatenate([x[key] for x in records]),device=self.device)
  log={}
  for key in records[0]['log']:
   if key in ['/success','/failure']:log[key]=sum(x['log'][key] for x in records)
   else:log[key]=sum(x['log'][key]*n for x,n in zip(records,self.sizes))/self.num_envs
  return self.obs,tensor('reward'),tensor('done'),dict(time_outs=tensor('timeouts'),log=log)
 def sync_sampler(self):
  events=self.events;self.events=[]
  if torch.distributed.is_initialized():
   gathered=[None]*torch.distributed.get_world_size();torch.distributed.all_gather_object(gathered,events);events=[e for batch in gathered for e in batch]
  self.sampler.update(events);self._broadcast('events',events)
 def close(self):
  for c,child in zip(self.connections,self.children):
   try:
    if child.poll() is None:c.send(('close',None))
   except Exception:pass
  for i,c in enumerate(self.connections):
   try:
    if c.poll(5):c.recv()
   except Exception:pass
   c.close()
  for c in self.children:
   if c.poll() is None:
    try:c.wait(timeout=30)
    except subprocess.TimeoutExpired:c.terminate()
   if c.poll() is None:
    try:c.wait(timeout=15)
    except subprocess.TimeoutExpired:c.kill();c.wait()
  self.connections=[]
  for log in self.logs:log.close()
  if hasattr(self,'socket_root') and self.socket_root.exists():shutil.rmtree(self.socket_root)
