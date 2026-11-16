"""Portable supervised launcher; run this process under tmux."""
import pathlib,subprocess,os,argparse,json,time,sys,signal,hashlib
root=pathlib.Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--gpus',default='0,1,2,3');p.add_argument('--mailer-envs',type=int,default=4096);p.add_argument('--output',type=pathlib.Path,default=pathlib.Path('runs/sgs'));p.add_argument('--iterations',type=int,default=10000);p.add_argument('--horizon',type=int,default=32);p.add_argument('--episode-steps',type=int,default=256);p.add_argument('--diagnostic',action='store_true');a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True);gpus=a.gpus.split(',');n=a.mailer_envs;assert n>0 and len(set(gpus))==len(gpus);child=None
import fcntl
lock=(a.output/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def atomic(d):
 q=a.output/'supervisor.tmp';q.write_text(json.dumps(dict(pid=os.getpid(),unix_s=time.time(),**d),indent=2));q.replace(a.output/'supervisor.json')
def checkpoint():
 f=a.output/'latest_checkpoint.json'
 if not f.exists():return None
 d=json.loads(f.read_text());assert d['world_size']==len(gpus)
 for r in d['files']:
  path=pathlib.Path(r['path']);assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256']
 return f
cmd=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc-per-node='+str(len(gpus)),str(root/'fr3_duo_sgs/train_mixed_sgs_ppo_v3.py'),'--index',str(root/'data_index'),'--output',str(a.output),'--num-envs',str(8*n),'--box-envs',str(4*n),'--mailer-envs',str(n),'--mailer-shards','4','--iterations',str(a.iterations),'--horizon',str(a.horizon),'--episode-steps',str(a.episode_steps),'--save-every','1']
if a.diagnostic:cmd+=['--diagnostic']
resume=checkpoint()
if resume:cmd+=['--resume',str(resume)]
env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpus,OMP_NUM_THREADS='2',MUJOCO_GL='osmesa',PYOPENGL_PLATFORM='osmesa',WARP_CACHE_PATH=str(root/'.cache/warp'))
def stop(sig,frame):raise KeyboardInterrupt
signal.signal(signal.SIGTERM,stop)
try:
 with (a.output/'train.log').open('a') as log:
  child=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  (a.output/'launch.json').write_text(json.dumps(dict(command=cmd,child_pid=child.pid,physical_gpus=gpus,envs_per_gpu=8*n,ng_free_target=.25,resume=str(resume) if resume else None),indent=2))
  while child.poll() is None:
   ranks=[]
   for f in a.output.glob('rank_*_status.json'):
    d=json.loads(f.read_text());ranks.append(d);assert d['state']!='failed',d.get('exception');assert time.time()-f.stat().st_mtime<1800,'Stale worker '+str(f)
   atomic(dict(state='running',child_pid=child.pid,ranks=ranks));time.sleep(10)
  assert child.returncode==0,'torchrun exit '+str(child.returncode)
 assert checkpoint() is not None;atomic(dict(state='complete'))
except BaseException as e:atomic(dict(state='failed',error=repr(e)));raise
finally:
 if child is not None and child.poll() is None:
  import psutil
  owned=psutil.Process(child.pid).children(recursive=True);os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=45)
  except subprocess.TimeoutExpired:
   for x in owned:
    try:x.kill()
    except psutil.NoSuchProcess:pass
   child.kill();child.wait()
