"""CPU validation of assets, installed native source, data IDs and NG sampling budget."""
import pathlib,sys,json,hashlib,numpy as np,importlib.metadata as md
root=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'fr3_duo_sgs'));from sgs_sampler import SuccessGuidedSampler
import newton,mujoco
native=pathlib.Path(newton.__file__).parent.parent
for name,h in json.loads((root/'upstream_native_hashes.json').read_text()).items():assert hashlib.sha256((native/name).read_bytes()).hexdigest()==h,'Native Newton source differs: '+name
idx=root/'data_index';m=json.loads((idx/'manifest.json').read_text());f=np.load(idx/'configurations.npz');flags=np.load(idx/'ng_free_flags.npz');kind=np.array([m['geometries'][int(g)]['kind'] for g in f['geometry_id']]);s=SuccessGuidedSampler(f['geometry_id']);s.configure_ng_budget(f['family'],flags['free_success'],kind)
report={}
for trial in ['fresh','mixed_history']:
 if trial=='mixed_history':
  rng=np.random.default_rng(91);s.history[:]=rng.integers(0,2,size=s.history.shape,dtype=np.uint8);s.count[:]=s.history.shape[1]
 w=s.weights()
 for k in ['box','mailer']:
  mask=(f['family']=='NG')&(kind==k);actual=float(w[mask&flags['free_success']].sum()/w[mask].sum());assert abs(actual-.25)<1e-12;report[trial+'_'+k]=actual
 assert np.isfinite(w).all() and (w>0).all()
cell=json.loads((root/'fr3_duo_sgs/cell_config.json').read_text());model=mujoco.MjModel.from_xml_path(cell['robot']['urdf']);assert model.nq==54
from pxr import Usd
usd=root/'fr3_duo_sgs/runtime/native_duo_usd/official_fr3_duo_dg5fm/official_fr3_duo_dg5fm.usda';stage=Usd.Stage.Open(str(usd));assert stage;prim=stage.GetDefaultPrim();prim.GetVariantSet('Physics').SetVariantSelection('mujoco');assert sum(x.GetTypeName()=='PhysicsRevoluteJoint' for x in stage.Traverse())==54
out=dict(state='pass',configuration_count=len(f['row']),ng_free_sampling_probability=report,robot_dof=model.nq,scope='CPU paths/assets/native SHA and sampling invariants; no fresh-server GPU rollout claimed',packages={k:md.version(k) for k in ['torch','newton','warp-lang','rsl-rl-lib']});(root/'preflight.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
