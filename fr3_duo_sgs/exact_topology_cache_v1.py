"""Reuse exact native self-contact CSR tables on verified disjoint copies of one topology."""
import pathlib,json,hashlib,numpy as np,warp as wp
from newton._src.geometry.tri_mesh_collision import TriMeshCollisionDetector as NativeDetector
import newton._src.solvers.vbd.solver_vbd as sv
ROOT=pathlib.Path(__file__).parent/'runtime/topology_equivalence_20261116_v1';P=ROOT/'native_topology_template.npz';proof=json.loads((ROOT/'topology_equivalence.json').read_text());assert proof['state']=='pass' and proof['all_native_filter_values_and_offsets_exact'];assert hashlib.sha256(P.read_bytes()).hexdigest()==proof['template_sha256']
with np.load(P) as f:T={k:f[k].copy() for k in f.files}
CACHE={}
class ExactCachedTopologyDetector(NativeDetector):
 def __init__(self,model,*args,**kwargs):
  eligible=not args and kwargs.get('topological_contact_filter_threshold',0)==2 and all(kwargs.get(k) is None for k in ['external_vertex_triangle_filtering_map','external_edge_edge_filtering_map','vertex_triangle_filtering_list','edge_filtering_list'])
  if eligible:
   n=model.particle_count//442;nt=len(T['tri_indices']);ne=len(T['edge_indices']);eligible=n>0 and model.particle_count==n*442 and model.tri_count==n*nt and model.edge_count==n*ne
  if eligible:
   tri=model.tri_indices.numpy().reshape(n,nt,3);edge=model.edge_indices.numpy().reshape(n,ne,4)
   # Validate every world, not only a template or shape count.
   shift=np.arange(n,dtype=np.int32)[:,None,None]*442
   assert np.array_equal(tri-shift,np.broadcast_to(T['tri_indices'],tri.shape)), 'Unexpected triangle topology'
   assert np.array_equal(edge-shift,np.broadcast_to(T['edge_indices'],edge.shape)), 'Unexpected edge topology'
   key=(str(model.device),n)
   if key not in CACHE:
    tables={}
    for prefix,rows,target in [('vertex_triangle',442,nt),('edge',ne,ne)]:
     base=T[prefix+'_values'];offset=T[prefix+'_offsets'];length=len(base)
     values=(base[None,:]+np.arange(n,dtype=np.int32)[:,None]*target).reshape(-1)
     offsets=np.concatenate([(offset[:-1][None,:]+np.arange(n,dtype=np.int32)[:,None]*length).reshape(-1),[n*length]]).astype(np.int32)
     tables[prefix+'_filtering_list']=wp.array(values,dtype=wp.int32,device=model.device);tables[prefix+'_filtering_list_offsets']=wp.array(offsets,dtype=wp.int32,device=model.device)
    CACHE[key]=tables
   kwargs.update(CACHE[key]);print(f'EXACT_TOPOLOGY_CSR_CACHE worlds={n} all_world_topology_verified=True',flush=True)
  super().__init__(model,*args,**kwargs)
def install_exact_topology_cache():sv.TriMeshCollisionDetector=ExactCachedTopologyDetector
