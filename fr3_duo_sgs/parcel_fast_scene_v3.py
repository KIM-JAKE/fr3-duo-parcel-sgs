"""Native material/contact physics, sorted contact traversal; allocation caps fail closed."""
import pathlib,json,hashlib
from parcel_native_scene import configuration as native_configuration,NewtonTrainingVBD
from newton.solvers import SolverVBD
from sorted_contact_vbd_v1 import SortedContactSolverVBD,install_sorted_gather
PROOF=pathlib.Path(__file__).parent/'runtime/sorted_equivalence_20261116_v1/kernel_equivalence.json'
proof=json.loads(PROOF.read_text());assert proof['state']=='pass'
assert proof['implementation_sha256']==hashlib.sha256(pathlib.Path(__file__).with_name('sorted_contact_vbd_v1.py').read_bytes()).hexdigest()
assert all(x['bitwise_equal'] for x in proof['metrics'].values())
install_sorted_gather()
from exact_topology_cache_v1 import install_exact_topology_cache
install_exact_topology_cache()
class NewtonSortedTrainingVBD(NewtonTrainingVBD):
 @classmethod
 def _create_solver(cls,model,solver_cfg):return SortedContactSolverVBD(model,**cls._filter_solver_kwargs(SolverVBD,solver_cfg))
def configuration(spec,n,device):
 cfg,scene=native_configuration(spec,n,device)
 if spec['kind']=='mailer':
  s=cfg.physics.solver_cfg;assert s.iterations==40 and s.particle_enable_self_contact
  assert cfg.dt==.002 and cfg.physics.num_substeps==2
  assert cfg.physics.collision_cfg.enable_rigid_soft_full_surface_contact
  s.class_type=NewtonSortedTrainingVBD;s.rigid_body_particle_contact_buffer_size=4096
  cfg.physics.collision_cfg.soft_contact_max=n*8192
 return cfg,scene

# Specialized face evaluation plus exact fixed-point shortcut: frozen geometric outputs are bitwise equal.
face_root=pathlib.Path(__file__).parent/'runtime/face_equivalence_20261116_v2'
face_proof=json.loads((face_root/'READY.json').read_text());assert face_proof['state']=='pass' and face_proof['cases']==4
face_sha=hashlib.sha256(pathlib.Path(__file__).with_name('specialized_face_contacts_v3.py').read_bytes()).hexdigest()
for case in face_proof['results']:
 assert case['proof']['implementation_sha256']==face_sha
 assert all(m['bitwise_equal'] for m in case['proof']['metrics'].values())
from specialized_face_contacts_v3 import install_specialized_faces
install_specialized_faces()
