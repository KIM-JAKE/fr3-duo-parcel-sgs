"""Training scene factory matching the reset campaign's native solver settings."""
import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.assets import DeformableObjectCfg
from isaaclab_newton.physics import NewtonCfg,VBDSolverCfg,NewtonSoftContactCfg,NewtonCollisionPipelineCfg
from isaaclab_newton.physics.vbd_manager import NewtonVBDManager
from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg
from isaaclab_newton.sim.spawners.materials import NewtonSurfaceDeformableBodyMaterialCfg
from newton_cell_cfg import scene_cfg,physics_cfg
from mailer_surface import spawn_mailer
class NewtonTrainingVBD(NewtonVBDManager):
 device='cuda:0'
 @classmethod
 def create_builder(cls,*a,**kw):
  b=super().create_builder(*a,**kw);original=b.add_usd
  def with_sdf(*aa,**kk):
   result=original(*aa,**kk)
   for s in b.shape_source:
    if hasattr(s,'build_sdf') and getattr(s,'sdf',None) is None:s.build_sdf(max_resolution=32,device=cls.device)
   return result
  b.add_usd=with_sdf;return b

def configuration(spec,n,device):
 soft=spec['kind']=='mailer';full=np.array(spec['size_m']);half=full*np.array([.31,.31,.21]) if soft else full/2
 cfg=physics_cfg(device,dense_contacts=True);NewtonTrainingVBD.device=device
 if soft:
  cfg=sim_utils.SimulationCfg(device=device,dt=.002,physics=NewtonCfg(solver_cfg=VBDSolverCfg(class_type=NewtonTrainingVBD,iterations=40,rigid_compliant_alm=True,particle_enable_self_contact=True,particle_self_contact_radius=.002,particle_self_contact_margin=.003,rigid_body_contact_buffer_size=256,rigid_body_particle_contact_buffer_size=8192),soft_contact_cfg=NewtonSoftContactCfg(soft_contact_ke=20000,soft_contact_kd=.01,soft_contact_mu=spec['friction']),collision_cfg=NewtonCollisionPipelineCfg(enable_rigid_soft_full_surface_contact=True),num_substeps=2));cfg.physics.solver_cfg.rigid_joint_linear_ke=1e7;cfg.physics.solver_cfg.rigid_joint_angular_ke=1e7
 scfg=scene_cfg(n,tuple(half),spec['mass_kg'],spec['friction'],contact_sensor=False)
 if soft:
  scfg.mailer=DeformableObjectCfg(prim_path='{ENV_REGEX_NS}/Mailer',spawn=sim_utils.MeshCuboidCfg(func=spawn_mailer,size=tuple(full),edge_refinement=1.,deformable_props=NewtonDeformableBodyPropertiesCfg(),visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.68,.73,.78)),physics_material=NewtonSurfaceDeformableBodyMaterialCfg(density=.25,particle_radius=.002,tri_ke=1500,tri_ka=1500,tri_kd=.001,edge_ke=.02,edge_kd=.001)),init_state=DeformableObjectCfg.InitialStateCfg(pos=(.65,.1,.6)));scfg.mailer.spawn.wrinkle=spec['wrinkle_m']
 return cfg,scfg
