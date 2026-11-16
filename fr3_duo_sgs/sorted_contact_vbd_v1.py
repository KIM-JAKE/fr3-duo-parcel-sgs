# Derived from Newton 1.6.1 particle_vbd_kernels.py (Apache-2.0), preserving contact evaluation and sum order.
import warp as wp
from newton.solvers import SolverVBD
from newton._src.solvers.vbd.particle_vbd_kernels import _eval_body_particle_contact,_eval_soft_ef_contact,gather_particle_body_contact_force_and_hessian as NATIVE_GATHER
import newton._src.solvers.vbd.solver_vbd as sv

@wp.kernel
def sort_particle_contact_lists(heads:wp.array[int], links:wp.array[int]):
 p=wp.tid()
 rest=heads[p]
 first=int(-1)
 tail=int(-1)
 while rest>=0:
  best=rest
  previous=int(-1)
  best_previous=int(-1)
  cursor=rest
  while cursor>=0:
   if cursor<best:
    best=cursor
    best_previous=previous
   previous=cursor
   cursor=links[cursor]
  following=links[best]
  if best_previous>=0:links[best_previous]=following
  else:rest=following
  links[best]=-1
  if tail>=0:links[tail]=best
  else:first=best
  tail=best
 heads[p]=first

@wp.kernel
def gather_sorted_particle_contacts(
    # inputs
    dt: float,
    particle_ids_in_color: wp.array[wp.int32],
    pos_anchor: wp.array[wp.vec3],
    pos: wp.array[wp.vec3],
    # body-particle contact
    friction_epsilon: float,
    particle_radius: wp.array[float],
    body_particle_contact_indices: wp.array[wp.vec3i],
    particle_contact_head: wp.array[int],
    particle_contact_next: wp.array[int],
    # per-contact soft AVBD parameters for body-particle contacts (shared with rigid side)
    body_particle_contact_penalty_k: wp.array[float],
    body_particle_contact_material_kd: wp.array[float],
    body_particle_contact_material_mu: wp.array[float],
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    contact_shape: wp.array[int],
    contact_body_pos: wp.array[wp.vec3],
    contact_body_vel: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    shape_margin: wp.array[float],
    contact_barycentric: wp.array[wp.vec3],
    # outputs
    particle_forces: wp.array[wp.vec3],
    particle_hessians: wp.array[wp.mat33],
):
    """Gather all body-contact contributions for one colored particle without output atomics.

    Contacts are consumed in increasing node order rather than linked-list order, so the
    summation order follows the contact buffer instead of the lists' insertion order. Selection
    scans the list once per consumed node (O(k^2) link reads for k contacts on one particle);
    k is small in practice and the scan cost is trivial next to one contact evaluation.
    """
    particle_index = particle_ids_in_color[wp.tid()]
    force = wp.vec3(0.0)
    hessian = wp.mat33(0.0)
    head = particle_contact_head[particle_index]

    node = head
    while node >= 0:
        contact_index = node // 3
        corner = node - 3 * contact_index
        corners = body_particle_contact_indices[contact_index]
        contact_ke = body_particle_contact_penalty_k[contact_index]
        contact_kd = body_particle_contact_material_kd[contact_index]
        contact_mu = body_particle_contact_material_mu[contact_index]

        if corners[1] < 0:
            contact_force, contact_hessian = _eval_body_particle_contact(
                particle_index,
                pos[particle_index],
                pos_anchor[particle_index],
                contact_index,
                contact_ke,
                contact_kd,
                contact_mu,
                friction_epsilon,
                particle_radius,
                shape_body,
                body_q,
                body_q_prev,
                body_qd,
                body_com,
                contact_shape,
                contact_body_pos,
                contact_body_vel,
                contact_normal,
                shape_margin,
                dt,
            )
            force += contact_force
            hessian += contact_hessian
        else:
            bary = contact_barycentric[contact_index]
            contact_force, contact_hessian, _cp_world = _eval_soft_ef_contact(
                contact_index,
                corners,
                bary,
                pos,
                pos_anchor,
                particle_radius,
                contact_ke,
                contact_kd,
                contact_mu,
                friction_epsilon,
                shape_body,
                body_q,
                body_q_prev,
                body_qd,
                body_com,
                contact_shape,
                contact_body_pos,
                contact_body_vel,
                contact_normal,
                shape_margin,
                dt,
            )
            weight = bary[corner]
            force += weight * contact_force
            hessian += (weight * weight) * contact_hessian

        node = particle_contact_next[node]

    # Plain write, no atomics: later spring/self-contact kernels accumulate on top.
    particle_forces[particle_index] = force
    particle_hessians[particle_index] = hessian

class SortedContactSolverVBD(SolverVBD):
 def _refresh_body_particle_contact_state(self,contacts,refresh):
  rebuild=contacts is not None and self.model.particle_count>0 and (refresh or not self._particle_contact_adjacency_initialized or self.body_particle_contact_penalty_k.shape[0]<contacts.soft_contact_max)
  super()._refresh_body_particle_contact_state(contacts,refresh)
  if rebuild:
   wp.launch(sort_particle_contact_lists,dim=self.model.particle_count,inputs=[self._particle_contact_head,self._particle_contact_next],device=self.device)

def install_sorted_gather():
 sv.gather_particle_body_contact_force_and_hessian=gather_sorted_particle_contacts
