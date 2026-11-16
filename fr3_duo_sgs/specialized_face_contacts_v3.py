# Derived from Newton 1.6.1 soft_contacts_sdf.py, Apache-2.0. CONVEX_MESH specialization only; all iterations and thresholds retained.
import warp as wp
from newton._src.geometry.soft_contacts_sdf import (
 TextureSDFData,ShapeFlags,GeoType,_is_analytic,_shape_frames,_soft_feature_aabb_misses_shape,
 _emit_soft_ef_contact,optimize_face_sdf,optimize_edge_sdf,eval_shape_sdf,create_soft_edge_contacts,SDF_EDGE_ITERS,SDF_FACE_ITERS,SDF_LS_ITERS,
)
@wp.func
def optimize_face_sdf_fixedpoint(
    geo: wp.int32,
    scale: wp.vec3,
    a: wp.vec3,
    b: wp.vec3,
    c: wp.vec3,
    shape_sdf_index: wp.int32,
    texture_sdf_table: wp.array[TextureSDFData],
    n_iter: wp.int32,
    ls_iter: wp.int32,
):
    """argmin phi over the soft triangle by Frank-Wolfe on the barycentric simplex (Macklin sec. 3).

    Each step picks the simplex vertex minimizing the linearized objective ``grad . corner`` (eq. 4)
    and line-searches phi toward it with :func:`optimize_edge_sdf`. Fixed ``n_iter`` / ``ls_iter``
    iterations -> graph-capturable. Returns ``(bary, x_local, phi, grad)``.
    """
    # Start at the centroid (interior). A corner start can strand Frank-Wolfe on a simplex edge
    # for non-smooth fields (e.g. a box-corner ridge), because the analytic gradient is single-axis
    # and never selects the third vertex. From the centroid, FW can move toward any vertex.
    bary = wp.vec3(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

    for _i in range(n_iter):
        previous_bary = bary
        x = bary[0] * a + bary[1] * b + bary[2] * c
        _phi_l, _phi_x, grad = eval_shape_sdf(geo, scale, x, shape_sdf_index, texture_sdf_table)
        # Frank-Wolfe vertex: argmin_k grad . corner_k (Macklin eq. 4).
        da = wp.dot(grad, a)
        db = wp.dot(grad, b)
        dc = wp.dot(grad, c)
        s = wp.vec3(1.0, 0.0, 0.0)
        if db <= da and db <= dc:
            s = wp.vec3(0.0, 1.0, 0.0)
        elif dc <= da and dc <= db:
            s = wp.vec3(0.0, 0.0, 1.0)
        target = s[0] * a + s[1] * b + s[2] * c
        gamma, _lx, _lphi, _lgrad = optimize_edge_sdf(
            geo, scale, x, target, shape_sdf_index, texture_sdf_table, ls_iter
        )
        bary = (1.0 - gamma) * bary + gamma * s
        if bary[0] == previous_bary[0] and bary[1] == previous_bary[1] and bary[2] == previous_bary[2]:
            break

    x = bary[0] * a + bary[1] * b + bary[2] * c
    _phi_l, phi, grad = eval_shape_sdf(geo, scale, x, shape_sdf_index, texture_sdf_table)
    return bary, x, phi, grad


@wp.kernel
def create_soft_convex_face_contacts(
    face_pairs: wp.array[wp.vec2i],
    particle_q: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    tri_indices: wp.array2d[wp.int32],
    shape_body: wp.array[wp.int32],
    shape_type: wp.array[wp.int32],
    shape_flags: wp.array[wp.int32],
    shape_transform: wp.array[wp.transform],
    shape_scale: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    shape_sdf_index: wp.array[wp.int32],
    texture_sdf_table: wp.array[TextureSDFData],
    shape_margin: wp.array[float],
    shape_gap: wp.array[float],
    shape_aabb_lower: wp.array[wp.vec3],
    shape_aabb_upper: wp.array[wp.vec3],
    sdf_face_iters: wp.int32,
    sdf_ls_iters: wp.int32,
    margin: float,
    tid_base: wp.int32,
    soft_contact_max: wp.int32,
    soft_contact_count: wp.array[wp.int32],
    soft_contact_tids: wp.array[wp.int32],
    soft_contact_particle: wp.array[wp.int32],
    soft_contact_indices: wp.array[wp.vec3i],
    soft_contact_barycentric: wp.array[wp.vec3],
    soft_contact_shape: wp.array[wp.int32],
    soft_contact_body_pos: wp.array[wp.vec3],
    soft_contact_body_vel: wp.array[wp.vec3],
    soft_contact_normal: wp.array[wp.vec3],
):
    """One thread per world-compatible (soft triangle, shape) pair. Minimizes the rigid SDF over the
    triangle interior and emits a unified ``(v0, v1, v2)`` face record if within margin. Pairs are
    precomputed world-filtered (like ``soft_rigid_contact_pairs``), so no per-thread world check is
    needed. ``tid_base`` is n_particle_pairs + n_edge_pairs (this pass's offset into the shared
    replay-tids array)."""
    tid = wp.tid()
    pair = face_pairs[tid]
    t = pair[0]
    shape_index = pair[1]
    if (shape_flags[shape_index] & ShapeFlags.COLLIDE_PARTICLES) == 0:
        return
    if shape_type[shape_index] != GeoType.CONVEX_MESH:
        return
    geo = wp.static(GeoType.CONVEX_MESH)
    sdf_idx = shape_sdf_index[shape_index]
    if (not _is_analytic(geo)) and sdf_idx < 0:
        # Mesh without a provisioned SDF: the legacy per-particle path still covers it, and the
        # pipeline already warned once about this at construction.
        return

    a_idx = tri_indices[t, 0]
    b_idx = tri_indices[t, 1]
    c_idx = tri_indices[t, 2]
    radius = wp.max(particle_radius[a_idx], wp.max(particle_radius[b_idx], particle_radius[c_idx]))

    a_w = particle_q[a_idx]
    b_w = particle_q[b_idx]
    c_w = particle_q[c_idx]
    feature_lower = wp.vec3(
        wp.min(a_w[0], wp.min(b_w[0], c_w[0])),
        wp.min(a_w[1], wp.min(b_w[1], c_w[1])),
        wp.min(a_w[2], wp.min(b_w[2], c_w[2])),
    )
    feature_upper = wp.vec3(
        wp.max(a_w[0], wp.max(b_w[0], c_w[0])),
        wp.max(a_w[1], wp.max(b_w[1], c_w[1])),
        wp.max(a_w[2], wp.max(b_w[2], c_w[2])),
    )
    if _soft_feature_aabb_misses_shape(
        geo,
        shape_index,
        shape_gap,
        shape_aabb_lower,
        shape_aabb_upper,
        feature_lower,
        feature_upper,
        margin,
        radius,
    ):
        return

    # _s suffix = shape-local frame (matching the X_*s transforms: b = body, w = world, s = shape).
    X_bs, X_ws, X_sw = _shape_frames(shape_body, body_q, shape_transform, shape_index)
    a_s = wp.transform_point(X_sw, a_w)
    b_s = wp.transform_point(X_sw, b_w)
    c_s = wp.transform_point(X_sw, c_w)
    scale = shape_scale[shape_index]
    # Per-shape contact margin (#2994), same threshold term as the legacy particle pass.
    s_margin = shape_margin[shape_index] if shape_margin.shape[0] > 0 else 0.0
    threshold = margin + s_margin + radius

    centroid_s = (a_s + b_s + c_s) / 3.0
    phi_c, _phi_c_a, _grad_c = eval_shape_sdf(geo, scale, centroid_s, sdf_idx, texture_sdf_table)
    # Conservative cull: the SDF is ~1-Lipschitz, so the triangle's minimum is >= phi_c minus the
    # farthest centroid-to-point distance, which is always a vertex. circumradius can be smaller than
    # that for non-equilateral triangles (e.g. 3-4-5: R=2.5 vs 2.85) and would drop valid contacts.
    reach = wp.max(wp.length(a_s - centroid_s), wp.max(wp.length(b_s - centroid_s), wp.length(c_s - centroid_s)))
    if phi_c > threshold + reach:
        return

    bary, x, phi, grad = optimize_face_sdf_fixedpoint(
        geo, scale, a_s, b_s, c_s, sdf_idx, texture_sdf_table, sdf_face_iters, sdf_ls_iters
    )
    if phi < threshold:
        y = x - phi * grad
        _emit_soft_ef_contact(
            tid,
            tid_base,
            soft_contact_max,
            soft_contact_count,
            soft_contact_tids,
            soft_contact_particle,
            soft_contact_indices,
            soft_contact_barycentric,
            soft_contact_shape,
            soft_contact_body_pos,
            soft_contact_body_vel,
            soft_contact_normal,
            wp.vec3i(a_idx, b_idx, c_idx),
            bary,
            shape_index,
            wp.transform_point(X_bs, y),
            wp.vec3(0.0, 0.0, 0.0),
            wp.transform_vector(X_ws, grad),
        )


@wp.kernel
def create_soft_other_face_contacts(
    face_pairs: wp.array[wp.vec2i],
    particle_q: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    tri_indices: wp.array2d[wp.int32],
    shape_body: wp.array[wp.int32],
    shape_type: wp.array[wp.int32],
    shape_flags: wp.array[wp.int32],
    shape_transform: wp.array[wp.transform],
    shape_scale: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    shape_sdf_index: wp.array[wp.int32],
    texture_sdf_table: wp.array[TextureSDFData],
    shape_margin: wp.array[float],
    shape_gap: wp.array[float],
    shape_aabb_lower: wp.array[wp.vec3],
    shape_aabb_upper: wp.array[wp.vec3],
    sdf_face_iters: wp.int32,
    sdf_ls_iters: wp.int32,
    margin: float,
    tid_base: wp.int32,
    soft_contact_max: wp.int32,
    soft_contact_count: wp.array[wp.int32],
    soft_contact_tids: wp.array[wp.int32],
    soft_contact_particle: wp.array[wp.int32],
    soft_contact_indices: wp.array[wp.vec3i],
    soft_contact_barycentric: wp.array[wp.vec3],
    soft_contact_shape: wp.array[wp.int32],
    soft_contact_body_pos: wp.array[wp.vec3],
    soft_contact_body_vel: wp.array[wp.vec3],
    soft_contact_normal: wp.array[wp.vec3],
):
    """One thread per world-compatible (soft triangle, shape) pair. Minimizes the rigid SDF over the
    triangle interior and emits a unified ``(v0, v1, v2)`` face record if within margin. Pairs are
    precomputed world-filtered (like ``soft_rigid_contact_pairs``), so no per-thread world check is
    needed. ``tid_base`` is n_particle_pairs + n_edge_pairs (this pass's offset into the shared
    replay-tids array)."""
    tid = wp.tid()
    pair = face_pairs[tid]
    t = pair[0]
    shape_index = pair[1]
    if (shape_flags[shape_index] & ShapeFlags.COLLIDE_PARTICLES) == 0:
        return
    geo = shape_type[shape_index]
    if geo == GeoType.CONVEX_MESH:
        return
    sdf_idx = shape_sdf_index[shape_index]
    if (not _is_analytic(geo)) and sdf_idx < 0:
        # Mesh without a provisioned SDF: the legacy per-particle path still covers it, and the
        # pipeline already warned once about this at construction.
        return

    a_idx = tri_indices[t, 0]
    b_idx = tri_indices[t, 1]
    c_idx = tri_indices[t, 2]
    radius = wp.max(particle_radius[a_idx], wp.max(particle_radius[b_idx], particle_radius[c_idx]))

    a_w = particle_q[a_idx]
    b_w = particle_q[b_idx]
    c_w = particle_q[c_idx]
    feature_lower = wp.vec3(
        wp.min(a_w[0], wp.min(b_w[0], c_w[0])),
        wp.min(a_w[1], wp.min(b_w[1], c_w[1])),
        wp.min(a_w[2], wp.min(b_w[2], c_w[2])),
    )
    feature_upper = wp.vec3(
        wp.max(a_w[0], wp.max(b_w[0], c_w[0])),
        wp.max(a_w[1], wp.max(b_w[1], c_w[1])),
        wp.max(a_w[2], wp.max(b_w[2], c_w[2])),
    )
    if _soft_feature_aabb_misses_shape(
        geo,
        shape_index,
        shape_gap,
        shape_aabb_lower,
        shape_aabb_upper,
        feature_lower,
        feature_upper,
        margin,
        radius,
    ):
        return

    # _s suffix = shape-local frame (matching the X_*s transforms: b = body, w = world, s = shape).
    X_bs, X_ws, X_sw = _shape_frames(shape_body, body_q, shape_transform, shape_index)
    a_s = wp.transform_point(X_sw, a_w)
    b_s = wp.transform_point(X_sw, b_w)
    c_s = wp.transform_point(X_sw, c_w)
    scale = shape_scale[shape_index]
    # Per-shape contact margin (#2994), same threshold term as the legacy particle pass.
    s_margin = shape_margin[shape_index] if shape_margin.shape[0] > 0 else 0.0
    threshold = margin + s_margin + radius

    centroid_s = (a_s + b_s + c_s) / 3.0
    phi_c, _phi_c_a, _grad_c = eval_shape_sdf(geo, scale, centroid_s, sdf_idx, texture_sdf_table)
    # Conservative cull: the SDF is ~1-Lipschitz, so the triangle's minimum is >= phi_c minus the
    # farthest centroid-to-point distance, which is always a vertex. circumradius can be smaller than
    # that for non-equilateral triangles (e.g. 3-4-5: R=2.5 vs 2.85) and would drop valid contacts.
    reach = wp.max(wp.length(a_s - centroid_s), wp.max(wp.length(b_s - centroid_s), wp.length(c_s - centroid_s)))
    if phi_c > threshold + reach:
        return

    bary, x, phi, grad = optimize_face_sdf_fixedpoint(
        geo, scale, a_s, b_s, c_s, sdf_idx, texture_sdf_table, sdf_face_iters, sdf_ls_iters
    )
    if phi < threshold:
        y = x - phi * grad
        _emit_soft_ef_contact(
            tid,
            tid_base,
            soft_contact_max,
            soft_contact_count,
            soft_contact_tids,
            soft_contact_particle,
            soft_contact_indices,
            soft_contact_barycentric,
            soft_contact_shape,
            soft_contact_body_pos,
            soft_contact_body_vel,
            soft_contact_normal,
            wp.vec3i(a_idx, b_idx, c_idx),
            bary,
            shape_index,
            wp.transform_point(X_bs, y),
            wp.vec3(0.0, 0.0, 0.0),
            wp.transform_vector(X_ws, grad),
        )


def launch_soft_ef_contacts(
    *,
    model,
    state,
    contacts,
    margin: float,
    device,
    edge_pairs: wp.array[wp.vec2i],
    face_pairs: wp.array[wp.vec2i],
    n_particle_pairs: int,
    shape_aabb_lower: wp.array[wp.vec3] | None = None,
    shape_aabb_upper: wp.array[wp.vec3] | None = None,
):
    """Launch the soft EDGE and FACE passes (the soft-particle pass is the legacy kernel).

    ``edge_pairs`` / ``face_pairs`` are precomputed world-compatible (soft feature, shape) index
    pairs (``wp.vec2i``), analogous to :func:`_build_soft_particle_rigid_contact_pairs` for the
    particle pass -- one thread per pair, so cross-world features never reach the kernel. Edge
    endpoints come straight from ``model.edge_indices``; no mesh adjacency is needed.

    ``n_particle_pairs`` is the particle pass's launch dim; combined with the edge-pair count it
    forms the static per-pass offsets into the shared replay-tids array, so the three passes never
    alias a tids slot. All passes share one soft counter, so launch order is immaterial (records
    self-describe via -1 padding; nothing reads by range)."""
    # Nothing to do when there are no candidate pairs -- covers the no-soft-mesh, no-shape, and
    # flag-off cases, since the pairs are built from the mesh edges and shapes.
    n_edge_pairs = int(edge_pairs.shape[0])
    n_face_pairs = int(face_pairs.shape[0])
    if n_edge_pairs == 0 and n_face_pairs == 0:
        return

    if shape_aabb_lower is None:
        # Isolated kernel tests can intentionally disable the broad rejection. Production collision
        # always supplies the current narrow-phase AABBs, so graph capture never allocates here.
        shape_aabb_lower = wp.empty(0, dtype=wp.vec3, device=device)
        shape_aabb_upper = wp.empty(0, dtype=wp.vec3, device=device)

    shape_args = [
        model.shape_body,
        model.shape_type,
        model.shape_flags,
        model.shape_transform,
        model.shape_scale,
        state.body_q,
        model._shape_sdf_index,
        model._texture_sdf_data,
        model.shape_margin,
        model.shape_gap,
        shape_aabb_lower,
        shape_aabb_upper,
    ]
    outputs = [
        contacts.soft_contact_count,
        contacts.soft_contact_tids,
        contacts.soft_contact_particle,
        contacts.soft_contact_indices,
        contacts.soft_contact_barycentric,
        contacts.soft_contact_shape,
        contacts.soft_contact_body_pos,
        contacts.soft_contact_body_vel,
        contacts.soft_contact_normal,
    ]

    if n_edge_pairs > 0:
        wp.launch(
            create_soft_edge_contacts,
            dim=n_edge_pairs,
            inputs=[
                edge_pairs,
                state.particle_q,
                model.particle_radius,
                model.edge_indices,
                *shape_args,
                SDF_EDGE_ITERS,
                margin,
                n_particle_pairs,
                contacts.soft_contact_max,
            ],
            outputs=outputs,
            device=device,
        )
    if n_face_pairs > 0:
      for face_kernel in [create_soft_convex_face_contacts,create_soft_other_face_contacts]:
        wp.launch(
            face_kernel,
            dim=n_face_pairs,
            inputs=[
                face_pairs,
                state.particle_q,
                model.particle_radius,
                model.tri_indices,
                *shape_args,
                SDF_FACE_ITERS,
                SDF_LS_ITERS,
                margin,
                n_particle_pairs + n_edge_pairs,
                contacts.soft_contact_max,
            ],
            outputs=outputs,
            device=device,
        )

def install_specialized_faces():
 import newton._src.sim.collide as collide
 collide.launch_soft_ef_contacts=launch_soft_ef_contacts
