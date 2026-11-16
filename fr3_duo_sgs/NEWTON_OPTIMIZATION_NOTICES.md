# Newton derived optimization kernels

The kernel bodies in `sorted_contact_vbd_v1.py` and `specialized_face_contacts_v1.py`, `specialized_face_contacts_v2.py`, `specialized_face_contacts_v3.py` derive from Newton 1.6.1's `particle_vbd_kernels.py` and `soft_contacts_sdf.py`.

SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers (particle kernels)
SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers (SDF contact kernels)
SPDX-License-Identifier: Apache-2.0

Original files and license are retained in the installed Newton package. See https://www.apache.org/licenses/LICENSE-2.0 for the license text. Modifications: sort contact incidence once, traverse in the original increasing-node summation order; specialize an identical face kernel by geometry type; stop repeated pure SDF iterations only at an exactly unchanged barycentric fixed point. Physics mesh, material, contact thresholds, timestep, solver iterations, and SDF iteration maxima are retained.

The topology cache reuses exact native CSR filtering values after validating the full instanced connectivity. It does not change the topology exclusion rule.
