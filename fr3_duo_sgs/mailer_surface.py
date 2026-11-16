"""Closed triangulated mailer membrane with a thin sealed rim."""
import numpy as np
import trimesh


def pillow_mesh(width=.24, length=.32, thickness=.05, nx=13, ny=17, wrinkle=.004):
    vertices=[]
    for sign in (1,-1):
        for j in range(ny):
            v=2*j/(ny-1)-1
            for i in range(nx):
                u=2*i/(nx-1)-1
                shape=max(0.,(1-u*u)*(1-v*v))**.45
                z=sign*(.001+thickness*.5*shape)
                z+=wrinkle*np.sin(12*u+4*v)*np.cos(14*v)*shape
                vertices.append((u*width*.5,v*length*.5,z))
    faces=[];n=nx*ny
    for layer in (0,1):
        for j in range(ny-1):
            for i in range(nx-1):
                a=layer*n+j*nx+i
                pair=[(a,a+1,a+nx+1),(a,a+nx+1,a+nx)]
                faces.extend(pair if layer==0 else [x[::-1] for x in pair])
    boundary=list(range(nx))+[j*nx+nx-1 for j in range(1,ny)]+list(range(n-2,n-nx-1,-1))+[j*nx for j in range(ny-2,0,-1)]
    for a,b in zip(boundary,boundary[1:]+boundary[:1]):
        faces.extend([(a,b+n,b),(a,a+n,b+n)])
    mesh=trimesh.Trimesh(vertices=np.array(vertices),faces=np.array(faces),process=False)
    mesh.fix_normals()
    if not mesh.is_watertight:raise RuntimeError("Mailer membrane is not closed")
    return mesh


def spawn_mailer(prim_path,cfg,translation=None,orientation=None,**kwargs):
    from isaaclab.sim.spawners.meshes.meshes import _spawn_mesh_geom_from_mesh
    mesh=pillow_mesh(*cfg.size,wrinkle=getattr(cfg,"wrinkle",.004))
    return _spawn_mesh_geom_from_mesh(prim_path,cfg,mesh,translation,orientation)
