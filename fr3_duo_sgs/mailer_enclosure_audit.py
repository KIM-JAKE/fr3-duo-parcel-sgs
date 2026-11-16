"""Geometric membrane/contents audit independent of solver contact candidates."""
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


def audit_enclosure(nodes,triangles,pose_xyzw,half,tolerance_m=.001):
    nodes=np.asarray(nodes);half=np.asarray(half);pose=np.asarray(pose_xyzw)
    rotation=Rotation.from_quat(pose[3:])
    corners=np.array([[x,y,z] for x in (-half[0],half[0])
                      for y in (-half[1],half[1]) for z in (-half[2],half[2])])
    corners=rotation.apply(corners)+pose[:3]
    face_points=nodes[triangles]
    relative=face_points[None]-corners[:,None,None,:]
    a,b,c=relative[:,:,0],relative[:,:,1],relative[:,:,2]
    la,lb,lc=[np.linalg.norm(x,axis=-1) for x in (a,b,c)]
    denominator=la*lb*lc+np.sum(a*b,axis=-1)*lc+np.sum(b*c,axis=-1)*la+np.sum(c*a,axis=-1)*lb
    winding=np.sum(2*np.arctan2(np.sum(a*np.cross(b,c),axis=-1),denominator),axis=-1)/(4*np.pi)
    mesh=trimesh.Trimesh(vertices=nodes,faces=triangles,process=False)
    _,distance,_=trimesh.proximity.closest_point_naive(mesh,corners)
    outside=np.where(abs(winding)>.5,0.,distance)
    # Probe vertices and triangle interiors against the analytic contents box.
    samples=[nodes,face_points.mean(axis=1)]
    for i in range(3):
        samples.append((face_points[:,i]+face_points[:,(i+1)%3])*.5)
        samples.append(face_points[:,i]*(2/3)+(face_points[:,(i+1)%3]+face_points[:,(i+2)%3])*(1/6))
    local=rotation.inv().apply(np.concatenate(samples)-pose[:3])
    delta=np.abs(local)-half
    signed=np.linalg.norm(np.maximum(delta,0.),axis=1)+np.minimum(np.max(delta,axis=1),0.)
    penetration=float(max(0.,-signed.min()))
    max_outside=float(outside.max())
    return dict(pass_enclosure=bool(max_outside<=tolerance_m and penetration<=tolerance_m),
        contents_corner_outside_max_m=max_outside,membrane_inside_contents_max_m=penetration,
        contents_corners_inside=int(np.sum(abs(winding)>.5)),tolerance_m=tolerance_m,
        method='closed-mesh solid angle + nearest surface + triangle interior box probes')
