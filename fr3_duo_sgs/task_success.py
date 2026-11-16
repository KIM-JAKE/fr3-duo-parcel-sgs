"""Task authority: whole parcel supported inside output belt, label facing up."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
CELL=json.loads(Path(__file__).with_name('cell_config.json').read_text())
CONTRACT_VERSION='conveyor_full_support_label_up_v3'
OUTPUT=CELL['station']['output'];C=np.array(OUTPUT['center_xyz_m']);H=np.array(OUTPUT['half_extents_m'])
ENTRY_Y=float(C[1]+H[1]);TOP_Z=float(C[2]+H[2]);LABEL_UP_MIN=.8
SIGNS=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])

def box_corners(pose_xyzw,half):
    pose=np.asarray(pose_xyzw);r=Rotation.from_quat(pose[...,3:]).as_matrix()
    return np.einsum('...ij,...kj->...ki',r,np.asarray(half)[...,None,:]*SIGNS)+pose[...,:3][...,None,:]

def box_entry_gap(pose_xyzw,half):
    """Trailing edge gap to full conveyor entry; positive means not fully entered."""
    return np.max(box_corners(pose_xyzw,half)[...,1],axis=-1)-ENTRY_Y

def mailer_entry_gap(nodes):
    return np.max(np.asarray(nodes)[...,1],axis=-1)-ENTRY_Y

def support_geometry(points):
    p=np.asarray(points);lo=np.min(p,axis=-2);hi=np.max(p,axis=-2)
    footprint=(lo[...,0]>=C[0]-H[0])&(hi[...,0]<=C[0]+H[0])&(lo[...,1]>=C[1]-H[1])&(hi[...,1]<=ENTRY_Y)
    supported=np.abs(lo[...,2]-TOP_Z)<=.015
    return footprint&supported

def box_success(pose_xyzw,half,label_local_normal=(0,0,1)):
    score=Rotation.from_quat(np.asarray(pose_xyzw)[...,3:]).apply(label_local_normal)[...,2]
    return support_geometry(box_corners(pose_xyzw,half))&(score>=LABEL_UP_MIN)

def mailer_label_score(nodes,triangles,label_mask):
    pts=np.asarray(nodes)[...,np.asarray(triangles)[np.asarray(label_mask)],:]
    n=np.cross(pts[...,1,:]-pts[...,0,:],pts[...,2,:]-pts[...,0,:]);area=np.linalg.norm(n,axis=-1)
    return np.sum(np.maximum(n[...,2],0),axis=-1)/np.maximum(np.sum(area,axis=-1),1e-12)

def mailer_success(nodes,triangles,label_mask):
    return support_geometry(nodes)&(mailer_label_score(nodes,triangles,label_mask)>=LABEL_UP_MIN)
