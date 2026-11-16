"""Float64 batched equivalents of existing reset geometry gates.

Uses the same winding, exact triangle distance and membrane probes as the
independent CPU audit. No physics threshold or sampling authority changes.
"""
import numpy as np
import torch
from task_success import CELL


def rotate(q,v):
    qv=q[...,:3]
    uv=torch.cross(qv,v,dim=-1)
    return v+2*(q[...,3:4]*uv+torch.cross(qv,uv,dim=-1))


def triangle_distance(p,faces):
    # p [B,Q,1,3], faces [B,1,F,3,3].
    a,b,c=faces.unbind(-2);ab=b-a;ac=c-a;ap=p-a
    d00=(ab*ab).sum(-1);d01=(ab*ac).sum(-1);d11=(ac*ac).sum(-1)
    d20=(ap*ab).sum(-1);d21=(ap*ac).sum(-1)
    det=(d00*d11-d01*d01).clamp_min(1e-30)
    v=(d11*d20-d01*d21)/det;w=(d00*d21-d01*d20)/det
    normal=torch.cross(ab,ac,dim=-1)
    plane=(ap*normal).sum(-1).abs()/torch.linalg.vector_norm(normal,dim=-1).clamp_min(1e-30)
    distance=torch.where((v>=0)&(w>=0)&(v+w<=1),plane,torch.inf)
    for x,y in ((a,b),(b,c),(c,a)):
        edge=y-x;t=((p-x)*edge).sum(-1)/(edge*edge).sum(-1).clamp_min(1e-30)
        d=torch.linalg.vector_norm(p-x-t.clamp(0,1)[...,None]*edge,dim=-1)
        distance=torch.minimum(distance,d)
    return distance.min(-1).values


def enclosure_batch(nodes,triangles,poses,half,device,chunk=64):
    nodes=np.asarray(nodes);poses=np.asarray(poses)
    out=[]
    tri=torch.as_tensor(triangles,device=device,dtype=torch.long)
    hh=torch.as_tensor(half,device=device,dtype=torch.float64)
    corners=hh*torch.tensor([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)],device=device)
    for start in range(0,len(nodes),chunk):
        nn=torch.as_tensor(nodes[start:start+chunk],device=device,dtype=torch.float64)
        pp=torch.as_tensor(poses[start:start+chunk],device=device,dtype=torch.float64)
        qq=pp[:,3:7];qq=qq/torch.linalg.vector_norm(qq,dim=-1,keepdim=True)
        cc=rotate(qq[:,None].expand(-1,8,-1),corners[None].expand(len(nn),-1,-1))+pp[:,None,:3]
        faces=nn[:,tri];relative=faces[:,None]-cc[:,:,None,None]
        a,b,c=relative.unbind(-2)
        la,lb,lc=[torch.linalg.vector_norm(v,dim=-1) for v in (a,b,c)]
        den=la*lb*lc+(a*b).sum(-1)*lc+(b*c).sum(-1)*la+(c*a).sum(-1)*lb
        winding=(2*torch.atan2((a*torch.cross(b,c,dim=-1)).sum(-1),den)).sum(-1)/(4*np.pi)
        distance=triangle_distance(cc[:,:,None],faces[:,None])
        outside=torch.where(abs(winding)>.5,0.,distance).max(-1).values
        probes=[nn,faces.mean(-2)]
        for i in range(3):
            probes.append((faces[:,:,i]+faces[:,:,(i+1)%3])*.5)
            probes.append(faces[:,:,i]*(2/3)+(faces[:,:,(i+1)%3]+faces[:,:,(i+2)%3])*(1/6))
        samples=torch.cat(probes,dim=1)-pp[:,None,:3]
        inv=qq.clone();inv[:,:3]*=-1
        local=rotate(inv[:,None].expand(-1,samples.shape[1],-1),samples)
        d=abs(local)-hh
        signed=torch.linalg.vector_norm(d.clamp_min(0),dim=-1)+d.max(-1).values.clamp_max(0)
        penetration=(-signed.min(-1).values).clamp_min(0)
        out.append(torch.stack([outside,penetration],dim=-1).cpu().numpy())
    metrics=np.concatenate(out)
    return (metrics<=.001).all(-1),metrics


def native_geometry_batch(audit,bodies,nodes=None,triangles=None,chunk=32):
    """Same native sample points/station test; exact AABB early-out for hands."""
    bodies=np.asarray(bodies);station=[];intrusion=[]
    device=audit.device
    for start in range(0,len(bodies),chunk):
        body=torch.as_tensor(bodies[start:start+chunk],device=device,dtype=torch.float64)
        bp=body[:,audit.anchors]
        q=bp[:,:,3:7];v=audit.local[None].expand(len(body),-1,-1)
        pts=rotate(q,v)+bp[:,:,:3]
        pc=pts[:,audit.collision];worst=torch.zeros(len(body),device=device,dtype=torch.float64)
        for key in ('flip_table','infeed','output'):
            s=CELL['station'][key]
            d=abs(pc-torch.tensor(s['center_xyz_m'],device=device))-torch.tensor(s['half_extents_m'],device=device)
            signed=torch.linalg.vector_norm(d.clamp_min(0),dim=-1)+d.max(-1).values.clamp_max(0)
            worst=torch.maximum(worst,(-signed.min(-1).values).clamp_min(0))
        station.append(worst.cpu().numpy())
        ih=np.zeros(len(body))
        if nodes is not None:
            nn=torch.as_tensor(nodes[start:start+chunk],device=device,dtype=torch.float64)
            ph=pts[:,audit.hand]
            candidate=((ph>=nn.min(1).values[:,None]-.001)&(ph<=nn.max(1).values[:,None]+.001)).all(-1)
            frame,index=torch.where(candidate)
            tri=torch.as_tensor(triangles,device=device,dtype=torch.long)
            depth=torch.zeros(len(body),device=device,dtype=torch.float64)
            for qstart in range(0,len(frame),2048):
                ff=frame[qstart:qstart+2048];ii=index[qstart:qstart+2048]
                query=ph[ff,ii];faces=nn[ff][:,tri]
                relative=faces-query[:,None,None];a,b,c=relative.unbind(-2)
                la,lb,lc=[torch.linalg.vector_norm(v,dim=-1) for v in (a,b,c)]
                den=la*lb*lc+(a*b).sum(-1)*lc+(b*c).sum(-1)*la+(c*a).sum(-1)*lb
                winding=(2*torch.atan2((a*torch.cross(b,c,dim=-1)).sum(-1),den)).sum(-1)/(4*np.pi)
                inside=abs(winding)>.5
                if bool(inside.any()):
                    distance=triangle_distance(query[inside,None,None],faces[inside,None]).squeeze(-1)
                    depth.scatter_reduce_(0,ff[inside],distance,reduce='amax',include_self=True)
            ih=depth.cpu().numpy()
        intrusion.append(ih)
    return np.concatenate(station),np.concatenate(intrusion)
