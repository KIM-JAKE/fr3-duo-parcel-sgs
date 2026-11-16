"""Public multi-env reset writes shared by generation and subsequent training.

q_target is a nominal motor reference. The material's recorded controller must
supply its recorded gravity mode; VBD uses native position-reference bias.
"""
import numpy as np,torch

def apply_scaled_rows(scene,rows,q_target):
    n=scene.num_envs;robot=scene['robot'];device=robot.device
    def t(value):return torch.as_tensor(value,device=device,dtype=torch.float32)
    for key in ('robot_q','robot_qd','parcel_pose_xyzw','parcel_twist'):
        if len(rows[key])!=n or not np.isfinite(rows[key]).all():raise ValueError('Invalid reset '+key)
    if not np.isfinite(q_target).all() or np.shape(q_target)!=(n,54):raise ValueError('Invalid target')
    robot.write_joint_position_to_sim_index(position=t(rows['robot_q']))
    robot.write_joint_velocity_to_sim_index(velocity=t(rows['robot_qd']))
    pose=t(rows['parcel_pose_xyzw']).clone();pose[:,:3]+=scene.env_origins
    scene['box'].write_root_pose_to_sim_index(root_pose=pose)
    scene['box'].write_root_velocity_to_sim_index(root_velocity=t(rows['parcel_twist']))
    if 'nodes' in rows:
        if not np.isfinite(rows['nodes']).all() or not np.isfinite(rows['nodal_vel']).all():raise ValueError('Nonfinite nodal reset')
        scene['mailer'].write_nodal_pos_to_sim_index(nodal_pos=t(rows['nodes'])+scene.env_origins[:,None,:])
        scene['mailer'].write_nodal_velocity_to_sim_index(nodal_vel=t(rows['nodal_vel']))
    scene.reset();robot.actuators.target_command.set_position_index(value=t(q_target))
