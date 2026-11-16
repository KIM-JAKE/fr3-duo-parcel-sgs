"""Single authority for the Isaac Lab 3.0 Newton rigid-box cell."""

from __future__ import annotations

import json
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg


ROOT = Path(__file__).resolve().parents[1]
CELL = json.loads((ROOT / "fr3_duo_sgs/cell_config.json").read_text())


def physics_cfg(device: str, dense_contacts: bool = False):
    solver = MJWarpSolverCfg(iterations=150, ls_iterations=100)
    if dense_contacts:
        solver.nconmax = 128
        solver.njmax = 1024
    return sim_utils.SimulationCfg(
        device=device, dt=CELL["reset"]["simulation_dt_s"],
        physics=NewtonCfg(solver_cfg=solver),
    )


def converted_usd() -> str:
    audit_path = ROOT / "fr3_duo_sgs/runtime/native_duo_usd/usd_audit.json"
    audit = json.loads(audit_path.read_text())
    matched = [x for x in audit["candidates"] if x["revolute_joint_count"] == 54 and not x["missing_expected_joints"]]
    if not matched:
        raise RuntimeError("Native Duo USD structural gate not passed")
    return matched[0]["usd"]


def scene_cfg(num_envs: int, box_size: tuple[float, float, float], box_mass: float,
              box_friction: float, contact_sensor: bool = False) -> InteractiveSceneCfg:
    """Create a scene with one fixed Duo and one rigid box per local environment."""
    usd = converted_usd()
    arm_home = CELL["robot"]["arm_home_rad"]
    robot_cfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=usd, variants={"Physics": "mujoco"}),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={f"{side}_fr3v2_joint{i}": value for side in ("left", "right")
                       for i, value in enumerate(arm_home, 1)}
        ),
        actuators={
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[r"(left|right)_fr3v2_joint[1-7]"],
                joint_effort_limit={r"(left|right)_fr3v2_joint[1-4]": 87.0,
                                    r"(left|right)_fr3v2_joint[5-7]": 12.0},
                joint_velocity_limit={r"(left|right)_fr3v2_joint[1-4]": 2.175,
                                      r"(left|right)_fr3v2_joint[5-7]": 2.61},
                stiffness={r"(left|right)_fr3v2_joint[1-4]": 600.0,
                           r"(left|right)_fr3v2_joint5": 250.0,
                           r"(left|right)_fr3v2_joint6": 150.0,
                           r"(left|right)_fr3v2_joint7": 50.0},
                damping={r"(left|right)_fr3v2_joint[1-4]": 50.0,
                         r"(left|right)_fr3v2_joint5": 30.0,
                         r"(left|right)_fr3v2_joint6": 25.0,
                         r"(left|right)_fr3v2_joint7": 15.0},
                armature={r"(left|right)_fr3v2_joint[1-2]": .6057,
                          r"(left|right)_fr3v2_joint[3-4]": .4625,
                          r"(left|right)_fr3v2_joint[5-7]": .2055},
            ),
            "hands": ImplicitActuatorCfg(
                joint_names_expr=[r"[lr]j_dg_.*"], joint_effort_limit=7.5,
                joint_velocity_limit=3.2, stiffness=20.0, damping=1.5,
                armature=0.01,
            ),
        },
    )

    def station(name, key):
        item = CELL["station"][key]
        return RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{name}",
            spawn=sim_utils.CuboidCfg(
                size=tuple(2*x for x in item["half_extents_m"]),
                rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
                collision_props=sim_utils.UsdPhysicsCollisionCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=.9, dynamic_friction=.8),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(item["center_xyz_m"])),
        )

    box_cfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box",
        spawn=sim_utils.CuboidCfg(
            size=tuple(2.0*float(x) for x in box_size),
            activate_contact_sensors=contact_sensor,
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            mass_props=sim_utils.MassCfg(mass=box_mass),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=box_friction, dynamic_friction=.9*box_friction),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(.72, .14, .27+float(box_size[2])+.04)),
    )

    @configclass
    class CellSceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=sim_utils.GroundPlaneCfg())
        light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2500))
        robot: ArticulationCfg = robot_cfg
        box: RigidObjectCfg = box_cfg
        flip_table: RigidObjectCfg = station("FlipTable", "flip_table")
        infeed: RigidObjectCfg = station("Infeed", "infeed")
        output: RigidObjectCfg = station("Output", "output")
        if contact_sensor:
            box_contact: ContactSensorCfg = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Box", update_period=0.0)

    return CellSceneCfg(num_envs=num_envs, env_spacing=3.0)
