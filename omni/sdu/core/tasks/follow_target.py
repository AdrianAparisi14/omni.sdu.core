from omni.isaac.manipulators import SingleManipulator
from omni.isaac.manipulators.grippers import ParallelGripper
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils import transformations as tf
from omni.isaac.core.utils import rotations as r
import omni.isaac.core.tasks as tasks
from typing import Optional
import numpy as np
import carb
import os

# Inheriting from the base class Follow Target
class FollowTarget(tasks.FollowTarget):
    def __init__(
        self,
        name: str = "ur5e_follow_target",
        target_prim_path: Optional[str] = None,
        target_name: Optional[str] = None,
        target_position: Optional[np.ndarray] = None,
        target_orientation: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
    ) -> None:
        tasks.FollowTarget.__init__(
            self,
            name=name,
            target_prim_path=target_prim_path,
            target_name=target_name,
            target_position=target_position,
            target_orientation=target_orientation,
            offset=offset,
        )
        return

    def set_robot(self) -> SingleManipulator:
        #TODO: change this to the robot usd file.
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find nucleus server with /Isaac folder")
        # asset_path = assets_root_path + "/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd"
        novo_omniverse_path =  os.getcwd() + "/novo_sim/omniverse/"
        asset_path = novo_omniverse_path+"robotiq_gripper/Novo-Nordisk/ur5e_2.usd"
        # asset_path = "/home/tkristiansen/omniverse/robotiq_gripper/Novo-Nordisk/ur5e_crp200.usd"
        add_reference_to_stage(usd_path=asset_path, prim_path="/World/ur5e")
        gripper = ParallelGripper(
            #We chose the following values while inspecting the articulation
            end_effector_prim_path="/World/ur5e/robotiq_base",
            joint_prim_names=["robotiq_finger_joint1", "robotiq_finger_joint2"],
            joint_opened_positions=np.array([0, 0]),
            joint_closed_positions=np.array([0.0185, 0.0185]),
            action_deltas=np.array([-0.0185, -0.0185]))

        T_Wb1 = np.genfromtxt("novo_sim/calibration_data/QuadArmA.csv",delimiter=",")
        T_r_z = [[-1, 0 ,0 ,0], [0, -1, 0, 0],[0,0,1,0],[0,0,0,1]]
        
        T_result = T_Wb1@T_r_z
        T_result =np.matmul(T_Wb1,T_r_z)

        pose = tf.pose_from_tf_matrix(T_result)
        manipulator = SingleManipulator(prim_path="/World/ur5e", name="ur5e_robot", end_effector_prim_name="wrist_3_link", position=pose[0], orientation=pose[1])



        #RRT
        # manipulator = SingleManipulator(prim_path="/World/ur5e", name="ur5e_robot", end_effector_prim_name="wrist_3_link")



        # manipulator = SingleManipulator(prim_path="/World/ur5e", name="ur5e_robot", end_effector_prim_name="wrist_3_link", position=pose[0], orientation=r.euler_angles_to_quat([0.,0.,1.796]))


        joints_default_positions = np.array([0.0,-1.5707,1.5707,-1.5707,-1.5707,0.0, 0.0, 0.0])
        # joints_default_positions = np.array([0.0,-1.5707,1.5707,-1.5707,-1.5707,0.0])
        manipulator.set_joints_default_state(positions=joints_default_positions)
        return manipulator

