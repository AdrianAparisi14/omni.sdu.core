# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
import omni.isaac.core.tasks as tasks
# from omni.isaac.franka import Franka
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.manipulators import SingleManipulator
from omni.isaac.manipulators.grippers import ParallelGripper
from typing import Optional
import numpy as np
import os

NOVO_DIRECTORY =  os.getcwd() + "/novo_sim/"
rmp_config_dir = os.path.join(NOVO_DIRECTORY,"robots/ur5e_assets")


class PickPlace(tasks.PickPlace):
    """[summary]

        Args:
            name (str, optional): [description]. Defaults to "franka_pick_place".
            cube_initial_position (Optional[np.ndarray], optional): [description]. Defaults to None.
            cube_initial_orientation (Optional[np.ndarray], optional): [description]. Defaults to None.
            target_position (Optional[np.ndarray], optional): [description]. Defaults to None.
            cube_size (Optional[np.ndarray], optional): [description]. Defaults to None.
            offset (Optional[np.ndarray], optional): [description]. Defaults to None.
        """

    def __init__(
        self,
        name: str = "pick_place",
        cube_initial_position: Optional[np.ndarray] = None,
        cube_initial_orientation: Optional[np.ndarray] = None,
        target_position: Optional[np.ndarray] = None,
        cube_size: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
        robot: SingleManipulator = None
    ) -> None:
        self.robot = robot
        tasks.PickPlace.__init__(
            self,
            name=name,
            cube_initial_position=cube_initial_position,
            cube_initial_orientation=cube_initial_orientation,
            target_position=target_position,
            cube_size=cube_size,
            offset=offset,
        )
        return

    def set_robot(self) -> SingleManipulator:
        if self.robot == None:
            add_reference_to_stage(usd_path= NOVO_DIRECTORY + "omniverse/robotiq_gripper/Novo-Nordisk/ur5e_crp200.usd",prim_path="/World/ur5e")
            gripper = ParallelGripper(
                end_effector_prim_path="/World/ur5e/crp_200_gripper2/crg200_base",
                joint_prim_names=["joint_left", "joint_right"],
                joint_opened_positions=np.array([0, 0]),
                joint_closed_positions=np.array([0.04, 0.04]),
                action_deltas=np.array([-0.04, -0.04]))
            self.robot = SingleManipulator(prim_path="/World/ur5e", name="ur5e_robot", end_effector_prim_name="wrist_3_link", gripper=gripper)
            joints_default_positions = np.array([0.0,-1.5707,1.5707,-1.5707,-1.5707,0.0, 0.0, 0.0])
            self.robot.set_joints_default_state(positions=joints_default_positions)
            print("No manipulator: Set to default")
        return self.robot 
