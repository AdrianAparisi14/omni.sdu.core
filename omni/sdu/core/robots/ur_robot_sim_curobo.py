# omniverse imports
import carb
import lula
from pxr import Gf

from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.core.utils.rotations import quat_to_euler_angles
from omni.isaac.motion_generation import LulaKinematicsSolver
from omni.isaac.motion_generation import ArticulationTrajectory
from omni.isaac.motion_generation.lula.trajectory_generator import LulaCSpaceTrajectoryGenerator
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.rotations import rot_matrix_to_quat
from omni.isaac.core.articulations.articulation import Articulation
from omni.sdu.core.utilities import utils as ut

# other imports
import numpy as np
import os
import math
from typing import Optional, Sequence
from scipy.spatial.transform import Rotation as R
from omni.sdu.core.trajectory.trajectory import jtraj

from ur_ikfast import ur_kinematics

import torch

# cuRobo
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_configs_path, get_assets_path, join_path, load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig


class URRobotSim(XFormPrim):
    """Class for instantiating a simulated UR robot.

    Args:
        Articulation (_type_): _description_
    """

    def __init__(
        self,
        robot_type: str = "ur5e",
        prim_path: str = "/World/",
        name: str = "ur5e_robot",
        initial_joint_q: np.ndarray = np.array(
            [0.0, -1.5707, -1.5707, -1.5707, 1.5707, 0.0]
        ),
        calibrated_urdf_path: str = None,
        position: Optional[Sequence[float]] = None,
        orientation: Optional[Sequence[float]] = None,
        scale: Optional[Sequence[float]] = [1.0, 1.0, 1.0],
        end_effector_name: str = "tool0",
        end_effector_offset: Optional[Sequence[float]] = [0, 0, 0],
    ) -> None:
        self._robot_type = robot_type
        self._prim_path = prim_path + "/" + robot_type
        self._name = name
        if self._name is None:
            self._name = prim_path.split("/")[-1]
        self._initial_joint_q = initial_joint_q
        self._calibrated_urdf_path = calibrated_urdf_path
        self._world_position = position
        self._world_orientation = orientation
        self._scale = scale
        self._end_effector_name = end_effector_name
        self._end_effector_offset = np.array(end_effector_offset)
        self._end_effector = None
        self._controller_handle = None
        XFormPrim.__init__(self, prim_path="/World/" + name, name=name)

    def initialize(self, physics_sim_view=None) -> None:
        """To be called before using this class after a reset of the world

        Args:
            physics_sim_view (_type_, optional): _description_. Defaults to None.
        """
        XFormPrim.initialize(self, physics_sim_view=physics_sim_view)
        return

    def initialize_controller(self):
        self._controller_handle = Articulation(
            prim_path=self._prim_path,
            position=self._world_position,
            translation=None,
            orientation=self._world_orientation,
            scale=self._scale,
            visible=None,
        )
        if self._end_effector:
            initial_state = np.append(self._initial_joint_q, np.zeros(2))
        else:
            initial_state = self._initial_joint_q
        self._controller_handle.initialize()
        self._controller_handle.set_joints_default_state(positions=initial_state)
        self.init_motion_control()
        self._controller_handle.post_reset()
        self.initialized = True

    def init_motion_control(self):
        # Used for placing the robot
        # robot_prim = XFormPrim(prim_path=self._prim_path)
        # print(self._prim_path)
        # robot_base_isaac_pose = robot_prim.get_world_pose()
        # print("robot_base_isaac_pose (pos): ", robot_base_isaac_pose[0])
        # print("robot_base_isaac_pose (rot): ", quat_to_euler_angles(robot_base_isaac_pose[1], degrees=True))

        rmp_config_dir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), self._robot_type + "_assets"
        )
        robot_description_path = os.path.join(
            rmp_config_dir, self._robot_type + "_robot_description.yaml"
        )

        # Init tensor device type
        self._tensor_args = TensorDeviceType()

        # Load curobo robot config
        config_file = load_yaml(join_path(get_robot_configs_path(), self._robot_type+'.yml'))
        urdf_file = config_file["robot_cfg"]["kinematics"]["urdf_path"]  # Send global path starting with "/"
        base_link = config_file["robot_cfg"]["kinematics"]["base_link"]
        ee_link = config_file["robot_cfg"]["kinematics"]["ee_link"]

        # Override URDF file if specified
        if self._calibrated_urdf_path:
            print(
               "NOTICE! using custom calibrated urdf path: ",
               self._calibrated_urdf_path,
            )
            urdf_file = self._calibrated_urdf_path
        else:
            urdf_file = config_file["robot_cfg"]["kinematics"]["urdf_path"]  # Send global path starting with "/"

        self._kinematics = LulaKinematicsSolver(
            robot_description_path=robot_description_path, urdf_path=urdf_file
        )

        self._joint_space_trajectory_generator = LulaCSpaceTrajectoryGenerator(
            robot_description_path=robot_description_path, urdf_path=join_path(get_assets_path(), urdf_file)
        )

        robot_cfg = RobotConfig.from_basic(urdf_file, base_link, ee_link, self._tensor_args)
        ik_config = IKSolverConfig.load_from_robot_config(
            robot_cfg,
            world_model=None,
            tensor_args=self._tensor_args,
            num_seeds=1,
            position_threshold=0.005,
            rotation_threshold=0.05,
            self_collision_check=False,
            self_collision_opt=False,
            use_cuda_graph=True,
            high_precision=False,
            regularization=True
        )
        self._ik_solver = IKSolver(ik_config)
        self._ur_kin = ur_kinematics.URKinematics('ur10e')
     
    def post_reset(self) -> None:
        if self._controller_handle:
            self._controller_handle.post_reset()
        return

    def get_robot_type(self):
        return self._robot_type

    def moveL(
        self,
        target_position: Optional[Sequence[float]] = None,
        target_orientation: Optional[Sequence[float]] = None,
    ):
        # TODO: Should be removed
        action = self._cspace_controller.forward(
            target_end_effector_position=target_position,
            target_end_effector_orientation=target_orientation,
        )
        self._articulation_controller.apply_action(action)

    def moveJ(self, joint_q):
        """Move robot to desired joint position

        Args:
            joint_q (array): desired joint position
        Return:
        """
        #if not len(joint_q) > 6:
        #    joint_q = np.concatenate((joint_q, np.array([0.0, 0.0])), axis=0)
        
        self.apply_articulationAction(joint_positions=np.array(joint_q))
        return

    def moveJ_with_params(self, joint_q, joint_velocities, joint_accelerations):
        """Move robot to desired joint position

        Args:
            joint_q (array): desired joint position
        Return:
        """
        # if not len(joint_q) > 6:
        #    joint_q = np.concatenate((joint_q, np.array([0.0, 0.0])), axis=0)
        # if not len(joint_velocities) > 6:
        #    joint_velocities = np.concatenate((joint_velocities, np.array([0.0, 0.0])), axis=0)
        
        self.apply_articulationAction(
            joint_positions=np.array(joint_q),
            joint_velocities=np.array(joint_velocities)
        )
        return
    
    def get_interpolated_traj_cartesian_space(
        self, target_pose, velocity, seed_q=None
    ):
        print("get_interpolated_traj_cartesian_space(): ")
        print("seed_q:", seed_q)
        current_q = self.get_joint_positions()[:6]

        # INVKIN_FAST

        # pose_quat = np.concatenate((np.array(target_pose[0:3]), rot_matrix_to_quat(R.from_rotvec(target_pose[3:6]).as_matrix()))).tolist()

        # q_solution = None
        # if seed_q is not None:
        #     q_solution = self._ur_kin.inverse(ee_pose=pose_quat, all_solutions=False, q_guess=np.array(seed_q))
        # else:
        #     q_solution = self._ur_kin.inverse(ee_pose=pose_quat, all_solutions=False, q_guess=np.zeros(6))

        # print("q_solution: ", q_solution)
        # target_q = q_solution

        # seed_q_tensor = torch.from_numpy(np.array(seed_q))
        seed_q_tensor = torch.as_tensor(
            np.array(seed_q), device=self._tensor_args.device, dtype=self._tensor_args.dtype
        ).unsqueeze(0)
        print("seed_q_tensor:", seed_q_tensor)

        goal_pos_quat = np.concatenate((np.array(target_pose[0:3]), R.from_rotvec(np.array(target_pose[3:6])).as_quat())).tolist()

        result = None
        goal_pose = Pose.from_list(goal_pos_quat, q_xyzw=True)

        print("goal_pose.position: ", goal_pose.position)
        print("goal_pose.quaternion: ", goal_pose.quaternion)

        if seed_q is not None:
            result = self._ik_solver.solve_single(goal_pose, retract_config=seed_q_tensor, seed_config=seed_q_tensor)  #, seed_config=seed_q_tensor)
        else:
            result = self._ik_solver.solve_single(goal_pose)

        torch.cuda.synchronize()

        if result.success:
            print(
                "IK completed: Poses: "
                + str(goal_pose.batch)
                + " Time(s): "
                + str(result.solve_time)
            )
        else:
            print("IK failed")

        q_solution = result.solution[result.success]
        target_q = q_solution.cpu().numpy()[0]
    
        print("current q:", current_q)
        print("target q:", target_q)

        # calculate trajectory duration based on provided velocity and leading axis movement.
        max_dist = 0.0
        for i in range(0, 6):
            max_dist = max(max_dist, math.fabs(current_q[i] - target_q[i]))
        trajectory_duration = max_dist / velocity

        t_array = np.linspace(0.0, trajectory_duration, num=60)
        j_traj = jtraj(current_q, target_q, t_array).q
        c_space_points = j_traj
        c_space_points_timestamps = t_array
        trajectory = self._joint_space_trajectory_generator.compute_timestamped_c_space_trajectory(
            c_space_points, c_space_points_timestamps, interpolation_mode="linear"
        )

        if trajectory is None:
            print("No trajectory could be generated!")

        # Use the ArticulationTrajectory wrapper to run Trajectory on UR10 robot Articulation
        physics_dt = (
            1 / 60.0
        )  # Physics steps are fixed at 60 fps when running a standalone script
        articulation_trajectory = ArticulationTrajectory(
            self._controller_handle, trajectory, physics_dt
        )
        return articulation_trajectory


    def get_interpolated_traj_joint_space(
        self, current_q, target_q, velocity
    ):
        # Truncate to the six robot joints
        current_q = current_q[:6]
        target_q = target_q[:6]
        print("current q:", current_q)
        print("target q:", target_q)

        max_dist = 0.0
        for i in range(0, 6):
            max_dist = max(max_dist, math.fabs(current_q[i] - target_q[i]))
        trajectory_duration = max_dist / velocity

        t_array = np.linspace(0.0, trajectory_duration, num=60)
        j_traj = jtraj(current_q, target_q, t_array).q
        c_space_points = j_traj
        c_space_points_timestamps = t_array

        trajectory = self._joint_space_trajectory_generator.compute_timestamped_c_space_trajectory(
            c_space_points, c_space_points_timestamps, interpolation_mode="linear"
        )

        if trajectory is None:
            print("No trajectory could be generated!")

        physics_dt = (
            1 / 60.0
        )  # Physics steps are fixed at 60 fps
        articulation_trajectory = ArticulationTrajectory(
            self._controller_handle, trajectory, physics_dt
        )
        return articulation_trajectory

    def get_initial_joint_position(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self._initial_joint_q

    def set_joint_velocities(self, velocity):
        if self._end_effector is not None:
            velocities = np.full(8, velocity)
        else:
            velocities = np.full(6, velocity)
        self._controller_handle.set_joint_velocities(velocities)

    def movePose3(self, target_pose=lula.Pose3):
        """Solving IK problem for target pose. Applies joints articulation

        Args:
            target_pose (lula.Pose3): _description_
        Return:
        """
        joint_pos = self.get_inverse_kinematics(target_pose)
        if self.gripper is not None:
            # TODO: Assumes parallel gripper. Must be more generic
            joint_pos = np.concatenate((joint_pos, np.array([0.0, 0.0])), axis=0)
        self.apply_articulationAction(joint_positions=joint_pos)
        return

    def apply_articulationAction(
        self,
        joint_positions: Optional[Sequence[float]] = None,
        joint_velocities: Optional[Sequence[float]] = None,
        joint_efforts: Optional[Sequence[float]] = None,
        joint_indices: Optional[Sequence[float]] = None,
    ):
        action = ArticulationAction(
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            joint_efforts=joint_efforts,
            joint_indices=joint_indices,
        )
        self._controller_handle.apply_action(action)

    @property
    def end_effector(self) -> RigidPrim:
        """
        Returns:
            RigidPrim: end effector of the manipulator (can be used to get its world pose, angular velocity..etc).
        """
        return self._end_effector

    def get_tcp(self) -> lula.Pose3:
        """Compute tcp pose with end_effector_offset

        Returns:
            lula.Pose3:
        """
        # TODO: Make it generic for any joint robot with or without gripper
        # Assumes 6 joint robot
        tcp = self._kinematics.compute_forward_kinematics(
            frame_name=self._end_effector_name,
            joint_positions=self._controller_handle.get_joint_positions()[:6],
        )
        # offset = lula.Pose3.from_translation(self._end_effector_offset)
        flange = lula.Pose3(lula.Rotation3(np.array(tcp[1])), np.array(tcp[0]))
        # tcp = flange * offset
        return flange

    def get_joint_positions(self):
        return self._controller_handle.get_joint_positions()

    def get_forward_kinematics(
        self, joint_positions: Optional[Sequence[float]] = None
    ) -> lula.Pose3:
        """Compute tcp pose with end_effector_offset

        Returns:
            lula.Pose3:
        """
        # TODO: Make it generic for any joint robot with or without gripper
        # Assumes 6 joint robot
        tcp = self._kinematics.compute_forward_kinematics(
            frame_name=self._end_effector_name, joint_positions=joint_positions
        )
        offset = lula.Pose3.from_translation(self._end_effector_offset)
        flange = lula.Pose3(lula.Rotation3(np.array(tcp[1])), np.array(tcp[0]))
        tcp = flange * offset
        return tcp

    def get_inverse_kinematics(
        self, target_pose=lula.Pose3, seed: Optional[Sequence[float]] = None
    ):
        """Compute the joint positions to a target pose

        Args:
            target_pose (lula.Pose3):

        Returns:
            np.array(): Target joint position
        """
        # TODO: Make it generic for any joint robot with or without gripper
        # Assumes 6 joint robot
        offset_pose = lula.Pose3.from_translation(np.array(self._end_effector_offset))
        goal = target_pose * offset_pose.inverse()
        if seed is None:
            seed_joint_pos = self._controller_handle.get_joint_positions()[:6]
        else:
            seed_joint_pos = seed

        print("seed_joint_pos:", seed_joint_pos)
        target_joint_position, success = self._kinematics.compute_inverse_kinematics(
            frame_name=self._end_effector_name,
            warm_start=seed_joint_pos,
            target_position=goal.translation,
            target_orientation=ut.pose3_to_quat(goal),
        )
        if success is False:
            carb.log_error(
                "Inverse Kinematics solver could not find a solution to target pose: "
            )
            print(target_pose)
            return seed_joint_pos
        else:
            return np.array(target_joint_position)
