import carb
import lula

#from spatialmath.spatialvector import SE3
from typing import Optional, Sequence

from omni.sdu.core.robots.ur5e import UR5E
from omni.sdu.core.utilities import utils as ut
from omni.sdu.core.trajectory.trajectory import (ctraj, jtraj)


class Task():
    def __init__(self, steps, time_steps: int = 200, task_type = None, gripper_state : str = "open") -> None:
        self.steps = steps 
        self.t = 0
        self.final_step = time_steps
        self.task_type = task_type
        self.gripper_state = gripper_state

    def execute_step(self):
        step = self.steps[self.t]
        self.t += 1
        return step

    def is_done(self):
        if self.t == self.final_step:
            return True
        else:
            return False

    def reset(self):
        self.t = 0


class Tasks():
    def __init__(self, robot : UR5E):
        self._t = 0
        self.task_buffer = []
        self.default_joint_position = robot._initial_joint_q
        self.gripper_state = "open"
        self.task_type = []
        self.robot = robot
        self.last_pose = []

    def get_last_joint_q(self):
        last_joint_q = []
        if len(self.task_buffer) > 0:
            last_task = self.task_buffer[-1]
            last_joint_q = last_task.steps[-1]
        else:
            last_joint_q = self.default_joint_position
        return last_joint_q

    def get_gripper_state(self, gripper_state):
        if len(self.task_buffer) > 0 and gripper_state is None:
            return self.task_buffer[-1].gripper_state
        elif len(self.task_buffer) == 0 and gripper_state is None:
            return "open"
        else:
            return gripper_state

    def add_ctraj(self, desired_pose : Optional[Sequence[float]] = None, time_step : int = 200, gripper_state : str = None):
        if desired_pose is None:
            carb.log_error("Could not add ctraj, missing correct input")
        else:
            last_joint_q = self.get_last_joint_q()
            poses = ctraj(SE3(self.last_pose.matrix()), SE3(desired_pose.matrix()), time_step)
            steps = []
            for pose in poses:
                last_joint_q = self.robot.get_inverse_kinematics(target_pose=ut.get_pose3(trans=pose.t,rot_mat=pose.R), 
                                                                    seed = last_joint_q) 
                steps.append(last_joint_q)
            self.task_buffer.append(Task(steps=steps, time_steps=time_step,task_type="ctraj", gripper_state=self.get_gripper_state(gripper_state)))
            self.last_pose = desired_pose

    def add_jtraj(self, desired_pose : Optional[Sequence[float]] = None, time_step : int = 200, gripper_state : str = None):
        """Adds a joint trajectory to the task. 
        Args:
            desired_pose (Optional[Sequence[float]], optional): _description_. Defaults to None.
            time_step (int, optional): _description_. Defaults to 200.
            gripper_state (str, optional): _description_. Defaults to "open".
        """
        if desired_pose is None:
            carb.log_error("Could not add jtraj, missing correct input")
        else:
            last_joint_q = self.get_last_joint_q()
            desired_joint_q = self.robot.get_inverse_kinematics(target_pose = desired_pose, seed = last_joint_q)
            steps = jtraj(last_joint_q,desired_joint_q, time_step).q
            self.task_buffer.append(Task(steps=steps, time_steps=time_step,task_type="jtraj",gripper_state=self.get_gripper_state(gripper_state)))
            self.last_pose = desired_pose


    def add_attact_fingertip(fingertip_prim_path):
        pass

    def add_detact_fingertip():
        pass

    def add_replace_fingertip():
        pass

    def add_pick_and_place():
        pass

    def open_gripper(self, time_steps : int = 40):
        self.add_jtraj(desired_pose=self.last_pose, time_step=time_steps ,gripper_state="open")

    def close_gripper(self, time_steps : int = 40):
        self.add_jtraj(desired_pose=self.last_pose, time_step=time_steps ,gripper_state="close")

    def reset(self):
        for i in self.task_buffer:
            i.reset()
        self._t = 0

    def next_step(self):
        if self._t < len(self.task_buffer):
            task = self.task_buffer[self._t]
            self.robot.moveJ(task.execute_step())
            if task.gripper_state == "open":
                self.robot._gripper.open()
            else: 
                self.robot.close_gripper()
            if task.is_done():
                self._t += 1
        return None 