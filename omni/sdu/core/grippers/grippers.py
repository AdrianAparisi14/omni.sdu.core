from omni.isaac.manipulators.grippers import ParallelGripper
import numpy as np
from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units
from omni.isaac.core.utils.prims import get_prim_at_path
import omni.kit.commands
from pxr import Usd, Sdf
from omni.sdu.utils.utilities import utils as ut
from omni.isaac.core.utils.nucleus import get_assets_root_path 

NOVO_NUCLEUS_ASSETS = "omniverse://sdur-nucleus.tek.sdu.dk/Projects/novo/Assets/"

def attach_gripper(robot_prim_path, gripper_prim_path, gripper_name, gripper_base):
    """Attaches the gripper to the robot.
        This function assumes the robot is a UR robot defined by Isaac Sim standards.

    Args:
        robot_prim_path (str): Prim path of the desired UR robot 
        gripper_prim_path (str): Prim path of the gripper 
        gripper_name (str): XForm name of the gripper 
        gripper_base (str): XForm name of the gripper base 

    Returns:
        str: Return the new prim path of the gripper when attached to the robot 
    """
    omni.kit.commands.execute('RemovePhysicsComponent',
        usd_prim=get_prim_at_path(robot_prim_path + "/tool0"),
        component='PhysicsRigidBodyAPI',
        multiple_api_token=None)
    ut.move_prim(gripper_prim_path,robot_prim_path + "/tool0/" + gripper_name)
    omni.kit.commands.execute('RemoveRelationshipTarget',
        relationship=get_prim_at_path(robot_prim_path+"/flange/flange_tool0").GetRelationship('physics:body1'),
        target=Sdf.Path(robot_prim_path+'/tool0'))
    omni.kit.commands.execute('AddRelationshipTarget',
        relationship=get_prim_at_path(robot_prim_path + "/flange/flange_tool0").GetRelationship('physics:body1'),
        target=Sdf.Path(robot_prim_path+'/tool0/' + gripper_name + "/" + gripper_base))
    gripper_prim_path = robot_prim_path + "/tool0/" + gripper_name
    return gripper_prim_path 


class CRG200(ParallelGripper):
    """CRG200 custom gripper, attaches the gripper to the robot

    Args:
        ParallelGripper (_type_): _description_
    """
    def __init__(self, robot_prim_path) -> None:
        self._gain = 1
        prim_path = "/World/crg200" 
        add_reference_to_stage(usd_path= NOVO_NUCLEUS_ASSETS + "Grippers/crg200/crg200.usd", prim_path=prim_path)
        self.gripper_base = "crg200_base"
        self.gripper_prim_path= attach_gripper(robot_prim_path=robot_prim_path, 
                                               gripper_prim_path=prim_path,
                                               gripper_name="crg200",
                                               gripper_base=self.gripper_base)
        
        self.left_finger_prim_path = self.gripper_prim_path + "/crg200_finger_left"
        self.right_finger_prim_path = self.gripper_prim_path + "/crg200_finger_right"
        self._end_effector_prim_path = self.gripper_prim_path+"/crg200_base"
        self._end_effector_offset = [0,0,0.165]
        joint_prim_names=["joint_left", "joint_right"]
        joint_opened_positions=np.array([0.0, 0.0]) / get_stage_units()
        joint_closed_positions=np.array([0.04, 0.04]) / get_stage_units()
        action_deltas=np.array([-0.04, -0.04]) / get_stage_units()
        ParallelGripper.__init__(self,
            end_effector_prim_path= self._end_effector_prim_path,
            joint_prim_names=joint_prim_names,
            joint_opened_positions=joint_opened_positions,
            joint_closed_positions=joint_closed_positions,
            action_deltas=action_deltas)
        return


class CRG200Sim(ParallelGripper):
    """CRG200 custom gripper, attaches the gripper to the robot

    Args:
        ParallelGripper (_type_): _description_
    """
    def __init__(self, gripper_prim_path) -> None:
        self.gripper_prim_path = gripper_prim_path
        self.left_finger_prim_path = self.gripper_prim_path + "/crg200_finger_left"
        self.right_finger_prim_path = self.gripper_prim_path + "/crg200_finger_right"
        self._end_effector_prim_path = self.gripper_prim_path+"/crg200_base"
        self._end_effector_offset = [0,0,0.165]
        joint_prim_names=["joint_left", "joint_right"]
        joint_opened_positions=np.array([0.0, 0.0]) / get_stage_units()
        joint_closed_positions=np.array([0.04, 0.04]) / get_stage_units()
        action_deltas=np.array([-0.04, -0.04]) / get_stage_units()
        ParallelGripper.__init__(self,
            end_effector_prim_path= self._end_effector_prim_path,
            joint_prim_names=joint_prim_names,
            joint_opened_positions=joint_opened_positions,
            joint_closed_positions=joint_closed_positions,
            action_deltas=action_deltas)
        return

        
class ROBOTIQ(ParallelGripper):
    """Robotiq hand-e custom gripper, attaches the gripper to the robot

    Args:
        ParallelGripper (_type_): _description_
    """
    def __init__(self, robot_prim_path) -> None:
        prim_path = "/World/hand_e" 
        self._gain = 0.6
        add_reference_to_stage(usd_path= NOVO_NUCLEUS_ASSETS + "Grippers/Robotiq/hand_e.usd", prim_path=prim_path)
        self.gripper_base = "hand_base"
        self.gripper_prim_path= attach_gripper(robot_prim_path=robot_prim_path, 
                                               gripper_prim_path=prim_path,
                                               gripper_name="hand_e",
                                               gripper_base=self.gripper_base)

        self.left_finger_prim_path = self.gripper_prim_path + "/finger_left"
        self.right_finger_prim_path = self.gripper_prim_path + "/finger_right"
        self._end_effector_prim_path = self.gripper_prim_path+"/" + self.gripper_base
        self._end_effector_offset = [0,0,0.18]
        joint_prim_names=["joint_left", "joint_right"]
        joint_opened_positions=np.array([0.0, 0.0]) / get_stage_units()
        joint_closed_positions=np.array([0.02, 0.02]) / get_stage_units()
        action_deltas=np.array([-0.02, -0.02]) / get_stage_units()
        ParallelGripper.__init__(self,
            end_effector_prim_path= self._end_effector_prim_path,
            joint_prim_names=joint_prim_names,
            joint_opened_positions=joint_opened_positions,
            joint_closed_positions=joint_closed_positions,
            action_deltas=action_deltas)
        return 

def attach_gripper_to_robot(robot,robot_prim_path, gripper_type):
    robot._gripper = gripper_type(robot_prim_path)
    robot._end_effector_prim_path = robot._gripper._end_effector_prim_path
        