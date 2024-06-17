import copy
import typing
import numpy as np
from scipy.spatial.transform import Rotation as R

from omni.isaac.motion_generation.lula import utils
from omni.isaac.core.prims.xform_prim import XFormPrim
import omni.graph.core as og
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.prims import get_prim_at_path
import omni.graph.tools.ogn as ogn
import lula
from pxr import Usd, UsdGeom, Gf
import omni.kit.commands


def indicate_node_exec_running(node: og.Node):
    """
    Indicate by selection that an action graph node is executing.

    Args:
        node (og.Node): The node which executing.
    """
    ctx = omni.usd.get_context()
    # Select node in action graph
    current_selections = ctx.get_selection().get_selected_prim_paths()
    new_selections = copy.deepcopy(current_selections)
    new_selections.append(node.get_prim_path())
    omni.kit.commands.execute('SelectPrimsCommand',
                              old_selected_paths=current_selections,
                              new_selected_paths=new_selections,
                              expand_in_stage=False)


def indicate_node_exec_ended(node: og.Node):
    """
    Indicate by de-selection that an action graph node is no longer executing.

    Args:
        node (og.Node): The node which is no longer executing.
    """
    ctx = omni.usd.get_context()
    # Deselect node in Action graph
    current_selections = ctx.get_selection().get_selected_prim_paths()
    new_selections = copy.deepcopy(current_selections)
    if node.get_prim_path() in new_selections:
        new_selections.remove(node.get_prim_path())
        omni.kit.commands.execute('SelectPrimsCommand',
                                  old_selected_paths=current_selections,
                                  new_selected_paths=new_selections,
                                  expand_in_stage=False)


def get_list_of_robot_names():
    robot_names = []
    robots = get_prim_at_path("/World/Robots").GetChildren()
    if len(robots) > 0:
        for child in robots:
            robot_names.append(child.GetName())
    return robot_names


def toggle_visibility(prim: Usd.Prim = None, prim_path: str = None):
    prim_path = prim_path
    if prim_path is None:
        if prim is not None:
            prim_path = prim.GetPath()
        else:
            raise Exception("toggle_Visibility need either prim or prim_path")
    omni.kit.commands.execute(
        "ToggleVisibilitySelectedPrims", selected_paths=[prim_path]
    )


def toggle_run_as_simulated(prim: Usd.Prim, state: bool):
    children = prim.GetAllChildren()
    for child in children:
        try:
            child.GetAttribute("inputs:runInSimulation").Set(
                state
            )  # for some reason does not work
        except:
            pass
    return


def remove_collisions_recursively(prim: Usd.Prim):
    children = prim.GetAllChildren()
    collision = prim.GetAttribute("physics:collisionEnabled")
    try:
        collision.Set(False)
    except:
        pass
    for child in children:
        remove_collisions_recursively(child)
    return


def move_prim(path_from, path_to):
    """Move a prim from one tree to a different tree

    Args:
        path_from (str): Path of the prim to be moved
        path_to (_type_): Path of the target prim
    """
    omni.kit.commands.execute(
        "MovePrim",
        path_from=path_from,
        path_to=path_to,
        keep_world_transform=False,
        destructive=False,
    )


def get_pose3(trans=None, rot_mat=None, rot_quat=None) -> lula.Pose3:
    """Generate a pose3 from translation and rotation matrix or quaternion

    Args:
        trans (np.array([3,1])): translation (XYZ). Defaults to None.
        rot_mat (np.array[3,3]): Rotation matrix. Defaults to None.
        rot_quat (np.array([4,1])): Unit Quaternion ([w,x,y,z]). Defaults to None.

    Returns:
        lula.Pose3: _description_
    """
    return utils.get_pose3(trans=trans, rot_mat=rot_mat, rot_quat=rot_quat)


def get_prim_pose3(prim: XFormPrim):
    """Get Pose3 from Prim

    Args:
        prim (XFormPrim): Prim

    Returns:
        lula.Pose3: _description_
    """
    pose = prim.get_world_pose()
    return get_pose3(trans=pose[0], rot_quat=pose[1])


def pose3_to_quat(pose3):
    """Returns unit quaternion from pose3

    Args:
        pose (pose3):

    Returns:
        np.array([w, x, y, z]): Unit quaternion
    """
    return np.array(
        [pose3.rotation.w(), pose3.rotation.x(), pose3.rotation.y(), pose3.rotation.z()]
    )


def get_ur_pose_from_isaac_pose(pose):
    """Returns a pose compatible with the UR pose representation (xyz, axis angle)

    Args:
        isaac pose [xyz, quat]

    Returns:
        list [x, y, z, RX, RY, RZ]
    """
    position = pose[0]
    axis_angle = R.from_quat(pose[1]).as_rotvec()

    return np.concatenate((position, axis_angle)).tolist()


def get_world_transform_xform(
    prim: Usd.Prim,
) -> typing.Tuple[Gf.Vec3d, Gf.Rotation, Gf.Vec3d]:
    """
    Get the local transformation of a prim using Xformable.
    See https://graphics.pixar.com/usd/release/api/class_usd_geom_xformable.html
    Args:
        prim: The prim to calculate the world transformation.
    Returns:
        A tuple of:
        - Translation vector.
        - Rotation quaternion, i.e. 3d vector plus angle.
        - Scale vector.
    """
    xform = UsdGeom.Xformable(prim)
    time = Usd.TimeCode.Default()  # The time at which we compute the bounding box
    world_transform: Gf.Matrix4d = xform.ComputeLocalToWorldTransform(time)
    translation: Gf.Vec3d = world_transform.ExtractTranslation()
    rotation: Gf.Rotation = world_transform.ExtractRotation()
    scale: Gf.Vec3d = Gf.Vec3d(
        *(v.GetLength() for v in world_transform.ExtractRotationMatrix())
    )
    return translation, rotation, scale


def get_isaac_pose_from_ur_pose(ur_pose):
    """
    Get the equivalent UR pose from an isaac pose
    Args:
        ur_pose: list: x,y,z, rotvec.
    Returns:
        A list of:
            - x, y, z, quat
    """

    return ur_pose[0:3], R.from_rotvec(ur_pose[3:6]).as_quat()


def define_pose_in_reference_frame(db, prim_path, pose_db, pose_name):
    # create a prim_path with the reference frame path as root (ref_path/pose_name + unique node handle)
    # TODO: Change Node handle to the unique node name (better for debugging/ the user)
    pose = pose_db.get_pose_from_db(db, pose_name)
    frame_prim = XFormPrim(prim_path, name=pose_name + str(db.node.get_handle()))
    pos, quat = get_isaac_pose_from_ur_pose(pose)
    frame_prim.set_world_pose(position=pos, orientation=quat)
    return pose


def get_robot_pose_from_frame(frame_prim_path, robot_name):
    """
    Gets the pose of a frame in base coordinates
    """
    frame_prim = XFormPrim(prim_path=frame_prim_path)
    robot_prim = XFormPrim(prim_path="/World/Robots/" + robot_name)
    T_world_base = np.eye(4)
    robot_base_isaac_pose = robot_prim.get_world_pose()
    robot_base_isaac_pos = robot_base_isaac_pose[0]
    robot_base_isaac_quat = robot_base_isaac_pose[1]
    T_world_base[:3, 3] = robot_base_isaac_pos
    T_world_base[:3, :3] = R.from_quat(
        np.array(
            [
                robot_base_isaac_quat[1],
                robot_base_isaac_quat[2],
                robot_base_isaac_quat[3],
                robot_base_isaac_quat[0],
            ]
        )
    ).as_matrix()
    print("T_world_base: \n", T_world_base)
    T_world_base_rotvec = R.from_matrix(
        T_world_base[:3, :3]
    ).as_rotvec()  # Only for printing
    print("T_world_base rotvec: ", T_world_base_rotvec)

    frame_isaac_pose = frame_prim.get_world_pose()
    frame_isaac_pose_pos = frame_isaac_pose[0]
    frame_isaac_pose_quat = frame_isaac_pose[1]
    T_world_frame = np.eye(4)
    T_world_frame[:3, 3] = frame_isaac_pose_pos
    T_world_frame[:3, :3] = R.from_quat(
        np.array(
            [
                frame_isaac_pose_quat[1],
                frame_isaac_pose_quat[2],
                frame_isaac_pose_quat[3],
                frame_isaac_pose_quat[0],
            ]
        )
    ).as_matrix()
    print("T_world_frame: \n", T_world_frame)
    T_world_frame_rotvec = R.from_matrix(
        T_world_frame[:3, :3]
    ).as_rotvec()  # Only for printing
    print("T_world_frame rotvec: ", T_world_frame_rotvec)

    # get frame in robot base
    T_base_world = np.eye(4)
    T_base_world[:3, :3] = T_world_base[:3, :3].T
    T_base_world[:3, 3] = np.dot(-T_world_base[:3, :3].T, T_world_base[:3, 3])

    T_base_frame = T_base_world @ T_world_frame
    print("T_base_frame: \n", T_base_frame)

    r = R.from_matrix(T_base_frame[:3, :3])
    T_base_frame_pos = T_base_frame[:3, 3]
    T_base_frame_rotvec = r.as_rotvec()
    print("T_base_frame rotvec: ", T_base_frame_rotvec)
    cartesian_pose = np.concatenate((T_base_frame_pos, T_base_frame_rotvec)).tolist()

    return cartesian_pose


def get_pose_from_frame_in_world(frame_prim_path):
    """
    Gets the pose of a frame in world coordinates
    """
    frame_prim = XFormPrim(prim_path=frame_prim_path)
    frame_isaac_pose = frame_prim.get_world_pose()
    frame_isaac_pose_pos = frame_isaac_pose[0]
    frame_isaac_pose_quat = frame_isaac_pose[1]
    frame_isaac_pose_rotvec = R.from_quat(
        np.array(
            [
                frame_isaac_pose_quat[1],
                frame_isaac_pose_quat[2],
                frame_isaac_pose_quat[3],
                frame_isaac_pose_quat[0],
            ]
        )
    ).as_rotvec()
    
    final_frame_pose = np.concatenate((frame_isaac_pose_pos, frame_isaac_pose_rotvec)).tolist()
    return final_frame_pose


def get_pose_from_reference_frame(db, prim_path, pose_name):
    frame_prim = XFormPrim(
        prim_path=prim_path, name=pose_name + str(db.node.get_handle())
    )
    cartesian_pose = get_ur_pose_from_isaac_pose(frame_prim.get_world_pose())
    # cartesian_pose[3:6] =  pose_db.get_pose_from_db(db, db.inputs.pose)[3:6]
    return cartesian_pose


def get_pose_in_reference_frame(db, pose_db, reference_prim_path, pose_name):
    prim_path = reference_prim_path + "/" + pose_name + str(db.node.get_handle())
    # check if a pose with the unique name alredy exists and use it to generate cartesian pose
    if is_prim_path_valid(prim_path):
        return get_pose_from_reference_frame(db, prim_path, pose_name)
    else:
        return define_pose_in_reference_frame(db, prim_path, pose_db, pose_name)


def create_dynamic_attribute(node, attrib_name, attrib_port_type, attrib_type):
    """This function creates a dynamic attribute for a node and adds it to the gui

    Args:
        node: object linking to the node
        attrib_name (string): name of the attribute to be created
        attrib_port_type (og.AttributePortType): og.AttributePortType.ATTRIBUTE_PORT_TYPE_INPUT or
        og.AttributePortType.ATTRIBUTE_PORT_TYPE_OUTPUT
            attrib_type_name = "string" / "int" / ....
        attrib_type (og.AttributeType): og.AttributeType.type_from_ogn_type_name(attrib_type_name)

    Returns:
        _type_: _description_
    """

    if (
        attrib_name == "execOut"
        and attrib_port_type == og.AttributePortType.ATTRIBUTE_PORT_TYPE_OUTPUT
        and attrib_type.get_ogn_type_name() == "execution"
    ):
        # Unhide outputs:execOut instead of creating it
        node.get_attribute("outputs:execOut").set_metadata(
            ogn.MetadataKeys.HIDDEN, None
        )
        return

    new_attribute = og.Controller.create_attribute(
        node, attrib_name, attrib_type, attrib_port_type
    )
    if new_attribute is None:
        return

    # Not used for our case, but kept just in case
    # if attrib_type.get_type_name() == "prim" and attrib_port_type in (
    #     og.AttributePortType.ATTRIBUTE_PORT_TYPE_OUTPUT,
    #     og.AttributePortType.ATTRIBUTE_PORT_TYPE_STATE,
    # ):
    #     # For bundle output/state attribs, the default UI name contains the port type, so we set it here instead
    #     def make_ui_name(attrib_name: str):
    #         parts_out = []
    #         words = attrib_name.replace("_", " ").split(" ")
    #         for word in words:  # noqa: PLR1702
    #             if word.islower() or word.isupper():
    #                 parts_out += [word]
    #             else:
    #                 # Mixed case.
    #                 # Lower-case followed by upper-case breaks between them. E.g. 'usdPrim' -> 'usd Prim'
    #                 # Upper-case followed by lower-case breaks before them. E.g: 'USDPrim' -> 'USD Prim'
    #                 # Combined example: abcDEFgHi -> abc DE Fg Hi
    #                 sub_word = ""
    #                 uppers = ""
    #                 for c in word:
    #                     if c.isupper():
    #                         if not uppers:  # noqa: SIM102
    #                             if sub_word:
    #                                 parts_out += [sub_word]
    #                                 sub_word = ""
    #                         uppers += c
    #                     else:
    #                         if len(uppers) > 1:
    #                             parts_out += [uppers[:-1]]
    #                         sub_word += uppers[-1:] + c
    #                         uppers = ""

    #                 if sub_word:
    #                     parts_out += [sub_word]
    #                 elif uppers:
    #                     parts_out += [uppers]

    #         # Title-case any words which are all lower case.
    #         parts_out = [part.title() if part.islower() else part for part in parts_out]
    #         return " ".join(parts_out)

    #         new_attribute.set_metadata(ogn.MetadataKeys.UI_NAME, make_ui_name(attrib_name))


def remove_dynamic_attribute(attrib: str = "inputs:Gripper_IP"):
    success = og.Controller.remove_attribute(attrib)
    if not success:
        return
