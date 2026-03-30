import argparse
import threading
import time
from multiprocessing import Array, Lock, Value

import logging_mp
import numpy as np

logging_mp.basicConfig(level=logging_mp.INFO)
logger_mp = logging_mp.getLogger(__name__)

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from sshkeyboard import listen_keyboard, stop_listening
from teleimager.image_client import ImageClient
from televuer import TeleVuerWrapper

# for simulation
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,  # dds
    ChannelPublisher,
)
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

from teleop.robot_control.robot_arm import (
    G1_23_ArmController,
    G1_29_ArmController,
    H1_2_ArmController,
    H1_ArmController,
)
from teleop.robot_control.robot_arm_ik import G1_23_ArmIK, G1_29_ArmIK, H1_2_ArmIK, H1_ArmIK
from teleop.utils.episode_writer import EpisodeWriter
from teleop.utils.ipc import IPC_Server
from teleop.utils.motion_switcher import LocoClientWrapper, MotionSwitcher


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def _rotation_to_axis_angle(rot: np.ndarray) -> np.ndarray:
    trace_val = float(np.trace(rot))
    cos_theta = np.clip((trace_val - 1.0) * 0.5, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    if theta < 1e-8:
        return np.zeros(3)

    if np.pi - theta < 1e-5:
        # Near 180 deg, use diagonal terms for numerical stability.
        axis = np.sqrt(np.maximum((np.diag(rot) + 1.0) * 0.5, 0.0))
        if axis[0] < 1e-6 and axis[1] < 1e-6 and axis[2] < 1e-6:
            axis = np.array([1.0, 0.0, 0.0])
        else:
            axis = axis / (np.linalg.norm(axis) + 1e-12)
        return axis * theta

    axis = np.array(
        [
            rot[2, 1] - rot[1, 2],
            rot[0, 2] - rot[2, 0],
            rot[1, 0] - rot[0, 1],
        ]
    ) / (2.0 * np.sin(theta))
    return axis * theta


def _axis_angle_to_rotation(axis_angle: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(axis_angle))
    if theta < 1e-8:
        return np.eye(3) + _skew(axis_angle)
    axis = axis_angle / theta
    k = _skew(axis)
    return np.eye(3) + np.sin(theta) * k + (1.0 - np.cos(theta)) * (k @ k)


def scale_wrist_pose(pose: np.ndarray, ref_pose: np.ndarray, pos_scale: float, rot_scale: float) -> np.ndarray:
    delta = np.linalg.inv(ref_pose) @ pose
    delta_t = delta[:3, 3] * pos_scale
    delta_rot = delta[:3, :3]
    delta_axis_angle = _rotation_to_axis_angle(delta_rot)
    scaled_rot = _axis_angle_to_rotation(delta_axis_angle * rot_scale)

    scaled_delta = np.eye(4)
    scaled_delta[:3, :3] = scaled_rot
    scaled_delta[:3, 3] = delta_t
    return ref_pose @ scaled_delta


def publish_reset_category(category: int, publisher):  # Scene Reset signal
    msg = String_(data=str(category))
    publisher.Write(msg)
    logger_mp.info(f"published reset category: {category}")


def publish_run_command(command, publisher):
    msg = String_(data=str(command))
    publisher.Write(msg)


# state transition
START = False  # Enable to start robot following VR user motion
STOP = False  # Enable to begin system exit procedure
READY = False  # Ready to (1) enter START state, (2) enter RECORD_RUNNING state
RECORD_RUNNING = False  # True if [Recording]
RECORD_TOGGLE = False  # Toggle recording state
#  -------        ---------                -----------                -----------            ---------
#   state          [Ready]      ==>        [Recording]     ==>         [AutoSave]     -->     [Ready]
#  -------        ---------      |         -----------      |         -----------      |     ---------
#   START           True         |manual      True          |manual      True          |        True
#   READY           True         |set         False         |set         False         |auto    True
#   RECORD_RUNNING  False        |to          True          |to          False         |        False
#                                ∨                          ∨                          ∨
#   RECORD_TOGGLE   False       True          False        True          False                  False
#  -------        ---------                -----------                 -----------            ---------
#  ==> manual: when READY is True, set RECORD_TOGGLE=True to transition.
#  --> auto  : Auto-transition after saving data.


def on_press(key):
    global STOP, START, RECORD_TOGGLE
    if key == "r":
        START = True
    elif key == "q":
        START = False
        STOP = True
    elif key == "s" and START == True:
        RECORD_TOGGLE = True
    else:
        logger_mp.warning(f"[on_press] {key} was pressed, but no action is defined for this key.")


def get_state() -> dict:
    """Return current heartbeat state"""
    global START, STOP, RECORD_RUNNING, READY
    return {
        "START": START,
        "STOP": STOP,
        "READY": READY,
        "RECORD_RUNNING": RECORD_RUNNING,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # basic control parameters
    parser.add_argument("--frequency", type=float, default=30.0, help="control and record 's frequency")
    parser.add_argument(
        "--input-mode",
        type=str,
        choices=["hand", "controller"],
        default="hand",
        help="Select XR device input tracking source",
    )
    parser.add_argument(
        "--display-mode",
        type=str,
        choices=["immersive", "ego", "pass-through"],
        default="immersive",
        help="Select XR device display mode",
    )
    parser.add_argument(
        "--arm", type=str, choices=["G1_29", "G1_23", "H1_2", "H1"], default="G1_29", help="Select arm controller"
    )
    parser.add_argument(
        "--ee",
        type=str,
        choices=["dex1", "dex3", "inspire_ftp", "inspire_dfx", "brainco"],
        help="Select end effector controller",
    )
    parser.add_argument(
        "--img-server-ip",
        type=str,
        default="192.168.123.164",
        help="IP address of image server, used by teleimager and televuer",
    )
    parser.add_argument(
        "--network-interface",
        type=str,
        default=None,
        help="Network interface for dds communication, e.g., eth0, wlan0. If None, use default interface.",
    )
    parser.add_argument(
        "--ee-translation-scale",
        type=float,
        default=1.5,
        help="Scale factor for EE translation from XR wrist pose to robot EE target pose.",
    )
    parser.add_argument(
        "--ee-rotation-scale",
        type=float,
        default=1.2,
        help="Scale factor for EE rotation angle (axis-angle) from XR wrist pose to robot EE target pose.",
    )
    parser.add_argument(
        "--control-mode",
        type=str,
        choices=["loco", "manip", "loco-manip"],
        default="loco-manip",
        help="Control arbitration mode: loco only, manip only, or loco + manip.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process XR/VR data normally but skip sending any robot actuation commands.",
    )
    # mode flags
    parser.add_argument("--motion", action="store_true", help="Enable motion control mode")
    parser.add_argument("--headless", action="store_true", help="Enable headless mode (no display)")
    parser.add_argument("--sim", action="store_true", help="Enable isaac simulation mode")
    parser.add_argument(
        "--ipc", action="store_true", help="Enable IPC server to handle input; otherwise enable sshkeyboard"
    )
    parser.add_argument("--affinity", action="store_true", help="Enable high priority and set CPU affinity mode")
    # record mode and task info
    parser.add_argument("--record", action="store_true", help="Enable data recording mode")
    parser.add_argument("--task-dir", type=str, default="./utils/data/", help="path to save data")
    parser.add_argument("--task-name", type=str, default="pick cube", help="task file name for recording")
    parser.add_argument("--task-goal", type=str, default="pick up cube.", help="task goal for recording at json file")
    parser.add_argument(
        "--task-desc", type=str, default="task description", help="task description for recording at json file"
    )
    parser.add_argument(
        "--task-steps",
        type=str,
        default="step1: do this; step2: do that;",
        help="task steps for recording at json file",
    )

    args = parser.parse_args()
    logger_mp.info(f"args: {args}")
    dry_run = args.dry_run

    try:
        # setup dds communication domains id
        if args.sim:
            ChannelFactoryInitialize(1, networkInterface=args.network_interface)
        else:
            ChannelFactoryInitialize(0, networkInterface=args.network_interface)

        # ipc communication mode. client usage: see utils/ipc.py
        if args.ipc:
            ipc_server = IPC_Server(on_press=on_press, get_state=get_state)
            ipc_server.start()
        # sshkeyboard communication mode
        else:
            listen_keyboard_thread = threading.Thread(
                target=listen_keyboard,
                kwargs={
                    "on_press": on_press,
                    "until": None,
                    "sequential": False,
                },
                daemon=True,
            )
            listen_keyboard_thread.start()

        # image client
        img_client = ImageClient(host=args.img_server_ip, request_bgr=True)
        camera_config = img_client.get_cam_config()
        logger_mp.debug(f"Camera config: {camera_config}")
        xr_need_local_img = not (args.display_mode == "pass-through" or camera_config["head_camera"]["enable_webrtc"])

        # televuer_wrapper: obtain hand pose data from the XR device and transmit the robot's head camera image to the XR device.
        tv_wrapper = TeleVuerWrapper(
            use_hand_tracking=args.input_mode == "hand",
            binocular=camera_config["head_camera"]["binocular"],
            img_shape=camera_config["head_camera"]["image_shape"],
            # maybe should decrease fps for better performance?
            # https://github.com/unitreerobotics/xr_teleoperate/issues/172
            # display_fps=camera_config['head_camera']['fps'] ? args.frequency? 30.0?
            display_mode=args.display_mode,
            zmq=camera_config["head_camera"]["enable_zmq"],
            webrtc=camera_config["head_camera"]["enable_webrtc"],
            webrtc_url=f"https://{args.img_server_ip}:{camera_config['head_camera']['webrtc_port']}/offer",
        )

        motion_switcher = None
        loco_wrapper = None
        run_command_publisher = None
        sim_stand_height = 0.75
        control_mode = args.control_mode
        arm_motion_mode = args.motion and (not args.sim)
        controller_input = args.input_mode == "controller"
        enable_loco_control = args.motion and args.input_mode == "controller" and control_mode in ("loco", "loco-manip")
        enable_arm_control = control_mode in ("manip", "loco-manip")
        lock_lower_body_for_arm = control_mode != "loco-manip"
        arm_ctrl = None
        arm_ik = None
        logger_mp.info(
            f"Control mode={control_mode}, enable_loco={enable_loco_control}, "
            f"enable_arm={enable_arm_control}, lock_lower_body_for_arm={lock_lower_body_for_arm}"
        )
        if dry_run:
            logger_mp.info("[DRY-RUN] Enabled: XR/VR data is processed, but no control commands will be sent.")
        # motion mode (G1: Regular mode R1+X, not Running mode R2+A)
        if args.motion:
            if args.sim:
                logger_mp.info("Simulation mode: skip MotionSwitcher; keep arm lowcmd on rt/lowcmd.")
            else:
                if dry_run:
                    logger_mp.info("[DRY-RUN] Skip Exit_Debug_Mode RPC before motion.")
                else:
                    # Ensure robot is switched out of debug mode before sending motion/loco RPC commands.
                    motion_switcher = MotionSwitcher()
                    status, result = motion_switcher.Exit_Debug_Mode()
                    logger_mp.info(f"Exit debug mode for motion: {'Success' if status == 0 else 'Failed'}")
            if controller_input:
                if args.sim:
                    if dry_run:
                        logger_mp.info("[DRY-RUN] Skip simulation run_command publisher initialization.")
                    else:
                        run_command_publisher = ChannelPublisher("rt/run_command/cmd", String_)
                        run_command_publisher.Init()
                        logger_mp.info("Simulation run_command publisher initialized: rt/run_command/cmd")
                else:
                    if dry_run:
                        logger_mp.info("[DRY-RUN] Skip LocoClientWrapper initialization and startup RPC actions.")
                    else:
                        loco_wrapper = LocoClientWrapper()
                        logger_mp.info("LocoClientWrapper initialized for controller safety actions (damp/stop).")
                        if enable_loco_control:
                            move_mode_code = loco_wrapper.Exit_Damp_Mode()
                            logger_mp.info(f"Enter move mode at startup code={move_mode_code}")
        else:
            if args.sim:
                logger_mp.info("Simulation mode: skip MotionSwitcher.")
            else:
                if dry_run:
                    logger_mp.info("[DRY-RUN] Skip Enter_Debug_Mode RPC.")
                else:
                    motion_switcher = MotionSwitcher()
                    status, result = motion_switcher.Enter_Debug_Mode()
                    logger_mp.info(f"Enter debug mode: {'Success' if status == 0 else 'Failed'}")

        # arm
        if enable_arm_control:
            if args.arm == "G1_29":
                arm_ik = G1_29_ArmIK()
                if not dry_run:
                    arm_ctrl = G1_29_ArmController(
                        motion_mode=arm_motion_mode,
                        simulation_mode=args.sim,
                        lock_lower_body_joints=lock_lower_body_for_arm,
                    )
            elif args.arm == "G1_23":
                arm_ik = G1_23_ArmIK()
                if not dry_run:
                    arm_ctrl = G1_23_ArmController(
                        motion_mode=arm_motion_mode,
                        simulation_mode=args.sim,
                        lock_lower_body_joints=lock_lower_body_for_arm,
                    )
            elif args.arm == "H1_2":
                arm_ik = H1_2_ArmIK()
                if not dry_run:
                    arm_ctrl = H1_2_ArmController(
                        motion_mode=arm_motion_mode,
                        simulation_mode=args.sim,
                        lock_lower_body_joints=lock_lower_body_for_arm,
                    )
            elif args.arm == "H1":
                arm_ik = H1_ArmIK()
                if not dry_run:
                    arm_ctrl = H1_ArmController(simulation_mode=args.sim)
        else:
            logger_mp.info("Arm controller and IK are disabled by control mode.")

        if dry_run and enable_arm_control:
            logger_mp.info("[DRY-RUN] Arm IK is active, but arm controller command publishing is disabled.")

        # end-effector
        if args.ee == "dex3":
            from teleop.robot_control.robot_hand_unitree import Dex3_1_Controller

            left_hand_pos_array = Array("d", 75, lock=True)  # [input]
            right_hand_pos_array = Array("d", 75, lock=True)  # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array("d", 14, lock=False)  # [output] current left, right hand state(14) data.
            dual_hand_action_array = Array("d", 14, lock=False)  # [output] current left, right hand action(14) data.
            if not dry_run:
                hand_ctrl = Dex3_1_Controller(
                    left_hand_pos_array,
                    right_hand_pos_array,
                    dual_hand_data_lock,
                    dual_hand_state_array,
                    dual_hand_action_array,
                    simulation_mode=args.sim,
                )
        elif args.ee == "dex1":
            from teleop.robot_control.robot_hand_unitree import Dex1_1_Gripper_Controller

            left_gripper_value = Value("d", 0.0, lock=True)  # [input]
            right_gripper_value = Value("d", 0.0, lock=True)  # [input]
            dual_gripper_data_lock = Lock()
            dual_gripper_state_array = Array("d", 2, lock=False)  # current left, right gripper state(2) data.
            dual_gripper_action_array = Array("d", 2, lock=False)  # current left, right gripper action(2) data.
            if not dry_run:
                gripper_ctrl = Dex1_1_Gripper_Controller(
                    left_gripper_value,
                    right_gripper_value,
                    dual_gripper_data_lock,
                    dual_gripper_state_array,
                    dual_gripper_action_array,
                    simulation_mode=args.sim,
                )
        elif args.ee == "inspire_dfx":
            from teleop.robot_control.robot_hand_inspire import Inspire_Controller_DFX

            left_hand_pos_array = Array("d", 75, lock=True)  # [input]
            right_hand_pos_array = Array("d", 75, lock=True)  # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array("d", 12, lock=False)  # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array("d", 12, lock=False)  # [output] current left, right hand action(12) data.
            if not dry_run:
                hand_ctrl = Inspire_Controller_DFX(
                    left_hand_pos_array,
                    right_hand_pos_array,
                    dual_hand_data_lock,
                    dual_hand_state_array,
                    dual_hand_action_array,
                    simulation_mode=args.sim,
                )
        elif args.ee == "inspire_ftp":
            from teleop.robot_control.robot_hand_inspire import Inspire_Controller_FTP

            left_hand_pos_array = Array("d", 75, lock=True)  # [input]
            right_hand_pos_array = Array("d", 75, lock=True)  # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array("d", 12, lock=False)  # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array("d", 12, lock=False)  # [output] current left, right hand action(12) data.
            if not dry_run:
                hand_ctrl = Inspire_Controller_FTP(
                    left_hand_pos_array,
                    right_hand_pos_array,
                    dual_hand_data_lock,
                    dual_hand_state_array,
                    dual_hand_action_array,
                    simulation_mode=args.sim,
                )
        elif args.ee == "brainco":
            from teleop.robot_control.robot_hand_brainco import Brainco_Controller

            left_hand_pos_array = Array("d", 75, lock=True)  # [input]
            right_hand_pos_array = Array("d", 75, lock=True)  # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array("d", 12, lock=False)  # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array("d", 12, lock=False)  # [output] current left, right hand action(12) data.
            if not dry_run:
                hand_ctrl = Brainco_Controller(
                    left_hand_pos_array,
                    right_hand_pos_array,
                    dual_hand_data_lock,
                    dual_hand_state_array,
                    dual_hand_action_array,
                    simulation_mode=args.sim,
                )
        else:
            pass

        if dry_run and args.ee is not None:
            logger_mp.info("[DRY-RUN] End-effector command publishing is disabled.")

        # affinity mode (if you dont know what it is, then you probably don't need it)
        if args.affinity:
            import psutil

            p = psutil.Process(os.getpid())
            p.cpu_affinity([0, 1, 2, 3])  # Set CPU affinity to cores 0-3
            try:
                p.nice(-20)  # Set highest priority
                logger_mp.info("Set high priority successfully.")
            except psutil.AccessDenied:
                logger_mp.warning("Failed to set high priority. Please run as root.")

            for child in p.children(recursive=True):
                try:
                    logger_mp.info(f"Child process {child.pid} name: {child.name()}")
                    child.cpu_affinity([5, 6])
                    child.nice(-20)
                except psutil.AccessDenied:
                    pass

        # simulation mode
        if args.sim:
            reset_pose_publisher = None
            if not dry_run:
                reset_pose_publisher = ChannelPublisher("rt/reset_pose/cmd", String_)
                reset_pose_publisher.Init()
            else:
                logger_mp.info("[DRY-RUN] Skip simulation reset_pose publisher initialization.")
            from teleop.utils.sim_state_topic import start_sim_state_subscribe

            sim_state_subscriber = start_sim_state_subscribe()

        # record + headless / non-headless mode
        if args.record:
            recorder = EpisodeWriter(
                task_dir=os.path.join(args.task_dir, args.task_name),
                task_goal=args.task_goal,
                task_desc=args.task_desc,
                task_steps=args.task_steps,
                frequency=args.frequency,
                rerun_log=not args.headless,
            )

        logger_mp.info("----------------------------------------------------------------")
        logger_mp.info("🟢  Press [r] to start syncing the robot with your movements.")
        if args.record:
            logger_mp.info("🟡  Press [s] to START or SAVE recording (toggle cycle).")
        else:
            logger_mp.info("🔵  Recording is DISABLED (run with --record to enable).")
        logger_mp.info("🔴  Press [q] to stop and exit the program.")
        logger_mp.info("⚠️  IMPORTANT: Please keep your distance and stay safe.")
        logger_mp.info(
            f"EE pose scale: translation={args.ee_translation_scale}, rotation(axis-angle)={args.ee_rotation_scale}"
        )
        if dry_run:
            logger_mp.info("[DRY-RUN] Active: computed targets are not sent to robot/simulation outputs.")
        READY = True  # now ready to (1) enter START state
        while not START and not STOP:  # wait for start or stop signal.
            time.sleep(0.033)
            if camera_config["head_camera"]["enable_zmq"] and xr_need_local_img:
                head_img = img_client.get_head_frame()
                tv_wrapper.render_to_xr(head_img)

        logger_mp.info("---------------------🚀start Tracking🚀-------------------------")
        if enable_arm_control and arm_ctrl is not None:
            arm_ctrl.speed_gradual_max()
        left_wrist_ref_pose = None
        right_wrist_ref_pose = None
        damp_pressed_last = False
        in_damp_mode = False
        next_loco_log_time = 0.0
        # main loop. robot start to follow VR user's motion
        while not STOP:
            start_time = time.time()
            # get image
            if camera_config["head_camera"]["enable_zmq"]:
                if args.record or xr_need_local_img:
                    head_img = img_client.get_head_frame()
                if xr_need_local_img:
                    tv_wrapper.render_to_xr(head_img)
            if camera_config["left_wrist_camera"]["enable_zmq"]:
                if args.record:
                    left_wrist_img = img_client.get_left_wrist_frame()
            if camera_config["right_wrist_camera"]["enable_zmq"]:
                if args.record:
                    right_wrist_img = img_client.get_right_wrist_frame()

            # record mode
            if args.record and RECORD_TOGGLE:
                RECORD_TOGGLE = False
                if not RECORD_RUNNING:
                    if recorder.create_episode():
                        RECORD_RUNNING = True
                    else:
                        logger_mp.error("Failed to create episode. Recording not started.")
                else:
                    RECORD_RUNNING = False
                    recorder.save_episode()
                    if args.sim and (not dry_run) and (reset_pose_publisher is not None):
                        publish_reset_category(1, reset_pose_publisher)

            # get xr's tele data
            tele_data = tv_wrapper.get_tele_data()
            vx = -tele_data.left_ctrl_thumbstickValue[1] * 1.0
            vy = -tele_data.left_ctrl_thumbstickValue[0] * 1.0
            vyaw = -tele_data.right_ctrl_thumbstickValue[0] * 2.0
            if (
                args.ee == "dex3" or args.ee == "inspire_dfx" or args.ee == "inspire_ftp" or args.ee == "brainco"
            ) and args.input_mode == "hand":
                with left_hand_pos_array.get_lock():
                    left_hand_pos_array[:] = tele_data.left_hand_pos.flatten()
                with right_hand_pos_array.get_lock():
                    right_hand_pos_array[:] = tele_data.right_hand_pos.flatten()
            elif args.ee == "dex1" and args.input_mode == "controller":
                with left_gripper_value.get_lock():
                    left_gripper_value.value = tele_data.left_ctrl_triggerValue
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_ctrl_triggerValue
            elif args.ee == "dex1" and args.input_mode == "hand":
                with left_gripper_value.get_lock():
                    left_gripper_value.value = tele_data.left_hand_pinchValue
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_hand_pinchValue
            else:
                pass
            # print(f"tele_data: left_wrist_pose: {tele_data.left_wrist_pose}, right_wrist_pose: {tele_data.right_wrist_pose}")

            if controller_input and tele_data.right_ctrl_aButton:
                START = False
                STOP = True

            damp_pressed_now = controller_input and tele_data.left_ctrl_thumbstick and tele_data.right_ctrl_thumbstick
            if damp_pressed_now and (not damp_pressed_last):
                if dry_run:
                    logger_mp.info("[DRY-RUN] Dampen command requested; skipped command publish/RPC.")
                    in_damp_mode = True
                elif args.sim and run_command_publisher is not None:
                    publish_run_command([0.0, 0.0, 0.0, sim_stand_height], run_command_publisher)
                    logger_mp.info("Enter damp mode command published in simulation.")
                    in_damp_mode = True
                elif loco_wrapper is not None:
                    code = loco_wrapper.Enter_Damp_Mode()
                    logger_mp.info(f"Enter damp mode code={code}")
                    in_damp_mode = (code == 0)
            damp_pressed_last = damp_pressed_now

            # high level control
            if enable_loco_control:
                # https://github.com/unitreerobotics/xr_teleoperate/issues/135, control, limit velocity to within 0.3
                if dry_run:
                    now = time.time()
                    if now >= next_loco_log_time:
                        logger_mp.info(
                            f"[DRY-RUN] loco target vx={vx:.3f}, vy={vy:.3f}, vyaw={vyaw:.3f}, in_damp={in_damp_mode}"
                        )
                        next_loco_log_time = now + 1.0
                elif args.sim:
                    publish_run_command([vx, vy, vyaw, sim_stand_height], run_command_publisher)
                else:
                    if in_damp_mode and (abs(vx) > 1e-3 or abs(vy) > 1e-3 or abs(vyaw) > 1e-3):
                        recover_code = loco_wrapper.Exit_Damp_Mode()
                        logger_mp.info(f"Exit damp mode code={recover_code}")
                        in_damp_mode = False if recover_code == 0 else in_damp_mode
                    code = loco_wrapper.Move(vx, vy, vyaw)
                    now = time.time()
                    if now >= next_loco_log_time:
                        fsm_code, fsm_id = loco_wrapper.Get_Fsm_Id()
                        logger_mp.info(
                            f"Motion code={code}, fsm_code={fsm_code}, fsm_id={fsm_id}, "
                            f"vx={vx:.3f}, vy={vy:.3f}, vyaw={vyaw:.3f}, in_damp={in_damp_mode}"
                        )
                        next_loco_log_time = now + 1.0

            # get current robot state data and solve ik (disabled during locomotion-only test)
            if enable_arm_control and arm_ik is not None:
                if arm_ctrl is not None:
                    current_lr_arm_q = arm_ctrl.get_current_dual_arm_q()
                    current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()
                else:
                    current_lr_arm_q = np.zeros(14)
                    current_lr_arm_dq = np.zeros(14)

                left_wrist_pose = tele_data.left_wrist_pose
                right_wrist_pose = tele_data.right_wrist_pose
                if args.ee_translation_scale != 1.0 or args.ee_rotation_scale != 1.0:
                    if left_wrist_ref_pose is None or right_wrist_ref_pose is None:
                        left_wrist_ref_pose = left_wrist_pose.copy()
                        right_wrist_ref_pose = right_wrist_pose.copy()
                    left_wrist_pose = scale_wrist_pose(
                        left_wrist_pose, left_wrist_ref_pose, args.ee_translation_scale, args.ee_rotation_scale
                    )
                    right_wrist_pose = scale_wrist_pose(
                        right_wrist_pose, right_wrist_ref_pose, args.ee_translation_scale, args.ee_rotation_scale
                    )

                time_ik_start = time.time()
                sol_q, sol_tauff = arm_ik.solve_ik(
                    left_wrist_pose, right_wrist_pose, current_lr_arm_q, current_lr_arm_dq
                )
                time_ik_end = time.time()
                logger_mp.debug(f"ik:\t{round(time_ik_end - time_ik_start, 6)}")
                if (not dry_run) and (arm_ctrl is not None):
                    arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)
            else:
                current_lr_arm_q = np.zeros(14)
                current_lr_arm_dq = np.zeros(14)
                sol_q = np.zeros(14)
                sol_tauff = np.zeros(14)

            # record data
            if args.record:
                READY = recorder.is_ready()  # now ready to (2) enter RECORD_RUNNING state
                # dex hand or gripper
                if args.ee == "dex3" and args.input_mode == "hand":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:7]
                        right_ee_state = dual_hand_state_array[-7:]
                        left_hand_action = dual_hand_action_array[:7]
                        right_hand_action = dual_hand_action_array[-7:]
                        current_body_state = []
                        current_body_action = []
                elif args.ee == "dex1" and args.input_mode == "hand":
                    with dual_gripper_data_lock:
                        left_ee_state = [dual_gripper_state_array[0]]
                        right_ee_state = [dual_gripper_state_array[1]]
                        left_hand_action = [dual_gripper_action_array[0]]
                        right_hand_action = [dual_gripper_action_array[1]]
                        current_body_state = []
                        current_body_action = []
                elif args.ee == "dex1" and args.input_mode == "controller":
                    with dual_gripper_data_lock:
                        left_ee_state = [dual_gripper_state_array[0]]
                        right_ee_state = [dual_gripper_state_array[1]]
                        left_hand_action = [dual_gripper_action_array[0]]
                        right_hand_action = [dual_gripper_action_array[1]]
                        current_body_state = arm_ctrl.get_current_motor_q().tolist() if arm_ctrl is not None else []
                        current_body_action = [
                            vx,
                            vy,
                            vyaw,
                        ]
                elif (
                    args.ee == "inspire_dfx" or args.ee == "inspire_ftp" or args.ee == "brainco"
                ) and args.input_mode == "hand":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:6]
                        right_ee_state = dual_hand_state_array[-6:]
                        left_hand_action = dual_hand_action_array[:6]
                        right_hand_action = dual_hand_action_array[-6:]
                        current_body_state = []
                        current_body_action = []
                else:
                    left_ee_state = []
                    right_ee_state = []
                    left_hand_action = []
                    right_hand_action = []
                    current_body_state = []
                    current_body_action = []

                # arm state and action
                left_arm_state = current_lr_arm_q[:7]
                right_arm_state = current_lr_arm_q[-7:]
                left_arm_action = sol_q[:7]
                right_arm_action = sol_q[-7:]
                if RECORD_RUNNING:
                    colors = {}
                    depths = {}
                    if camera_config["head_camera"]["binocular"]:
                        if head_img is not None:
                            colors[f"color_{0}"] = head_img.bgr[
                                :, : camera_config["head_camera"]["image_shape"][1] // 2
                            ]
                            colors[f"color_{1}"] = head_img.bgr[
                                :, camera_config["head_camera"]["image_shape"][1] // 2 :
                            ]
                        else:
                            logger_mp.warning("Head image is None!")
                        if camera_config["left_wrist_camera"]["enable_zmq"]:
                            if left_wrist_img is not None:
                                colors[f"color_{2}"] = left_wrist_img.bgr
                            else:
                                logger_mp.warning("Left wrist image is None!")
                        if camera_config["right_wrist_camera"]["enable_zmq"]:
                            if right_wrist_img is not None:
                                colors[f"color_{3}"] = right_wrist_img.bgr
                            else:
                                logger_mp.warning("Right wrist image is None!")
                    else:
                        if head_img is not None:
                            colors[f"color_{0}"] = head_img.bgr
                        else:
                            logger_mp.warning("Head image is None!")
                        if camera_config["left_wrist_camera"]["enable_zmq"]:
                            if left_wrist_img is not None:
                                colors[f"color_{1}"] = left_wrist_img.bgr
                            else:
                                logger_mp.warning("Left wrist image is None!")
                        if camera_config["right_wrist_camera"]["enable_zmq"]:
                            if right_wrist_img is not None:
                                colors[f"color_{2}"] = right_wrist_img.bgr
                            else:
                                logger_mp.warning("Right wrist image is None!")
                    states = {
                        "left_arm": {
                            "qpos": left_arm_state.tolist(),  # numpy.array -> list
                            "qvel": [],
                            "torque": [],
                        },
                        "right_arm": {
                            "qpos": right_arm_state.tolist(),
                            "qvel": [],
                            "torque": [],
                        },
                        "left_ee": {
                            "qpos": left_ee_state,
                            "qvel": [],
                            "torque": [],
                        },
                        "right_ee": {
                            "qpos": right_ee_state,
                            "qvel": [],
                            "torque": [],
                        },
                        "body": {
                            "qpos": current_body_state,
                        },
                    }
                    actions = {
                        "left_arm": {
                            "qpos": left_arm_action.tolist(),
                            "qvel": [],
                            "torque": [],
                        },
                        "right_arm": {
                            "qpos": right_arm_action.tolist(),
                            "qvel": [],
                            "torque": [],
                        },
                        "left_ee": {
                            "qpos": left_hand_action,
                            "qvel": [],
                            "torque": [],
                        },
                        "right_ee": {
                            "qpos": right_hand_action,
                            "qvel": [],
                            "torque": [],
                        },
                        "body": {
                            "qpos": current_body_action,
                        },
                    }
                    if args.sim:
                        sim_state = sim_state_subscriber.read_data()
                        recorder.add_item(
                            colors=colors, depths=depths, states=states, actions=actions, sim_state=sim_state
                        )
                    else:
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions)

            current_time = time.time()
            time_elapsed = current_time - start_time
            sleep_time = max(0, (1 / args.frequency) - time_elapsed)
            time.sleep(sleep_time)
            logger_mp.debug(f"main process sleep: {sleep_time}")

    except KeyboardInterrupt:
        logger_mp.info("⛔ KeyboardInterrupt, exiting program...")
    except Exception:
        import traceback

        logger_mp.error(traceback.format_exc())
    finally:
        try:
            if (not dry_run) and enable_arm_control and arm_ctrl is not None:
                arm_ctrl.ctrl_dual_arm_go_home()
        except Exception as e:
            logger_mp.error(f"Failed to ctrl_dual_arm_go_home: {e}")

        try:
            if args.ipc:
                ipc_server.stop()
            else:
                stop_listening()
                listen_keyboard_thread.join()
        except Exception as e:
            logger_mp.error(f"Failed to stop keyboard listener or ipc server: {e}")

        try:
            img_client.close()
        except Exception as e:
            logger_mp.error(f"Failed to close image client: {e}")

        try:
            tv_wrapper.close()
        except Exception as e:
            logger_mp.error(f"Failed to close televuer wrapper: {e}")

        try:
            if (not dry_run) and (not args.motion) and (not args.sim) and (motion_switcher is not None):
                status, result = motion_switcher.Exit_Debug_Mode()
                logger_mp.info(f"Exit debug mode: {'Success' if status == 3104 else 'Failed'}")
        except Exception as e:
            logger_mp.error(f"Failed to exit debug mode: {e}")

        try:
            if args.sim:
                sim_state_subscriber.stop_subscribe()
        except Exception as e:
            logger_mp.error(f"Failed to stop sim state subscriber: {e}")

        try:
            if args.record:
                recorder.close()
        except Exception as e:
            logger_mp.error(f"Failed to close recorder: {e}")
        logger_mp.info("✅ Finally, exiting program.")
        exit(0)
