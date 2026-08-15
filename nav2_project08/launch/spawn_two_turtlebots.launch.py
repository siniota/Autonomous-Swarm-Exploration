"""
spawn_two_turtlebots.launch.py

Spawns two namespaced TurtleBot3 Burgers into a single Gazebo Classic world.

This is adapted directly from turtlebot3_gazebo's own multi_robot.launch.py
(ROBOTIS, Apache-2.0) rather than hand-rolled, because that file already
solves the two things that are easy to get wrong doing this from scratch:

  1. The plugin-bearing robot model is models/turtlebot3_<model>/model.sdf,
     NOT the plain URDF — the URDF alone has no diff-drive or lidar plugin,
     so spawning from it gives you a robot you can't drive and that
     publishes no /scan or /odom.
  2. The SDF's <odometry_frame>, <robot_base_frame>, and <frame_name> (scan)
     tags must be patched to namespaced values (robot1/odom, etc.) BEFORE
     spawning, and robot_state_publisher's frame_prefix must produce the
     exact same names, or the TF tree between the plugin and
     robot_state_publisher won't connect.

Run with:
  ros2 launch nav2_project08 spawn_two_turtlebots.launch.py

Requires TURTLEBOT3_MODEL to be exported (e.g. export TURTLEBOT3_MODEL=burger)
"""

import os
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import GroupAction, IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnShutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import PushRosNamespace


# ---- Robot roster: name, spawn pose -----------------------------------
ROBOTS = [
    {"name": "robot1", "x": -2.0, "y": -0.5},
    {"name": "robot2", "x": 2.0, "y": 0.5},
]


def generate_launch_description():
    turtlebot3_model = os.environ["TURTLEBOT3_MODEL"]  # raises clearly if unset

    tb3_gazebo_dir = get_package_share_directory("turtlebot3_gazebo")
    launch_file_dir = os.path.join(tb3_gazebo_dir, "launch")
    pkg_gazebo_ros = get_package_share_directory("gazebo_ros")

    model_folder = f"turtlebot3_{turtlebot3_model}"
    sdf_template_path = os.path.join(
        tb3_gazebo_dir, "models", model_folder, "model.sdf"
    )
    tmp_prefix = os.path.join(tb3_gazebo_dir, "models", model_folder, "tmp")

    world = os.path.join(tb3_gazebo_dir, "worlds", "turtlebot3_world.world")
    use_sim_time = LaunchConfiguration("use_sim_time", default="true")

    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, "launch", "gzserver.launch.py")
        ),
        launch_arguments={"world": world}.items(),
    )
    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, "launch", "gzclient.launch.py")
        )
    )

    ld = LaunchDescription([gzserver_cmd, gzclient_cmd])
    generated_sdf_paths = []

    for robot in ROBOTS:
        name = robot["name"]

        # Patch the SDF's plugin frame names so odom/base/scan TF frames are
        # namespaced per robot.
        tree = ET.parse(sdf_template_path)
        root = tree.getroot()
        for tag in root.iter("odometry_frame"):
            tag.text = f"{name}/odom"
        for tag in root.iter("robot_base_frame"):
            tag.text = f"{name}/base_footprint"
        for tag in root.iter("frame_name"):
            tag.text = f"{name}/base_scan"
        sdf_out_path = f"{tmp_prefix}_{name}.sdf"
        with open(sdf_out_path, "w") as f:
            f.write('<?xml version="1.0" ?>\n' + ET.tostring(root, encoding="unicode"))
        generated_sdf_paths.append(sdf_out_path)

        # frame_prefix argument gets '/' appended internally by
        # robot_state_publisher.launch.py -> produces 'robot1/base_footprint'
        # etc., matching the SDF patch above exactly.
        robot_state_publisher_cmd = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(launch_file_dir, "robot_state_publisher.launch.py")
            ),
            launch_arguments={
                "use_sim_time": use_sim_time,
                "frame_prefix": name,
            }.items(),
        )

        spawn_cmd = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(launch_file_dir, "multi_spawn_turtlebot3.launch.py")
            ),
            launch_arguments={
                "x_pose": str(robot["x"]),
                "y_pose": str(robot["y"]),
                "robot_name": f"{turtlebot3_model}_{name}",
                "namespace": name,
                "sdf_path": sdf_out_path,
            }.items(),
        )

        ld.add_action(
            GroupAction([PushRosNamespace(name), robot_state_publisher_cmd, spawn_cmd])
        )

    def _cleanup_tmp_sdfs(event, context):
        for p in generated_sdf_paths:
            try:
                os.remove(p)
            except OSError:
                pass

    ld.add_action(RegisterEventHandler(OnShutdown(on_shutdown=_cleanup_tmp_sdfs)))

    return ld
