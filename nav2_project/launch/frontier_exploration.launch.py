"""
frontier_launch.py
Launches the Multi-Robot Frontier Coordinator Node after Nav2 brings up.
"""

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='nav2_project08',
            executable='frontier_coordinator',
            name='frontier_coordinator',
            output='screen',
            parameters=[{'use_sim_time': True}]
        )
    ])

