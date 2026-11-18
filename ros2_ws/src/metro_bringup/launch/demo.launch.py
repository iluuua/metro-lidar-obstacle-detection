"""One-command demo: detector + ros2 bag play + console status (+ optional RViz)."""
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _rviz(context, config):
    """RViz shows the cloud the node relays (/metro/debug/cloud), so any input topic works."""
    return [Node(package='rviz2', executable='rviz2', condition=IfCondition(LaunchConfiguration('rviz')),
                 arguments=['-d', str(Path(config).with_name('metro.rviz'))])]


def generate_launch_description():
    config = str(Path(get_package_share_directory('metro_bringup')) / 'config' / 'detector.yaml')
    bag = LaunchConfiguration('bag')
    return LaunchDescription([
        DeclareLaunchArgument('bag', description='Path to a ROS 2 bag directory'),
        DeclareLaunchArgument('input_topic', default_value='auto'),
        DeclareLaunchArgument('rate', default_value='1.0'),
        DeclareLaunchArgument('rviz', default_value='false'),
        # A demo laptop may be slower than the train computer: accept answers up to 1 s old
        # instead of dropping them (detector.launch.py keeps 0.5 s).
        DeclareLaunchArgument('max_result_age_s', default_value='1.0'),
        Node(package='metro_perception', executable='metro_obstacle_detector', name='metro_obstacle_detector',
             output='screen', parameters=[config, {'input_topic': LaunchConfiguration('input_topic'),
                                                   'max_result_age_s': ParameterValue(
                                                       LaunchConfiguration('max_result_age_s'), value_type=float)}]),
        Node(package='metro_perception', executable='metro_status_echo', output='screen'),
        OpaqueFunction(function=_rviz, args=[config]),
        # Give the geometry worker time to start before the first cloud. A short read-ahead
        # queue: the player's clock runs while the queue fills, and 1000 clouds of 3-9 MB
        # (the default) take seconds to load, after which the first seconds come in a burst.
        TimerAction(period=5.0, actions=[ExecuteProcess(
            cmd=['ros2', 'bag', 'play', bag, '--rate', LaunchConfiguration('rate'),
                 '--read-ahead-queue-size', '20', '--disable-keyboard-controls'], output='screen')]),
    ])
