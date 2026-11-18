from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
from pathlib import Path


def _rviz(context, config):
    """RViz shows the cloud the node relays (/metro/debug/cloud), so any input topic works."""
    return [Node(package='rviz2', executable='rviz2', condition=IfCondition(LaunchConfiguration('rviz')),
                 arguments=['-d', str(Path(config).with_name('metro.rviz'))])]


def generate_launch_description():
    config = str(Path(get_package_share_directory('metro_bringup')) / 'config' / 'detector.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('config', default_value=config),
        DeclareLaunchArgument('rviz', default_value='false'),
        DeclareLaunchArgument('input_topic', default_value='auto'),
        # answers older than this are dropped; a slower demo computer may need 1.0
        DeclareLaunchArgument('max_result_age_s', default_value='0.5'),
        Node(package='metro_perception', executable='metro_obstacle_detector',
             name='metro_obstacle_detector', output='screen',
             parameters=[LaunchConfiguration('config'),
                         {'input_topic': LaunchConfiguration('input_topic'),
                          'max_result_age_s': ParameterValue(LaunchConfiguration('max_result_age_s'), value_type=float)}]),
        OpaqueFunction(function=_rviz, args=[config]),
    ])
