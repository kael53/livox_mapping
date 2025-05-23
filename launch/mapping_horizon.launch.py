from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.conditions import IfCondition
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_dir = get_package_share_directory('livox_mapping')
    rviz_config_file = os.path.join(pkg_dir, 'rviz_cfg', 'loam_livox.rviz')

    rviz_arg = DeclareLaunchArgument(
        'rviz',
        default_value='true',
        description='Whether to start RViz'
    )

    scan_registration = Node(
        package='livox_mapping',
        executable='loam_scanRegistration_horizon',
        name='scanRegistration_horizon',
        output='screen'
    )

    laser_mapping = Node(
        package='livox_mapping',
        executable='loam_laserMapping',
        name='laserMapping',
        output='screen',
        parameters=[{
            'map_file_path': ' ',
            'filter_parameter_corner': 0.2,
            'filter_parameter_surf': 0.4
        }]
    )

    livox_repub = Node(
        package='livox_mapping',
        executable='livox_repub',
        name='livox_repub',
        output='screen',
        remappings=[
            ('/livox/lidar', '/livox/lidar')
        ]
    )

    rviz_node = Node(
        condition=IfCondition(LaunchConfiguration('rviz')),
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file]
    )

    return LaunchDescription([
        rviz_arg,
        scan_registration,
        laser_mapping,
        livox_repub,
        rviz_node
    ])
