from setuptools import find_packages, setup
import os
from glob import glob


package_name = 'aerial_micro_inspection'


def simulation_data_files():
    entries = []
    simulation_root = 'simulation'
    for root, _, files in os.walk(simulation_root):
        if not files:
            continue
        rel = os.path.relpath(root, simulation_root)
        if rel == '.':
            target = os.path.join('share', package_name, simulation_root)
        else:
            target = os.path.join('share', package_name, simulation_root, rel)
        entries.append((target, [os.path.join(root, f) for f in files]))
    return entries


def config_data_files():
    entries = []
    config_root = 'config'
    for root, _, files in os.walk(config_root):
        if not files:
            continue
        rel = os.path.relpath(root, config_root)
        if rel == '.':
            target = os.path.join('share', package_name, config_root)
        else:
            target = os.path.join('share', package_name, config_root, rel)
        entries.append((target, [os.path.join(root, f) for f in files]))
    return entries

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ] + config_data_files() + simulation_data_files(),
    install_requires=['setuptools', 'aerial_micro_inspection_interfaces'], 
    zip_safe=True,
    maintainer='Hojat Mirtajadini',
    maintainer_email='sehomi755@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'calibrate = aerial_micro_inspection.calibrate:main',
            'camera_node = aerial_micro_inspection.camera_node:main',
            'gimbal_node = aerial_micro_inspection.gimbal_node:main',
            'gimbal_tracker = aerial_micro_inspection.gimbal_tracker:main',
            'color_detector = aerial_micro_inspection.color_detector:main',
            'surface_segmentor = aerial_micro_inspection.surface_segmentor:main',
            'micro_detector = aerial_micro_inspection.micro_detector:main',
            'px4_xrce_mission = aerial_micro_inspection.px4_xrce_mission:main',
            'structural_inspection_mission = aerial_micro_inspection.orthogonal_path_planning.structural_inspection_mission:main',
            'structural_inspection_planner = aerial_micro_inspection.orthogonal_path_planning.structural_inspection_planner:main',
            'save_image = aerial_micro_inspection.orthogonal_path_planning.save_image:main',
            'real_test_save_image = aerial_micro_inspection.orthogonal_path_planning.real_test_save_image:main',
            'gazebo_ros_bridge = aerial_micro_inspection.gazebo_ros_bridge:main'
        ],
    },
)
