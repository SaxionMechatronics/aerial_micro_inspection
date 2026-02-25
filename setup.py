from setuptools import find_packages, setup
import os
from glob import glob


package_name = 'ai_scanner'


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

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'config'), glob('config/**/*.yaml')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ] + simulation_data_files(),
    install_requires=['setuptools', 'ai_scanner_interfaces'], 
    zip_safe=True,
    maintainer='Hojat Mirtajadini',
    maintainer_email='sehomi755@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'calibrate = ai_scanner.calibrate:main',
            'camera_node = ai_scanner.camera_node:main',
            'gimbal_node = ai_scanner.gimbal_node:main',
            'gimbal_tracker = ai_scanner.gimbal_tracker:main',
            'color_detector = ai_scanner.color_detector:main',
            'surface_segmentor = ai_scanner.surface_segmentor:main',
            'px4_xrce_mission = ai_scanner.px4_xrce_mission:main',
            'gazebo_ros_bridge = ai_scanner.gazebo_ros_bridge:main'
        ],
    },
)
