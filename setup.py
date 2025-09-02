from setuptools import find_packages, setup
import os
from glob import glob


package_name = 'ai_scanner'

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
    ],
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
            'tree_trunk_detector = ai_scanner.tree_trunk_detector:main'
        ],
    },
)
