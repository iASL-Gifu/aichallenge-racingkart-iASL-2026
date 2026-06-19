from setuptools import find_packages
from setuptools import setup

setup(
    name='tiny_lidar_net_controller',
    version='0.0.0',
    packages=find_packages(
        include=('tiny_lidar_net_controller', 'tiny_lidar_net_controller.*')),
)
