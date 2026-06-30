from setuptools import find_packages
from setuptools import setup

setup(
    name='pilot_net_controller',
    version='0.0.0',
    packages=find_packages(
        include=('pilot_net_controller', 'pilot_net_controller.*')),
)
