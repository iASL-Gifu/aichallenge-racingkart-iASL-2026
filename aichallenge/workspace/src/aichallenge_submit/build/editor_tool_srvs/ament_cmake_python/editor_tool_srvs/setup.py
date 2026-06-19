from setuptools import find_packages
from setuptools import setup

setup(
    name='editor_tool_srvs',
    version='0.0.0',
    packages=find_packages(
        include=('editor_tool_srvs', 'editor_tool_srvs.*')),
)
