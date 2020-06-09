
from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['requests==2.20.0'] #required for python GCS client

setup(
    name='sustainable_model',
    version='1.0',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='model for text classification'
)
