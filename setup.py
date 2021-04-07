from setuptools import setup

setup(
    name='lhvqt',
    url='https://github.com/cwitkowitz/LHVQT',
    author='Frank Cwitkowitz',
    author_email='fcwitkow@ur.rochester.edu',
    packages=['lhvqt'],
    install_requires=['numpy', 'librosa', 'torch'],
    version='0.3.0',
    license='MIT',
    description='Fine-tuneable filterbank front-end which implements a HVQT',
    long_description=open('README.md').read()
)
