from setuptools import setup

setup(
    name='lhvqt',
    url='https://github.com/cwitkowitz/LHVQT',
    author='Frank Cwitkowitz',
    author_email='fcwitkow@ur.rochester.edu',
    packages=['lhvqt'],
    install_requires=['numpy', 'librosa', 'torch', 'matplotlib', 'soundfile'],
    version='0.4.0',
    license='MIT',
    description='Frontend filterbank learning module with HVQT initialization capabilities',
    long_description=open('README.md').read()
)
