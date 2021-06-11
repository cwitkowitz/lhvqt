from setuptools import setup, find_packages

# TODO - print(lhvqt.__version__) - what else am I missing?
# TODO - update for current torch version
# TODO - none-any.whl?

setup(
    name='lhvqt',
    url='https://github.com/cwitkowitz/LHVQT',
    author='Frank Cwitkowitz',
    author_email='fcwitkow@ur.rochester.edu',
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=['numpy', 'librosa', 'torch==1.4.0', 'matplotlib', 'soundfile'],
    version='0.4.2',
    license='MIT',
    description='Frontend filterbank learning module with HVQT initialization capabilities',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown'
)
