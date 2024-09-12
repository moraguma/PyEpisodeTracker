from setuptools import setup, find_namespace_packages

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='pyepisodetracker',
    long_description=long_description,
    long_description_content_type='text/markdown',
    version='0.1.0',    
    description='Episode Tracker module for Python',
    url='https://github.com/moraguma/PyEpisodeTracker',
    author='Moraguma',
    author_email='g170603@dac.unicamp.br',
    license='MIT',
    packages=find_namespace_packages(),
    install_requires=[
        'setuptools>=70.0.0',
        'torchbringer',
        'pygame',
    ],

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',  
        'Operating System :: POSIX :: Linux',    
        'Environment :: GPU :: NVIDIA CUDA',  
        'Programming Language :: Python'
    ],
)