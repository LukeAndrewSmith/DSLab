from setuptools import setup, find_packages

setup(
    name='ringdetector',
    version='0.0.1',
    description='Tree ring detection',
    author='Yutong (Leona) Xiang, Frederic Boesel, Luke Smith, Clement Guerner',
    packages=find_packages(),
    install_requires=['Pillow',
                      'scikit-learn',
                      'networkx',
                    'shapely',
                    'wandb',
                    'numpy',
                    'opencv-python',
                    'coloredlogs'
                    ],

    classifiers=[
        'Development Status :: 1 - Highly unstable research development ;)',
        'Programming Language :: Python :: 3',
    ],
)