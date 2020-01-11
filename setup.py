import sys
import subprocess
from setuptools import setup, find_packages
from distutils.version import LooseVersion

if sys.version_info.major != 3:
    print('This Python is only compatible with Python 3, but you are running '
          'Python {}. The installation will likely fail.'.format(sys.version_info.major))

# Check tensorflow installation to avoid
# breaking pre-installed tf gpu
install_tf, tf_gpu = False, False
try:
    import tensorflow as tf
    if tf.__version__ < LooseVersion('2.0.0'):
        install_tf = True
        # check if a gpu version is needed
        tf_gpu = tf.test.is_gpu_available()
except ImportError:
    install_tf = True
    # Check if a nvidia gpu is present
    for command in ['nvidia-smi', '/usr/bin/nvidia-smi', 'nvidia-smi.exe']:
        try:
            if subprocess.call([command]) == 0:
                tf_gpu = True
                break
        except IOError:  # command does not exist / is not executable
            pass

tf_dependency = []
if install_tf:
    tf_dependency = ['tensorflow-gpu>=2.0.0'] if tf_gpu else ['tensorflow>=2.0.0']
    if tf_gpu:
        print("A GPU was detected, tensorflow-gpu will be installed")


long_description = """

# Stable Baselines TF2 [Experimental]

"""

setup(name='stable_baselines',
      packages=[package for package in find_packages()
                if package.startswith('stable_baselines')],
      package_data={
          'stable_baselines': ['py.typed'],
      },
      install_requires=[
          'gym[atari,classic_control]>=0.10.9',
          'scipy',
          'joblib',
          'cloudpickle>=0.5.5',
          'opencv-python',
          'numpy',
          'pandas',
          'matplotlib',
          'tensorflow-probability>=0.8.0'
      ] + tf_dependency,
      extras_require={
        'tests': [
            'pytest',
            'pytest-cov',
            'pytest-env',
            'pytest-xdist',
            'pytype',
        ],
        'docs': [
            'sphinx',
            'sphinx-autobuild',
            'sphinx-rtd-theme'
        ]
      },
      description='A fork of OpenAI Baselines, implementations of reinforcement learning algorithms.',
      author='Ashley Hill',
      url='https://github.com/hill-a/stable-baselines',
      author_email='ashley.hill@u-psud.fr',
      keywords="reinforcement-learning-algorithms reinforcement-learning machine-learning "
               "gym openai baselines toolbox python data-science",
      license="MIT",
      long_description=long_description,
      long_description_content_type='text/markdown',
      version="3.0.0a0",
      )

# python setup.py sdist
# python setup.py bdist_wheel
# twine upload --repository-url https://test.pypi.org/legacy/ dist/*
# twine upload dist/*
