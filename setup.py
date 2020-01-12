import sys
from setuptools import setup, find_packages
from distutils.version import LooseVersion

if sys.version_info.major != 3:
    print('This Python is only compatible with Python 3, but you are running '
          'Python {}. The installation will likely fail.'.format(sys.version_info.major))


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
          'tensorflow-probability>=0.8.0',
          'tensorflow>=2.1.0'
      ],
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
      author_email='github@hill-a.me',
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
