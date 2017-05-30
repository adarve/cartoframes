"""
cartoframes
https://github.com/CartoDB/cartoframes
"""

# TODO: update this once carto-python is a module
# NOTE: first do `pip install -r requirements.txt` to get setup

import os
from codecs import open
from setuptools import setup

with open(os.path.join(os.path.abspath(os.path.dirname(__file__)),
                       'README.rst'),
          encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()

setup(
    name='cartoframes',
    version='0.2.1-beta',
    description='An experimental Python pandas interface for using CARTO',
    long_description=LONG_DESCRIPTION,
    url='https://github.com/CartoDB/cartoframes',
    author='Andy Eschbacher',
    author_email='andy@carto.com',
    license='BSD',
    classifiers=[
        'Development Status :: 3 - Beta',
        'Intended Audience :: Data Scientists',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6'
    ],
    keywords='data science maps spatial pandas carto',
    packages=['cartoframes'],
    install_requires=['pandas>=0.20.1',
                      'webcolors>=1.7.0',
                      'pyrestcli>=0.6.3',],
    package_data={
        '': ['LICENSE', 'CONTRIBUTORS',],
    },
)
