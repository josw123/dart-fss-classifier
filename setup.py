from setuptools import setup, find_packages
from os import path

import versioneer

NAME = 'dart-fss-classifier'

INSTALL_REQUIRES = (
    ['dart-fss>=0.2.0', 'tensorflow', 'konlpy']
)


with open(path.join('./', 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name=NAME,
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description='Plugin for dart-fss',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Sungwoo Jo',
    author_email='nonswing.z@gmail.com',
    url='https://github.com/josw123/dart-fss-classifier',
    license='GPLv3+',
    classifiers=[
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3 :: Only'
    ],
    packages=find_packages(),
    keywords=['fss', 'dart-fss', 'scrapping', 'plugin'],
    python_requires='>=3.5',
    package_data={},
    install_requires=INSTALL_REQUIRES,
    include_package_data=True
)
