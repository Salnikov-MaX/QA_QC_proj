import shutil

from setuptools import setup, find_packages


def read_file(path: str) -> str:
    with open(path, 'r') as f:
        return f.read()


setup(
    name='qaqcgeo',
    version='0.1',
    author='HW',
    author_email='info@hw.tpu.ru',
    description='Library for assessing the quality of geological data',
    long_description=read_file('readme.md'),
    long_description_content_type='text/markdown',
    url='hwtpu.ru',
    packages=find_packages(),
    install_requires=read_file('requirements_ubuntu.txt').split('\n'),
    classifiers=[
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'
    ],
    keywords='files speedfiles',
    project_urls={
        'GitHub': 'https://github.com/Salnikov-MaX/QA_QC_proj'
    },
    python_requires='>=3.9'
)

shutil.rmtree('qaqcgeo.egg-info')
shutil.rmtree('build')
