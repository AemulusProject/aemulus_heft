from setuptools import setup, find_packages
import os

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

extra_files = package_files('aemulus_heft/data/nn_weights')
extra_files.append('data/training_data.json')
extra_files.append('data/nn_cfg.yaml')

setup(
    name='aemulus_heft',
    version='1.1',
    packages=find_packages(),
    package_dir={'aemulus_heft' : 'aemulus_heft'},
    package_data={'aemulus_heft': extra_files},
    long_description=open('README.md').read(),
    )
