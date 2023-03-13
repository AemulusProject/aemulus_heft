from setuptools import setup, find_packages

setup(
    name='aemulus_heft',
    version='1.0',
    packages=find_packages(),
    package_dir={'aemulus_heft' : 'aemulus_heft'},
    package_data={'aemulus_heft': ['data/*']},
    long_description=open('README.md').read(),
    )
