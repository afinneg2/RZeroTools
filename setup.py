from setuptools import setup, Extension

setup(
    name='RZeroTools',
    version='0.1.0',
    author='AJA Finnegan',
    author_email='finnegan2.alex@gmail.com',
    packages=['RZeroTools',],
    license='GNU GENERAL PUBLIC LICENSE v3',
    long_description=open('README.md').read(),
    install_requires=['numpy >= 1.18.1', 'scipy >= 1.4.1', 'matplotlib >= 3.2.1', 'pandas >= 1.0.3',
		      'seaborn >= 0.10.0', "pymc3 >= 3.8" ],
)
