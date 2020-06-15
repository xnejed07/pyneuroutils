from setuptools import setup,find_packages

setup(
    name='pyneuroutils',
    version='1.0.9',
    packages=['pyneuroutils.statistics','pyneuroutils.transforms','pyneuroutils.utilities','pyneuroutils.datasets','pyneuroutils.mefd'],
    url='https://github.com/xnejed07/pyneuroutils',
    license='MIT',
    author='Petr Nejedly',
    author_email='xnejed07@gmail.com',
    description='', install_requires=['numpy', 'scikit-learn', 'matplotlib', 'pandas']
)
