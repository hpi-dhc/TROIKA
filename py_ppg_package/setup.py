from setuptools import setup, find_packages

setup(
    name='ppg_package',
    version='0.1.0',
    description='some PPG stuff',
    author='Jost Morgenstern',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'sklearn',
        'scipy',
        'pypg',
        'pyts',
        'plotly',
        'tqdm'
    ]
)