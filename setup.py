from setuptools import setup, find_packages

setup(
    name='connect_four',
    version='0.1.0',
    packages=find_packages(include=['Agents', 'Run']),
    # install_requires=[
    #     'pettingzoo',
    #     'tensorflow',
    #     'keras',
    #     'numpy',
    #     'jupyterlab',
    #     'tqdm'
    # ]
)