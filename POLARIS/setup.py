from setuptools import setup, find_packages

setup(
    name='polaris',
    version='0.0.0',
    description='Post-training recipe for advanced reasoning models.',
    author='Chenxin An',
    packages=find_packages(include=['deepscaler',]),
    install_requires=[
        'google-cloud-aiplatform',
        'latex2sympy2',
        'pylatexenc',
        'sentence_transformers',
        'tabulate',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache 2.0 License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)