from setuptools import setup, find_packages

setup(
    name='azure-speech-to-text',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A Python project to connect to the Azure Speech-to-Text service.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'azure-cognitiveservices-speech',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)