from setuptools import find_packages, setup

NAME = 'trustyai-ood'
VERSION = '1.0'

kserve_requires = [
    "kserve",
]

rhods_requires = [
    "kubernetes"
]

setup(
    name=NAME,
    version=VERSION,
    description='A Library for enabling models with Out-of-Distribution (OOD) detection capabilities',
    packages=find_packages(),
    url='https://github.ibm.com/Distributed-Data/ood',
    author='Shalisha Witherspoon',
    author_email='shalisha.witherspoon@ibm.com',
    install_requires=['torch==1.13.1', 'torchvision==0.14.1', 'tensorflow==2.11.0', 'ibm-cos-sdk==2.12.2',
                      'Flask==2.2.3', 'celery[redis]==5.2.7', 'marshmallow==3.19.0', 'apispec==6.2.0',
                      'apispec-webframeworks==0.5.2', 'torch-model-archiver>=0.7.1', 'onnx', 'numpy==1.23.5',
                      'onnx2pytorch', 'tf2onnx', 'python-dotenv', 'tqdm', 'kubernetes', 'torchextractor', 'ultralytics',
                       'kserve', 'openshift'],
    extras_require={
        "kserve": kserve_requires, "rhods": rhods_requires,
    },
    python_requires='>=3.8',
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
    ],
    zip_safe=False)







