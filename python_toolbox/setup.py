from setuptools import setup, find_namespace_packages

setup(
    name="python_toolbox",
    version="1.0",
    description="Custom base code module for python",
    url="https://github.com/SEOULTECH-AIS-gyounghun6612/AIS_python_toolbox.git",
    author="Choi_keonghun & Jun_eins",
    author_email="dev.gyounghun6612@gmail.com",
    packages=find_namespace_packages(),
    python_requires=">= 3.11.0",
    zip_safe=False
)
