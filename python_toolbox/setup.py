from setuptools import setup, find_namespace_packages


requirments_package = [
    'numpy',
    'opencv-python',
    'pyyaml',
    'flake8',
    'matplotlib']

dependency_links = []
package_opt = {}

setup(
    name="python_ex",
    version="1.1.0",
    description="Custom base code module for python",
    url="https://github.com/SEOULTECH-AIS-gyounghun6612/AIS_python_ex.git",
    author="Choi_keonghun & Jun_eins",
    author_email="dev.gyounghun6612@gmail.com",
    packages=find_namespace_packages(),
    python_requires=">= 3.10.0",
    dependency_links=dependency_links,
    install_requires=requirments_package,
    zip_safe=False
)
