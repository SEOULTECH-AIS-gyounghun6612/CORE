from setuptools import setup


requirments_package = [
    'numpy',
    'opencv-python',
    'pyyaml',
    'flake8']

dependency_links = []
package_opt = {"python_ex": ["data_file/*.json"]}

setup(
    name="python_ex",
    version="1.0.0",
    description="Custom base code module for python",
    url="https://github.com/SEOULTECH-AIS-gyounghun6612/AIS_python_ex.git",
    author="Choi_keonghun & Jun_eins",
    author_email="dev.gyounghun6612@gmail.com",
    packages=["python_ex"],
    python_requires=">= 3.7.0",
    dependency_links=dependency_links,
    install_requires=requirments_package,
    package_data=package_opt,
    zip_safe=False
)
