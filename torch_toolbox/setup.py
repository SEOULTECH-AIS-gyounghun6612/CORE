from setuptools import setup


requirments_package = [
    'python_ex',
    'torch',
    'einops',
    'torchsummary']

dependency_links = [
    "git+https://github.com/SEOULTECH-AIS-gyounghun6612/AIS_python_ex.git@ver_alpha"
]
package_opt = {}


setup(
    name="torch_ex",
    version="1.0.0",
    description="Custom base code module for pytorch",
    url="https://github.com/SEOULTECH-AIS-gyounghun6612/AIS_torch_ex.git",
    author="Choi_keonghun & Jun_eins",
    author_email="dev.gyounghun6612@gmail.com",
    packages=["torch_ex"],
    python_requires=">= 3.7.0",
    dependency_links=dependency_links,
    install_requires=requirments_package,
    zip_safe=False,
)
