from setuptools import setup

setup(
    name="AIStorch_ex",
    version="0.1.0",
    description="Custom base code module for python",
    url="https://github.com/SEOULTECH-AIS-gyounghun6612/torch_base_utils.git",
    author="Choi_keonghun & Jun_eins",
    author_email="dev.gyounghun6612@gmail.com",
    packages=["torch_utils"],
    zip_safe=False,
    install_requires=[
        "torch", "python-ex"
    ]
)
