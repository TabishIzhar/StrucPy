import pathlib
from setuptools import setup
from setuptools import find_packages
import sys

# sys.path.insert(0, './src')

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="StrucPy",
    version="1.0.0",
    description="Object Oriented Structural Analysis",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/TabishIzhar/StrucPy.git",
    author="Tabish Izhar",
    author_email="tizhar@iul.ac.in",
    license="GNU LESSER GENERAL PUBLIC LICENSE v2.1 or later (GNU LGPLv2.1)",
    classifiers=[
        "Programming Language :: Python :: 3.10",
    ],
    packages=["StrucPy"],
    # packages=find_packages(include=['src', 'src/StrucPy']),
    include_package_data=False,
    install_requires=["numpy", "pandas", "plotly", "ray", "openpyxl" ]
)