import io
import os
from setuptools import find_packages, setup

def read(*paths, **kwargs):
    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content

def read_requirements(path):
    return [
        line.strip()
        for line in read(path).split("\n")
        if not line.startswith(('"', "#", "-", "git+"))
    ]

setup(
    name="comp9517-ass2",
    version="1.0",
    description="Computer vision group project for COMP9517 23T3 Assignment 2",
    url="https://github.com/mmrakib/comp9517-ass2/",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests", ".git", ".github"]),
    install_requires=read_requirements("requirements.txt"),
    entry_points={
        "console_scripts": ["comp9517-ass2 = comp9517-ass2.__main__:main"]
    },
    extras_require={"test": read_requirements("requirements-test.txt")},
)
