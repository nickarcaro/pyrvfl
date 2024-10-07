from setuptools import setup, find_packages


setup(
    name="pyrvfl",
    version="1.0",
    authors="Nicolás Araya, Pablo Henríquez",
    author_email=["nicolas.arayac@mail.udp.cl"],
    description="Python implementation of Random Vector Functional Link (RVFL) networks",
    license="MIT",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/nickarcaro/pyrvfl",
    packages=find_packages(),
)
