import re

from setuptools import setup

version = re.search(
    '^__version__\\s*=\\s*"(.*)"',
    open('edastra/__init__.py').read(),
    re.M
).group(1)

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
      name="Ed Astra",
      version=version,
      author="Eduardo Nunes",
      author_email="dioph@pm.me",
      license="MIT",
      description="Random astro utilities",
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/dioph/edastra",
      packages=["edastra"],
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Intended Audience :: Science/Research",
      ],
)
