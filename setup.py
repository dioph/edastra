import re

import setuptools

version = re.search(
    '^__version__\\s*=\\s*"(.*)"', open("src/edastra/__init__.py").read(), re.M
).group(1)

with open("README.md", "r") as f:
    long_description = f.read()

install_requires = [
    "astropy",
    "emcee",
    "matplotlib",
    "scipy",
]
extras_require = {
    "dev": [
        "black==20.8b1",
        "flake8",
        "isort",
        "jupyter",
        "tox",
    ]
}

setuptools.setup(
    name="Ed Astra",
    version=version,
    author="Eduardo Nunes",
    author_email="dioph@pm.me",
    license="MIT",
    description="Random astro utilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dioph/edastra",
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    install_requires=install_requires,
    extras_require=extras_require,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
    ],
)
