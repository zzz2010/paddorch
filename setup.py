import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="paddorch", # Replace with your own username
    version="0.4.8",
    author="Zhizhuo Zhang",
    author_email="zzz2010@gmail.com",
    description="paddle implementation for pytorch interface",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zzz2010/paddle_torch",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
