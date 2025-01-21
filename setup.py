from setuptools import setup, find_packages
import os

# Read the requirements from requirements.txt
# def read_requirements():
#     requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
#     with open(requirements_path) as req_file:
#         return req_file.read().splitlines()

from setuptools import setup, find_packages
import os

# Read the requirements from requirements.txt
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    with open(requirements_path) as req_file:
        requirements = req_file.read().splitlines()
        # Exclude git+ lines from install_requires
        return [req for req in requirements if not req.startswith("git+")]

# Extract dependency links for git-based dependencies
def read_dependency_links():
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    with open(requirements_path) as req_file:
        requirements = req_file.read().splitlines()
        # Only include git+ lines
        return [req for req in requirements if req.startswith("git+")]

setup(
    name="BaldrApp",
    version="0.1.0",
    description="Simulating Baldr - the Zernike Wavefront Sensor for Asgard",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Benjamin Courtney-Barrer",
    author_email="benjamin.courtney-barrer@anu.edu.au",
    url="https://github.com/your_username/your_project",
    packages=find_packages(),
    install_requires=read_requirements(),  # Include requirements dynamically
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12.7",
)