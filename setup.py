"""Package the model and optional dissertation experiment modules."""
from setuptools import find_packages, setup

setup(
    packages=find_packages("src") + find_packages(include=["experiments", "experiments.*"]),
    package_dir={"": "src", "experiments": "experiments"},
    package_data={"experiments": ["**/*.json", "**/*.csv", "**/*.md", "**/*.txt"]},
)
