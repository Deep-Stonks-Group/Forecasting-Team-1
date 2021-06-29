from setuptools import find_packages, setup

setup(
    name = 'forecast-team-1',
    packages = find_packages(),
    include_package_data = True,
    dependency_links = [
     "git+https://github.com/Deep-Stonks-Group/PythonDataProcessing",
    ],
)
