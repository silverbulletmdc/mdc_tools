from setuptools import setup, find_packages

setup(
    name="mdc_tools",
    version="1.0",
    # keywords=("pytorch", "vehicle", "ReID"),
    # description="Vechile ReID utils implemented with pytorch",
    # long_description="",
    packages=find_packages(exclude=('examples', 'examples.*')),
)
