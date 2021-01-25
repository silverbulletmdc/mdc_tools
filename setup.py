from setuptools import setup, find_packages

setup(
    name="mdc_tools",
    version="1.2",
    author="Dechao Meng",
    author_email="dechao.meng@vipl.ict.ac.cn",
    url="https://github.com/silverbulletmdc/mdc_tools",
    # keywords=("pytorch", "vehicle", "ReID"),
    description="Personal tools",
    # long_description="",
    packages=find_packages(exclude=('examples', 'examples.*')),
)
