#Used to create a package and even deploy in pypi (just like pip install seaborn)

from setuptools import find_packages, setup
from typing import List

# function to get requirements from the file

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path:str)->List[str]:

    requirements = []

    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]  # to avoid \n when reading the next line

    # ignore -e .

    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)
    
    return requirements

# setup file informations

setup(
    name                = 'ml_package',
    version             = '0.0.1',
    author              = 'Harish Ahamed',
    author_email        = 'harishahamed26@gmail.com',
    packages            = find_packages(),
    install_requires    = get_requirements('requirements.txt') #[ pandas, numpy,... ]
)