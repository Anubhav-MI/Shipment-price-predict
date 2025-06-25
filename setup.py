from setuptools import find_packages,setup
from typing import List


def get_requirements()->List[str]:
    """
    this fucntion will return the list of requirements
    """
    requirement_list:List[str] =[]
    try:
        with open('requirements.txt','r') as file:
            #read line form the file
            lines = file.readlines()
            #process each line
            for line in lines:
                requirement = line.strip()
                #ignore empty lines and -e .
                if requirement and requirement != '-e .':
                    requirement_list.append(requirement)

    except FileNotFoundError:
        print("requirements.txt file not found.")
    return requirement_list

print(get_requirements())   

setup(
    name = "SHIPMENT PRICE PREDICTION",
    version="0.0.1",
    author="Ayush Raj",
    author_email="peka021@gmail.com",
    packages=find_packages(),
    install_requires = get_requirements()
    
)

