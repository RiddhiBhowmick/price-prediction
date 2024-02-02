from setuptools import setup,find_packages
from typing import List


def get_requirements(file_name:str)->List[str]:
    requirements=[]

    HYPHEN_DOT="-e."
    with open(file_name) as file:
        requirements=file.readlines()
        requirements=[req.replace("/n","")for req in requirements]

        if HYPHEN_DOT in requirements:
            requirements.remove(HYPHEN_DOT)
        
    return requirements




setup(
    name="end to end car price prediction ml project",
    version="0.0.1",
    author="riddhi bhomwick",
    author_email="riddhibhowmick45@gmail.com",
    packages=find_packages(),
   
    install_requires=get_requirements("requirements.txt")
 )
