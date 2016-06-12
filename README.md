# Archaea
This project data analysis through machine learning on cloud. We basically are building a PaaS for Machine Learning.
The idea is machine learning hosted as a service and any application developer to bring up API's to use them to train and build
their own neural networks and other machine learning tools.We will be building mobile SDK's for the same at later point in this project

# Project Vision
In the initial stages of the project we will basically concentrate on building solid machine learning algorithms with proper tests and
with highest efficiency. Once we are done with those we should move on to building architecture over this using FLASK and mySQL for exposing
this as PaaS. With the PaaS platform in place lets build end-to-end mobile apps to show the use of such an extensive machine learning platform

# Project Progress Tracking
Lets track the features and bugs for the project @ https://www.pivotaltracker.com/n/projects/1579509

# Git Flow.
We use the 'git flow' architecture for git projects.

FAQ:

    1. We will be doing all the development process in the develop branch
    2. Releases will be tracked from master branch
    3. We uses git flow architecture while development (for further
     reference: https://github.com/nvie/gitflow)

# Getting started:

    1. Clone the repo
    2. To create a feature use the command : git flow feature start <pivotal_feature_id>_nameOfTheFeature
    3. When all the development is completed do :

        git rebase master ,
        git reset --hard ORIG_HEAD ,
        git flow feature finish <_nameOfTheFeature>
        git push origin develop

# Installing dependencies
Please use the requirement.txt file for installing all dependencies while setup.
You can use the command **pip install requirement.txt**

# Development Practices

1. Use pip_install.sh to install new dependencies **eg:** ./pip_install.sh <module-name>
2. Before commiting  into the repo run all the tests and make sure that they are passing. You
can run the tests using the command **sudo nosetests**
3. Use the coding standard followed throughout the code base for naming variables


For any info/would like to contribute mail @ satis.vishnu@gmail.com/satis.vishnu@yahoo.com