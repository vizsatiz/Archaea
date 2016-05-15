# Archaea
This project data analysis through machine learning on cloud. We basically are building a PaaS for Machine Learning.
The idea is machine learning hosted as a service and any application developer to bring up API's to use them to train and build
their own neural networks and other machine learning tools.We will be building mobile SDK's for the same at later point in this project

#Languages
This project is based on python with implementations of different machine learning algorithms on cloud. Major focus will be on
image recognition with convolution Neural Networks with wavelet Mutli-Resolution Analysis (MRI) for feature extraction.
Other languages include Java, SQL and javascript, angularjs and HTML.


# Plan
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
    3. Development uses git flow architecture while development

Getting started:

    1. Clone the repo
    2. To create a feature use the command : git flow feature start <pivotal_feature_id>_nameOfTheFeature
    3. When all the development is completed do :

        git rebase master ,
        git reset --hard ORIG_HEAD ,
        git flow feature finish <_nameOfTheFeature>
        git push origin develop

For any info/would like to contribute mail @ satis.vishnu@gmail.com/satis.vishnu@yahoo.com