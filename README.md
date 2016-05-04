# Archaea
This project data analysis through machine learning on cloud. We basically are building a PaaS for Machine Learning.
The idea of machine learing hosted as a service and any application developer can bring up API's to use the same to train and build
a neural network of there own through our sdk's

# Vision
This project is based on python with implementations of different machine learning algorithms on cloud. Major focus will be on
image recognition with convolution Neural Networks with wavelet mutli resolution analysis for feature extraction.

# Coding Languages
In the initial stages of the project we will basically concentrate on building solid machine learning algorithms with proper tests and
with maximum efficiency. Once we are done with those we should move on to building architecture over this using FLASK and mySQL for exposing
this as service.Then when the services are exposed lets build a mobile app with the same using the services exposed and use it to
validate the idea.

# Project Progress Tracking
Lets track the features and bugs for the project @ https://www.pivotaltracker.com/n/projects/1579509

# Git Flow.
We use the git flow architecture for development purpose.

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

For any info/would like to contribute mail @ satis.vishnu@gmail.com