# This command file can be used to install packages and along with that saving the package name to
# requirements file.
#!/usr/bin/env bash
package_name=$1
requirements_file=$2
if [[ -z $requirements_file ]]
then
    requirements_file='./requirements.txt'
fi
sudo pip install $package_name && pip freeze | grep -i $package_name >> $requirements_file


