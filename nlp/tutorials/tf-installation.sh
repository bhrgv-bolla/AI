#!/usr/bin/env bash

################# INSTALL TENSOR FLOW ####################

check_pip() {
  echo "Checking for pip"
  pip > /dev/null 2> /dev/null && return 0 || return 1
}

if [ check_pip == 1 ]
then
  echo pip not found; installing pip
  sudo easy_install pip
else
  echo pip found
fi

echo Installing/ Upgrading virtual env
pip install --upgrade virtualenv > /dev/null
target_directory='~/tf-virtualenv'
if [ !$1 ]
then
  target_directory='~/tf-virtualenv'
else
  target_directory=$1
fi
echo target directory is $target_directory
mkdir -p $target_directory
# virtualenv --system-site-packages ~/tf-virtualenv
