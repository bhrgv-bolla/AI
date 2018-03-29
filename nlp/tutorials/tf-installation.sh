#!/usr/bin/env bash

################# INSTALL TENSOR FLOW ####################
USER_DIR=$(echo ~)
echo USER_DIR : $USER_DIR

check_pip() {
  echo "CHECKING FOR PIP"
  pip > /dev/null 2> /dev/null && return 0 || return 1
}

if [ check_pip == 1 ]
then
  echo pip not found; installing pip
  sudo easy_install pip
else
  echo pip found
fi

######## Target directory ###########
echo INSTALLING/ UPGRADING VIRTUAL ENV
conda install virtualenv
if [ !$1 ]
then
  TARGET_DIRECTORY=$USER_DIR/tf-virtualenv
else
  TARGET_DIRECTORY=$1
fi
echo TARGET_DIRECTORY is $TARGET_DIRECTORY
mkdir -p $TARGET_DIRECTORY

echo ***INSTALLING SYSTEM SITE PACKAGES***
virtualenv --system-site-packages $TARGET_DIRECTORY

source $TARGET_DIRECTORY/bin/activate

echo UPDATING PIP
easy_install -U pip

echo ****INSTALLING TENSOR FLOW****
pip install --upgrade tensorflow

echo SHORTCUT: alias tfenv=source $(TARGET_DIRECTORY)/bin/activate
