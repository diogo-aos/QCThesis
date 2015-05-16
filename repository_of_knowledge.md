# Writing
## Sublime: Default vs User settings
Use the *default* settings to see which options are available and then copy those you wish to modify to the *user* settings. This will keep your settings on upgrades.

tag: sublime

## Writing academic with markdown

http://blog.cigrainger.com/2014/07/pandoc-markdown.html

Use citer. Pandoc produces ugly HTML - see workaround.


# Programming
## git
`git clone [-b <mybranch>] [--single-branch] <rep> [dest_folder]`

This will clone a specific branch *mybranch* from a repository *rep* into the folder *dest_folder* and speficy that only that branch should be cloned (*--single-branch*).

`git remote set-url origin git@github.com/Username/Repository.git`

This will change https to ssh.

## Challenges
[HackerRank](https://www.hackerrank.com)

# Linux
## Mount with SSH
[Ubuntu wiki](https://help.ubuntu.com/community/SSHFS)
[digitalocean](https://www.digitalocean.com/community/tutorials/how-to-use-sshfs-to-mount-remote-file-systems-over-ssh)

tag: ssh,filesystem

## Keep SSH alive
tag:ssh

[link](http://www.maketecheasier.com/keep-ssh-connections-alive-in-linux/)

Edit or create the config file `$HOME/.ssh/config`

Add the following:
```
Host *
  ServerAliveInterval 60
```

This will send a package every 60 seconds of inactivity. Change as desired. Save and exit. Restart SSH service:

`sudo service ssh restart`

This may be applied system wide in the `/etc/ssh/ssh_config` config file.

## Share folders through Samba

[see this instructions](https://help.ubuntu.com/community/How%20to%20Create%20a%20Network%20Share%20Via%20Samba%20Via%20CLI%20%28Command-line%20interface/Linux%20Terminal%29%20-%20Uncomplicated,%20Simple%20and%20Brief%20Way!)

tag:samba,network
date:25-04-2015


## NVIDIA + Intel
use driver 331
[askUbuntu](http://askubuntu.com/questions/452556/how-to-set-up-nvidia-optimus-bumblebee-in-14-04)
http://askubuntu.com/questions/457446/ubuntu-14-04-nvidia-prime-is-it-supported-no

tag:nvidia-prime

## Installing ubuntu restricted drivers through command line
tag: linux,ubuntu

### For 12.04 and below

The additional drivers program has a command line interface, jockey-text:

Use

`jockey-text --list`

to get a list of available drivers and their status, then use

the init lines are required only for graphics drivers
```bash
sudo init 1
jockey-text --enable=DRIVER
sudo init 2
```

where DRIVER is the one you got from the list. For example:

`jockey-text --enable=firmware:b43`

To install the Broadcom B43 wireless driver.

For your graphics card, you will get a choice of the proprietary driver from the manufacturer and a free alternative. You have to either restart the system entirely (recommended) or restart the display server:
- log out and back in
- Depending on what display manager, you can use one of the following commands:
    + Default Ubuntu (with LightDM): `sudo restart lightdm `
    + Gnome (with GDM): `sudo restart gdm`
    + KDE (with KDM): `sudo restart kdm`
    + For MDM: `sudo restart mdm`

Note: From 12.10, Kubuntu also uses LightDM.



### For 14.04
`sudo ubuntu-drivers list`

Will show all the driver packages which apply to your current system. You can then

`sudo ubuntu-drivers autoinstall `

to install all the packages you need, or you can do:

`sudo ubuntu-drivers devices`

to show you which devices need drivers, and their corresponding package names.



# Python
## Profiling

This blog [post](https://zapier.com/engineering/profiling-python-boss/) describes nicely how to profile Python code. For quick reference, there is **cProfile** and **line_profiler**. The latter is not included by default in Anaconda.

In **cProfile** we just `import cProfile` and profile the code we want with `cProfile.run("testFunc(arg1,arg2)")`. The code should be written as a string so it can be evaluated by `exec`.

also [this](http://pynash.org/2013/03/06/timing-and-profiling.html)

## IPython connect to remote SSH
A notebook creates a kernel-id.json file in IPYTHONDIR/profile/security/. Get this file. Execute:

ipython qtconsole --existing kernel-id.json --ssh myusername@myserver


Requires: pexpect package

tag:ipython,ssh

# CUDA
## timeout CUDA display

## NumbaPro + CUDA
To install NumbaPro, the simplest way is to install Anaconda and then do

```
conda update conda
conda install accelerate
```

This will not have CUDA working though. The CUDA driver's installation is the responsibility of the user. At the time of writing the appropriate driver is CUDA 5.5. To install download from internet ([link for Linux 64](http://developer.download.nvidia.com/compute/cuda/5_5/rel/installers/cuda_5.5.22_linux_64.run)) and run the `.run` file.

# Testing
## virtualenv

[nice primer](http://simononsoftware.com/virtualenv-tutorial/)

*virtualenv* is useful to test Python programs in different environments, e.g. Python version, the existence or not of certain packages, etc.

Get started with:

```bash
user@host:$ mkdir -P virt_envs
user@host:$ virtualenv --no-site-packages virt1
```

`--no-site-packages` is so the virtual environment will not symlink to existing packages in the machine. This way, we get a bare clean environment to test on. To activate/desactivate the environment:

```bash
user@host:$ source virt_envs/virt1/bin/activate
(virt1)user@host:$ desactivate
```

# Android
## copy files with ADB
tag: android,wifi

Activate wireless adb on Android device.
On host execute `adb connect IP` where IP is the device's IP address. Then, to transfer files use `adb pull /path/to/remote/file /path/to/local` or `adb push /path/to/local /path/to/remote`.

## How I returned a Samsung W I8150 to stock

Odin
use clockworkrom (CWR) to wipe everything
burn new image

you can also use the odin to install CWR and then install a ROM from CWR or install an image directly from Odin.