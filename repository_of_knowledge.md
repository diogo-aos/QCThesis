# Writing
## Sublime: Default vs User settings
Use the *default* settings to see which options are available and then copy those you wish to modify to the *user* settings. This will keep your settings on upgrades.

tag: sublime

## Writing academic with markdown

http://blog.cigrainger.com/2014/07/pandoc-markdown.html

Use citer. Pandoc produces ugly HTML - see workaround.


# Programming
## Challenges
[HackerRank](https://www.hackerrank.com)

# Linux
## Mount with SSH
[Ubuntu wiki](https://help.ubuntu.com/community/SSHFS)
[digitalocean](https://www.digitalocean.com/community/tutorials/how-to-use-sshfs-to-mount-remote-file-systems-over-ssh)

tag: ssh,filesystem

## NVIDIA + Intel
use driver 331
[askUbuntu](http://askubuntu.com/questions/452556/how-to-set-up-nvidia-optimus-bumblebee-in-14-04)
http://askubuntu.com/questions/457446/ubuntu-14-04-nvidia-prime-is-it-supported-no

tag:nvidia-prime



# Python
## Profiling

This blog                           [post](https://zapier.com/engineering/profiling-python-boss/) describes nicely how to profile Python code. For quick reference, there is **cProfile** and **line_profiler**. The latter is not included by default in Anaconda.

In **cProfile** we just `import cProfile` and profile the code we want with `cProfile.run("testFunc(arg1,arg2)")`. The code should be written as a string so it can be evaluated by `exec`.

# CUDA
## timeout CUDA display

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

