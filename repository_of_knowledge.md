# Python
## Profiling

This blog                           [post](https://zapier.com/engineering/profiling-python-boss/) describes nicely how to profile Python code. For quick reference, there is **cProfile** and **line_profiler**. The latter is not included by default in Anaconda.

In **cProfile** we just `import cProfile` and profile the code we want with `cProfile.run("testFunc(arg1,arg2)")`. The code should be written as a string so it can be evaluated by `exec`.

# CUDA
## timeout CUDA display

