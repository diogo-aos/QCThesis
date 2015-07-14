# -*- coding: utf-8 -*-

"""
author: Diogo Silva
Utils.
"""

from timeit import default_timer as timer


class Timer():
    def __init__(self, ID = None):
    	if ID is not None:
  			self.id = str(ID)
        self.start = None
        self.end = None

    def tic(self):
        self.start = timer()

    def tac(self):
        self.end = timer()
        self.elapsed = self.end - self.start
        return self.elapsed

#    def __repr__(self):

    def __str__(self):
    	id_str = ""
    	if hasattr(self, 'id'):
    		id_str = self.id + " : "

		if hasattr(self, 'elapsed'):
		    return id_str + "Elapsed time: {} s, {} ms".format(self.elapsed, self.elapsed * 1000)
		else:
			return id_str + "No time recorded yet."


# def main():
# 	pass

# if __name__ == "__main__":
# 	main()







