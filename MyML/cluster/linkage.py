class SL():

	def __init__(self):
		pass

	def fit(self,mat,input_type="similarity"):

		self.n_samples = mat.shape[0]

		if input_type is "similarity":
			self.mat = mat
			self.cluster()


	def cluster(mat):

		