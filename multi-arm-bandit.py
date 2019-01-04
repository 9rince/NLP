import numpy as np 
import pprint
# import random


class bandits():

	def __init__(self,arms=10):
		self.arms = arms
		self.state = np.zeros(arms)
		self.rewards_mean = np.random.uniform(high=10,low=0,size=arms)
		self.qtable = np.zeros([self.arms,2])

	def pull(self,i):
		if self.states[i]==1:
			self.states[i] = 0
		else:
			self.states[i]=1

	def get_reward(self,i):

		return np.random.normal(loc=self.rewards_mean[i])


nitish = bandits()
print(nitish.state,'\n',nitish.rewards_mean,'\n',nitish.qtable)
