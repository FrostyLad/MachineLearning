
# Code from Chapter 10 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014


# The Genetic algorithm
# Comment and uncomment fitness functions as appropriate (as an import and the fitnessFunction variable)
import random
from matplotlib import pylab as pl
import numpy as np
import fitness as fit
import fourpeaks as fF


class ga:

	def __init__(self,stringLength,fitnessFunction,nEpochs,populationSize=100,mutationProb=-1,crossover='un',nElite=4,tournament=True):
		""" Constructor"""
		self.stringLength = stringLength
		
		# Population size should be even
		if np.mod(populationSize,2)==0:
			self.populationSize = populationSize
		else:
			self.populationSize = populationSize+1
		
		if mutationProb < 0:
			 self.mutationProb = 1/stringLength
		else:
			 self.mutationProb = mutationProb
			 	  
		self.nEpochs = nEpochs

		self.fitnessFunction = fitnessFunction

		self.crossover = crossover
		self.nElite = nElite
		self.tournment = tournament

		self.population = np.random.rand(self.populationSize,self.stringLength)
		self.population = np.where(self.population<0.5,0,1)
		
	def runGA(self):
		"""The basic loop"""
		pl.ion()
		#plotfig = pl.figure()
		bestfit = np.zeros(self.nEpochs)
		popresults = open("resultoutput.txt", "a")
		fitresults = open("fitnessresults.txt", "a")

		for i in range(self.nEpochs):
			print("----Epoch " + str(i) + "----")
			# Compute fitness of the population
			fitness = eval(self.fitnessFunction)(self.population)


			# Pick parents -- can do in order since they are randomised
			newPopulation = self.fps(self.population,fitness)

			# Apply the genetic operators
			if self.crossover == 'sp':
				newPopulation = self.spCrossover(newPopulation)
			elif self.crossover == 'un':
				newPopulation = self.uniformCrossover(newPopulation)
			newPopulation = self.mutate(newPopulation)

			# Apply elitism and tournaments if using
			if self.nElite>0:
				newPopulation = self.elitism(self.population,newPopulation,fitness)
	
			if self.tournment:
				newPopulation = self.tournament(self.population,newPopulation,fitness,self.fitnessFunction)
	
			self.population = newPopulation
			bestfit[i] = fitness.max()

			if (np.mod(i,100)==0):
				print (i, fitness.max())
			#pl.plot([i],[fitness.max()],'r+')

			popresults.write("Epoch" + str(i) + "\n")
			fitresults.write("Epoch" + str(i) + "\n")
			for popindex in range(len(newPopulation)):
				popresults.write(" ".join(item.astype('str') for item in newPopulation[popindex]))
				popresults.write("\n")
				fitresults.write(fitness[popindex].astype('str') + "\n")

		pl.plot(bestfit,'kx-')
		pl.show()



		popresults.close()
		fitresults.close()
	
	def fps(self,population,fitness):

		# Scale fitness by total fitness
		fitness = fitness/np.sum(fitness)
		fitness = 10.*fitness/fitness.max()
		
		# Put repeated copies of each string in according to fitness
		# Deal with strings with very low fitness
		j=0
		while j<len(fitness) and np.round(fitness[j])<1: 
			j = j+1
		
		newPopulation = np.kron(np.ones((int(np.round(fitness[j])),1)),population[j,:])



		# Add multiple copies of strings into the newPopulation
		for i in range(j+1,self.populationSize):
			if np.round(fitness[i])>=1:

				newPopulation = np.concatenate((newPopulation,np.kron(np.ones((int(np.round(fitness[i])),1)),population[i,:])),axis=0)


		# Shuffle the order (note that there are still too many)
		indices = np.arange(np.shape(newPopulation)[0])

		np.random.shuffle(indices)

		newPopulation = newPopulation[indices[:self.populationSize],:]
		return newPopulation	

	def spCrossover(self,population):
		# Single point crossover
		newPopulation = np.zeros(np.shape(population))
		crossoverPoint = np.random.randint(0,self.stringLength,self.populationSize)
		for i in range(0,self.populationSize,2):
			newPopulation[i,:crossoverPoint[i]] = population[i,:crossoverPoint[i]]
			newPopulation[i+1,:crossoverPoint[i]] = population[i+1,:crossoverPoint[i]]
			newPopulation[i,crossoverPoint[i]:] = population[i+1,crossoverPoint[i]:]
			newPopulation[i+1,crossoverPoint[i]:] = population[i,crossoverPoint[i]:]
		return newPopulation

	def uniformCrossover(self,population):
		# Uniform crossover
		newPopulation = np.zeros(np.shape(population))
		which = np.random.rand(self.populationSize,self.stringLength)
		which1 = which>=0.5
		for i in range(0,self.populationSize,2):
			newPopulation[i,:] = population[i,:]*which1[i,:] + population[i+1,:]*(1-which1[i,:])
			newPopulation[i+1,:] = population[i,:]*(1-which1[i,:]) + population[i+1,:]*which1[i,:]
		return newPopulation
		
	def mutate(self,population):
		# Mutation
		whereMutate = np.random.rand(np.shape(population)[0],np.shape(population)[1])
		population[np.where(whereMutate < self.mutationProb)] = 1 - population[np.where(whereMutate < self.mutationProb)]
		return population

	def elitism(self,oldPopulation,population,fitness):
		best = np.argsort(fitness)
		best = np.squeeze(oldPopulation[best[-self.nElite:],:])
		indices = np.arange(np.shape(population)[0])
		np.random.shuffle(indices)
		population = population[indices,:]
		population[0:self.nElite,:] = best
		return population

	def tournament(self,oldPopulation,population,fitness,fitnessFunction):
		newFitness = eval(self.fitnessFunction)(population)
		for i in range(0,np.shape(population)[0],2):
			f = np.concatenate((fitness[i:i+2],newFitness[i:i+2]),axis=0)
			#f = np.concatenate((fitness[i:i+2],newFitness[i:i+2]),axis=1)
			indices = np.argsort(f)
			if indices[-1]<2 and indices[-2]<2:
				population[i,:] = oldPopulation[i,:]
				population[i+1,:] = oldPopulation[i+1,:]
			elif indices[-1]<2:
				if indices[0]>=2:
					population[i+indices[0]-2,:] = oldPopulation[i+indices[-1]]
				else:
					population[i+indices[1]-2,:] = oldPopulation[i+indices[-1]]
			elif indices[-2]<2:
				if indices[0]>=2:
					population[i+indices[0]-2,:] = oldPopulation[i+indices[-2]]
				else:
					population[i+indices[1]-2,:] = oldPopulation[i+indices[-2]]
		return population
			
