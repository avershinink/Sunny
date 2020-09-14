#pragma once
#include <iostream>
#include "NeuronsLayer.h"

namespace SimpleUndimNeuralNetworkYlem
{
	class NeuralNetwork
	{
		friend std::ostream& operator<<(std::ostream &, NeuralNetwork &);
		friend std::istream& operator>>(std::istream &, NeuralNetwork &);
	public:
		NeuralNetwork();
		NeuralNetwork(int LayersCount, int * NeuronsByLayer, int InputsPerFirstLayerNeurons, ActivationFuncs::Funcs * ActivationsByLayers, double learningRate, double momentum, double decay);
		NeuralNetwork(int LayersCount, int * NeuronsByLayer, int InputsPerFirstLayerNeurons, SimpleUndimNeuralNetworkYlem::ActivationFuncs::Funcs *ActivationsByLayers, double * learningRates, double * momentums, double * decays);
		~NeuralNetwork();

		void Train(int Epochs, double targetAccuracy, std::istream & InputsOutputs, std::istream & TestingInputsOutputs);
		double Accuracy(std::istream & Inputs);

		void Feed(double *);
		void BackPropagate(double *);
		void UpdateWeights(double *);
		
		void GetOutputs(double* &) const;


		NeuronsLayer * * Layers_ = NULL;
	private:
		int layersCount_ = 0;

		void PrintArray(double * const arr, int size)
		{
			std::cout << "[ " << arr[0];
			for (int i = 1; i < size; i++)
				std::cout << ", " << arr[i];
			std::cout << "]";
		}
	};

}