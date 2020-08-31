#pragma once
#include "NeuronsLayer.h"

namespace SimpleUndimNeuralNetworkYlem
{
	class NeuralNetwork
	{
	public:
		NeuralNetwork();
		NeuralNetwork(int LayersCount, int * NeuronsByLayer, int InputsPerFirstLayerNeurons, ActivationFuncs::Funcs * ActivationsByLayers, double learningRate, double momentum, double decay);
		NeuralNetwork(int LayersCount, int * NeuronsByLayer, int InputsPerFirstLayerNeurons, SimpleUndimNeuralNetworkYlem::ActivationFuncs::Funcs *ActivationsByLayers, double * learningRates, double * momentums, double * decays);
		~NeuralNetwork();

		void Feed(double *);
		void BackPropagate(double *);
		void UpdateWeights(double *);
		
		double * GetOutputs(void) const;

		NeuronsLayer * * Layers_ = NULL;
	private:
		int layersCount_ = 0;
	};

}