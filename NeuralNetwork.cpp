#include "NeuralNetwork.h"
#include <fstream>
#include <iostream>
using std::cout;
using std::endl;

SimpleUndimNeuralNetworkYlem::NeuralNetwork::NeuralNetwork()
{
}

SimpleUndimNeuralNetworkYlem::NeuralNetwork::NeuralNetwork(int LayersCount, int * NeuronsByLayer, int InputsPerFirstLayerNeurons, SimpleUndimNeuralNetworkYlem::ActivationFuncs::Funcs *ActivationsByLayers, double learningRate, double momentum, double decay) :
	layersCount_(LayersCount)
{
	Layers_ = new SimpleUndimNeuralNetworkYlem::NeuronsLayer*[LayersCount];
	Layers_[0] = new SimpleUndimNeuralNetworkYlem::NeuronsLayer(NeuronsByLayer[0], InputsPerFirstLayerNeurons, learningRate, momentum, decay, ActivationsByLayers[0]);
	for (int i = 1; i < layersCount_; i++)
		Layers_[i] = new SimpleUndimNeuralNetworkYlem::NeuronsLayer(NeuronsByLayer[i], NeuronsByLayer[i - 1], learningRate, momentum, decay, ActivationsByLayers[i]);

}

SimpleUndimNeuralNetworkYlem::NeuralNetwork::NeuralNetwork(int LayersCount, int * NeuronsByLayer, int InputsPerFirstLayerNeurons, SimpleUndimNeuralNetworkYlem::ActivationFuncs::Funcs *ActivationsByLayers, double * learningRates, double * momentums, double * decays) :
	layersCount_(LayersCount)
{
	Layers_ = new SimpleUndimNeuralNetworkYlem::NeuronsLayer*[LayersCount];
	Layers_[0] = new SimpleUndimNeuralNetworkYlem::NeuronsLayer(NeuronsByLayer[0], InputsPerFirstLayerNeurons, learningRates[0], momentums[0], decays[0], ActivationsByLayers[0]);
	for (int i = 1; i < layersCount_; i++)
		Layers_[i] = new SimpleUndimNeuralNetworkYlem::NeuronsLayer(NeuronsByLayer[i], NeuronsByLayer[i - 1], learningRates[i], momentums[i], decays[i], ActivationsByLayers[i]);

}


SimpleUndimNeuralNetworkYlem::NeuralNetwork::~NeuralNetwork()
{
	cout << "Neural Network --> Desctructing NeuralNetwork" << endl;
	for (int i = 0; i < layersCount_; i++)
		delete Layers_[i];
	delete[] Layers_;
}

void SimpleUndimNeuralNetworkYlem::NeuralNetwork::Feed(double * inputs)
{
	Layers_[0]->Feed(inputs);
	for (int i = 1; i < layersCount_; i++)
	{
		double *outputsPtr = Layers_[i - 1]->GetNeuronsOutputs();
		Layers_[i]->Feed(outputsPtr);
		delete[] outputsPtr;
	}
}

void SimpleUndimNeuralNetworkYlem::NeuralNetwork::BackPropagate(double * TargetOutput)
{
	Layers_[layersCount_ - 1]->BackPropagate(TargetOutput);

	for(int l = layersCount_ - 2; l >= 0; l--)
	{
		double * backThis = new double[Layers_[l]->NeuronsCount_];
		for (int i = 0; i < Layers_[l]->NeuronsCount_; i++)
			backThis[i] = 0;
		for (int n = 0; n < Layers_[l]->NeuronsCount_; n++)
		{
			for (int pN = 0; pN < Layers_[l + 1]->NeuronsCount_; pN++)
			{
				backThis[n] += Layers_[l + 1]->Neurons_[pN]->delta_ * Layers_[l + 1]->Neurons_[pN]->weights_[n];
			}
		}
		Layers_[l]->BackPropagate(backThis);
		delete[] backThis;
	}
}

void SimpleUndimNeuralNetworkYlem::NeuralNetwork::UpdateWeights(double * Inputs)
{
	Layers_[0]->UpdateNeurons(Inputs);
	for (int i = 1; i < layersCount_; i++)
	{
		double *outputsPtr = Layers_[i - 1]->GetNeuronsOutputs();
		Layers_[i]->UpdateNeurons(outputsPtr);
		delete[] outputsPtr;
	}
}

double * SimpleUndimNeuralNetworkYlem::NeuralNetwork::GetOutputs(void) const
{
	return Layers_[layersCount_ - 1]->GetNeuronsOutputs();
}

std::ostream & SimpleUndimNeuralNetworkYlem::operator<<(std::ostream & dst, SimpleUndimNeuralNetworkYlem::NeuralNetwork &src)
{
	dst << src.layersCount_ << std::endl;
	for (int i = 0; i < src.layersCount_; i++)
		dst << *src.Layers_[i];
	return dst;
}

std::istream & SimpleUndimNeuralNetworkYlem::operator>>(std::istream & src, SimpleUndimNeuralNetworkYlem::NeuralNetwork &dst)
{

	for (int i = 0; i < dst.layersCount_; i++)
		delete dst.Layers_[i];
	src >> dst.layersCount_;
	delete[] dst.Layers_;

	dst.Layers_ = new NeuronsLayer*[dst.layersCount_];
	for (int i = 0; i < dst.layersCount_; i++)
	{
		NeuronsLayer *nl = new NeuronsLayer;
		src >> *nl;
		dst.Layers_[i] = nl;
	}
	return src;
}