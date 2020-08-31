
#include <iostream>
#include "NeuronsLayer.h"

namespace Sunny = SimpleUndimNeuralNetworkYlem;

Sunny::NeuronsLayer::NeuronsLayer(int NeuronsCount, int InputsPerNeuronCount, double LearningRate, double Momentum, double Decay, ActivationFuncs::Funcs UseActivation) :
	NeuronsCount_(NeuronsCount),
	InputsPerNeuronCount_(InputsPerNeuronCount),
	LayerCommonFunc_(UseActivation)
{
	Neurons_ = new Neuron*[NeuronsCount_];
	NeuronFunc Act = NULL;
	NeuronFunc Der = NULL;
	switch (LayerCommonFunc_)
	{
	case SimpleUndimNeuralNetworkYlem::ActivationFuncs::IdentityFunc:
		Act = ActivationFuncs::Identity;
		Der = ActivationFuncs::IdentityDerivative;
		break;
	case SimpleUndimNeuralNetworkYlem::ActivationFuncs::SigmoidFunc:
		Act = ActivationFuncs::Sigmoid;
		Der = ActivationFuncs::SigmoidDerivative;
		break;
	case SimpleUndimNeuralNetworkYlem::ActivationFuncs::ReLUFunc:
		Act = ActivationFuncs::ReLU;
		Der = ActivationFuncs::ReLUDerivative;
		break;
	case SimpleUndimNeuralNetworkYlem::ActivationFuncs::PReLUFunc:
		Act = ActivationFuncs::PReLU;
		Der = ActivationFuncs::PReLUDerivative;
		break;
	case SimpleUndimNeuralNetworkYlem::ActivationFuncs::HyperbolicTangentFunc:
		Act = ActivationFuncs::HyperbolicTangent;
		Der = ActivationFuncs::HyperbolicTangentDerivative;
		break;
	case SimpleUndimNeuralNetworkYlem::ActivationFuncs::SoftMaxFunc:
		// Act -- NULL this is layer level function
		Der = ActivationFuncs::SoftMaxDerivative;
		break;
	}

	for (int i = 0; i < NeuronsCount_; i++)
		Neurons_[i] = new Neuron(InputsPerNeuronCount_, LearningRate, Momentum, Decay, Act, Der);
	
}


SimpleUndimNeuralNetworkYlem::NeuronsLayer::NeuronsLayer(const NeuronsLayer & rhs)
{
	Copy(rhs);
}

SimpleUndimNeuralNetworkYlem::NeuronsLayer & SimpleUndimNeuralNetworkYlem::NeuronsLayer::operator=(const NeuronsLayer &rsh)
{
	if (this == &rsh)
		return *this;
	delete[] Neurons_;
	Copy(rsh);
	return *this;
}

SimpleUndimNeuralNetworkYlem::NeuronsLayer::~NeuronsLayer()
{
	for(int i = 0; i < NeuronsCount_; i++)
	    delete Neurons_[i];
	delete[] Neurons_;
}


void SimpleUndimNeuralNetworkYlem::NeuronsLayer::Copy(const SimpleUndimNeuralNetworkYlem::NeuronsLayer & rhs)
{
	NeuronsCount_ = rhs.NeuronsCount_;
	InputsPerNeuronCount_ = rhs.InputsPerNeuronCount_;

	Neurons_ = new Neuron*[NeuronsCount_];
	for (int i = 0; i < NeuronsCount_; i++)
		Neurons_[i] = rhs.Neurons_[i];

	LayerCommonFunc_ = rhs.LayerCommonFunc_;
	ActFunc_ = rhs.ActFunc_;
	ActDerFunc_ = rhs.ActDerFunc_;

	MaxNeuronNetSum_ = rhs.MaxNeuronNetSum_;
	LayerScaleFactor_ = rhs.LayerScaleFactor_;
}

void Sunny::NeuronsLayer::Feed(double * LayerInputs)
{
	for (int i = 0; i < NeuronsCount_; i++)
		Neurons_[i]->Feed(LayerInputs);

	if (LayerCommonFunc_ == ActivationFuncs::Funcs::SoftMaxFunc) // soft max is based on other Neurons in layer
	{
		double MaxNeuronNetSum_ = Neurons_[0]->net_sum_; // find maximal net sum among layer's neurons
		for (int i = 1; i < NeuronsCount_; i++)
			if (Neurons_[i]->net_sum_ > MaxNeuronNetSum_) MaxNeuronNetSum_ = Neurons_[i]->net_sum_;

		// calculate scaling factor -- sum of exp(each val - max)
		double LayerScaleFactor_ = 0.0;
		for (int i = 0; i < NeuronsCount_; i++)
			LayerScaleFactor_ += exp(Neurons_[i]->net_sum_ - MaxNeuronNetSum_);
		// calculate neurons activation based on max net sum and scale factor
		for (int i = 0; i < NeuronsCount_; i++)
			Neurons_[i]->activation_ = ActivationFuncs::SoftMax(Neurons_[i]->net_sum_, MaxNeuronNetSum_, LayerScaleFactor_);
	}
}

// @param1 - TargetOutput - array of expected outputs of layer's neurons
void Sunny::NeuronsLayer::BackPropagate(double * TargetOutput)
{
	for (int i = 0; i < NeuronsCount_; i++)
		Neurons_[i]->BackPropagate(TargetOutput[i]);
}

void Sunny::NeuronsLayer::UpdateNeurons(double * inputs)
{
	for (int i = 0; i < NeuronsCount_; i++)
		Neurons_[i]->UpdateWeights(inputs);
}

double * SimpleUndimNeuralNetworkYlem::NeuronsLayer::GetNeuronsOutputs(void)
{
	double * outs = new double[NeuronsCount_];
	for (int i = 0; i < NeuronsCount_; i++)
		outs[i] = Neurons_[i]->activation_;
	return outs;
}


std::ostream & Sunny::operator<<(std::ostream & DstStream, NeuronsLayer &prjNeuronLayer)
{
	DstStream << "[ " << prjNeuronLayer.Neurons_[0]->GetActivation();
	for (int i = 1; i < prjNeuronLayer.NeuronsCount_; i++)
		DstStream << ", " << prjNeuronLayer.Neurons_[i]->GetActivation();
	DstStream << " ]" << std::endl;
	DstStream << * prjNeuronLayer.Neurons_[0];
	return DstStream;
}