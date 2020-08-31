#include "Neuron.h"
#include <iostream>



SimpleUndimNeuralNetworkYlem::Neuron::Neuron(const Neuron & rhs)
{
	//std::cout << "Neuron --> Copy Constructor running from " << &(rhs) << " to " << &(*this) << std::endl;
	Copy(rhs);
}

	SimpleUndimNeuralNetworkYlem::Neuron::Neuron(const Neuron * rhs)
	{
		//std::cout << "Neuron --> from Neuron Pointer" << std::endl;
		Copy(*rhs);
	}

	SimpleUndimNeuralNetworkYlem::Neuron::Neuron(void)
	{
		//std::cout << "Neuron --> Default constructer " << &(*this) << std::endl;
	}

	SimpleUndimNeuralNetworkYlem::Neuron::Neuron(int inputsCount, double learningRate, double momentum, double decay) :
		inputsCount_(inputsCount),
		learningRate_(learningRate),
		momentum_(momentum),
		decay_(decay)
	{
		Init();
	}

	SimpleUndimNeuralNetworkYlem::Neuron::Neuron(int inputsCount, double learningRate, double momentum, double decay, NeuronFunc ActivationFunc, NeuronFunc ActivationDerivativeFunc) :
		inputsCount_(inputsCount),
		learningRate_(learningRate),
		momentum_(momentum),
		decay_(decay)
	{
		//std::cout << "Neuron --> Parametrized constructor " << &(*this) << std::endl;
		Init();
		this->ActivationFunc = ActivationFunc;
		this->ActivationDerivativeFunc = ActivationDerivativeFunc;
	}

	SimpleUndimNeuralNetworkYlem::Neuron::~Neuron()
	{
		//std::cout << "Neuron -->Destructing " << &(*this) << std::endl;
		delete[] weights_;
	}

	void SimpleUndimNeuralNetworkYlem::Neuron::Init()
	{

		ActivationFunc = NULL;
		ActivationDerivativeFunc = NULL;

		InitWeights();

		bias_ = 1;
		biasDelta_ = 0;
		biasPrevDelta_ = 0;

		delta_ = 0;
		prevDelta_ = 0;
		net_sum_ = 0;
		activation_ = 0;
	}

	void SimpleUndimNeuralNetworkYlem::Neuron::Feed(double* inputs)
	{
		net_sum_ = 0;
		for (int i = 0; i < inputsCount_; i++)
			net_sum_ += inputs[i] * weights_[i];

		net_sum_ += bias_ * biasWeight_;

		if(ActivationFunc)
			activation_ = ActivationFunc(net_sum_);
	}

	void SimpleUndimNeuralNetworkYlem::Neuron::InitWeights(void)
	{
		weights_ = new double[inputsCount_];

		for (int i = 0; i < inputsCount_; i++)
			weights_[i] = (rand() % 1000) / 1000.0;
		biasWeight_ = (rand() % 1000) / 1000.0;
	}


	void SimpleUndimNeuralNetworkYlem::Neuron::BackPropagate(double targetOutput)
	{
		if (ActivationDerivativeFunc)
		{
			double deviation = ActivationDerivativeFunc(activation_);
			delta_ = deviation * (targetOutput - activation_);
			biasDelta_ = deviation * 1;
		}
	}

	void SimpleUndimNeuralNetworkYlem::Neuron::UpdateWeights(double* inputs)
	{
		double learningDelta = 0.0;
		double biasLearningDelta = 0.0;
		for (int i = 0; i < inputsCount_; i++)
		{
			learningDelta = learningRate_ * delta_ * inputs[i];
			weights_[i] += learningDelta;
			weights_[i] += momentum_ * prevDelta_;
			weights_[i] -= decay_ * weights_[i];
			prevDelta_ = learningDelta;

			biasLearningDelta = learningRate_ * delta_ * 1;
			bias_ += biasLearningDelta;
			bias_ += momentum_ * biasPrevDelta_;
			bias_ -= decay_ * bias_;
			biasPrevDelta_ = learningDelta;
		}
	}

	double SimpleUndimNeuralNetworkYlem::Neuron::GetActivation(void) const
	{
		return activation_;
	}

	void SimpleUndimNeuralNetworkYlem::Neuron::PrintWeights(std::ostream &DstStream) const
	{
		DstStream << "[ " << weights_[0];
		for (int i = 1; i < inputsCount_; i++)
			DstStream << ", " << weights_[i];
		DstStream << "]" << std::endl;
	}

	void SimpleUndimNeuralNetworkYlem::Neuron::Copy(const Neuron & src)
	{
		inputsCount_ = src.inputsCount_;

		weights_ = new double[inputsCount_];
		for (int i = 0; i < inputsCount_; i++)
			weights_[i] = src.weights_[i];

		learningRate_ = src.learningRate_;
		momentum_ = src.momentum_;
		decay_ = src.decay_;

		net_sum_ = src.net_sum_;
		weights_ = src.weights_;
		activation_ = src.activation_;

		bias_ = src.bias_;
		biasWeight_ = src.biasWeight_;
		biasDelta_ = src.biasDelta_;
		biasPrevDelta_ = src.biasPrevDelta_;

		delta_ = src.delta_;
		prevDelta_ = src.prevDelta_;

		ActivationFunc = src.ActivationFunc;
		ActivationDerivativeFunc = src.ActivationDerivativeFunc;
	}

	SimpleUndimNeuralNetworkYlem::Neuron & SimpleUndimNeuralNetworkYlem::Neuron::operator=(const Neuron & rhs)
	{
		std::cout << "Neuron --> Operator= running" << std::endl;
		if (this == &rhs)
			return *this;
		delete[] weights_;
		Copy(rhs);

		return *this;
	}

	std::ostream& SimpleUndimNeuralNetworkYlem::operator<<(std::ostream &DstStream, Neuron &PrjNeuron)
	{
		DstStream << std::endl;
		DstStream << "================ NEURON ================" << std::endl;
		DstStream << "\tBias = " << PrjNeuron.bias_ << std::endl;
		DstStream << "\t\tbiasDelta_ = " << PrjNeuron.biasDelta_ << std::endl;
		DstStream << "\t\tbiasPrevDelta_ = " << PrjNeuron.biasPrevDelta_ << std::endl;

		//DstStream << "\tWeights: " << std::endl;
		//DstStream << "\t\t";
		//PrjNeuron.PrintWeights(DstStream);

		DstStream << "\tDelta = " << PrjNeuron.delta_ << std::endl;
		DstStream << "\tPrev Delta = " << PrjNeuron.prevDelta_ << std::endl;
		DstStream << "\tACTIVATION = " << PrjNeuron.activation_ << std::endl;
		DstStream << "========================================" << std::endl;
		return DstStream;
	}