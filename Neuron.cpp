#include "Neuron.h"
#include <iostream>

namespace Sunny = SimpleUndimNeuralNetworkYlem;

Sunny::Neuron::Neuron(const Neuron & rhs)
{
	//std::cout << "Neuron --> Copy Constructor running from " << &(rhs) << " to " << &(*this) << std::endl;
	Copy(rhs);
}

	Sunny::Neuron::Neuron(const Neuron * rhs)
	{
		//std::cout << "Neuron --> from Neuron Pointer" << std::endl;
		Copy(*rhs);
	}

	Sunny::Neuron::Neuron(void)
	{
		//std::cout << "Neuron --> Default constructer " << &(*this) << std::endl;
	}

	Sunny::Neuron::Neuron(int inputsCount, double learningRate, double momentum, double decay) :
		inputsCount_(inputsCount),
		learningRate_(learningRate),
		momentum_(momentum),
		decay_(decay)
	{
		Init();
	}

	Sunny::Neuron::Neuron(int inputsCount, double learningRate, double momentum, double decay, Sunny::ActivationFuncs::Funcs UseActivation) :
		inputsCount_(inputsCount),
		learningRate_(learningRate),
		momentum_(momentum),
		decay_(decay),
		activationFuncEnum_(UseActivation)
	{


		Init();
	}
	
	Sunny::Neuron::~Neuron()
	{
		//std::cout << "Neuron -->Destructing " << &(*this) << std::endl;
		delete[] weights_;
	}

	void Sunny::Neuron::Init()
	{

		Sunny::ActivationFuncs::SetActivationFunction(activationFuncEnum_, ActivationFunc, ActivationDerivativeFunc);

		InitWeights();

		bias_ = 1;
		biasDelta_ = 0;
		biasPrevDelta_ = 0;

		delta_ = 0;
		prevDelta_ = 0;
		net_sum_ = 0;
		activation_ = 0;
	}

	void Sunny::Neuron::Feed(double* inputs)
	{
		net_sum_ = 0;
		for (int i = 0; i < inputsCount_; i++)
			net_sum_ += inputs[i] * weights_[i];

		net_sum_ += bias_ * biasWeight_;

		if(ActivationFunc)
			activation_ = ActivationFunc(net_sum_);
	}

	void Sunny::Neuron::InitWeights(void)
	{
		weights_ = new double[inputsCount_];

		for (int i = 0; i < inputsCount_; i++)
			weights_[i] = (rand() % 1000) / 1000.0;
		biasWeight_ = (rand() % 1000) / 1000.0;
	}


	void Sunny::Neuron::BackPropagate(double targetOutput)
	{
		if (ActivationDerivativeFunc)
		{
			double deviation = ActivationDerivativeFunc(activation_);
			delta_ = deviation * (targetOutput - activation_);
			biasDelta_ = deviation * 1;
		}
	}

	void Sunny::Neuron::UpdateWeights(double* inputs)
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

	double Sunny::Neuron::GetActivation(void) const
	{
		return activation_;
	}

	void SimpleUndimNeuralNetworkYlem::Neuron::ShowInfo(std::ostream & dst) const
	{
		dst << std::endl;
		dst << "================ NEURON ================" << std::endl;
		dst << "\tBias = " << this->bias_ << std::endl;
		dst << "\t\tbiasDelta_ = " << this->biasDelta_ << std::endl;
		dst << "\t\tbiasPrevDelta_ = " << this->biasPrevDelta_ << std::endl;

		//dst << "\tWeights: " << std::endl;
		//dst << "\t\t";
		this->PrintWeights(dst);

		dst << "\tDelta = " << this->delta_ << std::endl;
		dst << "\tPrev Delta = " << this->prevDelta_ << std::endl;
		dst << "\tACTIVATION = " << this->activation_ << std::endl;
		dst << "========================================" << std::endl;
	}

	void Sunny::Neuron::PrintWeights(std::ostream &DstStream) const
	{
		DstStream << "[ " << weights_[0];
		for (int i = 1; i < inputsCount_; i++)
			DstStream << ", " << weights_[i];
		DstStream << "]" << std::endl;
	}

	void Sunny::Neuron::Copy(const Neuron & src)
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

	Sunny::Neuron & Sunny::Neuron::operator=(const Neuron & rhs)
	{
		std::cout << "Neuron --> Operator= running" << std::endl;
		if (this == &rhs)
			return *this;
		delete[] weights_;
		Copy(rhs);

		return *this;
	}

	std::ostream& Sunny::operator<<(std::ostream &DstStream, Neuron &PrjNeuron)
	{
		DstStream << PrjNeuron.activationFuncEnum_ << " ";
		DstStream << PrjNeuron.bias_ << " ";
		DstStream << PrjNeuron.biasWeight_ << " ";
		DstStream << PrjNeuron.learningRate_ << " ";
		DstStream << PrjNeuron.momentum_ << " ";
		DstStream << PrjNeuron.decay_ << " ";
		DstStream << PrjNeuron.inputsCount_ << std::endl;
		for (int i = 0; i < PrjNeuron.inputsCount_; i++)
			DstStream << PrjNeuron.weights_[i] << " ";
		DstStream << std::endl;

		return DstStream;
	}

	std::istream& Sunny::operator>>(std::istream& src, Sunny::Neuron& dst)
	{
		// set activation function and derivative
		int func = 0;
		src >> func;
		ActivationFuncs::Funcs f = Sunny::ActivationFuncs::IntToFuncs(func);
		ActivationFuncs::SetActivationFunction(f, dst.ActivationFunc, dst.ActivationDerivativeFunc);

		src >> dst.bias_;
		src >> dst.biasWeight_;
		src >> dst.learningRate_;
		src >> dst.momentum_;
		src >> dst.decay_;
		src >> dst.inputsCount_;

		delete[] dst.weights_;
		dst.weights_ = new double[dst.inputsCount_];
		for(int i = 0; i < dst.inputsCount_; i++)
			src >> dst.weights_[i];
		return src;
	}

