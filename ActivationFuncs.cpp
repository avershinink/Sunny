#include "ActivationFuncs.h"

namespace Sunny = SimpleUndimNeuralNetworkYlem;

Sunny::ActivationFuncs::Funcs Sunny::ActivationFuncs::IntToFuncs(int funcInt)
{
	switch (funcInt)
	{
	case 1:
		return Sunny::ActivationFuncs::Funcs::IdentityFunc;
	case 2:
		return Sunny::ActivationFuncs::Funcs::SigmoidFunc;
	case 3:
		return Sunny::ActivationFuncs::Funcs::ReLUFunc;
	case 4:
		return Sunny::ActivationFuncs::Funcs::PReLUFunc;
	case 5:
		return Sunny::ActivationFuncs::Funcs::HyperbolicTangentFunc;
	case 6:
		return Sunny::ActivationFuncs::Funcs::SoftMaxFunc;
	default:
		return Sunny::ActivationFuncs::Funcs::NONE;
	}
}


void Sunny::ActivationFuncs::SetActivationFunction(Sunny::ActivationFuncs::Funcs ActivationEnum, Sunny::ActivationFuncs::NeuronFunc &Act, Sunny::ActivationFuncs::NeuronFunc &ActDer, bool isNeuron)
{
	switch (ActivationEnum)
	{
	case Sunny::ActivationFuncs::IdentityFunc:
		Act = Sunny::ActivationFuncs::Identity;
		ActDer = Sunny::ActivationFuncs::IdentityDerivative;
		break;
	case Sunny::ActivationFuncs::SigmoidFunc:
		Act = Sunny::ActivationFuncs::Sigmoid;
		ActDer = Sunny::ActivationFuncs::SigmoidDerivative;
		break;
	case Sunny::ActivationFuncs::ReLUFunc:
		Act = Sunny::ActivationFuncs::ReLU;
		ActDer = Sunny::ActivationFuncs::ReLUDerivative;
		break;
	case Sunny::ActivationFuncs::PReLUFunc:
		Act = Sunny::ActivationFuncs::PReLU;
		ActDer = Sunny::ActivationFuncs::PReLUDerivative;
		break;
	case Sunny::ActivationFuncs::HyperbolicTangentFunc:
		Act = Sunny::ActivationFuncs::HyperbolicTangent;
		ActDer = Sunny::ActivationFuncs::HyperbolicTangentDerivative;
		break;
	case Sunny::ActivationFuncs::SoftMaxFunc:
		// Act -- NULL this is layer level function
		ActDer = Sunny::ActivationFuncs::SoftMaxDerivative;
		break;
	}
}

double Sunny::ActivationFuncs::Identity(double value)
{
	return value;
}

double Sunny::ActivationFuncs::IdentityDerivative(double value)
{
	return 1.0;
}

double Sunny::ActivationFuncs::Sigmoid(double value)
{
	return 1 / (1 + exp(-value));
}

double Sunny::ActivationFuncs::SigmoidDerivative(double value)
{
	return value * (1 - value);
}

// Rectified Linear Unit
double Sunny::ActivationFuncs::ReLU(double value)
{
	if (value < 0)
		return value;
	return value;
}

//Rectified Linear Unit Derivative
double Sunny::ActivationFuncs::ReLUDerivative(double value)
{
	if (value < 0)
		return 0.0;
	return 1.0;
}

//Parametric Rectified Linear Unit 
double Sunny::ActivationFuncs::PReLU(double value)
{
	if (value < 0)
		return alf * value;
	return value;
}

//Parametric Rectified Linear Unit Derivative
double Sunny::ActivationFuncs::PReLUDerivative(double value)
{
	if (value < 0)
		return alf;
	return 1.0;
}

double Sunny::ActivationFuncs::HyperbolicTangent(double value)
{
	return tanh(value);
}

double Sunny::ActivationFuncs::HyperbolicTangentDerivative(double value)
{
	return (1 - value) * (1 + value);
}

double Sunny::ActivationFuncs::SoftMax(double value, double netMax, double scale)
{
	return exp(value - netMax) / scale;
}

double Sunny::ActivationFuncs::SoftMaxDerivative(double value)
{
	return value * (1 - value);
}
