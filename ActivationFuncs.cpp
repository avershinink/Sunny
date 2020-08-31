#include "ActivationFuncs.h"

double SimpleUndimNeuralNetworkYlem::ActivationFuncs::Identity(double value)
{
	return value;
}

double SimpleUndimNeuralNetworkYlem::ActivationFuncs::IdentityDerivative(double value)
{
	return 1.0;
}

double SimpleUndimNeuralNetworkYlem::ActivationFuncs::Sigmoid(double value)
{
	return 1 / (1 + exp(-value));
}

double SimpleUndimNeuralNetworkYlem::ActivationFuncs::SigmoidDerivative(double value)
{
	return value * (1 - value);
}

// Rectified Linear Unit
double SimpleUndimNeuralNetworkYlem::ActivationFuncs::ReLU(double value)
{
	if (value < 0)
		return value;
	return value;
}

//Rectified Linear Unit Derivative
double SimpleUndimNeuralNetworkYlem::ActivationFuncs::ReLUDerivative(double value)
{
	if (value < 0)
		return 0.0;
	return 1.0;
}

//Parametric Rectified Linear Unit 
double SimpleUndimNeuralNetworkYlem::ActivationFuncs::PReLU(double value)
{
	if (value < 0)
		return alf * value;
	return value;
}

//Parametric Rectified Linear Unit Derivative
double SimpleUndimNeuralNetworkYlem::ActivationFuncs::PReLUDerivative(double value)
{
	if (value < 0)
		return alf;
	return 1.0;
}

double SimpleUndimNeuralNetworkYlem::ActivationFuncs::HyperbolicTangent(double value)
{
	return tanh(value);
}

double SimpleUndimNeuralNetworkYlem::ActivationFuncs::HyperbolicTangentDerivative(double value)
{
	return (1 - value) * (1 + value);
}

double SimpleUndimNeuralNetworkYlem::ActivationFuncs::SoftMax(double value, double netMax, double scale)
{
	return exp(value - netMax) / scale;
}

double SimpleUndimNeuralNetworkYlem::ActivationFuncs::SoftMaxDerivative(double value)
{
	return value * (1 - value);
}
