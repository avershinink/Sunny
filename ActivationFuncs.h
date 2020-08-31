#include <iostream>

namespace SimpleUndimNeuralNetworkYlem
{
	namespace ActivationFuncs
	{
		enum Funcs
		{
			NONE,
			IdentityFunc,
			SigmoidFunc,
			ReLUFunc,
			PReLUFunc,
			HyperbolicTangentFunc,
			SoftMaxFunc
		};

		double Identity(double value);
		double IdentityDerivative(double value);

		double Sigmoid(double value);
		double SigmoidDerivative(double value);

		// Rectified Linear Unit
		double ReLU(double value);
		//Rectified Linear Unit Derivative
		double ReLUDerivative(double value);

		const double alf = .01;
		//Parametric Rectified Linear Unit 
		double PReLU(double value);
		//Parametric Rectified Linear Unit Derivative
		double PReLUDerivative(double value);


		double HyperbolicTangent(double value);
		double HyperbolicTangentDerivative(double value);

		double SoftMax(double value, double netMax, double scale);
		double SoftMaxDerivative(double value);
	}
}