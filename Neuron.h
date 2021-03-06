#include <iostream>
#include "ActivationFuncs.h"

namespace SimpleUndimNeuralNetworkYlem
{
		class Neuron
		{
			friend class NeuronsLayer;
			friend class NeuralNetwork;
			friend std::ostream& operator<<(std::ostream&, Neuron&);
			friend std::istream& operator>>(std::istream&, Neuron&);
		public:
			

			// Standard Constructor
			Neuron(void);

			//@parem1<int>    -- number of inputs
			//@parem2<double> -- learningRate
			//@parem3<double> -- momentum
			//@parem4<double> -- decay
			Neuron(int, double, double, double);

			//@parem1<int>    -- number of inputs
			//@parem2<double> -- learningRate
			//@parem3<double> -- momentum
			//@parem4<double> -- decay
			//@parem5<Funcs> -- Activation/Derivative function
			Neuron(int inputsCount, double learningRate, double momentum, double decay, SimpleUndimNeuralNetworkYlem::ActivationFuncs::Funcs UseActivation);

			// Neuron copy constructor
			Neuron(const Neuron &);
			//// Constructor from pointer;
			Neuron(const Neuron *);

			// No comments
			~Neuron();

			// Feed an array of inputs into neuron. 
			// @param inputs -- neuron inputs array
			void Feed(double*);

			//Backpropagate neuron with expected outputs
			//@param targetOutput -- neuron expected aim output
			void BackPropagate(double);

			//Update neuron weights based on propagated state on neuron
			//@param inputs -- neuron entries 
			void UpdateWeights(double*);

			//@param -- output value accurancy
			void InitWeights(void);

			//Returns current output of the neuron
			double GetActivation(void) const;

			//Print neuron telemetry
			void ShowInfo(std::ostream & dst) const;

			//Copy assignment operator
			Neuron & operator=(const Neuron &);

		private:
			// Number of inputs into neuron
			int inputsCount_ = 0;
			double learningRate_;
			double momentum_;
			double decay_;

			double net_sum_;
			double* weights_ = NULL;

			double activation_;

			double bias_;
			double biasWeight_;
			double biasDelta_;
			double biasPrevDelta_;

			double delta_;
			double prevDelta_;

			ActivationFuncs::NeuronFunc ActivationFunc;
			ActivationFuncs::NeuronFunc ActivationDerivativeFunc;
			ActivationFuncs::Funcs activationFuncEnum_;

			void Init();
			void PrintWeights(std::ostream &) const;
			void Copy(const Neuron &);
		};

}