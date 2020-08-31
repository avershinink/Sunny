#include "Neuron.h"
#include "ActivationFuncs.h"

namespace SimpleUndimNeuralNetworkYlem
{
	class NeuronsLayer
	{
		friend std::ostream & operator<<(std::ostream &, NeuronsLayer &);
		friend class NeuralNetwork;
	public:

		NeuronsLayer(int NeuronsCount, int InputsPerNeuronCount, double LearningRate, double Momentum, double Decay, ActivationFuncs::Funcs);
		NeuronsLayer(const NeuronsLayer & rhs);

		NeuronsLayer & operator=(const NeuronsLayer &);


		~NeuronsLayer();

		void Feed(double *);
		void BackPropagate(double *);
		void UpdateNeurons(double *);

		double * GetNeuronsOutputs(void);
		int NeuronsCount_ = 0;

	private:
		int InputsPerNeuronCount_ = 0;

		Neuron * *Neurons_ = NULL;

		ActivationFuncs::Funcs LayerCommonFunc_ = ActivationFuncs::NONE;
		NeuronFunc ActFunc_ = NULL;
		NeuronFunc ActDerFunc_ = NULL;

		double MaxNeuronNetSum_ = 0;
		double LayerScaleFactor_ = 0;

		void Copy(const SimpleUndimNeuralNetworkYlem::NeuronsLayer & rhs);
	};
}