#include "Neuron.h"

namespace SimpleUndimNeuralNetworkYlem
{
	class NeuronsLayer
	{
		friend std::ostream & operator<<(std::ostream &, NeuronsLayer &);
		friend std::istream & operator>>(std::istream &, NeuronsLayer &);
		friend class NeuralNetwork;
	public:
		NeuronsLayer() {};
		NeuronsLayer(int NeuronsCount, int InputsPerNeuronCount, double LearningRate, double Momentum, double Decay, ActivationFuncs::Funcs);
		NeuronsLayer(const NeuronsLayer & rhs);

		NeuronsLayer & operator=(const NeuronsLayer &);


		~NeuronsLayer();

		void Feed(double *);
		void BackPropagate(double *);
		void UpdateNeurons(double *);
		void ShowInfo(std::ostream & dst) const;

		double * GetNeuronsOutputs(void);
		int NeuronsCount_ = 0;

	private:
		int InputsPerNeuronCount_ = 0;

		Neuron * *Neurons_ = NULL;

		ActivationFuncs::Funcs LayerCommonFunc_ = ActivationFuncs::NONE;
		ActivationFuncs::NeuronFunc ActFunc_ = NULL;
		ActivationFuncs::NeuronFunc ActDerFunc_ = NULL;

		double MaxNeuronNetSum_ = 0;
		double LayerScaleFactor_ = 0;

		void Copy(const SimpleUndimNeuralNetworkYlem::NeuronsLayer & rhs);
	};
}