#include "NeuralNetwork.h"
#include <fstream>
#include <iostream>
#include <ios>

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
	for (int i = 0; i < layersCount_; i++)
		delete Layers_[i];
	delete[] Layers_;
}

double SimpleUndimNeuralNetworkYlem::NeuralNetwork::Accuracy(std::istream & Inputs)
{
	// jump to stream begining
	Inputs.clear();
	Inputs.seekg(0);

	// accuracy calculation
	double rightOutsCount = 0;
	int testingSetSize = 0;
	
	// declare testing set arrays
	int InputSize = this->Layers_[0]->NeuronsCount_;
	int OtputSize = this->Layers_[layersCount_ - 1]->NeuronsCount_;
	double * testingDataInputs = new double[InputSize];
	double * testingDataOutputs = new double[OtputSize];
	double * NNOutput = new double[OtputSize];
	while (!Inputs.eof())
	{
		testingSetSize++;
		// prepare testing set arrays
		char str[4096];
		Inputs.getline(str, 4096);
		char* begin = str;
		char* end;
		errno = 0;
		int j = 0, k = 0;
		testingDataInputs[j++] = strtod(begin, &end);
		while (errno == 0 && end != begin && *end != '\0') {

			begin = end;
			if (j < InputSize)
				testingDataInputs[j++]  = strtod(begin, &end);
			else
				testingDataOutputs[k++] = strtod(begin, &end);
		}

		// feed the testing data into network 
		Feed(testingDataInputs);

		// get NN outpus
		GetOutputs(NNOutput);

		//get max ouput = answer, and max expected = right answer
		double maxOut = NNOutput[0];
		int    maxNeuronIndex = 0;
		double maxExpectedOut = testingDataOutputs[0];
		int    maxExpectedOutIndex = 0;
		for (int j = 1; j < Layers_[layersCount_ - 1]->NeuronsCount_; j++)
		{
			if (maxOut < NNOutput[j])
			{
				maxOut = NNOutput[j];
				maxNeuronIndex = j;
			}
			if (maxExpectedOut < testingDataOutputs[j])
			{
				maxExpectedOut = testingDataOutputs[j];
				maxExpectedOutIndex = j;
			}
		}

		//check the ansewer
		if (maxExpectedOutIndex == maxNeuronIndex)
			rightOutsCount++;
	}
	delete[] NNOutput;
	delete[] testingDataInputs;
	delete[] testingDataOutputs;
	return rightOutsCount / testingSetSize;
}

void SimpleUndimNeuralNetworkYlem::NeuralNetwork::Train(int Epochs, double targetAccuracy, std::istream & InputsOutputs, std::istream & TestingInputsOutputs)
{
	int InputSize = this->Layers_[0]->NeuronsCount_;
	int OtputSize = this->Layers_[layersCount_ - 1]->NeuronsCount_;
	double * learningDataInputs = new double[InputSize];
	double * learningDataOutputs = new double[OtputSize];
	for (int epoch = 0; epoch < Epochs; epoch++)
	{
		InputsOutputs.clear();
		InputsOutputs.seekg(0);
		double accuracy = 0;
		if (epoch % 100 == 0)
		{
			system("cls");
			accuracy = Accuracy(TestingInputsOutputs);
			std::cout << "Current accuracy is " << accuracy << " on epoch " << epoch << "/" << Epochs << std::endl;

			Feed(learningDataInputs);
			PrintArray(learningDataOutputs, OtputSize);
			Layers_[layersCount_ - 1]->ShowInfo(std::cout);
			std::cout << std::endl;

			if (accuracy >= targetAccuracy)
			{
				std::cout << "Learning epoch " << epoch << "/" << Epochs << " meets the target accuracy of " << targetAccuracy << std::endl;
				break;
			}
		}

		while(!InputsOutputs.eof())
		{
			// prepare inputs and outputs
			char str[4096];
			InputsOutputs.getline(str, 4096);
			char* begin = str; 
			char* end;
			errno = 0;
			int j = 0, k = 0;
			learningDataInputs[j++] = strtod(begin, &end);
			while (errno == 0 && end != begin && *end != '\0') {
				begin = end;
				if (j < InputSize)
					learningDataInputs[j++] = strtod(begin, &end);
				else
					learningDataOutputs[k++] = strtod(begin, &end);
			}

			Feed(learningDataInputs);
			BackPropagate(learningDataOutputs);
			UpdateWeights(learningDataInputs);

		}
	}
	delete[] learningDataInputs;
	delete[] learningDataOutputs;
}

void SimpleUndimNeuralNetworkYlem::NeuralNetwork::Feed(double * inputs)
{
	Layers_[0]->Feed(inputs);
	for (int i = 1; i < layersCount_; i++)
	{
		double *outputsPtr = new double[Layers_[i - 1]->NeuronsCount_];
		Layers_[i - 1]->GetNeuronsOutputs(outputsPtr);
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
	    double *outputsPtr = new double[Layers_[i -1]->NeuronsCount_];
		Layers_[i - 1]->GetNeuronsOutputs(outputsPtr);
		Layers_[i]->UpdateNeurons(outputsPtr);
		delete[] outputsPtr;
	}
}

void SimpleUndimNeuralNetworkYlem::NeuralNetwork::GetOutputs(double* &outs) const
{
	Layers_[layersCount_ - 1]->GetNeuronsOutputs(outs);
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