
// Cudafy.CudafyHelper
extern "C" __global__  void CudafyTrainOutputLayer(double learningRate,  double* inputData, int inputDataLen0,  double* outputData, int outputDataLen0,  double* expectedResuts, int expectedResutsLen0,  double* weights, int weightsLen0, int weightsLen1,  double* outNodeDelta, int outNodeDeltaLen0);
// Cudafy.CudafyHelper
extern "C" __global__  void CudafyTrainHiddenLayer(double learningRate,  double* inputData, int inputDataLen0,  double* outputData, int outputDataLen0,  double* weightsNextLayer, int weightsNextLayerLen0, int weightsNextLayerLen1,  double* weights, int weightsLen0, int weightsLen1,  double* inNodeDelta, int inNodeDeltaLen0,  double* outNodeDelta, int outNodeDeltaLen0);
// Cudafy.CudafyHelper
extern "C" __global__  void CudafyComputeOutput( double* trainingData, int trainingDataLen0,  double* weight, int weightLen0, int weightLen1,  double* outputResult, int outputResultLen0);

// Cudafy.CudafyHelper
extern "C" __global__  void CudafyTrainOutputLayer(double learningRate,  double* inputData, int inputDataLen0,  double* outputData, int outputDataLen0,  double* expectedResuts, int expectedResutsLen0,  double* weights, int weightsLen0, int weightsLen1,  double* outNodeDelta, int outNodeDeltaLen0)
{
	int x = blockIdx.x;
	bool flag = x < outputDataLen0;
	if (flag)
	{
		int i = 0;
		outNodeDelta[(x)] = (outputData[(x)] - expectedResuts[(x)]) * outputData[(x)] * (1.0 - outputData[(x)]);
		for (i = 0; i < weightsLen0 - 1; i++)
		{
			weights[(x) * weightsLen1 + ( i)] -= learningRate * outNodeDelta[(x)] * inputData[(i)];
		}
		weights[(x) * weightsLen1 + ( i)] -= learningRate * outNodeDelta[(x)];
	}
}
// Cudafy.CudafyHelper
extern "C" __global__  void CudafyTrainHiddenLayer(double learningRate,  double* inputData, int inputDataLen0,  double* outputData, int outputDataLen0,  double* weightsNextLayer, int weightsNextLayerLen0, int weightsNextLayerLen1,  double* weights, int weightsLen0, int weightsLen1,  double* inNodeDelta, int inNodeDeltaLen0,  double* outNodeDelta, int outNodeDeltaLen0)
{
	int x = blockIdx.x;
	bool flag = x < outputDataLen0;
	if (flag)
	{
		double num = 0.0;
		int i = 0;
		for (i = 0; i < weightsNextLayerLen0; i++)
		{
			num += inNodeDelta[(i)] * weightsNextLayer[(i) * weightsNextLayerLen1 + ( x)];
		}
		outNodeDelta[(x)] = num * outputData[(x)] * (1.0 - outputData[(x)]);
		for (i = 0; i < weightsLen1 - 1; i++)
		{
			weights[(x) * weightsLen1 + ( i)] -= learningRate * outNodeDelta[(x)] * inputData[(i)];
		}
		weights[(x) * weightsLen1 + ( i)] -= learningRate * outNodeDelta[(x)];
	}
}
// Cudafy.CudafyHelper
extern "C" __global__  void CudafyComputeOutput( double* trainingData, int trainingDataLen0,  double* weight, int weightLen0, int weightLen1,  double* outputResult, int outputResultLen0)
{
	int x = blockIdx.x;
	bool flag = x < weightLen0;
	if (flag)
	{
		double num = 0.0;
		int i = 0;
		for (i = 0; i < trainingDataLen0; i++)
		{
			num += trainingData[(i)] * weight[(x) * weightLen1 + ( i)];
		}
		num += weight[(x) * weightLen1 + ( i)];
		outputResult[(x)] = 1.0 / (1.0 + exp(-num));
	}
}
