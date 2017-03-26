
// CudafyByExample.add_loop_gpu
extern "C" __global__  void adder(double learningRate,  double* inputData, int inputDataLen0,  double* outputData, int outputDataLen0,  double* expectedResuts, int expectedResutsLen0,  double* weights, int weightsLen0, int weightsLen1,  double* outNodeDelta, int outNodeDeltaLen0);

// CudafyByExample.add_loop_gpu
extern "C" __global__  void adder(double learningRate,  double* inputData, int inputDataLen0,  double* outputData, int outputDataLen0,  double* expectedResuts, int expectedResutsLen0,  double* weights, int weightsLen0, int weightsLen1,  double* outNodeDelta, int outNodeDeltaLen0)
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
