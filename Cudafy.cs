/* 
 * This software is based upon the book CUDA By Example by Sanders and Kandrot
 * and source code provided by NVIDIA Corporation.
 * It is a good idea to read the book while studying the examples!
*/
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;
using mlDemo;

namespace Cudafy
{
    public class CudafyHelper
    {
        static CudafyModule km;

        static GPGPU gpu;
        static CudafyHelper()
        {
             km = CudafyTranslator.Cudafy();

             gpu = CudafyHost.GetDevice(CudafyModes.Target, CudafyModes.DeviceId);
            gpu.LoadModule(km);

        }
        public static void TrainOutputLayer(double learningRate, double[] inputData, double[] outputData, double[] expectedResuts, double[,] weights, double[] outNodeDelta)
        {
            

            // copy the arrays 'a' and 'b' to the GPU
            var dev_inputData = gpu.CopyToDevice(inputData);
            var dev_outputData = gpu.CopyToDevice(outputData);
            var dev_expectedResuts = gpu.CopyToDevice(expectedResuts);
            var dev_weights = gpu.CopyToDevice(weights);
            var dev_outNodeDelta = gpu.CopyToDevice(outNodeDelta);
            // launch add on N threads
            gpu.Launch(outNodeDelta.Length, 1).CudafyTrainOutputLayer(learningRate, dev_inputData, dev_outputData, dev_expectedResuts, dev_weights, dev_outNodeDelta);
            gpu.Synchronize();
            // copy the array 'c' back from the GPU to the CPU
            gpu.CopyFromDevice(dev_outNodeDelta, outNodeDelta);
            gpu.CopyFromDevice(dev_weights, weights);


            // free the memory allocated on the GPU
            gpu.FreeAll();
            
        }

        [Cudafy]
        public static void CudafyTrainOutputLayer(GThread thread, double learningRate, double[] inputData, double[] outputData, double[] expectedResuts, double[,] weights, double[] outNodeDelta)
        {
            int tid = thread.blockIdx.x;
            if (tid < outputData.Length) {
                int l = 0;
                outNodeDelta[tid] = (outputData[tid] - expectedResuts[tid]) * outputData[tid] * (1 - outputData[tid]);
                for (l = 0; l < weights.GetLength(0) - 1; l++)
                {
                    weights[tid, l] = weights[tid, l] - learningRate * outNodeDelta[tid] * inputData[l];
                }
                weights[tid, l] = weights[tid, l] - learningRate * outNodeDelta[tid];
            }
               
        }


        public static void TrainHiddenLayer(double learningRate, double[] inputData, double[] outputData, double[,] weightsNextLayer, double[,] weights, double[] inNodeDelta, double[] outNodeDelta)
        {
            

            // copy the arrays 'a' and 'b' to the GPU
            var dev_inputData = gpu.CopyToDevice(inputData);
            var dev_outputData = gpu.CopyToDevice(outputData);
            var dev_weightsNextLayer = gpu.CopyToDevice(weightsNextLayer);
            var dev_weights = gpu.CopyToDevice(weights);
            var dev_outNodeDelta = gpu.Allocate(outNodeDelta);
            var dev_inNodeDelta = gpu.Allocate(inNodeDelta);

            // launch add on N threads
            gpu.Launch(outputData.Length, 1).CudafyTrainHiddenLayer(learningRate, dev_inputData, dev_outputData, dev_weightsNextLayer, dev_weights, dev_inNodeDelta, dev_outNodeDelta);
            gpu.Synchronize();
            // copy the array 'c' back from the GPU to the CPU
            gpu.CopyFromDevice(dev_outNodeDelta, outNodeDelta);
            gpu.CopyFromDevice(dev_weights, weights);


            // free the memory allocated on the GPU
            gpu.FreeAll();

        }

        [Cudafy]
        public static void CudafyTrainHiddenLayer(GThread thread, double learningRate, double[] inputData, double[] outputData, double[,] weightsNextLayer, double[,] weights, double[] inNodeDelta, double[] outNodeDelta)
        {
            int tid = thread.blockIdx.x;
            if (tid < outputData.Length)
            {
                double accumulateErrorDelta = 0;
                int l = 0;
                for (l = 0; l < weightsNextLayer.GetLength(0); l++)
                {
                    accumulateErrorDelta += inNodeDelta[l] * weightsNextLayer[l, tid];
                }
                outNodeDelta[tid] = accumulateErrorDelta * outputData[tid] * (1 - outputData[tid]);
                for (l = 0; l < weights.GetLength(1) - 1; l++)
                {
                    weights[tid, l] = weights[tid, l] - learningRate * outNodeDelta[tid] * inputData[l];
                }
                weights[tid, l] = weights[tid, l] - learningRate * outNodeDelta[tid];
            }

        }

        public static void ComputeOutput(double[] trainingData, double[,] weight, double[] outputResult)
        {
            var dev_outputResult = gpu.Allocate(outputResult);
            // copy the arrays 'a' and 'b' to the GPU
            var dev_trainingData = gpu.CopyToDevice(trainingData);
            var dev_weight = gpu.CopyToDevice(weight);

            // launch add on N threads
            gpu.Launch(weight.GetLength(0), 1).CudafyComputeOutput(dev_trainingData, dev_weight, dev_outputResult);
            gpu.Synchronize();
            // copy the array 'c' back from the GPU to the CPU
            gpu.CopyFromDevice(dev_outputResult, outputResult);



            // free the memory allocated on the GPU
            gpu.FreeAll();

        }

        [Cudafy]
        public static void CudafyComputeOutput(GThread thread, double[] trainingData, double[,] weight, double[] outputResult)
        {
            int tid = thread.blockIdx.x;
            if (tid < weight.GetLength(0))
            {
                double z = 0.0;
                int i = 0;
                for (i = 0; i < trainingData.Length; i++)
                {
                    z += trainingData[i] * weight[tid, i];
                }
                z += weight[tid, i];
                outputResult[tid] = 1.0 / (1.0 + Math.Exp(-z));
            }


        }
    }
}
