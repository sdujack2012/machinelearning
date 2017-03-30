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
using System.Runtime.InteropServices;

namespace Cudafy
{
    public class CudafyHelper
    {
        static CudafyModule km;

        static GPGPU gpu;
        static CudafyHelper()
        {
             km = CudafyTranslator.Cudafy();

             gpu = CudafyHost.GetDevice(eGPUType.Cuda);
            gpu.LoadModule(km);

        }
        public static void TrainOutputLayer(double learningRate, double[] inputData, double[] outputData, double[] expectedResuts, double[,] weights, double[] outNodeDelta)
        {
            


            //// capture the start time
            //gpu.StartTimer();
            // copy the arrays 'a' and 'b' to the GPU
            var dev_inputData = gpu.Allocate(inputData);
            var dev_outputData = gpu.Allocate(outputData);
            var dev_expectedResuts = gpu.Allocate(expectedResuts);
            var dev_weights = gpu.Allocate(weights);
            var dev_outNodeDelta = gpu.Allocate(outNodeDelta);


            var host_inputDataPtr = Marshal.UnsafeAddrOfPinnedArrayElement(inputData, 0);
            var host_outputDataPtr = Marshal.UnsafeAddrOfPinnedArrayElement(outputData, 0);
            var host_expectedResutsPtr = Marshal.UnsafeAddrOfPinnedArrayElement(expectedResuts, 0);
            var host_weightsPtr = Marshal.UnsafeAddrOfPinnedArrayElement(weights, 0);
            var host_outNodeDeltaPtr = Marshal.UnsafeAddrOfPinnedArrayElement(outNodeDelta, 0);


            gpu.CopyToDeviceAsync(host_inputDataPtr, 0, dev_inputData, 0, inputData.Length, 1);
            gpu.CopyToDeviceAsync(host_outputDataPtr, 0, dev_outputData, 0, outputData.Length, 1);
            gpu.CopyToDeviceAsync(host_expectedResutsPtr, 0, dev_expectedResuts, 0, expectedResuts.Length, 1);
            gpu.CopyToDeviceAsync(host_weightsPtr, 0, dev_weights, 0, weights.Length, 1);
            gpu.CopyToDeviceAsync(host_outNodeDeltaPtr, 0, dev_outNodeDelta, 0, outNodeDelta.Length, 1);

            gpu.LaunchAsync(outputData.Length / 256 + 1, 256, 1, "CudafyTrainOutputLayer", learningRate, dev_inputData, dev_outputData, dev_expectedResuts, dev_weights, dev_outNodeDelta);
            //gpu.Launch(N / 256, 256, 1).kernel(dev_a0, dev_b0, dev_c0);
            //gpu.Launch(N / 256, 256, 2).kernel(dev_a1, dev_b1, dev_c1);
            gpu.CopyFromDeviceAsync(dev_outNodeDelta, 0, host_outNodeDeltaPtr, 0, outNodeDelta.Length, 1);
            gpu.CopyFromDeviceAsync(dev_weights, 0, host_weightsPtr, 0, weights.Length,1);

            gpu.SynchronizeStream(1);




            //float time = gpu.StopTimer();
            //Console.WriteLine("TrainOutputLayer Time:" + time);


            

            
            // free the memory allocated on the GPU
            gpu.FreeAll();
            gpu.DestroyStream(1);
        }

        [Cudafy]
        public static void CudafyTrainOutputLayer(GThread thread, double learningRate, double[] inputData, double[] outputData, double[] expectedResuts, double[,] weights, double[] outNodeDelta)
        {
            int tid = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
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

            //// capture the start time
            //gpu.StartTimer();
            // copy the arrays 'a' and 'b' to the GPU
            var dev_inputData = gpu.Allocate(inputData);
            var dev_outputData = gpu.Allocate(outputData);
            var dev_weightsNextLayer = gpu.Allocate(weightsNextLayer);
            var dev_weights = gpu.Allocate(weights);
            var dev_inNodeDelta = gpu.Allocate(inNodeDelta);
            var dev_outNodeDelta = gpu.Allocate(outNodeDelta);

            var host_inputDataPtr = Marshal.UnsafeAddrOfPinnedArrayElement(inputData, 0);
            var host_outputDataPtr = Marshal.UnsafeAddrOfPinnedArrayElement(outputData, 0);
            var host_weightsNextLayerPtr = Marshal.UnsafeAddrOfPinnedArrayElement(weightsNextLayer, 0);
            var host_weightsPtr = Marshal.UnsafeAddrOfPinnedArrayElement(weights, 0);
            var host_inNodeDeltaPtr = Marshal.UnsafeAddrOfPinnedArrayElement(inNodeDelta, 0);
            var host_outNodeDeltaPtr = Marshal.UnsafeAddrOfPinnedArrayElement(outNodeDelta, 0);

            gpu.CopyToDeviceAsync(host_inputDataPtr, 0, dev_inputData, 0, inputData.Length, 1);
            gpu.CopyToDeviceAsync(host_outputDataPtr, 0, dev_outputData, 0, outputData.Length, 1);
            gpu.CopyToDeviceAsync(host_weightsNextLayerPtr, 0, dev_weightsNextLayer, 0, weightsNextLayer.Length, 1);
            gpu.CopyToDeviceAsync(host_weightsPtr, 0, dev_weights, 0, weights.Length, 1);
            gpu.CopyToDeviceAsync(host_outNodeDeltaPtr, 0, dev_outNodeDelta, 0, outNodeDelta.Length, 1);
            gpu.CopyToDeviceAsync(host_inNodeDeltaPtr, 0, dev_inNodeDelta, 0, inNodeDelta.Length, 1);

            gpu.LaunchAsync(outputData.Length / 256 + 1, 256, 1, "CudafyTrainHiddenLayer", learningRate, dev_inputData, dev_outputData, dev_weightsNextLayer, dev_weights, dev_inNodeDelta, dev_outNodeDelta);
            gpu.CopyFromDeviceAsync(dev_outNodeDelta, 0, host_outNodeDeltaPtr, 0, outNodeDelta.Length, 1);
            gpu.CopyFromDeviceAsync(dev_weights, 0, host_weightsPtr, 0, weights.Length, 1);

            gpu.SynchronizeStream(1);

            //float time = gpu.StopTimer();
            //Console.WriteLine("TrainHiddenLayer Time:" + time);

            // free the memory allocated on the GPU
            gpu.FreeAll();
            gpu.DestroyStream(1);

        }

        [Cudafy]
        public static void CudafyTrainHiddenLayer(GThread thread, double learningRate, double[] inputData, double[] outputData, double[,] weightsNextLayer, double[,] weights, double[] inNodeDelta, double[] outNodeDelta)
        {
            int tid = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
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
            // capture the start time
            //gpu.StartTimer();
            var dev_outputResult = gpu.Allocate(outputResult);
            // copy the arrays 'a' and 'b' to the GPU
            var dev_trainingData = gpu.Allocate(trainingData);
            var dev_weight = gpu.Allocate(weight);

            var host_outputResultPtr = Marshal.UnsafeAddrOfPinnedArrayElement(outputResult, 0);
            var host_trainingDataPtr = Marshal.UnsafeAddrOfPinnedArrayElement(trainingData, 0);
            var host_weightPtr = Marshal.UnsafeAddrOfPinnedArrayElement(weight, 0);
           
            gpu.CopyToDeviceAsync(host_outputResultPtr, 0, dev_outputResult, 0, outputResult.Length, 1);
            gpu.CopyToDeviceAsync(host_trainingDataPtr, 0, dev_trainingData, 0, trainingData.Length, 1);
            gpu.CopyToDeviceAsync(host_weightPtr, 0, dev_weight, 0, weight.Length, 1);
            
            gpu.LaunchAsync(weight.GetLength(0) / 256 + 1, 256, 1, "CudafyComputeOutput", dev_trainingData, dev_weight, dev_outputResult);
            gpu.CopyFromDeviceAsync(dev_outputResult, 0, host_outputResultPtr, 0, outputResult.Length, 1);


            gpu.SynchronizeStream(1);

            //float time = gpu.StopTimer();
            //Console.WriteLine("TrainHiddenLayer Time:" + time);

            // free the memory allocated on the GPU
            gpu.FreeAll();
            gpu.DestroyStream(1);



        }

        [Cudafy]
        public static void CudafyComputeOutput(GThread thread, double[] trainingData, double[,] weight, double[] outputResult)
        {
            int tid = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
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
