using Cudafy;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
namespace mlDemo
{
    class Program
    {
        static void Main(string[] args)
        {
            var macine = new Machine(new int[] { 784, 30, 10 }, 0.1);
            var currentImage = 0;
            var trainBytes = new byte[60000][];
            var expectedResults = new byte[60000];

            var testBytes = new byte[10000][];
            var testResults = new byte[10000];


            FileStream fsImage = new FileStream(Directory.GetCurrentDirectory() + "\\train-images.idx3-ubyte",
                                   FileMode.Open,
                                   FileAccess.Read);
            BinaryReader brImage = new BinaryReader(fsImage);
            brImage.ReadBytes(16);

            FileStream fsLabel = new FileStream(Directory.GetCurrentDirectory() + "\\train-labels.idx1-ubyte",
                                   FileMode.Open,
                                   FileAccess.Read);
            BinaryReader brfsLabel = new BinaryReader(fsLabel);
            brfsLabel.ReadBytes(8);


            while (brImage.BaseStream.Position != brImage.BaseStream.Length)
            {
                trainBytes[currentImage] = brImage.ReadBytes(784);
                expectedResults[currentImage] = brfsLabel.ReadByte();
                currentImage++;
            }
            brfsLabel.Close();
            brImage.Close();
            Console.WriteLine("Total Training Data:" + currentImage);


            fsImage = new FileStream(Directory.GetCurrentDirectory() + "\\t10k-images.idx3-ubyte",
                                   FileMode.Open,
                                   FileAccess.Read);
            brImage = new BinaryReader(fsImage);
            brImage.ReadBytes(16);

            fsLabel = new FileStream(Directory.GetCurrentDirectory() + "\\t10k-labels.idx1-ubyte",
                                   FileMode.Open,
                                   FileAccess.Read);
            brfsLabel = new BinaryReader(fsLabel);
            brfsLabel.ReadBytes(8);

            currentImage = 0;
            while (brImage.BaseStream.Position != brImage.BaseStream.Length && currentImage <= 10000)
            {
                testBytes[currentImage] = brImage.ReadBytes(28 * 28);
                testResults[currentImage] = brfsLabel.ReadByte();
                currentImage++;
            }
            brfsLabel.Close();
            brImage.Close();
            Console.WriteLine("Total Test Data:" + currentImage);

            for (int i = 0; i < 60000; i++)
            {

                var expectedResult = new double[10];
                expectedResult[expectedResults[i]] = 1;
                macine.Train(trainBytes[i].Select(x => System.Convert.ToDouble(x) / 255).ToArray(), expectedResult);
            }
            var correctCount = 0;
            for (int i = 0; i < testBytes.Length; i++)
            {
                var expectedResult = new double[10];
                expectedResult[testResults[i]] = 1;
                var actualdResult = macine.ComputeOutput(testBytes[i].Select(x => System.Convert.ToDouble(x) / 255).ToArray());
                Console.WriteLine("#######################");
                Console.WriteLine("Correct Result:" + string.Join(",", expectedResult.Select(x => string.Format("{0:N2}", x))));
                Console.WriteLine("Actual Result: " + string.Join(",", actualdResult.Select(x => string.Format("{0:N2}", x))));
                double maxValue = actualdResult.Max();
                int maxIndex = actualdResult.ToList().IndexOf(maxValue);
                if (testResults[i] == maxIndex)
                {
                    correctCount++;
                }
            }
            Console.WriteLine("#######################");
            Console.WriteLine("Correct Rate:" + string.Format("{0:N2}", (double)correctCount / 10000));
            Console.Read();

        }
    }

    class SGDHlper
    {



        public static double Output(double z)
        {
            return 1.0 / (1.0 + Math.Exp(-z));
        }
        public static double[] ComputeOutput(double[] trainingData, double[,] weight)
        {
            var outputResult = new double[weight.GetLength(0)];
            for (int k = 0; k < weight.GetLength(0); k++)
            {
                double z = 0.0;
                int i = 0;
                for (i = 0; i < trainingData.Length; i++)
                {
                    z += trainingData[i] * weight[k, i];
                }
                z += weight[k, i];
                outputResult[k] = SGDHlper.Output(z);
            }

            return outputResult;
        }
        
        public static void TrainOutputLayer(double learningRate, double[] inputData, double[] outputData, double[] expectedResuts, double[,] weights, double[] outNodeDelta)
        {


            CudafyHelper.TrainOutputLayer( learningRate, inputData,outputData, expectedResuts, weights,outNodeDelta);
            
        }

        public static void TrainHiddenLayer(double learningRate, double[] inputData, double[] outputData, double[,] weightsNextLayer, double[,] weights, double[] inNodeDelta, double[] outNodeDelta)
        {

            CudafyHelper.TrainHiddenLayer( learningRate, inputData,outputData,weightsNextLayer,weights,inNodeDelta,  outNodeDelta);

        }

        public static T[,] Make2DArray<T>(T[] input, int height, int width)
        {
            T[,] output = new T[height, width];
            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    output[i, j] = input[i * width + j];
                }
            }
            return output;
        }


    }
    class Machine
    {
        public double[][,] Weights { set; get; }
        private double LearningRate;
        private int NumOfInputs;
        private int NumOfOutput;
        public Machine(int[] layers, double learningRate)
        {
            LearningRate = learningRate;
            NumOfInputs = layers[0];
            NumOfOutput = layers.Last();
            Random random = new Random();
            Weights = new double[layers.Length - 1][,];
            for (int i = 1; i < layers.Length; i++)
            {
                Weights[i - 1] = new double[layers[i], layers[i - 1] + 1];
                Array.Copy(SGDHlper.Make2DArray(Enumerable.Repeat(0, layers[i] * (layers[i - 1] + 1)).Select(x => (random.NextDouble() - 0.5) * 2).ToArray(), layers[i - 1] + 1, layers[i]), Weights[i - 1], layers[i] * (layers[i - 1] + 1));
            }
        }
        public void Train(double[] trainingData, double[] expectedResuts)
        {
            var newWeights = new double[Weights.Length][,];
            for (int i = 0; i < newWeights.Length; i++)
            {
                newWeights[i] = new double[Weights[i].GetLength(0), Weights[i].GetLength(1)];
                Array.Copy(Weights[i], newWeights[i], Weights[i].GetLength(0) * Weights[i].GetLength(1));
            }

            var inputDataForLayers = new double[Weights.Length + 1][];
            var count = 0;
            inputDataForLayers[0] = trainingData;
            Weights.ToList().ForEach(y => {
                inputDataForLayers[count + 1] = SGDHlper.ComputeOutput(inputDataForLayers[count], y);
                count++;
            });
            var outputDelta = new double[expectedResuts.Length];
            SGDHlper.TrainOutputLayer(LearningRate, inputDataForLayers[inputDataForLayers.Length - 2], inputDataForLayers[inputDataForLayers.Length - 1], expectedResuts, newWeights[newWeights.Length - 1], outputDelta);

            var inNodeDelta = outputDelta;

            for (int i = newWeights.Length - 2; i >= 0; i--)
            {
                outputDelta = new double[newWeights[i].GetLength(0)];
                SGDHlper.TrainHiddenLayer(LearningRate, inputDataForLayers[i], inputDataForLayers[i + 1], Weights[i + 1], newWeights[i], inNodeDelta, outputDelta);
                inNodeDelta = outputDelta;
            }

            Weights = newWeights;

        }

        public double[] ComputeOutput(double[] trainingData)
        {
            var inputDataForLayers = ComputeOutputForEachLayers(trainingData);
            return inputDataForLayers.Last();


        }

        public double[][] ComputeOutputForEachLayers(double[] trainingData)
        {
            var inputDataForLayers = new double[Weights.Length + 1][];
            inputDataForLayers[0] = trainingData;
            var count = 0;
            Weights.ToList().ForEach(weight => {
                inputDataForLayers[count + 1] = SGDHlper.ComputeOutput(inputDataForLayers[count], weight);
                count++;
            });


            return inputDataForLayers;


        }

    }
}
