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
            var macine = new Machine(new int[] { 784, 60,30, 10 }, 0.1);
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
                if (testResults[i]== maxIndex)
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
        public static double ComputeOutput(double[] trainingData, double[] weight)
        {
            double z = 0.0;
            int i = 0;
            for (i = 0; i < trainingData.Length; i++)
            {
                z += trainingData[i] * weight[i];
            }
            z += weight[i];
            return SGDHlper.Output(z);
        }
        
        public static void TrainOutputLayer(double learningRate, double[] inputData, double[] outputData, double[] expectedResuts, double[][] weights, double[] outNodeDelta)
        {



            Parallel.For(0, outputData.Length, i =>
           {
               int l = 0;
               outNodeDelta[i] = (outputData[i] - expectedResuts[i]) * outputData[i] * (1 - outputData[i]);
               for (l = 0; l < weights[i].Length - 1; l++)
               {
                   weights[i][l] = weights[i][l] - learningRate * outNodeDelta[i] * inputData[l];
               }
               weights[i][l] = weights[i][l] - learningRate * outNodeDelta[i];
           });
        }

        public static void TrainHiddenLayer(double learningRate, double[] inputData, double[] outputData, double[][] weightsNextLayer, double[][] weights, double[] inNodeDelta, double[] outNodeDelta)
        {



            Parallel.For(0, outputData.Length, i =>
            {
                double accumulateErrorDelta = 0;
                int l = 0;
                for (l = 0; l < weightsNextLayer.Length; l++)
                {
                    accumulateErrorDelta += inNodeDelta[l] * weightsNextLayer[l][i];
                }
                outNodeDelta[i] = accumulateErrorDelta * outputData[i] * (1 - outputData[i]);
                for (l = 0; l < weights[i].Length - 1; l++)
                {
                    weights[i][l] = weights[i][l] - learningRate * outNodeDelta[i] * inputData[l];
                }
                weights[i][l] = weights[i][l] - learningRate * outNodeDelta[i];
            });
        }


    }
    class Machine
    {
        public double[][][] Weights { set; get; }
        private double LearningRate;
        private int NumOfInputs;
        private int NumOfOutput;
        public Machine(int[] layers, double learningRate)
        {
            LearningRate = learningRate;
            NumOfInputs = layers[0];
            NumOfOutput = layers.Last();
            Random random = new Random();
            Weights = new double[layers.Length - 1][][];
            for (int i = 1; i < layers.Length; i++)
            {
                Weights[i-1] = new double[layers[i]][];
                for (int j = 0; j < Weights[i-1].Length; j++)
                {

                    Weights[i-1][j] = Enumerable.Repeat(0, layers[i - 1] + 1).Select(num => (random.NextDouble() - 0.5) * 2).ToArray();
                }
            }
        }
        public void Train(double[] trainingData, double[] expectedResuts)
        {
            var newWeights = new double[Weights.Length][][];
            for (int i = 0; i < newWeights.Length; i++)
            {
                newWeights[i] = new double[Weights[i].Length][];
                for (int j = 0; j < newWeights[i].Length; j++)
                {
                    newWeights[i][j] = new double[Weights[i][j].Length];
                    for (int k = 0; k < newWeights[i][j].Length; k++)
                    {
                        newWeights[i][j][k] = Weights[i][j][k];
                    }
                }
            }

            var inputDataForLayers = new double[Weights.Length+1][];
            var count = 0;
            inputDataForLayers[0] = trainingData;
            Weights.ToList().ForEach(y => {
                inputDataForLayers[count+1] = y.Select(x => SGDHlper.ComputeOutput(inputDataForLayers[count], x)).ToArray();
                count++;
            });
            var outputDelta = new double[expectedResuts.Length];
            SGDHlper.TrainOutputLayer(LearningRate, inputDataForLayers[inputDataForLayers.Length-2], inputDataForLayers[inputDataForLayers.Length - 1], expectedResuts, newWeights[newWeights.Length - 1], outputDelta);

            var inNodeDelta = outputDelta;

            for (int i = newWeights.Length - 2; i >= 0; i--)
            {
                outputDelta = new double[newWeights[i].Length];
                SGDHlper.TrainHiddenLayer(LearningRate, inputDataForLayers[i], inputDataForLayers[i+1], Weights[i + 1], newWeights[i], inNodeDelta, outputDelta);
                inNodeDelta = outputDelta;
            }

            Weights= newWeights;

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
            Weights.ToList().ForEach(weights => {
                inputDataForLayers[count + 1] = weights.Select(x => SGDHlper.ComputeOutput(inputDataForLayers[count], x)).ToArray();
                count++;
            });
           

            return inputDataForLayers;


        }

    }
}
