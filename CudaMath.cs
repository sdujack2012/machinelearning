using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;
using System;
using System.Collections.Generic;
using System.Linq;

namespace CudaMath.Double
{
    public static class TwoDimensional
    {
        private const int blockSide = 16;
        private static dim3 blockSize = new dim3(blockSide, blockSide);
        private static GPGPU gpu;
        private static CudafyModule module;

        static TwoDimensional()
        {
            module = CudafyTranslator.Cudafy();
            gpu = CudafyHost.GetDevice(CudafyModes.Target, CudafyModes.DeviceId);
        }

        public static double[,] Add(this double[,] matrix, int add)
        {
            if (matrix.Length < 1)
                return new double[0, 0];

            int x = matrix.GetLength(0);
            int y = matrix.GetLength(1);

            double[,] result = new double[x, y];

            gpu.LoadModule(module);

            double[,] gpuMatrix = gpu.Allocate(matrix);
            double[,] gpuResult = gpu.Allocate(result);

            gpu.CopyToDevice(matrix, gpuMatrix);
            gpu.CopyToDevice(result, gpuResult);

            gpu.Launch(getGridSize(x, y), blockSize, "addSingle", gpuMatrix, add, gpuResult);

            gpu.Synchronize();

            gpu.CopyFromDevice(gpuResult, result);

            gpu.FreeAll();

            return result;
        }

        public static double[,] Add(this double[,] left, double[,] right)
        {
            if (left.Length < 1 || right.Length < 1)
                return new double[0, 0];

            int fields = Math.Min(left.GetLength(0), right.GetLength(1));
            int x = right.GetLength(0);
            int y = left.GetLength(1);

            double[,] result = new double[x, y];

            gpu.LoadModule(module);

            double[,] gpuLeft = gpu.Allocate(left);
            double[,] gpuRight = gpu.Allocate(right);
            double[,] gpuResult = gpu.Allocate(result);

            gpu.CopyToDevice(left, gpuLeft);
            gpu.CopyToDevice(right, gpuRight);
            gpu.CopyToDevice(result, gpuResult);

            gpu.Launch(getGridSize(x, y), blockSize, "addMatrix", gpuLeft, gpuRight, fields, gpuResult);

            gpu.Synchronize();

            gpu.CopyFromDevice(gpuResult, result);

            gpu.FreeAll();

            return result;
        }

        public static double[,] Multiply(this double[,] matrix, int multiplicator)
        {
            if (matrix.Length < 1)
                return new double[0, 0];

            int x = matrix.GetLength(0);
            int y = matrix.GetLength(1);

            double[,] result = new double[x, y];

            gpu.LoadModule(module);

            double[,] gpuMatrix = gpu.Allocate(matrix);
            double[,] gpuResult = gpu.Allocate(result);

            gpu.CopyToDevice(matrix, gpuMatrix);
            gpu.CopyToDevice(result, gpuResult);

            gpu.Launch(getGridSize(x, y), blockSize, "multiplySingle", gpuMatrix, multiplicator, gpuResult);

            gpu.Synchronize();

            gpu.CopyFromDevice(gpuResult, result);

            gpu.FreeAll();

            return result;
        }

        public static double[,] Multiply(this double[,] left, double[,] right)
        {
            if (left.Length < 1 || right.Length < 1)
                return new double[0, 0];

            int fields = Math.Min(left.GetLength(0), right.GetLength(1));
            int x = right.GetLength(0);
            int y = left.GetLength(1);

            double[,] result = new double[x, y];

            gpu.LoadModule(module);

            double[,] gpuLeft = gpu.Allocate(left);
            double[,] gpuRight = gpu.Allocate(right);
            double[,] gpuResult = gpu.Allocate(result);

            gpu.CopyToDevice(left, gpuLeft);
            gpu.CopyToDevice(right, gpuRight);
            gpu.CopyToDevice(result, gpuResult);

            gpu.Launch(getGridSize(x, y), blockSize, "multiplyMatrix", gpuLeft, gpuRight, fields, gpuResult);

            gpu.Synchronize();

            gpu.CopyFromDevice(gpuResult, result);

            gpu.FreeAll();

            return result;
        }

        public static double[,] Subtract(this double[,] left, double[,] right)
        {
            if (left.Length < 1 || right.Length < 1)
                return new double[0, 0];

            int fields = Math.Min(left.GetLength(0), right.GetLength(1));
            int x = right.GetLength(0);
            int y = left.GetLength(1);

            double[,] result = new double[x, y];

            gpu.LoadModule(module);

            double[,] gpuLeft = gpu.Allocate(left);
            double[,] gpuRight = gpu.Allocate(right);
            double[,] gpuResult = gpu.Allocate(result);

            gpu.CopyToDevice(left, gpuLeft);
            gpu.CopyToDevice(right, gpuRight);
            gpu.CopyToDevice(result, gpuResult);

            gpu.Launch(getGridSize(x, y), blockSize, "subtractMatrix", gpuLeft, gpuRight, fields, gpuResult);

            gpu.Synchronize();

            gpu.CopyFromDevice(gpuResult, result);

            gpu.FreeAll();

            return result;
        }

        public static double[,] Subtract(this double[,] matrix, int add)
        {
            if (matrix.Length < 1)
                return new double[0, 0];

            int x = matrix.GetLength(0);
            int y = matrix.GetLength(1);

            double[,] result = new double[x, y];

            gpu.LoadModule(module);

            double[,] gpuMatrix = gpu.Allocate(matrix);
            double[,] gpuResult = gpu.Allocate(result);

            gpu.CopyToDevice(matrix, gpuMatrix);
            gpu.CopyToDevice(result, gpuResult);

            gpu.Launch(getGridSize(x, y), blockSize, "subtractSingle", gpuMatrix, add, gpuResult);

            gpu.Synchronize();

            gpu.CopyFromDevice(gpuResult, result);

            gpu.FreeAll();

            return result;
        }

        [Cudafy]
        private static void addMatrix(GThread thread, double[,] left, double[,] right, int fields, double[,] result)
        {
            int x = (blockSide * thread.blockIdx.x) + thread.threadIdx.x;
            int y = (blockSide * thread.blockIdx.y) + thread.threadIdx.y;

            if (x < result.GetLength(0) && y < result.GetLength(1))
                result[x, y] = left[x, y] + right[x, y];
        }

        [Cudafy]
        private static void addSingle(GThread thread, double[,] matrix, int add, double[,] result)
        {
            int x = (blockSide * thread.blockIdx.x) + thread.threadIdx.x;
            int y = (blockSide * thread.blockIdx.y) + thread.threadIdx.y;

            if (x < result.GetLength(0) && y < result.GetLength(1))
                result[x, y] = matrix[x, y] + add;
        }

        private static dim3 getGridSize(int x, int y)
        {
            return new dim3(((x - (x % blockSide)) / blockSide) + 1, ((y - (y % blockSide)) / blockSide) + 1);
        }

        [Cudafy]
        private static void multiplyMatrix(GThread thread, double[,] left, double[,] right, int fields, double[,] result)
        {
            int x = (blockSide * thread.blockIdx.x) + thread.threadIdx.x;
            int y = (blockSide * thread.blockIdx.y) + thread.threadIdx.y;

            if (x < result.GetLength(0) && y < result.GetLength(1))
            {
                double tempResult = 0;

                for (int offset = 0; offset < fields; offset++)
                    tempResult += left[offset, y] * right[x, offset];
            }
        }

        [Cudafy]
        private static void multiplySingle(GThread thread, double[,] matrix, int multiplicator, double[,] result)
        {
            int x = (blockSide * thread.blockIdx.x) + thread.threadIdx.x;
            int y = (blockSide * thread.blockIdx.y) + thread.threadIdx.y;

            if (x < result.GetLength(0) && y < result.GetLength(1))
                result[x, y] = matrix[x, y] * multiplicator;
        }

        [Cudafy]
        private static void subtractMatrix(GThread thread, double[,] left, double[,] right, int fields, double[,] result)
        {
            int x = (blockSide * thread.blockIdx.x) + thread.threadIdx.x;
            int y = (blockSide * thread.blockIdx.y) + thread.threadIdx.y;

            if (x < result.GetLength(0) && y < result.GetLength(1))
                result[x, y] = left[x, y] - right[x, y];
        }

        [Cudafy]
        private static void subtractSingle(GThread thread, double[,] matrix, int subtract, double[,] result)
        {
            int x = (blockSide * thread.blockIdx.x) + thread.threadIdx.x;
            int y = (blockSide * thread.blockIdx.y) + thread.threadIdx.y;

            if (x < result.GetLength(0) && y < result.GetLength(1))
                result[x, y] = matrix[x, y] - subtract;
        }
    }
}