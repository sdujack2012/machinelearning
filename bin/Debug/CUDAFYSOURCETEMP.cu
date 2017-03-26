
// CudaMath.Double.TwoDimensional
extern "C" __global__  void addMatrix( double* left, int leftLen0, int leftLen1,  double* right, int rightLen0, int rightLen1, int fields,  double* result, int resultLen0, int resultLen1);
// CudaMath.Double.TwoDimensional
extern "C" __global__  void addSingle( double* matrix, int matrixLen0, int matrixLen1, int add,  double* result, int resultLen0, int resultLen1);
// CudaMath.Double.TwoDimensional
extern "C" __global__  void multiplyMatrix( double* left, int leftLen0, int leftLen1,  double* right, int rightLen0, int rightLen1, int fields,  double* result, int resultLen0, int resultLen1);
// CudaMath.Double.TwoDimensional
extern "C" __global__  void multiplySingle( double* matrix, int matrixLen0, int matrixLen1, int multiplicator,  double* result, int resultLen0, int resultLen1);
// CudaMath.Double.TwoDimensional
extern "C" __global__  void subtractMatrix( double* left, int leftLen0, int leftLen1,  double* right, int rightLen0, int rightLen1, int fields,  double* result, int resultLen0, int resultLen1);
// CudaMath.Double.TwoDimensional
extern "C" __global__  void subtractSingle( double* matrix, int matrixLen0, int matrixLen1, int subtract,  double* result, int resultLen0, int resultLen1);

// CudaMath.Double.TwoDimensional
extern "C" __global__  void addMatrix( double* left, int leftLen0, int leftLen1,  double* right, int rightLen0, int rightLen1, int fields,  double* result, int resultLen0, int resultLen1)
{
	int num = 16 * blockIdx.x + threadIdx.x;
	int num2 = 16 * blockIdx.y + threadIdx.y;
	bool flag = num < resultLen0 && num2 < resultLen1;
	if (flag)
	{
		result[(num) * resultLen1 + ( num2)] = left[(num) * leftLen1 + ( num2)] + right[(num) * rightLen1 + ( num2)];
	}
}
// CudaMath.Double.TwoDimensional
extern "C" __global__  void addSingle( double* matrix, int matrixLen0, int matrixLen1, int add,  double* result, int resultLen0, int resultLen1)
{
	int num = 16 * blockIdx.x + threadIdx.x;
	int num2 = 16 * blockIdx.y + threadIdx.y;
	bool flag = num < resultLen0 && num2 < resultLen1;
	if (flag)
	{
		result[(num) * resultLen1 + ( num2)] = matrix[(num) * matrixLen1 + ( num2)] + (double)add;
	}
}
// CudaMath.Double.TwoDimensional
extern "C" __global__  void multiplyMatrix( double* left, int leftLen0, int leftLen1,  double* right, int rightLen0, int rightLen1, int fields,  double* result, int resultLen0, int resultLen1)
{
	int num = 16 * blockIdx.x + threadIdx.x;
	int num2 = 16 * blockIdx.y + threadIdx.y;
	bool flag = num < resultLen0 && num2 < resultLen1;
	if (flag)
	{
		double num3 = 0.0;
		for (int i = 0; i < fields; i++)
		{
			num3 += left[(i) * leftLen1 + ( num2)] * right[(num) * rightLen1 + ( i)];
		}
		result[(num) * resultLen1 + ( num2)] = num3;
	}
}
// CudaMath.Double.TwoDimensional
extern "C" __global__  void multiplySingle( double* matrix, int matrixLen0, int matrixLen1, int multiplicator,  double* result, int resultLen0, int resultLen1)
{
	int num = 16 * blockIdx.x + threadIdx.x;
	int num2 = 16 * blockIdx.y + threadIdx.y;
	bool flag = num < resultLen0 && num2 < resultLen1;
	if (flag)
	{
		result[(num) * resultLen1 + ( num2)] = matrix[(num) * matrixLen1 + ( num2)] * (double)multiplicator;
	}
}
// CudaMath.Double.TwoDimensional
extern "C" __global__  void subtractMatrix( double* left, int leftLen0, int leftLen1,  double* right, int rightLen0, int rightLen1, int fields,  double* result, int resultLen0, int resultLen1)
{
	int num = 16 * blockIdx.x + threadIdx.x;
	int num2 = 16 * blockIdx.y + threadIdx.y;
	bool flag = num < resultLen0 && num2 < resultLen1;
	if (flag)
	{
		result[(num) * resultLen1 + ( num2)] = left[(num) * leftLen1 + ( num2)] - right[(num) * rightLen1 + ( num2)];
	}
}
// CudaMath.Double.TwoDimensional
extern "C" __global__  void subtractSingle( double* matrix, int matrixLen0, int matrixLen1, int subtract,  double* result, int resultLen0, int resultLen1)
{
	int num = 16 * blockIdx.x + threadIdx.x;
	int num2 = 16 * blockIdx.y + threadIdx.y;
	bool flag = num < resultLen0 && num2 < resultLen1;
	if (flag)
	{
		result[(num) * resultLen1 + ( num2)] = matrix[(num) * matrixLen1 + ( num2)] - (double)subtract;
	}
}
