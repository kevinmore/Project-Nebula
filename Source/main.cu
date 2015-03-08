#include <stdio.h>
#include <string.h>
#include <Windows.h>
#include <cuda_runtime.h>

__global__ void Calc(int a, int b, int *c)
{
	for (int i = 0; i < 9999999999; ++i)
	{
		*c = a + b;
	}
}

int main()
{
	int a, b, c;
	int *dev_c;
	scanf("%d%d", &a, &b);
	cudaMalloc(&dev_c, sizeof(int));
	Calc<<<1, 1>>>(a, b, dev_c);
	cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
	printf("%d\n", c);
	system("pause");
	return 0;

}