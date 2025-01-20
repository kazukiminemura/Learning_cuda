#include <stdio.h>
#include <math.h>
// #inlcude <thrust/device.h>
// #include <thrust/device_ptr.h>

// Device parameter
const int NUM_SM = 56;
const int WARPS_PER_SM = 64;
const int WARPS_PER_BLOCK = 32;
const int NUM_BLOCKS = NUM_SM * WARPS_PER_SM / WARPS_PER_BLOCK;
const int THREADS_PER_WARP = 32;
const int THREADS_PER_BLOCK = WARPS_PER_BLOCK * THREADS_PER_WARP;

__global__ void ReductionCascading(
    int* inArray, unsigned int numElements, int* halfwayResult)
{
  // thread id in each block
  const unsigned int tid = threadIdx.x;
  // warp id in each block
  const unsigned int wid = tid / THREADS_PER_WARP;
  // lane id = thread id in each warp
  const unsigned int lid = tid % THREADS_PER_WARP;
  // sequnetial thread id in the kernel
  const unsigned int sid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numTotalThreads = gridDim.x * blockDim.x;

  int item1 = 0;
  __shared__ int x[WARPS_PER_BLOCK * THREADS_PER_WARP]; // array for each thread sum
  __shared__ int y[WARPS_PER_BLOCK];                    // array for each warp sum
  const int numIter = 1 + (numElements - 1) / numTotalThreads; // round up
  // Main
  for(int i = 0; i < numIter - 1; i++){ // sum corresponding data in inArray
    item1 += inArray[numTotalThreads*i + sid];
  }
  // Last: some threads have one more data
  int idx = numTotalThreads * (numIter - 1) + sid;
  if(idx < numElements){
    item1 += inArray[idx];
  }
  x[tid] = item1;
  __syncwarp(); // sync for threads in a warp

  // Reduction for a warp
  x[tid] += x[tid + 16];
  __syncwarp();
  x[tid] += x[tid + 8];
  __syncwarp();
  x[tid] += x[tid + 4];
  __syncwarp();
  x[tid] += x[tid + 2];
  __syncwarp();
  x[tid] += x[tid + 1];
  if( 0 == lid){
    y[wid] = x[tid];
  }
  __syncthreads(); // sync with all threads in a block

  // Reduction for a block
  if(tid < 32){
    y[tid] += y[tid + 16];
    __syncwarp();
    y[tid] += y[tid + 8];
    __syncwarp();
    y[tid] += y[tid + 4];
    __syncwarp();
    y[tid] += y[tid + 2];
    __syncwarp();
    y[tid] += y[tid + 1];
  }
  if( 0 == tid){
    halfwayResult[blockIdx.x] = y[tid];
  }

  return;
}


int main()
{
  int logNumElements = 25; // input_size = 2^25
  int numElements = 1 << logNumElements;
  int* h_input = new int[numElements]; // on host (CPU)
  int* h_halfwayResult = new int[NUM_BLOCKS]; //
  int* h_output = 0;
  int* d_input;                       // on device (GPU)
  int* d_halfwayResult;

  cudaMalloc(&d_input, numElements * sizeof(int));
  cudaMalloc(&d_halfwayResult, NUM_BLOCKS * sizeof(int));
  srand(0);

  for( unsigned int i = 0; i < numElements; ++i){
    h_input[i] = (int)(rand() % 2);
  }
  int answer = 0; // final answer
  for (int i = 0; i < numElements; i++){
    answer += h_input[i];
  }

  // Measure the elapsed time
  cudaEvent_t start, stop;
  float time_ms;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Copy data from host to device
  cudaMemcpy(d_input, h_input, numElements * sizeof(int), cudaMemcpyHostToDevice);

  // Main
  printf("numElements = %d (2^%d)\n", numElements, logNumElements);
  cudaEventRecord(start, 0);
  ReductionCascading <<< NUM_BLOCKS, THREADS_PER_BLOCK >>> 
      (d_input, numElements, d_halfwayResult);
  
  // Copy data from device to host
  cudaMemcpy(h_halfwayResult, d_halfwayResult, NUM_BLOCKS * sizeof(int), cudaMemcpyDeviceToHost);

  h_output = 0;
  for (int i = 0; i < NUM_BLOCKS; i++){
    h_output += h_halfwayResult[i];
  }

  // elapsed time
  cudaEventRecord(stop, 0);
  // wait until cudaEventRecord (asyncronous function) is completed
  cudaEventSynchronize(stop);

  // Display results
  printf("Our result = %d, answer = %d\n", h_output, answer);
  cudaEventElapsedTime(&time_ms, start, stop);
  printf(" exe time: %f ms\n", time_ms);

  // Thrust Library
  // int h_outputThrust;
  // thrust::device_ptr<int> d_input_ptr = thrust::device_pointer_cast(d_input);
  // cudaEventRecord(start, 0);
  // h_outputThrust = thrust::reduce(d_input_ptr, d_input_ptr + numElements);
  // cudaEventRecord(stop, 0);
  // // Wait until cudaEventRecord is completed.
  // // (Note: cudaEventRecord is an asynchronous function)
  // cudaEventSynchronize(stop);
  // printf("Thrust result = %d, answer = %d\n", h_outputThrust, answer);
  // cudaEventElapsedTime(&time_ms, start, stop);
  // printf(" exe time: %f ms\n", time_ms);

  // Free memory
  delete [] h_input;
  delete [] h_halfwayResult;
  cudaFree(d_input);
  cudaFree(d_halfwayResult);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}
