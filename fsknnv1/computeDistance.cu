/*
** Copyright (C) 2011 Centre for Bioinformatics, Biomarker Discovery and 
** Information-Based Medicine
** http://www.newcastle.edu.au/research-centre/cibm/
**  
** Code for Paper:
** FS-kNN: A Fast and Scalable kNN Computation Technique using GPU,
** Ahmed Shamsul Arefin, Carlos Riveros, Regina Berrettaand Pablo Moscato,
** Email: ASA∗ - ahmed.arefin@newcastle.edu.au; CR - Carlos.Riveros@newcastle.edu.au ; RB - regina.berretta.@newcastle.edu.au ; PM
** pablo.moscato@newcastle.edu.au
** 
**   
** This program is free software; you can redistribute it and/or modify
** it under the terms of the GNU General Public License as published by
** the Free Software Foundation; either version 2 of the License, or
** (at your option) any later version.
**
** This program is distributed in the hope that it will be useful,
** but WITHOUT ANY WARRANTY; without even the implied warranty of
** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
** GNU General Public License for more details.
**
** You should have received a copy of the GNU General Public License
** along with this program; if not, write to the Free Software
** Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
**
** computeDistance.cu // requires CUDA API to compile
**
**
*/



#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <sys/time.h>

#define b 16
#define MaxValue 4294967295


/*  Implements kNNKernel Algorithm
 * 
 * Input Parameters:
 * 
 * Da 			: Distance matrix chunk (chunkSize x chunkSize)
 * Gka			: An array of 3-tuples {source,target,weight} to store the kNN graph (Gka)
 * Maxka		: An array that contains the index of the farthest nearest neighbours of each node (in a chunk)
 * chunkSize    : Total number of rows in a chunk
 * nRow         : Total rows in the original distance matrix 
 * nExtraRow	: Extra rows added to fit all the chunks
 * split        : split ID 
 * chunk        : chunk ID
 * splitSize    : Total number of chunks in a split
 * chunkSize    : Total number of rows in a chunk
 * tid          : Openmp threadID (segment/gpu ID)
 * k            : Integer value of k
 * 
 */



__global__ void kNNKernel (float* Da,  float3* Gka, int *Maxka, int chunkSize, int nExtraRow, int split, int chunk, int splitSize,  int nRow, int tid, int segmentSize, int k)
{

	
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (i<chunkSize)
	
	{
		
		int beginSearch=(split-tid*segmentSize)*k*chunkSize+i*k;
		int endSearch=(split-tid*segmentSize)*k*chunkSize+i*k+k;
		
		for(int j=0;j<chunkSize;j++){
			if(split==chunk){
					if(i!=j){
							if ( (nExtraRow!=0) && ((split>=splitSize-1)||(chunk>=splitSize-1)) && (i>=chunkSize-nExtraRow||j>=chunkSize-nExtraRow) )
									continue;

							if((*(Gka +  *(Maxka+i))).z>*(Da+chunkSize*i+j)){
								(*(Gka + *(Maxka+i))).z=*(Da+chunkSize*i+j);
								(*(Gka + *(Maxka+i))).x=split*chunkSize+i;
								(*(Gka + *(Maxka+i))).y=chunk*chunkSize+j;

								for (int index = beginSearch;index <endSearch;index ++){
									if((*(Gka+index)).z >(*(Gka + *(Maxka+i))).z)
									*(Maxka+i)=index;
								}
							} 
					}				
			} 
			
			else
			
			{
							if ( (nExtraRow!=0) && (j>=chunkSize-nExtraRow &&  chunk>=splitSize-1) || (nExtraRow!=0) && (i>=chunkSize-nExtraRow && split>=splitSize-1)   ) 
								continue;
							
							if((*(Gka +  *(Maxka+i))).z>*(Da+chunkSize*i+j)){
							
								(*(Gka + *(Maxka+i))).z=*(Da+chunkSize*i+j);
								(*(Gka + *(Maxka+i))).x=split*chunkSize+i;
								(*(Gka + *(Maxka+i))).y=chunk*chunkSize+j;

								for (int index = beginSearch; index < endSearch; index ++){
											if((*(Gka+index)).z >(*(Gka + *(Maxka+i))).z)
												*(Maxka+i)= index;
								} 
							} 
			  } 
		} 
	} 
	
}



/*
 * 
 * Distance kernel (Implements Pearsons Correlation) (adapted from Chang et al. (2009))
 * We compute a distance matrix chunk (chunkSize x chunkSize)
 *  
 * Input Parameters:
 * Ina          : Input matrix (nRow x nCol)
 * Da 			: Distance matrix chunk (chunkSize x chunkSize)
 * chunkSize    : Total number of rows in a chunk (must be multiple of b)
 * split        : split ID 
 * chunk        : chunk ID
 * nExtraCol    : No of extra columns added afterwards
 * nColExtended : Total number of columns in the extended input matrix
 * 
 * 
 */

__global__ void PearsonDistanceKernel(float *Ina, float *Da, int chunkSize,  int split, int chunk, int nRow, int nCol, int nExtraCol, int nColExtended){
  
	
  __shared__ float Xs[b][b];
  __shared__ float Ys[b][b]; 
  
  
  int bx = blockIdx.x, by = blockIdx.y;
  int tx = threadIdx.x, ty = threadIdx.y;
  
  

 int xBegin = (bx + (chunkSize/b)*chunk) * b * nCol;
 int yBegin = (by + (chunkSize/b)*split) * b * nCol;
 
   
  
  
  int yEnd = yBegin + nCol - 1;
  int x, y, i, index;
  
  float a1, a2, a3, a4, a5;
  float avgX, avgY, varX, varY, cov, rho;
  
  a1 = a2 = a3 = a4 = a5 = 0.0;
  for(y=yBegin,x=xBegin;y<=yEnd;y+=b,x+=b){
    
	Ys[ty][tx] = Ina[y + ty*nCol + tx];
    Xs[tx][ty] = Ina[x + ty*nCol + tx];
  
    __syncthreads();
    for(i=0;i<b;i++){

    	
   if (nExtraCol!=0 && (y>=(yBegin +(nColExtended-b))) && (i>= nCol-(nColExtended-b)))  continue;
	   
      a1 += Xs[i][tx];
      a2 += Ys[ty][i];
      a3 += Xs[i][tx] * Xs[i][tx];
      a4 += Ys[ty][i] * Ys[ty][i];
      a5 += Xs[i][tx] * Ys[ty][i];
    }
    __syncthreads();
  }
  avgX = a1/nCol;
  avgY = a2/nCol;
  varX = (a3-avgX*avgX*nCol)/(nCol-1);
  varY = (a4-avgY*avgY*nCol)/(nCol-1);
  cov = (a5-avgX*avgY*nCol)/(nCol-1);
  rho = cov/sqrtf(varX*varY);
  index = by*b*chunkSize + ty*chunkSize + bx*b + tx;
  Da[index] = rho;

}




/*
 * knn function to represent the fsknn algorithm.
 * 
 * 
 * 
 * Input prameters:
 * 
 * filenameOut  : Output File name
 * Ina          : Input matrix (nRow x nCol)
 * nRow         : Total number of rows in the input matrix
 * nCol         : Total number of columns in the input matrix
 * chunkSize    : Total number of rows/columns in a chunk of the distance matrix
 * k            : The integer value of k.
 * 
 * Ouput:
 * 
 * 
 * Graph(kNN) will be writen in (filenameOut).knn file in the following format:
 * 
 * number of vertices,  number of edges
 * source, target, weight
 * ....
 * source, target, weight
 * 
 * 
 */

int knn(char *filenameOut, float *Ina, int nRow, int nCol, int chunkSize, int k)
{

	
	
	/* define thread and blocks for Distance Kernel */
	dim3 blocksD(chunkSize/b,chunkSize/b);
	dim3 threadsD(b,b);

	
	/* define thread and blocks for kNN Kernel */
	int threadsK = 512;
	int blocksK = chunkSize/threadsK+1;
	
	
	
	
	/* Declare local variables */
	
	float3 *Gka;  // An array of 3-tuples {source,target,weight} to store the kNN graph (Gka)
	int nRowExtended; 
	int nColExtended;
	int nExtraCol;
	int nExtraRow;
	int splitSize;
	int segmentSize; 
	
	
		
	struct tm *current;
	time_t now;

	
	/* Extend the columns */
	nColExtended=nCol;
	
	while (nColExtended%b!=0){
			nColExtended++;
	}
	
	nExtraCol = nColExtended-nCol;
	
	/* Extend the rows */
	nRowExtended = nRow;
	while (nRowExtended%chunkSize!=0){
		nRowExtended++;
	}
		
	nExtraRow=nRowExtended-nRow;
	

	
	/* Identify total number of chunks ina split  */
	splitSize=nRowExtended/chunkSize;
	
	
	/* Allocate memory to store the kNN graph*/
	Gka = (float3*)malloc(sizeof(float3)*nRowExtended*k);

	
	/* Initialise weights to INT_MAX*/
	for (int i = 0; i < nRowExtended*k; i++)
	{
			Gka[i].z=MaxValue;
	}
	
   
	
	/* Setup multi-GPUs  */
		
	int nGpu=0;
	cudaGetDeviceCount(&nGpu);

	if(nGpu<1){
		printf("no CUDA capable devices were detected!\n");
		return 0;
	}
	
	if(nGpu>splitSize)
		nGpu=splitSize;
	

	segmentSize = splitSize/nGpu;
	
	
	printf("number of host CPUs:\t%d\n", omp_get_num_procs());
    printf("number of CUDA devices:\t%d\n", nGpu);
    for(int i = 0; i < nGpu; i++)
	{
	     cudaDeviceProp dprop;
	     cudaGetDeviceProperties(&dprop, i);
	     printf("   %d: %s\n", i, dprop.name);
	}
	printf("---------------------------\n");


   printf("Total no of chunks in each split (splitSize):\t%d\n", splitSize);
   printf("Total no of splits in each segment (segmentSize):\t%d\n", segmentSize);
   
   
   /* Implementation of the FSkNN algorithm   */  
   

   omp_set_num_threads(nGpu);
   #pragma omp parallel
    	{
	   	   float *Da;       	
	   	   int *Maxka;
	   	   float3 *Gka_sub;
	   	   unsigned int Gka_sub_bytes;
	   	       	
	   	   
	   	   float *dev_Ina,  *dev_Da;
	   	   int *dev_maxArray;
	   
	   	   
	       
	   	   int tid = omp_get_thread_num();
	   	   int numthread = omp_get_num_threads();
	   	   int gpuid = -1;
    	
	   	   cudaSetDevice(tid);
	   	   cudaGetDevice(&gpuid);
	   	   
	   	   Da = (float*)malloc(sizeof(float)*chunkSize*chunkSize);
	   	   Maxka = (int*)malloc(sizeof(int)*chunkSize);
     	
	   	   cudaMalloc((void **) &dev_Ina, nRow*nCol*sizeof(float));
	   	   cudaMalloc((void **) &dev_Da, chunkSize*chunkSize*sizeof(float));
	   	   cudaMalloc((void **) &dev_maxArray, chunkSize*sizeof(int));
	
	   	   cudaMemcpy(dev_Ina, Ina, nRow*nCol*sizeof(float),cudaMemcpyHostToDevice);
	   	   cudaMemcpy(dev_Da, Da, chunkSize*chunkSize*sizeof(float),cudaMemcpyHostToDevice);
	   	   
	   	   printf("CPU thread %d (of %d) uses CUDA device %d\n", tid, numthread, gpuid);
    	
	   	   int splitBegin = tid*segmentSize;
	   	   int splitEnd= splitBegin+segmentSize;


	   	   if ((splitSize-1)-splitEnd <= segmentSize){
	   		   splitEnd=splitSize;
    	   }
	
	   	   Gka_sub = Gka + (tid*chunkSize*k)*(splitSize/numthread);
	   	   Gka_sub_bytes = (((tid+1)*chunkSize*k*(splitEnd-splitBegin))-(tid*chunkSize*k*(splitEnd-splitBegin)))*sizeof(float3);
	       
		   float3 *dev_Gka_sub=0;
	       cudaMalloc((void **)&dev_Gka_sub,Gka_sub_bytes);
	       
	       cudaMemset (dev_Gka_sub,0,Gka_sub_bytes);
	       cudaMemcpy (dev_Gka_sub,Gka_sub,Gka_sub_bytes,cudaMemcpyHostToDevice);

    	  for (int split=splitBegin;split<splitEnd;split++){

    		  for (int r=0;r<chunkSize;r++){
  		  	  	  	  *(Maxka + r)=(split-tid*segmentSize)*k*chunkSize+r*k;
   		  	  }

    		  cudaMemcpy(dev_maxArray, Maxka, chunkSize*sizeof(int),cudaMemcpyHostToDevice);

    		  for (int chunk=0;chunk<splitSize;chunk++){
			
    			  PearsonDistanceKernel<<<blocksD,threadsD>>>(dev_Ina,  dev_Da, chunkSize, split, chunk,  nRow,nCol,nExtraCol, nColExtended);
				  kNNKernel<<<blocksK,threadsK>>>(dev_Da,  dev_Gka_sub, dev_maxArray, chunkSize, nExtraRow, split, chunk, splitSize, nRow, tid, segmentSize, k);
			  }  
    	  }  
	
    	  cudaMemcpy(Gka_sub, dev_Gka_sub, Gka_sub_bytes,cudaMemcpyDeviceToHost);
    	  
    	  cudaFree(dev_Ina); 
    	  cudaFree(dev_Da);
    	  cudaFree(dev_maxArray);
    	  cudaFree(dev_Gka_sub);
	
    } // OpenMP loop
	
  	  time(&now);
	  current = localtime(&now);
	  printf("\n\nqNN computation finished at time : %i:%i:%i\n", current->tm_hour, current->tm_min, current->tm_sec);

	  FILE * pFile;
	  printf ("Writing....output File Name = %s",filenameOut);
	  pFile = fopen (filenameOut,"w");
	    
	  if (pFile!=NULL)  {
	    	  fprintf(pFile,"%d %d\n",nRow,nRow*k);
	    	  for (int i = 0; i < nRow*k; i++)
	    	  	  {
	    		  	  fprintf(pFile,"%d %d %f\n",int(Gka[i].x), int(Gka[i].y), Gka[i].z);
	    	  	  }
	    	  fclose (pFile);
	      }
	 return 0;

}



