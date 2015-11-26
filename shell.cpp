#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void shell_sort(float *array, int N){

	int a, i, j, colIndex, currentColNum; 
	
		
	for(a = 0; a < N; a++)
	{
		array[a] = random();
	}	

    int colsNums[] = { 33936, 19001, 8358, 2878, 997, 413, 156, 34, 9, 1};
	
    for (colIndex=0; colIndex < 10; colIndex++)
    {
        currentColNum = colsNums[colIndex];
        for (i=currentColNum; i<N; i++)
        {
            float value = array[i];
            j=i;
            while (j >= currentColNum && array[j-currentColNum] > value)
            {
                array[j] = array[j-currentColNum];
                j=j-currentColNum;
            }
            array[j] = value;
        }
	
    }
}


int min(float * proc, int * count, int local_size, int numprocs){ 
	int i = 0;
	int min = 0;
	while(count[i] >= local_size-1 && i < numprocs){
		i++;
	}
	//printf("i:%d\n", i);
	if(i < numprocs){
		min = i;
		for(; i < numprocs; i++){
			if(min > proc[i] && count[i] < local_size-1)
				min = i;
		}
	}
	//printf("asdasd: %d\n",min);
	return min;
}

int main(int argc, char **argv){

	float N = 100000;
	int myid, numprocs;

	int i, j, k, l, start, end, sent = 0, my;
	float timeStart, timeEnd;
	float * A;

	
	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&myid);
	int local_size = N/numprocs;
	//buffer will be filled with the data comming from other proc.
	float * buffer = (float *)malloc (sizeof(float)*1);
	
	//SUB_ARRAY IS THE LOCAL ARRAY THAT EACH PROCESS WORK IN
	float * sub_array = (float *)malloc (sizeof(float)*local_size);
	
	if(myid == 0){
		timeStart = MPI_Wtime();	
	}
	
	//EVERYBODY CREATES/INITIALIZE OWN SECTION
	for(i = 0; i < local_size; i++){
		//sub_array[i] = random() / ((myid+1)*(myid+1));
		double t = (double)i / (double)N;
		sub_array[i] = t*t*sin(2000*t*t);
	}
	//EVERYBODY CREATES/INITIALIZE OWN SECTION
	if(myid == 0){
		timeEnd = MPI_Wtime();
		printf("creation time %f", (timeEnd - timeStart));
		
	}
	//EVERY PROCESS SORTS ITS OWN SETION.
	shell_sort(sub_array, local_size);
	
		
	if(myid == 0){
		A = (float *)malloc(sizeof(float)*N);
		float * proc = (float *)malloc((sizeof(float)*numprocs));
		int * count = (int *)malloc((sizeof(int)*numprocs));
		proc[0] = sub_array[0];
		count[0]++;
		for(i = 1; i < numprocs; i++){
			float * min = (float *)malloc(sizeof(float)*1);
			MPI_Recv ((void *)min, 1, MPI_FLOAT, i, 0xACE5, MPI_COMM_WORLD, NULL);
			//printf("i: %d, %f\n",i , min[0]);
			proc[i] = min[0];
			count[i]++;
		}
		
			
		int next_index = 0;
		int index;
		float *buff;
	
		while(next_index < N){
			index = min(proc, count, local_size, numprocs);
		
			A[next_index] = proc[index];
			next_index++;
			//printf("NEXT_INDEX: %d, EKLENEN: %f, FROM: %d\n",next_index, proc[index], index);
			if(index != 0){
				MPI_Recv ((void *)buffer, 1, MPI_FLOAT, index, 0xACE5, MPI_COMM_WORLD, NULL);
				proc[index] = buffer[0];
				count[index] ++;
			}
			else{
				proc[0] = sub_array[count[0]];
				count[0]++;
			}
		
		}
		//shell_sort(A,N);
	
	}
	else{
		int counter = 0;
		
		while(counter < local_size){ 
			buffer[0] = sub_array[counter];
			MPI_Send ((void *)buffer, 1, MPI_FLOAT, 0, 0xACE5, MPI_COMM_WORLD);
			counter++;
		}
	}
	
	if(myid == 0){
		timeEnd = MPI_Wtime();
		printf("TIME IT TAKES IN SECONDS IS: %f\n", (timeEnd - timeStart));
	}
	
	/*if(myid == 0){
		FILE *dosya;
		dosya = fopen("./out", "w");
		for(i = 0; i < N; i++)
			fprintf(dosya, "%f\n", A[i]);
	}*/
	
	MPI_Finalize();
	
}