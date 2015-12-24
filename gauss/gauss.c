/*                   Gaussian Elimination case study.
                             C+MPI version. */

/* This file contains the C+MPI code for a Gaussian elimination
   program. It was produced using the APM methodology. This version
   of the program distributes the data in a cyclic fashion and uses
   pipelining optimisations in the back substitution phase. */

/* When this program is run, the user enters the "coordinates" of the
   set of linear equations to be solved in the form of the values of
   the appropriate matrix, A, and vector, b.  The program returns the
   values of the vector, x, which solve the equations. */

/* To run this program you need access to the MPI library */

/* NOTE: Due to problems with C misbehaving, the matrix in this
   program is stored in a 1D array, rather than a 2D matrix, but it
   still works correctly. */

/* -------------------------------- Header -------------------------------- */

#include <stdio.h>
#include <mpi.h>

#define SIZE 100
#define PROCSIZE 50

  /* global variables */
int p;              /* number of processors */
int matrixsize;     /* number of linear equations, ie. size of matrix */
int pid;            /* processor id of the current processor */
int row_nos[PROCSIZE]; /* the row nos of the rows in the current proc */
int no_in_proc;     /* number of rows held in the current processor */

  /* input/output functions */
void enterint(int *var);
void entermatrixcyc(int n, int p, float var[]); 
/* the variable must be 2 dimensional since each proc stores several rows
    because of the cyclic storage */
void output(float values[]);

void gauss(float m[], float x[]);

void deal_with_jth_row(float matrix[], int j);
void put_0_in_jth_pos(float rowj[], int j, float row[], int rowno);

void solve_ith_eqn_init (float *sum, float row[]);
void solve_ith_eqn_g (int j, float *sum, float row[], float new);
void solve_ith_eqn_h (int rowno, float *sum, float row[]);

void pipeline (void (*init) (float *, float *),
	       void (*g) (int, float *, float *, float),
	       void (*h) (int, float *, float *),
	       float local_vals[], float res[]);

void makecyclic(float inarray[], float outarray[]);
void makecyclicmatrix(float inarray[], float outarray[]);
int mod(int x, int y);

/* ------------------------ Start of main program -------------------------- */

main(int argc, char *argv[])
{
  int errcode,row,i,j, maxblocksize;
  float m[PROCSIZE*(SIZE+1)]; 
     /* combine values from a and b which are in this proc */
  float result[PROCSIZE]; /* results */

    /*start MPI and set up global variables*/
  errcode = MPI_Init (&argc, &argv);

  MPI_Comm_size (MPI_COMM_WORLD, &p);
  MPI_Comm_rank (MPI_COMM_WORLD, &pid);

  enterint(&matrixsize);

  i=0;
  for(row=pid; row<=matrixsize; row=row+p)
    {
      row_nos[i] = row+1;
      i++;
    }

  if (matrixsize%p == 0)
    maxblocksize = matrixsize/p;
  else
    maxblocksize = matrixsize/p+1;

  if (pid<=(matrixsize-1)%p)
    no_in_proc = maxblocksize;
  else
    no_in_proc = maxblocksize-1;

    /* enter and broadcast/scatter other input values */
  entermatrixcyc(matrixsize, p, m); 

  gauss(m, result); 

  output(result); 

    /* finish MPI */
  errcode = MPI_Finalize ();
}

/* ---------------------- Input/output functions ------------------------ */

void enterint(int *var)
/* prompts for and receives an integer from the input and broadcasts it*/
{
  if (pid == 0)
    {
      printf("Enter size of matrix: ");
      scanf("%d",var);
    }
  MPI_Bcast(var, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

void entermatrixcyc(int n, int p, float var[])
/* Prompts for and receives an n*n matrix (a) and an n*1 vector (b)
   from the input. It then distributes the rows of this combined
   matrix in a cyclic fashion across p processors, storing it in var
   which is a (flattened) 2D matrix. */
{
  float input[SIZE*(SIZE+1)], cyclic[SIZE*(SIZE+1)]; 
    /* stores the values temporarily until they are scattered */
  int i,j,count,no_rows;
  int sizes[p], displs[p];  /* for use of scatter code */

    /* enter the matrix and the vector and make cyclic (all in proc 0) */
  if (pid == 0)
    {
      /* enter the matrix into input - leaving one space for the vector 
	 after each row */
      count = 0;
      printf("Enter the elements of the matrix.\n");
      printf("All floats please.\n");
      for (i=1; i<=n; i++)
	{
	  /* receive values in row i */
	  for (j=1;j<=n; j++)
	    {
	      printf("Enter a[%d][%d]: ", i, j);
	      scanf("%f", &input[count]);
	      count++;
	    }
	  count++;
	}

      /* enter the vector into the spaces that were left */
      for (i=1; i<=n; i++)
	{
	  printf("Enter b[%d]: ", i);
	  scanf("%f", &input[i*n+i-1]);
	  /* ie. (i-1)*(n+1)+n 
	     - the indices of the spaces at the end of each row. */
	}

      /* make input cyclic */
      makecyclicmatrix(input, cyclic);
    }

  /* set up sizes and displs variables for the scattering */
  for (i=0;i<n%p;i++)
    sizes[i] = (n/p + 1)*(n+1);
  for (i=n%p;i<p;i++)
    sizes[i] = (n/p)*(n+1);

  displs[0] = 0;
  for (i=1;i<p;i++)
    displs[i] = displs[i-1] + sizes[i-1];

    /* scatter it */
  MPI_Scatterv(cyclic, sizes, displs, MPI_FLOAT, var, SIZE*SIZE, 
               MPI_FLOAT, 0, MPI_COMM_WORLD);
}

void output(float values[])
/* Prints its parameter, values */
{
  int elt;

  if (pid == 0)
    printf("The solution vector is x = \n [");

    /* for each element in turn, print it */
  for (elt = 0; elt < matrixsize-1; elt++)
    {
      if (pid == elt%p)
	{
          printf("%f, ", values[elt/p]);
          fflush(NULL); /* the flush is necessary as the output is done from
                           processor 0 and without the flush, all the output
                           will only be gathered at the end. */
	}
      MPI_Barrier (MPI_COMM_WORLD); /* NB. This might not be very efficient */
    }

    /* final row */
  if (pid == (matrixsize-1)%p)
    printf("%f]\n", values[(matrixsize-1)/p]);
}

/* ------------------------ Gaussian Elimination -------------------------- */

void gauss(float m[], float x[])
/* converts m to lower triangular form, and stores the solution to
   this system of equations in x */
{
  int j;

    /* reduce to upper triangular */
  /* for each row (j) except the last, make all the rows below it have
     0 in their jth column - leave other rows alone. */
  for (j=1; j<matrixsize; j++)
    deal_with_jth_row(m, j);

    /* back substitution */
  /* use pipelining (function - pipeline) to solve one equation 
     after the other */
  pipeline(solve_ith_eqn_init, solve_ith_eqn_g, solve_ith_eqn_h, m, x);
}

/* ----------- Functions for reduction to upper triangular form ---------- */

void deal_with_jth_row(float matrix[], int j)
/* Changes matrix. Makes all of its rows below row j have 0 in 
   columns i (i<=j). Leaves the rest of the matrix alone. */
{
  float rowj[SIZE+1];
  int i;

    /* Store jth row in rowj - takes place in the proc where this row is.
       The local index of this row is (j-1)/p. */
  if (pid == mod(j-1,p))
      for(i=0;i<matrixsize+1;i++)
	rowj[i] = matrix[((j-1)/p)*(matrixsize+1)+i];

    /* now broadcast it */
  MPI_Bcast(rowj, matrixsize+1, MPI_FLOAT, mod(j-1,p), MPI_COMM_WORLD);

  /* for each row, put 0 in its jth position */
  for (i=0;i<no_in_proc;i++)
    put_0_in_jth_pos(rowj, j, &matrix[i*(matrixsize+1)], row_nos[i]);
}

void put_0_in_jth_pos(float rowj[], int j, float row[], int rowno)
/* Changes row by putting 0 in its jth position, if the row is
   below rowj (ie. if rowno>j) and leaves it alone otherwise.  It does
   this through a linear combination with rowj.
   This function is used by deal_with_jth_row.
   It is a local function - no communication. */
{
  int i;
  float multiplier;

  multiplier = -(row[j-1]/rowj[j-1]);

  if (rowno > j)
    for (i=0; i<matrixsize+1; i++)
      row[i] = multiplier*rowj[i] + row[i];
}

/* --------------------- Back Substitution Functions --------------------- */

/* These are parts of a local subsidiary function (ie. no communication),
   solve_ith_eqn. Together they give the value of xi given the ith
   row of the (upper triangular) matrix and xj (j>i). They store
   this value into sum. */

void solve_ith_eqn_init (float *sum, float row[])
/* Sets up sum by storing bi (ie. the value of b in the current row) in it. */
{
  *sum = row[matrixsize];
}

void solve_ith_eqn_g (int j, float *sum, float row[], float new)
/* g does the intermediate calculations. It adds to the sum
   an expression involving new (which is one of the xi values).

   new is part of the set of values received from the proc before pid in
   the pipline. j is the counter of which value of these
   we're on - starting at 1 - this counts from the right
   as the 1st value received is x_n, then x_(n-1), etc. */
{
  *sum = *sum + (-new*(row[matrixsize-j]));
}

void solve_ith_eqn_h (int rowno, float *sum, float row[])
/* h finishes the calculation off by doing the non-incremental part of
   the calculation. rowno counts from 0. */
{
  *sum = *sum/row[rowno];
}


/* ---------------------------- Pipelining ------------------------------- */

void pipeline (void (*init) (float *, float *),
                   void (*g) (int, float *, float *, float),
                   void (*h) (int, float *, float *),
                   float local_vals[], float res[])

/* Creates a pipeline with the processors - each processor passes on
   already received values; calculates a new value from them
   (incrementally) using init, g and h; and then passes on this new value.
   This particular version is tailored (and optimised) for a Cyclic
   distribution.
   The local values are distributed cyclically so each processor
   has several local values, stored in an array, with each row
   stored straight after the last.
   The result is stored in res (cyclicly through the 
   set of processors) - this means that each processor has several
   of its values and so res is an array. */
{
  float tmp, sum;
  int startelt, endelt, no_seen, no_to_receive;
  float accum[matrixsize];
  MPI_Status status;
  int n = matrixsize;  /* for an abbreviation */
  int x, i, j;

    /* set up start of pipeline - no g part */
  if (pid==(n-1)%p)
    {
      (*init)(&sum, &local_vals[((n-1)/p)*(n+1)]);
      (*h)(n-1, &sum, &local_vals[((n-1)/p)*(n+1)]);
      res[(n-1)/p] = sum;
      MPI_Send(&sum, 1, MPI_FLOAT, mod(pid-1,p), 0, MPI_COMM_WORLD);
      accum[0] = sum;
      no_seen = 1;
    }
  else
    no_seen = 0;

    /*calculations of numbers of elts */
  if (n<p)
    {
    if (pid < n-1)
      startelt = 0;
    else
      startelt = -1;
    }
  else if (n%p == 0)
    {
      if (pid < (n-1)%p)
	startelt = n/p - 1;
      else
	startelt = n/p - 2;
    }
  else if (pid > n%p - 2)
    startelt = n/p - 1;
  else
    startelt = n/p;

  if (pid>0)
    endelt = 0;
  else
    endelt = 1;

    /* main part of pipeline */
  for (x=startelt; x>=endelt; x--)
    {
      (*init)(&sum, &local_vals[x*(n+1)]);  /* initialise sum */

        /* do calc g for already seen values */
      for (i=0; i<no_seen; i++)
        (*g)(i+1, &sum, &local_vals[x*(n+1)], accum[i]);

        /* receive and send values one at a time and do rest of g */
      if (no_seen==0)
        no_to_receive = (n-pid-1)%p;
      else
        no_to_receive = p-1;

      for (j=0; j<no_to_receive; j++)
        {
          MPI_Recv(&tmp, 1, MPI_FLOAT, (pid+1)%p, 0, MPI_COMM_WORLD, &status);

            /* After the 1st time round, don't send on the 1st val */
          if ((j==0) && ((no_seen>0)||((no_seen==0)&&(pid==n%p))))
	    ;
	  else
	    MPI_Send (&tmp, 1, MPI_FLOAT, mod(pid-1,p), 0, MPI_COMM_WORLD);

          no_seen++;
          accum[no_seen-1] = tmp;
          (*g)(no_seen, &sum, &local_vals[x*(n+1)], tmp);
        }

        /* final calculation */
      (*h)(pid+p*x, &sum, &local_vals[x*(n+1)]); 
      res[x] = sum;
      MPI_Send(&sum, 1, MPI_FLOAT, mod(pid-1,p), 0, MPI_COMM_WORLD);

      no_seen++;
      accum[no_seen-1] = sum;
    }

    /* end of pipeline */
  if (pid==0)
    {   /* initialise and do g for already seen values */
      (*init)(&sum, &local_vals[0]);

      for (i=0; i<no_seen; i++)
	(*g)(i+1, &sum, &local_vals[0], accum[i]);

        /* receive values one at a time, doing rest of g */
      if (no_seen==0)
        no_to_receive = (n-pid-1)%p;
      else 
        no_to_receive = p-1;

      for (j=0; j<no_to_receive; j++)
        {
          MPI_Recv(&tmp, 1, MPI_FLOAT, 1, 0, MPI_COMM_WORLD, &status);
          no_seen++;
          (*g)(no_seen, &sum, &local_vals[0], tmp);
        }

        /* final calculation */
      (*h)(0, &sum, &local_vals[0]);
      res[0] = sum;
    }
}

/* ------------------------ Auxiliary functions -------------------------- */

void makecyclic(float inarray[], float outarray[])
/* outarray = inarray cyclically distributed and then flattened.
   NB. If outarray was 2 dimensional, then everything in outarray[i]
   would correspond to values which will go to proc i.
   Assumes the arrays to have length matrixsize. */
{
  int maxblocksize, counter,i,j,index;

  if (matrixsize%p == 0)
    maxblocksize = matrixsize/p;
  else
    maxblocksize = matrixsize/p+1;

  for (counter=0;counter<matrixsize;counter++)
    {
      i = counter%p;
      j = counter/p;
      if (i<=(matrixsize-1)%p)
	index = maxblocksize*i+j;
      else
	index = maxblocksize*((matrixsize-1)%p+1) + 
	  (matrixsize/p)*(i - ((matrixsize-1)%p+1)) + j;
      outarray[index] = inarray[counter];
    }
}

void makecyclicmatrix(float inarray[], float outarray[])
/* same as makecyclic above, but operates on 2D matrices and moves the
   rows around, instead of on 1D arrays and moving the elements
   around. However the arrays are passed as one dimensional arrays. 
   This is because C was bringing up strange errors when I tried to
   do it with proper 2D arrays.

   Assumes there to be matrixsize rows each with (matrixsize+1) elements
   in it. */
{
  int maxblocksize, counter,i,j,k,index;
  float x;

  if (matrixsize%p == 0)
    maxblocksize = matrixsize/p;
  else
    maxblocksize = matrixsize/p+1;

  /* for each row, move its elements into outarray */
  for (counter=0;counter<matrixsize;counter++)
    {
      /* calculate the location of the row*/
      i = counter%p;
      j = counter/p;
      if (i<=(matrixsize-1)%p)
	index = maxblocksize*i+j;
      else
	index = maxblocksize*((matrixsize-1)%p+1) + 
	  (matrixsize/p)*(i - ((matrixsize-1)%p+1)) + j;

      /* move the elements in row counter */
      for(k=0;k<=matrixsize;k++)
	outarray[index*(matrixsize+1)+k] = inarray[counter*(matrixsize+1)+k];
    }
}

int mod(int x, int y)
/* Returns x`mod`y - this version deals correcly with negative x */
{
  while (x<0)
    x=x+y;
  
  x = x%y;
  return x;
}
