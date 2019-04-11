# Tau

## Introduction

Tau is a collection of tools for profiling and tracing parallel programs written
in Fortran, C, C++, Java and Python.

Using Tau, you can critically assess the performance of your application,
identify bottlenecks and bugs and generally improve your code. Tau traces the
execution of your parallel program and reports information such as the time
spent in each function (or even in a specific loop) or the number of calls to a
given function. This information is available not only for entities (functions,
classes, templates, etc.) defined in your own code, but also for those of some
external libraries (such as MPI).

## Using Tau on Colosse

Tau is available on Colosse for the GCC and Intel compilers, and works with
OpenMPI. To add tau 2.22.1 to your environment, use one of the following pairs
of commands:

```
module load compilers/gcc/4.4.2 mpi/openmpi/1.4.3_gcc
module load tools/tau/2.21.1_gcc

module load compilers/intel/11.1.059 mpi/openmpi/1.4.3_intel
module load tools/tau/2.21.1_intel
```

Tau is compiled with support for profiling, tracing (using the internal Tau
library), and includes support for the following external software:

- Programing Database Toolkit (PDT)
- Binary File Descriptor (BFD)
- OpenMPI
- OpenMP and OPARI2
- Performance API (PAPI)
- Dyninst (only for GCC)
- Dwarf
- VTF (trace format)
- OTF (trace format)

These programs are automatically added to your environment. No additional
command is therefore necessary to use them. In addition, multiple Tau versions
are installed in parallel to provide all profiling and tracing tools from a
single module. Tau automatically selects the correct version for the requested
task. If you need more control over Tau options or want to select a particular
version, you can use environment variables and program flags as described in the
Tau manual.

If you need a specific Tau feature that is not currently available in the
current setup on Colosse, do not hesitate to contact us. We will be pleased to
add it.

## Documentation

[Tau documentation](http://www.cs.uoregon.edu/Research/tau/docs.php) is
available on the project [Web site](http://www.cs.uoregon.edu/Research/tau/). We
recommend reading the user guide, especially the first section which describes
the instrumentation process, and also the section *Some Common Application
Scenarios*.

## Tutorial

The remainder of this page is dedicated to a short Tau tutorial. To illustrate
basic Tau functions, we will use a trivial parallel program that computes π
using a Monte-Carlo approach. All examples were tested on Colosse and aim to
make you familiar with the use of Tau on this supercomputer.

### Introduction

The number π is the constant linking the diameter of the circle to its
circumference. π is a transcendental number, meaning that it cannot be computed
using simple algebraic operations (sums, products, roots, etc.). However,
several approaches can be used to approximate π with more or less precision. We
will use one such approach to write a program that computes π using multiple
parallel processes.

![π from a square and an inscribed circle](/square_circle.png)

Our Monte-Carlo technique uses the ratio between the surface of a square and
that of an inscribed circle (see figure). If a large number of points are
selected at random inside that square, the ratio of the number of points inside
the circle to the total number of points with be equal to the ratio between the
surface of the circle and that of the square: p/n = πr²/(2r)², where p is
the number of points inside the circle and n is the total number of points. We
can then isolate π = 4p/n.

Our program will therefore generate a great number of points inside the square
(by randomly selecting numbers between 0.0 and 1.0 for the X and Y coordinates
of each point). The number of points to test will be specified by a command-line
argument. For instance, to compute 8 000 000 points using 8 MPI processes:

```
mpirun -np 8 ./pi-mc 8000000
```

An additional requirement is that our program must print an estimate of π on
standard output at regular intervals. Starting with a naive implementation, we
will improve our program using the information provided by Tau, until we obtain
optimal performance.

### First version, a naive implementation

The first version of our program is very simple. Each process tests one point
and communicates the result back to the process that manages the workload. This
manager process then prints a π estimate. This is repeated until the desired
number of points have been tested:

```
/* pi-mc-1.c */

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "mpi.h"

/*
 * Generate a random point in the square, then test and return whether it is
 * inside the circle.
 */
bool generate_point()
{
	double x; // X coordinate of point inside the square
	double y; // Y coordinate of point inside the square
	x = (double) rand() / RAND_MAX;
	y = (double) rand() / RAND_MAX;
	return (sqrt(x * x + y * y) <= 1.0);
}

/*
 * Estimate Pi over MPI using a Monte Carlo algorithm, printing an estimate
 * after each iteration
 *
 *     n:    Number of random points to test (the higher, the more precise the
 *           estimate will be)
 *     comm: MPI communicator
 */
void pi_monte_carlo(int n, MPI_Comm comm)
{
	int rank; // Rank inside the MPI communicator
	int size; // Size of the MPI communicator
	int pw;   // 0 or 1, was the point inside the circle for this rank and this
	          // iteration?
	int pc;   // Number of points inside the circle, summed for all ranks for
	          // this cycle of iteration
	int p;     // Total number of points inside the circle (pi = 4*p/n)
	int m;     // Number of points tested up to now
	double pi; // Pi estimate

	/* Initialize variables and check that the number of points is valid.
	 * We simplify the algorithm by forcing n to be a multiple of the number
	 * of processes.
	 */
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);
	p = 0;
	if (n % size != 0)
	{
		if (rank == 0)
		{
			printf("Invalid number of points: %d\n", n);
			printf("Must be a multiple of: %d\n", size);
		}
		return;
	}

	// Compute pi
	for (m = size; m <= n; m += size)
	{
		pw = (int) generate_point();
		MPI_Reduce(&pw, &pc, 1, MPI_INT, MPI_SUM, 0, comm);
		// Rank 0 prints the estimate
		if (rank == 0)
		{
			p += pc;
			pi = (double) 4 * p / m;
			printf("%d %.20f\n", m, pi);
		}
	}
}

int main(int argc, char* argv[])
{
	MPI_Init(&argc, &argv);
	pi_monte_carlo(atoi(argv[1]), MPI_COMM_WORLD);
	MPI_Finalize();
	return 0;
}
```

We can compile and test this program with:

```
module load compilers/gcc/4.4.2 mpi/openmpi/1.4.3_gcc
mpicc -lm -o pi-mc-1 pi-mc-1.c
mpirun -np 8 ./pi-mc-1 8000000 > pi-mc-1.out
tail pi-mc-1.out
```

### Dynamic instrumentation

At this stage, it is already possible to use Tau with our program thanks to
dynamic instrumentation, a Tau feature to profile a program without recompiling
it. The information thus gathered is limited, but still makes it possible to
judge certain aspects of the parallel performance of an application. Dynamic
instrumentation is not only simple, it is the only alternative when access to
the source code is not possible, such as is often the case for commercial
software. Dynamic instrumentation allows one to study memory usage, input-output
and GPU usage, amongst others.

Instrumentation and profiling happen at runtime through `tau_exec`. We
can execute our program again using `tau_exec`, and study input-output.

```
module load tools/tau/2.21.1_gcc
tau_exec -io -- mpirun -np 8 ./pi-mc-1 8000000 > mpi-mc-1.out
```

After execution, profiling information is stored in a file named
`profile.0.0.0`. It can be analysed using `pprof`, which produces
a summary of function usage in our program:

```
pprof > pprof.out
```

File `pprof.out` contains the compiled results:

```
Reading Profile files in profile.*

NODE 0;CONTEXT 0;THREAD 0:
---------------------------------------------------------------------------------------
%Time    Exclusive    Inclusive       #Call      #Subrs  Inclusive Name
              msec   total msec                          usec/call
---------------------------------------------------------------------------------------
100.0       35,200       43,990           1      748909   43990242 .TAU application
 14.0        6,145        6,145      484557           0         13 write()
  6.0        2,638        2,638      263887           0         10 read() [THROTTLED]
  0.0        0.916        0.916          13           0         70 fopen()
  0.0        0.667        0.667          54           0         12 writev()
  0.0        0.596        0.596         116           0          5 readv()
  0.0        0.412        0.463         113           6          4 close()
  0.0         0.46         0.46          20           0         23 send()
  0.0        0.397        0.397          15           0         26 fclose()
  0.0        0.382        0.382          34           0         11 pipe()
  0.0        0.286        0.286          35           0          8 open()
  0.0        0.161        0.161          16           0         10 recv()
  0.0        0.102        0.102          16           0          6 accept()
  0.0        0.094        0.094           4           0         24 recvfrom()
  0.0        0.092        0.092           8           0         12 socket()
  0.0        0.086        0.086           2           0         43 read()
  0.0        0.056        0.056           5           0         11 fopen64()
  0.0         0.04         0.04           5           0          8 fscanf()
  0.0        0.025        0.025           5           0          5 connect()
  0.0        0.022        0.022           2           0         11 fprintf()
  0.0        0.019        0.019           1           0         19 socketpair()
  0.0        0.008        0.008           3           0          3 lseek()
  0.0        0.008        0.008           2           0          4 rewind()
  0.0        0.006        0.006           2           0          3 bind()
---------------------------------------------------------------------------------------

[...]
```

Even with no information about MPI calls or the internal workings of the
program, we can already identify a performance problem. 14 % of execution time
is spent in the `write()` function to print π estimates. Ideally,
input-output functions should represent a negligible portion of the runtime,
which should be entirely dedicated to actually computing π. One solution is to
print estimates less often.

### Compile time instrumentation

Before reviewing our code, we can study the first version of our program using
compile time instrumentation. This will gather much more detailed information,
not only about the function calls inside our program, but also about the time
spent in MPI functions. Compile time instrumentation requires the use of
`taucc`, `taucxx` or `tauf90`. Once the program has been
compiled with one of these, it can be profiled through execution. In our case:

```
taucc -lm -o pi-mc-1 pi-mc-1.c
mpirun -np 8 ./pi-mc-1 8000000 > pi-mc-1.out
pprof > pprof.out
```

This time, eight `profile.*.0.0` files are created (one for each MPI
process). Using `pprof`, we compile this information and get the
following measurements, which are averaged over all MPI processes:

```
Reading Profile files in profile.*

[...]

FUNCTION SUMMARY (mean):
---------------------------------------------------------------------------------------
%Time    Exclusive    Inclusive       #Call      #Subrs  Inclusive Name
              msec   total msec                          usec/call
---------------------------------------------------------------------------------------
100.0           16       31,472           1           1   31472907 .TAU application
 99.9       0.0725       31,456           1           3   31456223 main
 96.1        4,140       30,241           1      987503   30241926 pi_monte_carlo
 82.7       26,027       26,027      875000           0         30 MPI_Reduce()
  3.8        1,200        1,200           1           0    1200720 MPI_Init()
  0.2           54           54     12500.1           0          4 MPI_Reduce() [THROTTLED]
  0.1           19           19      100001           0          0 generate_point [THROTTLED]
  0.0           13           13           1           0      13504 MPI_Finalize()
  0.0      0.00025      0.00025           1           0          0 MPI_Comm_rank()
  0.0      0.00025      0.00025           1           0          0 MPI_Comm_size()
```

We note that MPI communications (calls to `MPI_Reduce()`) represent the
bulk of execution time, which is of course not what we want. We could reduce the
amount of communication by computing a large number of points (rather than
proceeding one at a time) in each process before sending the results back to the
managing process that prints the estimate. This solution will also significantly
reduce the amount of input-output, solving at the same time the previously
identified problem.

### Second version, fewer MPI calls

The second version of our program uses each process to compute 10 000 points
inside the square before transmitting the results to the manager process. This
is repeated until the desired number of points have been tested:

```
/* pi-mc-2.c */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "mpi.h"

// Number of points to test at a time
#define np 10000

/*
 * Generate points inside the square and test if they are also inside the
 * circle. Then, return the number of points inside the circle.
 *
 *     n: Number of points to test
 */
int generate_points(int n)
{
	int i;
	int p = 0;
	double x;
	double y;
	for (i = 0; i < n; ++i)
	{
		x = (double) rand() / RAND_MAX;
		y = (double) rand() / RAND_MAX;
		if (sqrt(x * x + y * y) <= 1.0)
		{
			p++;
		}
	}
	return p;
}

/*
 * Estimate Pi over MPI using a Monte Carlo algorithm, printing an estimate
 * at regular intervals
 *
 *     n:    Number of random points to test (the higher, the more precise the
 *           estimate will be)
 *     comm: MPI communicator
 */
void pi_monte_carlo(long int n, MPI_Comm comm)
{
	int rank;   // Rank inside the MPI communicator
	int size;   // Size of the MPI communicator
	int pw;     // Number of points inside the circle for this rank and this
	            // estimate cycle
	int p;      // Total number of points inside the circle (pi = 4*p/n), summed
	            // for all ranks and estimate cycles
	int pc;     // Number of points inside the circle, summed for all ranks for
	            // this cycle of iteration
	long int m; // Number of points tested up to now
	double pi;  // Pi estimate

	/* Initialize variables and check that the number of points is valid.
	 * We simplify the algorithm by forcing n to be a multiple of the number
	 * of points per estimate cycle.
	 */
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);
	p = 0;
	if (n % (size * np) != 0)
	{
		if (rank == 0)
		{
			printf("Invalid number of points: %d\n", n);
			printf("Must be a multiple of: %d\n", size * np);
		}
		return;
	}

	// Compute pi
	for (m = size * np; m <= n; m += size * np)
	{
		pw = generate_points(np);
		MPI_Reduce(&pw, &pc, 1, MPI_INT, MPI_SUM, 0, comm);
		// Rank 0 prints the estimate
		if (rank == 0)
		{
			p += pc;
			pi = (double) 4 * p / m;
			printf("%d %.20f\n", m, pi);
		}
	}
}

int main(int argc, char* argv[])
{
	MPI_Init(&argc, &argv);
	pi_monte_carlo(atol(argv[1]), MPI_COMM_WORLD);
	MPI_Finalize();
	return 0;
}
```

We can instrument and profile this code:

```
taucc -lm -o pi-mc-2 pi-mc-2.c
mpirun -np 8 ./pi-mc-2 8000000 > pi-mc-2.out
pprof > pprof.out
```

```
Reading Profile files in profile.*

[...]

FUNCTION SUMMARY (mean):
---------------------------------------------------------------------------------------
%Time    Exclusive    Inclusive       #Call      #Subrs  Inclusive Name
              msec   total msec                          usec/call
---------------------------------------------------------------------------------------
100.0           19        1,367           1           1    1367477 .TAU application
 98.6       0.0709        1,348           1           3    1348449 main
 89.0        1,216        1,216           1           0    1216556 MPI_Init()
  9.3            7          127           1        2002     127602 pi_monte_carlo
  5.4           73           73        1000           0         74 MPI_Reduce()
  3.3           45           45        1000           0         46 generate_points
  0.3            4            4           1           0       4220 MPI_Finalize()
  0.0     0.000375     0.000375           1           0          0 MPI_Comm_rank()
  0.0     0.000375     0.000375           1           0          0 MPI_Comm_size()
```

We immediately notice that the program completes much faster. However, analysis
with Tau shows that the program now spends most of its time inside the
`MPI_Init()` function. How is this possible, knowing that our MPI
environment is no different than before? The answer is that our program is now
sufficiently fast for initialisation time not to be negligible compared to the
time spent computing π. To obtain reliable performance measurements, we need to
increase the number of points to compute such that the time spent initialising
the MPI environment becomes again negligible:

```
taucc -lm -o pi-mc-2 pi-mc-2.c
mpirun -np 8 ./pi-mc-2 8000000000 > pi-mc-2.out
pprof > pprof.out
```

```
Reading Profile files in profile.*

[...]

FUNCTION SUMMARY (mean):
---------------------------------------------------------------------------------------
%Time    Exclusive    Inclusive       #Call      #Subrs  Inclusive Name
              msec   total msec                          usec/call
---------------------------------------------------------------------------------------
100.0           18       51,823           1           1   51823376 .TAU application
100.0        0.068       51,804           1           3   51804915 main
 97.6          501       50,575           1      200002   50575714 pi_monte_carlo
 88.4       45,787       45,787      100000           0        458 generate_points
  8.3        4,287        4,287      100000           0         43 MPI_Reduce()
  2.4        1,225        1,225           1           0    1225538 MPI_Init()
  0.0            3            3           1           0       3595 MPI_Finalize()
  0.0     0.000375     0.000375           1           0          0 MPI_Comm_rank()
  0.0            0            0           1           0          0 MPI_Comm_size()
```

The time spent initialising MPI is now small enough to allow us to critically
assess our program. We see that the amount of time spent in MPI communications
has considerably decreased compared to what it was in the first version.
However, this time is still too important considering that our problem is
trivial. We might be tempted to increase `np` to compute more points at a
time. However, increasing `np` to 100000 has only minimal effects.

### Third version, using non-blocking MPI calls

The problem we observe here is due to the use of blocking MPI calls. When
function `MPI_Reduce()` is called, it has to wait for all processes to
send their results before continuing. Similarly, all processes need to wait for
all data to be transmitted before they can resume their work. This causes delays
when the processes are not perfectly synchronised, which is invariably the case
on multi-tasking operating systems that use process schedulers.

The solution is to use non-blocking MPI communications. In the third version of
our program, each process computes 10000 points at a time, sends the results to
the manager, and immediately starts computing a new batch without waiting.
Periodically, the process checks if the manager is done receiving the data. The
manager compiles the results as they come in:

```
/* pi-mc-3.c */

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "mpi.h"

// Rank in charge of coordinating all processes
#define manager_rank 0

// Tags used in MPI calls to segregate messages relating to results from the
// stop signal sent to workers when enough points have been accumulated.
#define tag_results 0
#define tag_stop 1

// Number of points to generate per iteration
#define np 10000

// Encapsulate an MPI message, including the data buffer and objects required
// for non-blocking messages.
typedef struct
{
	int data[2];
	MPI_Request request;
	MPI_Status status;
}
Message;

// Test if a previously sent message has completed
bool test(Message* message)
{
	int flag;
	MPI_Test(&message->request, &flag, &message->status);
	return (bool) flag;
}

// Cancel a previously sent message
void cancel(Message* message)
{
	MPI_Cancel(&message->request);
}

// Non-blocking send of results from a worker to the manager rank
void isend_results(Message* message, int p, int n, MPI_Comm comm)
{
	message->data[0] = p;
	message->data[1] = n;
	MPI_Isend(&message->data, 2, MPI_INT, manager_rank, tag_results, comm,
			  &message->request);
}

// Non-blocking receive of a rank’s results by the manager rank
void irecv_results(Message* message, int rank, MPI_Comm comm)
{
	MPI_Irecv(&message->data, 2, MPI_INT, rank, tag_results, comm,
			  &message->request);
}

// Blocking send of the stop signal to all ranks
void send_stop(Message* message, MPI_Comm comm)
{
	int size;
	int i;
	MPI_Comm_size(comm, &size);
	message->data[0] = 0;
	for (i = 0; i < size; ++i)
	{
		MPI_Send(&message->data, 1, MPI_INT, i, tag_stop, comm);
	}
}

// Non-blocking receive of the stop signal by a worker
void irecv_stop(Message* message, MPI_Comm comm)
{
	MPI_Irecv(&message->data, 1, MPI_INT, manager_rank, tag_stop, comm,
			  &message->request);
}

int generate_points(int n)
{
	int i;
	int p = 0;
	double x;
	double y;
	for (i = 0; i < n; ++i)
	{
		x = (double) rand() / RAND_MAX;
		y = (double) rand() / RAND_MAX;
		if (sqrt(x * x + y * y) <= 1.0)
		{
			p++;
		}
	}
	return p;
}

/*
 * Estimate Pi over MPI using a Monte Carlo algorithm, printing an estimate
 * at regular intervals
 *
 *     n:     Number of random points to test (the higher, the more precise the
 *            estimate will be)
 *     comm:  MPI communicator
 */
void pi_monte_carlo(long int n, MPI_Comm comm)
{
	// Variables used by all ranks
	bool finished;               // Has this rank finished all his job?
	bool working;                // Is this rank currently computing points to
	                             // test?
	bool managing;               // Is this rank currently managing the results?
	int rank;                    // Rank inside the MPI communicator
	int pw;                      // Number of points inside the circle on this
	                             // rank since the last message sent to the
								 // manager
	int nw;                      // Number of points tested on this rank since
	                             // the last message sent to the manager
	Message results_out_message; // Message for sending results to manager
	Message stop_in_message;     // Message to receive stop signal from manager

	// Variables used only by the manager rank
	int p;                        // Total number of points inside the circle
	long int m;                        // Total number of points tested up to now
	long int mp;                       // The number of points accumulated for the
	                              // last printed estimate
	int size;                     // Size of the MPI communicator
	int i;                        // Loop counter
	Message* results_in_message;  // Array of messages for receiving results
	Message stop_out_message;     // Message to sent stop signal

	// Variable and communication initialization
	finished = false;
	working = true;
	MPI_Comm_rank(comm, &rank);
	managing = (rank == manager_rank);
	// All ranks prepare to receive the stop signal
	irecv_stop(&stop_in_message, comm);
	// All ranks compute a first set of points and send it to the manager
	isend_results(&results_out_message, generate_points(np), np, comm);
	pw = 0;
	nw = 0;
	if (managing)
	{
		p = 0;
		m = 0;
		mp = 0;
		MPI_Comm_size(comm, &size);
		results_in_message = malloc(size * sizeof(Message));
		for (i = 0; i < size; ++i)
		{
			// Manager prepares to receive results from all ranks
			irecv_results(&results_in_message[i], i, comm);
		}
	}

	// Main loop
	while (! finished)
	{
		if (working)
		{
			// If the stop signal has been received, we are done computing
			// points. We cancel pending results message.
			if (test(&stop_in_message))
			{
				working = false;
				cancel(&results_out_message);
			}
			// We generate another round of points. Then, we check if our
			// previous results have been sent to the manager. If that is
			// the case, we begin sending another batch of results.
			else
			{
				pw += generate_points(np);
				nw += np;
				if (test(&results_out_message))
				{
					isend_results(&results_out_message, pw, nw, comm);
					pw = 0;
					nw = 0;
				}
			}
		}
		if (managing)
		{
			// We send the stop signal to all ranks as soon as we have tested
			// enough points. We cancel pending results reception messages.
			if (m >= n)
			{
				send_stop(&stop_out_message, comm);
				for (i = 0; i < size; ++i)
				{
					cancel(&results_in_message[i]);
				}
				managing = false;
			}
			// Collate results from all workers. If we have received
			// new results, we add them to our current count. We then prepare to
			// receive more.
			for (i = 0; i < size; ++i)
			{
				if (test(&results_in_message[i]))
				{
					p += results_in_message[i].data[0];
					m += results_in_message[i].data[1];
					irecv_results(&results_in_message[i], i, comm);
				}
			}
			// We print a new pi estimate every time we have received new
			// results.
			if (m != mp)
			{
				printf("Pi is estimated as %.14f after testing %d points\n",
					   (double) 4 * p / m, m);
				mp = m;
			}
		}
		if (! (working | managing))
		{
			finished = true;
		}
	}
}

int main(int argc, char* argv[])
{
	MPI_Init(&argc, &argv);
	pi_monte_carlo(atol(argv[1]), MPI_COMM_WORLD);
	MPI_Finalize();
	return 0;
}
```

If we instrument and profile this new version like the previous ones, we obtain:

```
Reading Profile files in profile.*

[...]

FUNCTION SUMMARY (mean):
---------------------------------------------------------------------------------------
%Time    Exclusive    Inclusive       #Call      #Subrs  Inclusive Name
              msec   total msec                          usec/call
---------------------------------------------------------------------------------------
100.0           17       47,908           1           1   47908900 .TAU application
100.0       0.0666       47,891           1           3   47891230 main
 97.4          649       46,678           1      274391   46678917 pi_monte_carlo
 95.8       45,887       45,887      100682           0        456 generate_points
  2.5        1,208        1,208           1           0    1208418 MPI_Init()
  0.2           25           77      100001      100001          1 test [THROTTLED]
  0.1           18           57     61202.9     61202.9          1 isend_results
  0.1           52           52      100001           0          1 MPI_Test() [THROTTLED]
  0.1           38           38     61202.9           0          1 MPI_Isend()
  0.0            3            7     12500.1       12500          1 irecv_results [THROTTLED]
  0.0            3            3           1           0       3829 MPI_Finalize()
  0.0            3            3     12500.1           0          0 MPI_Irecv() [THROTTLED]
  0.0        0.011       0.0286           1           1         29 irecv_stop
  0.0       0.0155       0.0155       0.875           0         18 MPI_Irecv()
  0.0        0.003        0.015           2           2          8 cancel
  0.0        0.012        0.012           2           0          6 MPI_Cancel()
  0.0     0.000625      0.00725       0.125       1.125         58 send_stop
  0.0      0.00662      0.00662           1           0          7 MPI_Send()
  0.0            0            0           1           0          0 MPI_Comm_rank()
  0.0            0            0        0.25           0          0 MPI_Comm_size()
```

Data exchange through MPI now occupies a negligible portion of execution time,
which is what we aim for in a trivial parallel problem that was correctly
optimised. The latency that was caused by process synchronisation is gone and
all time is now dedicated to computing π, thanks to non-blocking communication.

### Algorithmic considerations

We have seen how Tau makes it possible to identify and correct performance
problems. However, choosing the right approach and selecting the best algorithm
for a given problem is more important than optimising a particular program. For
instance, even though our π estimator is now properly optimised, the Monte-Carlo
approach it uses is several orders of magnitude slower than what is possible
using a Taylor series development.
