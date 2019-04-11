# Tau

## Introduction

Tau est un ensemble d’outils pour le profilage et le traçage visant l’analyse
des performances de programmes parallèles écrits en Fortran, C, C++, Java et
Python.

Grâce à Tau, vous pouvez porter un regard critique sur les performances de votre
application, identifier les goulots d’étranglement et les bogues qui la
ralentissent, et améliorer votre code. Tau traque l’exécution de votre programme
parallèle et fourni des informations telles que le temps passé dans chaque
fonction (ou même dans une boucle particulière) ou encore le nombre d’appels à
une fonction. Cette information est disponible non seulement pour les entités
(fonctions, classes, gabarits, etc.) définies dans votre programme, mais aussi
pour les fonctions de certaines bibliothèques externes (MPI, par exemple).

## Utiliser Tau sur Colosse

Tau est disponible sur Colosse pour les compilateurs GCC et Intel, et fonctionne
avec OpenMPI. Pour ajouter Tau 2.21.1 à votre environnement, utilisez un des
deux exemples suivants :

```
module load compilers/gcc/4.4.2 mpi/openmpi/1.4.3_gcc
module load tools/tau/2.21.1_gcc

module load compilers/intel/11.1.059 mpi/openmpi/1.4.3_intel
module load tools/tau/2.21.1_intel
```

Tau est compilé avec support pour le profilage, le traçage (bibliothèque Tau
interne), ainsi que le support pour les logiciels externes suivants :

- Programing Database Toolkit (PDT)
- Binary File Descriptor (BFD)
- OpenMPI
- OpenMP et OPARI2
- Performance API (PAPI)
- Dyninst (seulement pour le compilateur GCC)
- Dwarf
- VTF (format de traçage)
- OTF (format de traçage)

Ces logiciels sont automatiquement ajoutés à votre environnement. Aucune
commande supplémentaire n’est donc requise pour les utiliser. De même, de
multiples installations parallèles de Tau permettent d’utiliser les fonctions de
traçage, de profilage, etc. à partir d’un seul module. Tau choisit
automatiquement la version appropriée pour la tâche demandée. Si vous souhaitez
davantage de contrôle sur le comportement de Tau ou le choix d’une version
particulière de la bibliothèque, vous pouvez utiliser des variables
d’environnement et les options appropriées des programmes, tel que décrit dans
le manuel de Tau.

Si vous avez besoin d’une fonction particulière de Tau qui n’est pas disponible
dans l’installation actuelle sur Colosse, n’hésitez pas à nous contacter. Il
nous fera plaisir de l’ajouter.

## Documentation

La [documentation de Tau](http://www.cs.uoregon.edu/Research/tau/docs.php) est
disponible sur le [site Web](http://www.cs.uoregon.edu/Research/tau/) du projet.
Nous vous recommandons la lecture du guide de l’utilisateur, particulièrement la
première section, qui décrit le processus d’instrumentation, ainsi que la
section « *Some Common Application Scenarios* ».

## Tutoriel

Le reste de cette page est consacré à un court tutoriel sur Tau. Nous
utiliserons un programme parallèle trivial pour illustrer ses fonctions de
base : le calcul de π par une approche de type Monte-Carlo. Tous les exemples de
code et commandes ont été testés sur Colosse et visent à vous rendre familier
avec l’utilisation de Tau sur ce super-ordinateur.

### Introduction

Le nombre π est la constante reliant le diamètre du cercle à sa circonférence.
Le nombre π est transcendant, c’est-à-dire qu’il ne peut être calculé par des
opérations algébriques simples (sommes, produits, racines, etc.). Plusieurs
approches permettent toutefois d’approximer π avec plus ou moins de précision.
Nous utiliserons l'une de ces approches pour écrire un programme calculant π sur
plusieurs processus parallèles.

![π à partir d’un carré et d’un cercle inscrit](/square_circle.png)

Notre approche (dite de type Monte-Carlo) utilise le rapport entre l’aire d’un
carré et celle d’un cercle inscrit (voir figure). Si l’on place au hasard un
grand nombre de points à l’intérieur de ce carré, le rapport entre le nombre de
points à l’intérieur du cercle et le nombre total de points sera égal au rapport
entre l’aire du cercle et celle du carré : p/n = πr²/(2r)², où p est le
nombre de points dans le cercle et n le nombre total de points. On peut isoler π
= 4p/n.

Notre programme placera donc au hasard un grand nombre de points dans le carré
(en tirant des nombres aléatoires entre 0.0 et 1.0 pour les composantes en X et
en Y de chaque point). Le nombre de points à tester sera spécifié par un
argument donné en ligne de commande, par exemple (pour 8 000 000 points et 8
processus MPI) :

```
mpirun -np 8 ./pi-mc 8000000
```

Une contrainte supplémentaire de notre programme est qu’il devra imprimer sur la
sortie standard un estimé de π à intervalles réguliers. Débutant avec une
implémentation naïve, nous améliorerons notre programme grâce aux informations
fournies par Tau, jusqu’à l’obtention de performances optimales.

### Première version, une implémentation naïve

La première version de notre programme est particulièrement simple. Chaque
processus teste un point et transmet le résultat au processus gérant le travail.
Ce dernier imprime alors un estimé de π. Ces instructions sont répétées jusqu’à
ce que le nombre désiré de points ont été testés :

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

On peut compiler puis tester ce programme avec :

```
module load compilers/gcc/4.4.2 mpi/openmpi/1.4.3_gcc
mpicc -lm -o pi-mc-1 pi-mc-1.c
pirun -np 8 ./pi-mc-1 8000000 > pi-mc-1.out
tail pi-mc-1.out
```

### Instrumentation dynamique

À ce point, il est déjà possible d’utiliser Tau avec notre programme, grâce à
l’instrumentation dynamique, une fonction de Tau permettant de profiler un
programme sans avoir à le modifier ou à le recompiler. L’information ainsi
obtenue est moins détaillée qu’avec les autres techniques, mais elle permet
néanmoins de juger certains aspects des performances d’une application
parallèle. L’instrumentation dynamique est non seulement simple, mais aussi la
seule alternative lorsque l’accès au code source n’est pas possible, comme c’est
fréquemment le cas pour les logiciels commerciaux. L’instrumentation dynamique
permet d’étudier la consommation de mémoire, les entrées-sorties et
l’utilisation du GPU, entre autres.

L’instrumentation se fait pendant l’exécution du programme par l’utilitaire
`tau_exec`. Exécutons à nouveau notre programme, cette fois par le biais de
`tau_exec`, afin d’étudier les entrées-sorties réalisées par notre programme :

```
module load tools/tau/2.21.1_gcc
tau_exec -io -- mpirun -np 8 ./pi-mc-1 8000000 > mpi-mc-1.out
```

Après l’exécution, l’information de profilage est stockée dans un fichier nommé
`profile.0.0.0`. Ce fichier peut être analysé par l’utilitaire
`pprof`, dont la sortie débute par un résumé des fonctions exécutées dans
notre programme :

```
pprof > pprof.out
```

Le fichier `pprof.out` contient les résultats colligés :

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

Même sans information sur les appels MPI ou le déroulement interne du programme,
nous remarquons déjà que notre programme a un problème de performance. La
fonction `write()`, utilisée pour imprimer l’estimé de π, représente 14 %
du temps d’exécution. Or, les entrées-sorties devraient idéalement être une
portion négligeable du temps d’exécution, que l’on souhaite entièrement dédié à
calculer π. Une solution pourrait être d’imprimer l’estimé moins fréquemment.

### Instrumentation à la compilation

Avant de réviser notre code, étudions notre première version avec la technique
d’instrumentation à la compilation. Celle-ci nous permettra d’obtenir des
informations beaucoup plus détaillées, non seulement sur les appels à
l’intérieur de notre code, mais aussi sur le temps passé dans les fonctions MPI.
On instrumente à la compilation avec les utilitaires `taucc`,
`taucxx` et `tauf90`. On profile ensuite le code en l’exécutant.
Dans notre cas :

```
taucc -lm -o pi-mc-1 pi-mc-1.c
mpirun -np 8 ./pi-mc-1 8000000 > pi-mc-1.out
pprof > pprof.out
```

On obtient cette fois huit fichiers `profile.*.0.0`, soit un par
processus MPI. L’utilitaire `pprof` collige cette information et nous la
présente, avec une moyenne sur tous les nœuds à la fin :

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

On remarque que les communications MPI (appels à `MPI_Reduce()`)
représentent l’essentiel du temps d’exécution du programme, ce qui n’est
évidemment pas l’objectif recherché. On pourrait diminuer le nombre de
communications nécessaires en calculant un grand nombre de points (plutôt qu’un
seul) sur chaque processus avant de renvoyer cette information au nœud de
gestion qui imprime l’estimé. Cette solution réduirait également la quantité
d’entrées-sorties, réglant du même coup le problème identifié précédemment.

### Deuxième version, diminution du nombre d’appels MPI

La deuxième version de notre programme utilise chaque processus pour calculer 10
000 points dans le carré, avant de transmettre le résultat au processus gérant
le travail. Ces instructions sont répétées jusqu’à ce que le nombre désiré de
points ait été testé :

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

Instrumentons et profilons ce code :

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

On remarque immédiatement que l’exécution du programme est beaucoup plus rapide.
Toutefois, l’analyse avec Tau montre que le programme passe maintenant
l’essentiel de son temps dans la fonction `MPI_Init()`. Comment cela
est-il possible sachant que notre environnement MPI n’a pas changé ? La réponse
est que le programme est désormais suffisamment rapide pour que le temps
d’initialisation ne soit plus négligeable par rapport au temps passé à calculer
π. Afin d’obtenir une mesure précise des performances, il faut augmenter le
nombre de points à calculer pour que la proportion du temps passé à initialiser
MPI soit négligeable :

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

Le temps passé à initialiser MPI est maintenant assez limité pour nous permettre
d’évaluer la situation. On remarque que la proportion du temps passé dans les
communications MPI a considérablement diminué par rapport à la première version
du programme. Toutefois, ce temps est encore beaucoup trop important,
considérant que notre problème est trivial. On pourrait être tenté d’augmenter
la valeur de `np` pour calculer davantage de points à la fois. Toutefois,
augmenter `np` à 100000 n’a qu’un effet minime.

### Troisième version, utilisation de fonctions MPI non bloquantes

Le problème que l’on observe ici est dû à l’utilisation de communications MPI
bloquantes. Lorsque la fonction `MPI_Reduce()` est appelée, elle doit
attendre que tous les processus envoient leur résultats avant de poursuivre. À
leur tour, les processus attendent que leurs données soient reçues avant de
poursuivre. Cela cause des délais lorsque les processus ne terminent pas
exactement en même temps, ce qui est inévitable avec les systèmes d’exploitation
multi-tâches utilisant un ordonnanceur.

La solution est d’utiliser des communications MPI non bloquantes. Dans la
troisième version de notre programme, chaque processus calcule 10000 points à la
fois puis envoie les résultats au processus de gestion. Il poursuit ensuite le
travail sans attendre, et vérifie périodiquement si le processus de gestion a
terminé la réception des résultats. Le processus de gestion, pour sa part,
collige les résultats des processus au fur et à mesure de leur réception :

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

Si l’on instrumente et profile cette nouvelle version comme pour les
précédentes, on obtient :

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

L’échange de résultats par MPI occupe désormais une portion négligeable du temps
d’exécution, ce à quoi on s’attend pour un programme trivialement parallèle
convenablement optimisé. Le temps de latence auparavant perdu à synchroniser les
processus est maintenant dédié au calcul de π, grâce à l’utilisation de
communications non bloquantes.

### Considérations algorithmiques

Nous avons vu que Tau permet d’identifier les problèmes de performance et
facilite leur correction. Toutefois, il importe de rappeler que le choix d’une
approche et d’algorithmes appropriés pour chaque problème est autrement plus
important que l’optimisation d’un programme donné. Par exemple, même si notre
programme calculant π est maintenant optimisé, notre algorithme Monte-Carlo est
plus lent par plusieurs ordres de grandeur qu’une approche par développement de
séries mathématiques…
