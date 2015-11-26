EXECS=shell
MPICC?=mpic++

all: ${EXECS}

shell: shell.cpp
	${MPICC} -o shell shell.cpp

clean:
	rm ${EXECS}