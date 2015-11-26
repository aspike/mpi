EXECS=gauss
MPICC?=mpic++

all: ${EXECS}

gauss: gauss.cpp
	${MPICC} -o gauss gauss.cpp

clean:
	rm ${EXECS}
