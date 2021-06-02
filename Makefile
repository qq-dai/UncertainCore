CC = g++ "-std=c++11" -O3 -fopenmp -g
CONFIG = -lgsl -lgslcblas -lm -lgmp
objects = graph.o uncertain-core.o uncertain-core-topdown.o uncertain-core-main.o

.PHONY : clean

uncertain-core: $(objects)
#	$(CC) -o uncertain-core $(objects)
	$(CC) -L/usr/local/bin -o uncertain-core $(objects) $(CONFIG)
%.o:%.cpp
	$(CC) -c $^

clean:
	rm -f *.o uncertain-core
