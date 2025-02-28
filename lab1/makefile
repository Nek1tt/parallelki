CC = g++
SRC = src/lab1.cpp

all: float double

float:
	$(CC) -DUSE_FLOAT $(SRC) -o sin_float

double:
	$(CC) -DUSE_DOUBLE $(SRC) -o sin_double

clean:
	rm -f sin_float sin_double