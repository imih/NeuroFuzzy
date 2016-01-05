PROJECT = nenr
SOURCE = main.cc neuro_fuzzy_network.cc

CC = g++
CFLAGS =-Wall -g -c -std=c++11
LDFLAGS = 
OBJECTS = ${SOURCE:.cpp=.o}

all: $(SOURCE) $(PROJECT)

$(PROJECT): $(OBJECTS)
		$(CC) $(OBJECTS) -std=c++11 -o $(PROJECT)

.cpp.o:
		$(CC) $(CFLAGS) $< -o $@

