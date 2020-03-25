






output: network.o main.o reader.o
	g++ network.o main.o reader.o -o output

network.o: network.cpp network.hpp 
	g++ -c network.cpp 

reader.o: reader.cpp reader.hpp network.hpp
	g++ -c reader.cpp

main.o: main.cpp network.hpp
	g++ -c main.cpp


clean:
	rm -f *.o 

