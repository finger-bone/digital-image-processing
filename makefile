all:
	clang++ -std=c++23 -O2 -o main.out main.cxx

clean:
	rm -rf main.out main.dSYM
