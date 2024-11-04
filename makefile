all:
	clang++ -std=c++23 -o main main.cxx

clean:
	rm -rf main main.dSYM output