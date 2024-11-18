all:
	clang++ -std=c++23 -O2 -o main main.cxx

clean:
	rm -rf main main.dSYM main output && mkdir output
