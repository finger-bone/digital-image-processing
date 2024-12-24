all:
	clang++ -std=c++20 -O3 -o main main.cxx

clean:
	rm -rf main.dSYM output && mkdir output
