all:
	clang++ -std=c++23 -o main main.cxx

clean:
	rm -rf main main.dSYM

format:
	find . -regex '.*\.\(h\|hpp\|c\|cpp\|cxx\)' -exec clang-format -i {} +