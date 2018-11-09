all: clean polyfit

polyfit:  polyfit.c
	gcc -g -Wall -O2 -I/usr/include polyfit.c -lgslcblas -lgsl -lm -o polyfit

clean:
	rm -f polyfit
