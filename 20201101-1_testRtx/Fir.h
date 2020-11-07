/*
 * Fir.h
 *
 *  Created on: 2020/11/07
 *      Author: gtezz
 */

#ifndef FIR_H_
#define FIR_H_

#include <iostream>
#include <cmath>

class Fir {
public:
	static const int depth = 12;

private:
	static const int taps = (1 << depth);
	static const int mask = (1 << depth) - 1;
	long long x[1 << depth];
	int ptr = 0;

	void triangular();
	void sinc();
	void clear();

public:
	long long k[1 << depth];
	Fir() {
		clear();
		sinc();
	}
	virtual ~Fir() {
		// TODO Auto-generated destructor stub
	}

	short int block(const short int u[]) {
		long long a = 0;
		for (int i = 0; i < 1 << depth; i++) {
			a += u[i] * k[i];
		}
		return a >> 64 - 16;
	}

	short int put(short int u) {
		int mptr = ptr & mask;
		short int retval = x[mptr] >> 64 - 16;
		x[mptr] = 0;
		for (int i = 0; i < 1 << depth; i++) {
			x[(ptr + i) & mask] += u * k[i];
		}
		ptr++;
		return retval;
	}
};

inline void Fir::triangular() {
	for (int i = 0; i < taps; i++)
		k[i] = 2.0 * (1LL << (64 - 16 - depth)) * (1.0 - (float) ((i)) / taps);
}


inline void Fir::sinc() {

//	function [x]=filt_sinc(n,fl)

//x=sinc(n,fl)
//Calculate n samples of the function sin(2*pi*fl*t)/(pi*t)
//for t=-n/2:n/2 (i.e. centered around the origin).
//  n  :Number of samples
//  fl :Cut-off freq. of assoc. low-pass filter in Hertz
//  x  :Samples of the sinc function

	int n = taps;
	double fl = 200.0/48000.0;
	double pi = atan2(0, -1);

	double no2 = (n - 1) / 2;
	double wl = fl * 2 * pi;

	for (int i = 0; i < taps; i++) {
		double no2Now = -no2 + i;
		double xn = sin(wl * no2Now);
		double xd = pi * no2Now;
		double xx = no2Now == 0 ? 2 * fl : xn / xd;
		k[i] = xx * (1LL << (64 - 16));

//		std::cerr << i << ": " << xx << std::endl;
	}

}

inline void Fir::clear() {
	for (int i = 0; i < taps; i++)
		x[i] = 0;
}

#endif /* FIR_H_ */
