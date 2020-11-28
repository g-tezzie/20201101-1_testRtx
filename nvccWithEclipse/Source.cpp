//============================================================================
// Name        : testCpp.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <fstream>
#include <iostream>
#include <ctime>

#include "Fir.h"

using namespace std;

static const int buflen = 100e6;
char ioRaw[buflen];
char oRaw[buflen];
short int raw[2][buflen / 4];
short int rawo[2][buflen / 4];

Fir lFir = Fir();
Fir rFir = Fir();


int __cdecl  maine();

int main() {
	time_t start = time(nullptr);

	ifstream in = ifstream("in.raw", std::ios::binary);
	// basic_ifstream<char> in = basic_ifstream<char>("in.raw", std::ios::binary);
	ofstream out = ofstream("out.raw", std::ios::binary);
	ofstream outg = ofstream("outg.raw", std::ios::binary);

	in.read(ioRaw, buflen);
	int samples = in.gcount();
	cerr << "Octets read: " << samples << endl;

	for (int i = 0; i < (samples >> 2); i++) {
		char ll, lh, rl, rh;
		ll = ioRaw[i * 4];
		lh = ioRaw[i * 4 + 1];
		rl = ioRaw[i * 4 + 2];
		rh = ioRaw[i * 4 + 3];

		raw[0][i] = static_cast<short int>(lh << 8) + ll;
		raw[1][i] = static_cast<short int>(rh << 8) + rl;
	}


	{
		if (0 != maine()) return 1;

		for (int i = 0; i < (samples >> 2); i++) {

			oRaw[i * 4] = rawo[0][i];
			oRaw[i * 4 + 1] = (rawo[0][i] >> 8);
			oRaw[i * 4 + 2] = (rawo[1][i]);
			oRaw[i * 4 + 3] = (rawo[1][i] >> 8);
		}

		outg.write(oRaw, samples);
		cerr << difftime(time(nullptr), start) << endl;
	}

	{

#pragma loop(hint_parallel(0))
#pragma loop(ivdep) // ivdep will force this through.
		for (int i = 0; i < (samples >> 2); i++) {
			rawo[0][i] = lFir.block(&raw[0][i]);
			rawo[1][i] = rFir.block(&raw[1][i]);
		}
	}

	{
		for (int i = 0; i < (samples >> 2); i++) {

			ioRaw[i * 4] = rawo[0][i];
			ioRaw[i * 4 + 1] = (rawo[0][i] >> 8);
			ioRaw[i * 4 + 2] = (rawo[1][i]);
			ioRaw[i * 4 + 3] = (rawo[1][i] >> 8);
		}

		out.write(ioRaw, samples);
	}

	cerr << difftime(time(nullptr), start) << endl;

	cerr << "!!!Hello World!!!" << endl; // prints !!!Hello World!!!


	return 0;
}
