/*
** Copyright (C) 2011 Centre for Bioinformatics, Biomarker Discovery and Information-Based Medicine
** http://www.newcastle.edu.au/research-centre/cibm/
**
** Code for Paper:
** FS-kNN: A Fast and Scalable kNN Computation Technique using GPU,
** Ahmed Shamsul Arefin, Carlos Riveros, Regina Berrettaand Pablo Moscato,
** Email: ASA∗ - ahmed.arefin@newcastle.edu.au; CR - Carlos.Riveros@newcastle.edu.au ; RB - regina.berretta.@newcastle.edu.au ; PM
** pablo.moscato@newcastle.edu.au
**
**
** This program is free software; you can redistribute it and/or modify
** it under the terms of the GNU General Public License as published by
** the Free Software Foundation; either version 2 of the License, or
** (at your option) any later version.
**
** This program is distributed in the hope that it will be useful,
** but WITHOUT ANY WARRANTY; without even the implied warranty of
** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
** GNU General Public License for more details.
**
** You should have received a copy of the GNU General Public License
** along with this program; if not, write to the Free Software
** Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
**
** computedistance.h
*/

#ifndef COMPUTEDISTANCE_H_
#define COMPUTEDISTANCE_H_

	#include <fstream>
	#include <string>
	#include <boost/foreach.hpp>
	#include <iostream>
	#include <string>
	#include <string.h>
	#include <boost/tokenizer.hpp>
	#include <sstream>
	#include <boost/lexical_cast.hpp>
	#include <vector>
	#include <cstdio>
	#include <cstdlib>
	#include <cstring>
	#include <cmath>
	#include <cstring>
	#include <algorithm>
	#include <string>


	#define MaxValue 4294967295
	#define MaxRow 8192
	#define MaxCol 128

	using std::vector;
	using namespace std;
	int nRow; //Total Number rows in the input matrix
	int nCol; //Total Number columns in the input matrix
	std::string line;
	std::string token;
	vector < string > nameExp;
	vector < string > nameGenes;
	vector < string > classExp;
	std::string filenameIn;
	std::string filenameOutS;
	std::string filenameNode;
	char *filenameOut;
	struct tm *current;
	time_t now;
	float *Ina; // Holds the input matrix
	float *Da;  // Holds chunk of the original distance matrix

	int type; // 2: distance matrix, 1: gene expression
	int k;
	int chunkSize;
	int knn(char *name,float *Ina, int nRow, int nCol, int chunkSize, int k);


#endif /* COMPUTEDISTANCE_H_ */
