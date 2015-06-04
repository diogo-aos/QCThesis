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
** main.cpp // requires CUDA API, Boost library to compile
**
**
*/




/* Requires boost library from boost.org*/

#include <fstream>
#include <string>
#include <boost/foreach.hpp>
#include <iostream>
#include <string>
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
#include <time.h>


#include "computeDistance.h"


using std::vector;
using namespace std;

/*
 * This function has been implemented using boost tokenizer.
 * Purpose: Load a Microarray data in CIBM (http://cibm.newcastle.edu.au/ format and
 *
 * Input: microarray_cibm_format.txt
 *
 * <MicroarrayData>
 * 10	6
 * F_1	1	4.5	2	 7.3  8.21	5.71
 * F_2	1	3	8	 6.66 7.19	1.06
 * F_3	3	3	9	 5	  6.67	7.73
 * F_4	4	4	1	 4	  3.64	2.92
 * F_5	8	7	1.5	 4	  4.91	6.15
 * F_6	8	6	3	 2.4  7.36	4.76
 * F_7	3	8	3	 3.2  5.31	5.49
 * F_8	2	1	4.02 5.8  3.08	7.15
 * F_9	4	1	3.7	 2.2  1.31	1.48
 * F_10	-3	0	3	 1	  3.82	2.7
 * <SamplesNames>
 * 	C1	C2	C3	C4	C5	C6
 * <SamplesClasses>
 * 	1	1	1	0	0	0
 * <EndOfFile>
 *
 *
 * Output: none, store the matrix in a single dimensional global array input[]
 * the value of k will be set as k= int(log(n))+1
 *
 */

void readMicroarray(std::ifstream& filename){

		int i,j;

		typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
		getline(filename,line);
		boost::char_separator<char> sep("	,. \n");
		tokenizer tok(line, sep);
		tokenizer::iterator  it=tok.begin();
		token = *it;
		nRow=atoi(token.c_str());

		/* Initialise k=log(n)+1 */

		k = (int)log((double)nRow)+1;

		/* Initialise chunkSize */

		/*if (nRow > 0  && nRow <=25000)
			chunkSize =4096;
		else if (nRow > 25000  && nRow <=100000)
			chunkSize = 16384;
		else if (nRow > 100000  && nRow <=2000000)*/
			//chunkSize = 65536;
			chunkSize = 16384;
		/*else
		cout<< "Data can not be loaded into GPU's in-memory, please use the external memory version"<<endl;
*/

		it++;
		token = *it;
		nCol=atoi(token.c_str());

		cout <<"Features (nRow) = "<<nRow << "  Samples(nCol) = "<<nCol<<endl;

		Ina = (float*)malloc(sizeof(float)*nRow*nCol);

		std::ofstream out (filenameNode.c_str());
		out<<nRow<<endl;


		for (i=0;i<nRow;i++){

			getline(filename,line);
			boost::char_separator<char> sep("	");
			tokenizer tok(line, sep);
			tokenizer::iterator  it=tok.begin();
			token = *it;
			nameGenes.push_back(token);
			out <<nameGenes[i]<<endl;
			it++;

			for (j=0;j<nCol;j++){
				token = *it;
				*(Ina+(nCol*i+j)) =  atof(token.c_str());
				it++;
				}
			}

		cout<<"Data file loaded successfully...!";

		{
			getline(filename,line);
			boost::char_separator<char> sep("	,. \n");
			tokenizer tok(line, sep);
			tokenizer::iterator  it=tok.begin();
			token = *it;

			{
				getline(filename,line);
				boost::char_separator<char> sep("	");
				tokenizer tok(line, sep);
				tokenizer::iterator  it=tok.begin();

				for (i=0;i<nCol;i++){
					token = *it;
					nameExp.push_back(token);
					it++;
				}
			}
		}


		{
			getline(filename,line);
			boost::char_separator<char> sep("	,. \n");
			tokenizer tok(line, sep);
			tokenizer::iterator  it=tok.begin();
			token = *it;
			{

				getline(filename,line);
				boost::char_separator<char> sep("	");
				tokenizer tok(line, sep);
				tokenizer::iterator  it=tok.begin();

				for (i=0;i<nCol;i++){
					token = *it;
					classExp.push_back(token);
					it++;
				}
			}
		}

		out.close();
}


/* This function decides if the input is an Microarray of Distance Matrix.
 * We will only concider the Mircoarrays for this program
 *
 * Input: filename
 * Output : none, readMicroarray(filename) will store the store the matrix
 * in a single dimensional global array input[]
 *
 */


void readInput(std::ifstream& filename, string filenameIn) {

		typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
		getline(filename,line);
		boost::char_separator<char> sep("	,. \n");
		tokenizer tok(line, sep);
		tokenizer::iterator  it=tok.begin();
		token = *it;

		if (token.compare("<MicroarrayData>")==0){
			readMicroarray(filename);
			type = 1;
			cout<<"\nType = Microarray\n";
		}

		else if (token.compare("<DistanceData>") == 0){
		//	readDistance(filename);
			type = 2;
			cout<<"\nType = DistanceData\n";
		}

		else
			cout<<"Data not defined";
	}






/* Main function */


int main(int argc, char *argv[])
{


	filenameIn = argv[1];
	cout<<"File Name (input)= "<<filenameIn<<std::endl;


	std::ifstream inputStream (filenameIn.c_str());

	filenameOutS = filenameIn + ".knn";
	filenameNode = filenameIn + ".node";

	filenameOut = new char[filenameOutS.size()+1];
	filenameOut[filenameOutS.size()]=0;
	memcpy(filenameOut,filenameOutS.c_str(),filenameOutS.size());

	time(&now);
	current = localtime(&now);

	printf("\nCalculation starts at time : %i:%i:%i\n", current->tm_hour, current->tm_min, current->tm_sec);


	readInput(inputStream, filenameIn);
	knn(filenameOut, Ina, nRow, nCol, chunkSize, k);


	time(&now);
	current = localtime(&now);
	printf("\nOutput stored at time : %i:%i:%i\n", current->tm_hour, current->tm_min, current->tm_sec);

	return 0;
}

