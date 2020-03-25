/**
 * Header file for the reader class - Contains declarations
 * for a reader that reads in network structure, possible weight values, 
 * and training data 
 * 
 * The reader handles file I/O, reads in training data, and exports weights
 * at the end. 
 * 
 * @author Kailash Ranganathan
 * @version March 21, 2020
 */



#pragma once

#ifndef READER_H
#define READER_H

#include <string>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>

using namespace std; 
/**
 * Helper method to read in the config file values
 * (hyperparameters) (only functional method rn)
 */
void readConfigFile(string config);
/**
 *  Exports the weights to a file given by the filename
 * 
 */
void exportWeights(vector<vector<vector<double> > > weights, string fileName);

/* class Reader
{
   int numIter; 
   int* layerSizes; 
   int numLayers; 
   double** inputs; 
   double truths[];
   ifstream fileIn; 

   
   vector<vector<vector<double> > > weights;


   

   public:
      vector<vector<vector<double> > > getWeights();
      double* getMetaData();

      double** getTrainingData();
      double* getTruths();

      Reader(string fileName); 

      
   private:
      void readConfigFile(); 
      void readWeights();
      void readTrainingData();



      


}
 */




#endif /* READER_H */