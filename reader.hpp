/*
 * Header file for the reader class - Contains declarations
 * for a Reader that reads in network structure, possible weight values, 
 * and training data 
 * 
 * The reader handles file I/O, reads in training data, and exports weights
 * to a file at the end of training
 * 
 * @author Kailash Ranganathan
 * @version March 21, 2020
 */



#pragma once      //include guard

#ifndef READER_H
#define READER_H

#include <string>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>

using namespace std; 
/*
 * Helper method to read in the config file values
 * (hyperparameters) (only functional method rn)
 */
void readConfigFile(string config);

/*
 *  Exports the weights to a file given by the filename
 */
void exportWeights(vector<vector<vector<double> > > weights, string fileName);


/*
 * The Reader class contains implementations for getting training data
 * and network configuration data from a properly formatted file. Using various
 * helper methods, it first reads in the number of training sets, a 0 or 1 depending on
 * whether the user wants to input weights, and the number of layers. It then reads in the
 * given number of layers, optional weights, and the training data. All the values are properly
 * interpreted and stored in their various data structures for usage in training the network. 
 * 
 * Note - the Reader is generalized for an N layer network. 
 * (and will properly calculate how many weights it has to read in that context)
 * Also, there should always be spaces between numbers and no empty lines between data. 
 * 
 */
class Reader
{
   int numTrain; 
   int numLayers; 
   int* layerSizes; 
   double** inputs; 
   double** truths;
   double* test; 
   int numOutputs; 
   int numInputs; 
   int hasWeights; 
   int testOrTrain; 
   vector<vector<vector<double> > > weightsRead;

   private:
      void readWeights(ifstream& fin);
      void readTrainingData(ifstream& fin);
      void readMetaData(ifstream& fin);
      void readTestData(string testFile);


   public:
      vector<vector<vector<double> > > getWeights();
      int* getMetaData();
      int* getLayerSizes();
      double* getTest();

      double** getTrainingData();
      double** getTruths();

      Reader(string fileName, string configFile, string testFile);   

};    //class Reader


#endif /* READER_H */