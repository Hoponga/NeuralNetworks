/**
 * Implementation of reader functionalities - currently only
 * for reading in the config file and exporting weights
 * 
 * @author Kailash Ranganathan
 * @version 3/21/20
 */


#include "reader.hpp"
#include <fstream>
#include <iostream>

using namespace std; 


/**
 * Retrieving already defined values from other source files.
 */
extern double lambda;
extern int maxIter;
extern double randomWeightMin;
extern double randomWeightMax;
extern double minError;
extern string outputFile; 


/**
 * This function is a helper method to read in values of the
 * hyperparameters from the given config file. It takes in the filepath
 * (name of the file unless its not in the root folder) and opens
 * a file input stream and parses the file. The file MUST have only one
 * token, be it the name of a hyperparameter or its value, on each line. 
 * After each successive hyperparameter name should follow its value. 
 * @param config the name of the config file
 * note - All the values are stored back into the hyperparameter variables
 * defined in the header file. 
 */
void readConfigFile(string config)
{
   
   ifstream confstream(config);  //Opening file input stream
   string currentArg; 

   while (!confstream.eof())
   {
      /**
       * Gets the hyperparameter at the current line
       * and the value at the next line. 
       */ 
      getline(confstream, currentArg);

      string value;
      getline(confstream,value);
      double val = atof(value.c_str());
      

      /*
       * Parsing of the configuration files. The valid expressions
       * are lambda, maxIter, minWeight, and maxWeight. Their values
       * must be on the line following the hyperparameter name. 
       */ 
      if (currentArg == "lambda")
      {
         lambda = val;
      }
      else if (currentArg == "maxIter")
      {
         maxIter = val;
      }
      else if (currentArg == "minWeight")
      {
         randomWeightMin = val;
      }
      else if (currentArg == "maxWeight")
      {
         randomWeightMax = val;
      }
      else if (currentArg == "minError")
      {
         minError = val;
      }
      
      
   }

}  //readConfig(string filename) method


/**
 * Exports the given weights to a file with the name of the parameter
 * @param weights the weights to export
 * @param filename the filename of the weights file. 
 */
void exportWeights(vector<vector<vector<double> > > weights, string fileName)
{
   ofstream fout(fileName);

   /**
    * Iterates over the weights array and outputs the weights
    * one by one - each "layer" of weights corresponds to a line
    * in the file. 
    */ 
   for(int n = 0; n < weights.size(); n++)            //Iterating over the layers
   {
      for (int j = 0; j < weights[n].size(); j++)     //Iterating over the source layer
      {
         for(int i = 0; i < weights[n][j].size(); i++)//Iterating over the destination layer
         {
            fout << weights[n][j][i] << " ";

         }
         
      }
      fout << endl; 
   }
   fout.close();

}  //exportWeights method
