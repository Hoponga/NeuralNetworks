/*
 * This file contains the declaration of the Network class and its
 * functionalities as well as the the activation function and its derivative. 
 * 
 * @author Kailash Ranganathan
 * @version 2/17/20
 *
 */

#pragma once

/*
 * Unusual inputs - the time library is used for completely random seeded
 * number generation for weights
 */
#ifndef NETWORK_H
#define NETWORK_H




#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <vector> 
#include <string> 
#include <math.h>

using namespace std;


extern double lambda;
extern int maxIter;
extern double randomWeightMin;
extern double randomWeightMax;
extern double minError;





/*
 * These functions are general utilities that are not part of
 * the network object 
 * 
 */
 
double activation(double x);
double derivative(double x);
double randomGenerator(double min, double max);


/*
 * Class description for a perceptron
 * of variable inputs and hidden layer nodes
 * The perceptron should follow the structure A-B-1 
 * but the run(input) method works for a general number of hidden layers
 * (One output only). Constructing a network currently requires all the weights values
 * to be known and formatted into the proper structure of this network. 
 * 
 * Usage: Network net = Network(numHidden, numInput, hiddenLayerSizes, startingWeights)
 * numHidden and numInput are both integers, hiddenLayerSizes is a vector of integers representing
 * the size of all the hidden layers in the perceptron (currently generalized but should only be one)
 * and weights is a three dimensional array of double values having the same shape as the perceptron's
 * structure
 */
class Network
{
   /*
    * The contents of a network object. The "number" variables are used to hold the structure
    * of the perceptron. 
    */
   int nHidden, nActivation, nOutput; 
   int nLayers; 
   vector<vector<vector<double> > > weights; 
   vector<vector<vector<double> > > deltaWeights; 
   double** layers; 
   int* hiddenLayerSizes; 
   double truth; 
   double outputValue;  //Because the network only has one input, the outputValue is not an array

   private:
      void fillWeights(double min, double max);
      


   public:
      Network(int hidden, int input, int* layerSizes, int hasWeights, vector<vector<vector<double> > >& weightsInput);
      

      void setTruth(double truthValue);
      
      double run(double inputValues[]);
      void updateWeights();
      double error();
      vector<vector<vector<double> > > getWeights();
      

      /*
       * Currently, the destructor for the Network class is not functional
       */ 
      ~Network() 
      {
         free(layers);
         free(hiddenLayerSizes);

      }

}; //Network class declarations - implementation is below 




#endif /* NETWORK_H */








