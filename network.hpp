/*
 * This file contains the declaration of the Network class and its
 * functionalities as well as the the activation function and its derivative. 
 * The network class follows the structure of an n layer network. 
 * 
 * @author Kailash Ranganathan
 * @version 2/17/20
 *
 */


#pragma once      //include guard

#ifndef NETWORK_H
#define NETWORK_H


#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <vector> 
#include <string> 
#include <math.h>

using namespace std;


/*
 * Global variables storing the hyperparameter values
 */
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
 * The training (backpropagation) and the run(input) method for forward propagation
 * work for a general number of hidden layers.
 * Constructing a network currently requires the weights array to already be 
 * formatted (values do not have to be known) but this is done by the driver method
 * 
 * Usage: Network net = Network(numLayers, layerSizes[], hasWeights (0 or 1), weightsArray)
 * numHidden and numInput are both integers, hiddenLayerSizes is a vector of integers representing
 * the size of all the hidden layers in the perceptron (currently generalized but should only be one
 * for training to work)
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
   int* layerSizes; 
   double* truth; 
   double* outputs;  //Because the network only has one input, the outputValue is not an array
   double** theta; 
   double** omega; 
   double** psi; 

   private:
      void fillWeights(double min, double max);

   public:
      Network(int numLayers, int* layerSizesInp, int hasWeights, vector<vector<vector<double> > >& weightsInput);
      void setTruth(double* truthValue);
      double* run(double inputValues[]);
      void updateWeights();
      double error();
      vector<vector<vector<double> > > getWeights();
      ~Network();

}; //Network class declarations




#endif /* NETWORK_H */








