
/*
 * This file contains the main method for running the network and 
 * reading in input from a file that gives the parameters of the 
 * network as well as the training of the network using backpropagation. All definitions are 
 * done in the network header file and training works for a generalized number and
 * sizes of layers. 
 * 
 * Network class services: 
 * double* run(double inputVals[]) for forward propagation, void updateWeights() for backward
 * propagation, double error() for error calculation, double activation(double x) for f(x), 
 * double derivative(double x) for f'(x), double randomGenerator(double min, double max) 
 * to give a random number in the given range, and fillWeights(double min, double max) to fill the
 * network's weights with random values in the given range. 
 * 
 * Note: f(x) corresponds to the activation function used in training the network. 
 * 
 * @author Kailash Ranganathan
 * @version 2/17/20
 * 
 */ 


#include <iostream>
#include <fstream> 
#include "network.hpp"
#include <string>
#include <stdlib.h>

/*
 * Default hyperparamter values (can be overridden in the config file)
 */
double lambda = 0.1; 
int maxIter = 50000;
double randomWeightMin = -0.7;
double randomWeightMax = 0.7;
double minError = 0.001;

/*
 * This method sets all of the weights in the network to 
 * random numbers given in the specified range
 * @param range the range of numbers that limits the RNG of the
 * weights
 */
void Network::fillWeights(double min, double max)
{
   for (int n = 0; n < nLayers-1; n++)            // Iterates over the weight layers
   {
      for (int i = 0; i < layerSizes[n+1]; i++)   // Iterates over the destinations  
      {
         for (int j = 0; j < layerSizes[n]; j++)  // Iterates over the sources 
         {
            /*
             * Random number generator - uses the random generator function
             * to get a number in the given range
             */
            double randNum = randomGenerator(min, max);
            //cout << "Weight " << n << j << i << ": " << randNum << endl; 

            weights[n][j][i] = randNum;
            
         }                                          
         
      }
       
   }
}  //fillWeights(int range) method definition


/*
 * Returns the weights array used by the network
 * @return a shallow copy of the weights (should only be used for printing)
 */
vector<vector<vector<double> > > Network::getWeights()
{
   return weights; 
}



/*
 * Constructor for the Network class - initializes all the backend arrays and size parameters
 * of the network
 * 
 * @param numLayers the number of layers this network has
 * @param layerSizes a vector where each element represents the size of that respective layer
 * including inputs as first layer and output as final layer
 * For example, layerSizes[4] would return the size of the 4TH hidden layer
 * @param hasWeights checks if the weights array is already filled. If so, it does not
 * fill the weights with random numbers, and vice versa. 
 * @param weightsInput the weights are represented as a three dimensional vector where the 
 * first dimension represents the layer number, the second dimension
 * represents its source neuron, and the third dimension represents its destination. For example
 * a weight as the 3rd element of a source's weights array would be going to the 3rd hidden node
 * in the next layer. 
 * 
 */
Network::Network(int numLayers, int* layerSizesInp, int hasWeights, vector<vector<vector<double> > >& weightsInput)
{   
   
   srand(time(NULL));  
   
   nLayers = numLayers; 
   
   nHidden = nLayers - 2; 
   //Defining the sizes of instance variables for the network 
   nOutput = layerSizesInp[nLayers-1]; 

   /*
    * Storing the values of the constructor parameters
    */
                 
   nActivation = layerSizesInp[0]; 
   layerSizes = layerSizesInp; 
   weights = weightsInput; 
   deltaWeights = weightsInput; 
   /*
    * The layers jagged array holds the activation values for the hidden and output 
    * layers. During network forward propagation, the first element of "layer" 
    * is set to the input values. 
    */ 
   layers = new double*[nLayers];
   theta = new double*[nLayers-1];
   omega = new double*[nLayers-1];
   psi = new double*[nLayers-1];
   
   /*
    * This for loop allocates memory for the layers array
    * as well as the psi, omega, and theta backend arrays used
    * in backpropagation.
    */ 
   for (int n = 0; n < nLayers; n++)  //Iterating over the layers
   {
      layers[n] = new double [layerSizesInp[n]]; 

      /*
       * In this for loop, I also allocate memory for the 
       * theta, omega, and psi backend arrays for backpropagation. 
       * All of these are n-1 long as they start at the first hidden layer, 
       * so they are declared as such
       */
      if (n < nLayers-1)
      {
         theta[n] = new double[layerSizesInp[n+1]];
         omega[n] = new double[layerSizesInp[n+1]];
         psi[n] = new double[layerSizesInp[n+1]];

      }
   }  //for (int n = 0; n < nLayers; n++)
   
   /*
    * Fills weights randomly because the user did not provide a set
    * of weights. 
    */
   if (hasWeights == 0)
   {
      fillWeights(randomWeightMin, randomWeightMax);

   }


}  //Network class constructor

/*
 * The run function takes in a vector of input values and feeds them 
 * through the network and returns the output layer values
 * For each hidden layer neuron, its new value is calculated by calculating the dot product
 * of the weights vector and the input vector and running them through the activation function.
 * 
 * Theta values for backpropagation are also calcualted on the fly. My forward propgation
 * only requires one loop (the small one at the start is just to populate the first layer with
 * the new activation values and is not part of forward propagation)
 * 
 * @param inputValues the vector of input values to run this network over
 * @return the value of the network's single output node after feeding the input values
 * through the network
 * 
 */
double* Network::run(double inputValues[])
{
   /*
    * This for loop adds the input values into the network's
    * backend layers array and is NOT a part of forward propagation
    */
   for (int k = 0; k < nActivation; k++) //Iterating over the number of activation
   {
      layers[0][k] = inputValues[k];     //Setting the values of the first layer to the 
                                         //given input values 
      
   }
   
   /*
    * Loop for forward propagation - generalized for n layers
    * Serves two purposes - to calculate new activation values and
    * to calculate new theta values
    */
   for (int n = 0; n < nLayers - 1; n++)           //Iterates over the hidden layers
   {
      for (int i = 0; i < layerSizes[n+1]; i++)    //Iterates over the destination layer
      {
         double newValue = 0.0;                    //The new sum value of the current hidden layer node
         
         for (int j = 0; j < layerSizes[n]; j++)   //Iterates over the source layer
         {
            /*
             * Calculating dot product - multiplies the weight
             * by its given source node 
             * n, j, i is used for generalized amount
             * of hidden layers
             */
            newValue += weights[n][j][i] * layers[n][j]; 
         }

         layers[n+1][i] = activation(newValue);

         /**
          * Calculating theta values on the fly
          */
         theta[n][i] = newValue; 
         
      }  //for (int i = 0; i < layerSizes[n+1]; i++)

   }     //for (int n = 0; n < nLayers-1; n++)
   
   /*
    * Storing output layer values and returning the layer
    */
   outputs = layers[nLayers-1];

   return layers[nLayers - 1]; 

}  //Network::run(input) method 

/*
 * Currently, the error function is the sum of squares of the difference between
 * respective truth and output values all multiplied by 0.5. 
 * @return the error between the inputted truth value and the calculated output value
 * by the perceptron
 * 
 */
double Network::error()
{
   /*
    * The error is calculated as half the sum over i of (Ti-Fi)^2
    */
   double total = 0.0; 
   for (int i = 0; i < nOutput; i++)      //Looping over the output layer
   {
      
      double diff = outputs[i] - truth[i]; 
      omega[nLayers-2][i] = -diff; 
      psi[nLayers-2][i] = omega[nLayers-2][i] * derivative(theta[nLayers-2][i]);
      total += diff*diff; 


   }
   total *= 0.5; 
   
   return total; 
   
}  //double Network::error()

/*
 * Sets the internal truth variable
 * to the input parameter
 * @param truthValue the new value of the perceptron's input
 * truth output (what it should output)
 */ 
void Network::setTruth(double* truthValue)
{
   truth = truthValue;
   return; 

}


/*
 * The updateWeights function uses the backpropgation formula for
 * "i" outputs to update the weights given the error and lambda hyperparameter.
 * The delta(weights) are stored in a separate vector with the same dimensions
 * as the weights array and are used to update the weights. 
 * My backpropagation works for a generalized number of layers, so only
 * two backwards for loops are used. 
 * 
 */
void Network::updateWeights()
{ 

   /*
    * This loop iterates backwards(starting at the 2nd to last layer)
    * and calculatesthe omega, psi, and deltaWeights values for each layer. 
    * Delta weights calculated from the previous (n+1) layer are applied
    * and delta weights for the nth layer are calculated
    * Note - for generalized backprop, I'm using j as the current "source layer,"
    * i as the next/destination layer, and k as the previous layer (n) but indexing n
    * from the first k layer (so k is layers[n], j is layers[n+1], and i is layers[n+2])
    */
   for (int n = nLayers-3; n >= 0; n--)                     //Iterating over the layers starting from
   {                                                        //Last hidden layer
      
            
      for (int j = 0; j < layerSizes[n+1]; j++)             //Iterating over source layer
      {
         /*
          * Omegas indices are shifted by one as it starts at the FIRST
          * hidden layer, so 
          * it would be [n][j] as opposed to [n+1][j]
          */ 
         omega[n][j] = 0.0; 

         for (int i = 0; i < layerSizes[n+2]; i++)          //Iterating over destination layer to calculate omega
         {  
            /*
             * Calculating omega(j) = sum over J of psi(i)*weights(ji)
             * After this, the deltaWeights is calculated tactically right
             * before incrementing its corresponding weights on the fly so the
             * PREVIOUS values are still used but they're done in the same for loop
             * for optimization.
             */
            omega[n][j] += psi[n+1][i] * weights[n+1][j][i];
            deltaWeights[n+1][j][i] = lambda * psi[n+1][i] * layers[n+1][j];
            weights[n+1][j][i] += deltaWeights[n+1][j][i];

         }
          
         psi[n][j] = omega[n][j] * derivative(theta[n][j]); //Derived from the formula for psi

      }  //for (int j = 0; j < layerSizes[n+1]; j++) - "source layer" 
            
      
   }     //for (int n = nLayers - 3; n >= 0; n--)    - backpropagating through the network

   /*
    * By the backpropagation algorithm, the first weights layer
    * requires an extra deltaWeights calculation and increment. This second
    * for loop does just that. 
    */
   for (int m = 0; m < layerSizes[0]; m++)
   {
      for (int k = 0; k < layerSizes[1]; k++)
      {
         deltaWeights[0][m][k] = lambda * psi[0][k] * layers[0][m];
         weights[0][m][k] += deltaWeights[0][m][k];
      }
   }
   
   return; 

}  //updateWeights() method for backpropgation

/*
 * Random generator currently generates a random number
 * in the given range above the "min" parameter
 * and below the "max" parameter
 * @param min the minimum boundary for the RNG
 * @param max the maximum boundary for the RNG
 * @return a random number between min and max
 * 
 */
double randomGenerator(double min, double max)
{
   return ((double) rand() / (RAND_MAX))*(max-min) + min;   //(rand() / RAND_MAX) gives a value in the
                                                            //standard range 0 to 1

} 

/*
 * This function holds the activation function that is applied over
 * the dot products in my network
 * @param x the input into the activation function f(x)
 * @return the value of f(x) - currently f(x) = x
 */
double activation(double x)
{
   return 1.0/(1.0+exp(-x));
}

/*
 * This function holds the value of the derivative function of the given
 * activation for this network. 
 * @param x the x value to execute the derivative function at
 * @return the value of df(x)/dx where f(x) is the activation function
 */
double derivative(double x)
{
   double activationAtX = activation(x);
   return activationAtX*(1.0-activationAtX);
}

/*
 * Destructor for the network class - frees up 
 * space allocated for the various array instances
 */
Network::~Network()
{
   free(layerSizes);
   free(layers);
   free(truth);
   free(outputs);

}















