
/**
 * This file contains the main method for running the network and 
 * reading in input from a file that gives the parameters of the 
 * network as well as the training of the network using steepest descent. All definitions are 
 * done in the network header file
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

/**
 * Default hyperparamter values (can be overriden in the config file)
 */
double lambda = 5.0; 
int maxIter = 10000;
double randomWeightMin = -2.0;
double randomWeightMax = 2.0;
double minError = 0.001;

/**
 * This method sets all of the weights in the network to 
 * random numbers given in the specified range
 * @param range the range of numbers that limits the RNG of the
 * weights
 */
void Network::fillWeights(double min, double max)
{
   

   for (int n = 0; n < nLayers-1; n++)                  // Iterates over the hidden layer nodes
   {
      for (int i = 0; i < hiddenLayerSizes[n+1]; i++)   // Iterates over the destinations  
      {
         for (int j = 0; j < hiddenLayerSizes[n]; j++)  // Iterates over the sources 
         {
            /**
             * Random number generator - uses the random generator function
             * to get a number in the given range
             */
            double randNum = randomGenerator(min, max);
            cout << "Weight " << n << j << i << ": " << randNum << endl; 

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



/**
 * Constructor for the Network class - initializes all the vectors and size parameters
 * of the network
 * @param hidden the number of hidden layers
 * @param input the number of inputs this network can handle(first layer)
 * @param layerSizes a vector where each element represents the size of that respective layer
 * including inputs as first layer and output as final layer
 * For example, layerSizes[4] would return the size of the 4TH hidden layer
 * @param hasWeights checks if the weights array is already filled. If so, it does not
 * fill the weights with random numbers, and vice versa. 
 * @param weightsInput the weights are represented as a three dimensional vector where the 
 * first dimension represents the layer it is in (the "weights layer"), the second dimension
 * represents its source neuron, and the third dimension represents its destination. For example
 * a weight as the 3rd element of a source's weights array would be going to the 3rd hidden node
 * in the next layer. 
 * 
 */
Network::Network(int hidden, int input, int* layerSizes, int hasWeights, vector<vector<vector<double> > >& weightsInput)
{   
   
   srand(time(NULL));  
   
   nLayers = (sizeof(layerSizes)/sizeof(*layerSizes)) + 1; 
   
   nHidden = nLayers - 2; 
   //Defining the sizes of instance variables for the network 
   nOutput = 1; 

   /*
    * Storing the values of the constructor parameters
    */
                 
   nActivation = input; 
   hiddenLayerSizes = layerSizes; 
   weights = weightsInput; 
   deltaWeights = weightsInput; 
   /*
    * The layers vector holds the activation values for the hidden and output 
    * layers. During network forward propagation, the first element of "layer" 
    * is set to the input values. 
    */ 
   layers = new double*[nLayers];
   
   for (int n = 0; n < nLayers; n++)  //Allocating the hidden layers
   {
      layers[n] = new double[layerSizes[n]]; 
      //cout << layerSizes[n] << endl; 

   }
   
   /**
    * Fills weights randomly because the user did not provide a set
    * of weights. 
    */
   if (hasWeights == 0)
   {
      fillWeights(randomWeightMin, randomWeightMax);

   }

}  //Network class constructor

/**
 * The run function takes in a vector of input values and feeds them 
 * through the network and returns the output value (currently, the output layer
 * only has one neuron, so the return value is just a single value representing
 * the value of the 1 output node)
 * For each hidden layer neuron, its new value is calculated by calculating the dot product
 * of the weights vector and the input vector and running them through the activation function
 * (currently f(x) = x)
 * @param inputValues the vector of input values to run this network over
 * @return the value of the network's single output node after feeding the input values
 * through the network
 * 
 */
double Network::run(double inputValues[])
{

   for (int i = 0; i < nActivation; i++) //Iterating over the number of activation
   {
      layers[0][i] = inputValues[i];    //Setting the values of the first layer to the 
                                        //given input values 
   }
   

   for (int n = 0; n < nLayers - 1; n++)                 //Iterates over the hidden layers
   {
      for (int i = 0; i < hiddenLayerSizes[n+1]; i++)    //Iterates over the destination layer
      {
         double newValue = 0.0;                          //The new value of the current hidden layer node
         
         for (int j = 0; j < hiddenLayerSizes[n]; j++)   //Iterates over the source layer
         {
            /*
             * Calculating dot product - multiplies the weight
             * by its given source node 
             * n, j, i is used for generalized amount
             * of hidden layers
             * Also prints the current weight. 
             */
            //cout << "weight " << n << j << i << " is " << weights[n][j][i] << endl;
            newValue += weights[n][j][i] * layers[n][j];  
         }
         layers[n+1][i] = activation(newValue);
         

      }

   }  //for (int n = 0; n < nLayers-1; n++)
   

   outputValue = layers[nLayers-1][0];    //Storing the output value for error
   
   
   /* At the moment, the last layer only
    * has one element, so run returns
    * the value of the singular output neuron
    */
   return layers[nLayers - 1][0]; 

}  //Network::run(input) method 


/**
 * Currently, the error function is just a difference between the truth value
 * and the actual output value (E = T - F)
 * @return the error between the inputted truth value and the calculated output value
 * by the perceptron
 * 
 */
double Network::error()
{
   double diff = truth - outputValue; 
   
   return 0.5*diff*diff;
   
}

/*
 * Sets the internal truth variable
 * to the input parameter
 * @param truthValue the new value of the perceptron's input
 * truth output (what it should output)
 */ 
void Network::setTruth(double truthValue)
{
   truth = truthValue;

}


/**
 * The updateWeights function uses the steepest descent formula for
 * one output to update the weights given the error and lambda hyperparameter.
 * The delta(weights) are stored in a separate vector with the same dimensions
 * as the weights array and are used to update the weights. 
 * 
 */
void Network::updateWeights()
{
   
   /**
    * One for loop is used to iterate over the weight(j0) and the
    * weights(kj) by iterating over the j's. For each j, that respective
    * partial(E, weight(j0)) and partial(E, weight(kj))
    * 
    */
   for (int j = 0; j < hiddenLayerSizes[nLayers-2]; j++)
   {
      /**
       * dotProductSum is used to accumulate the sum over the J's of
       * w(J0)*h(J)
       * Currently, weight J0 is used because there is only one output. 
       * currentDeltaWeightOne is used to hold partial of E with respect to weight j0
       * 
       */
      double currentDeltaWeightOne = (outputValue - truth)*layers[1][j];
      double dotProductSum = 0.0; 
      

      for (int J = 0; J < hiddenLayerSizes[1]; J++)   //Iterates over J for the dot product summation
      {
         dotProductSum += weights[nLayers-2][J][0]*layers[1][J];

      }
      
      currentDeltaWeightOne *= derivative(dotProductSum);
      deltaWeights[nLayers-2][j][0] =  currentDeltaWeightOne;  //Storing the value of this partial in delta weights array
      
      
      /**
       * This for loop iterates over the k's for the hidden layer
       * weights to calculate the partial of E with respect to weight(kj)
       * 
       */
      for (int k = 0; k < hiddenLayerSizes[0]; k++)            //Dot product sum over the K's (activation * weight)
      {
         /**
          * Same format as above - currentDeltaWeightTwo holds to current partial and
          * dotProductSumForK is used to sum over the K's a(k)*weight(kj)
          * 
          */
         double currentDeltaWeightTwo = layers[0][k]*(outputValue - truth)*weights[nLayers-2][j][0];
         double dotProductSumForK = 0.0;

         for (int K = 0; K < hiddenLayerSizes[0]; K++)
         {
            
            
            dotProductSumForK += weights[0][K][j]*layers[0][K];

         }
         currentDeltaWeightTwo *= derivative(dotProductSum)*derivative(dotProductSumForK);
         deltaWeights[0][k][j] =  currentDeltaWeightTwo;
         

      }  // for (int k = 0; k < hiddenLayerSizes[0]; k++)

   }     // for (int j = 0; k < hiddenLayerSizes[nLayers-2]; j++)



   /**
    * This for loop takes all the delta weights calculated above
    * and applies them to the weights array.
    */
   for (int n = 0; n < nLayers - 1; n++)               //Iterating over the layers
   {
      for (int i = 0; i < hiddenLayerSizes[n+1]; i++)  //Iterating over the destination layer
      {
         for (int j = 0; j < hiddenLayerSizes[n]; j++) //Iterating over the source layer
         {
            /**
             * weight(nji) is updated by adding the deltaWeight multiplied by 
             * the lambda (not currently adaptive) and negating that for
             * gradient descent as opposed to ascent. 
             */
            weights[n][j][i] += -lambda*deltaWeights[n][j][i];
        
         }
      }
   }  // for (int n = 0; n < nLayers - 1; n++)
   
}




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















