/**
 * Main method for training the network after reading
 * in the input files. Uses the Network class functionalities
 * for running network and updating weights via steepest descent. 
 * 
 * @author Kailash Ranganathan
 * @version March 20, 2020
 *
 */


#include "network.hpp"
#include "reader.hpp"
#include <iostream>
#include <fstream> 
#include <string>
#include <stdlib.h>
#include <algorithm>

using namespace std; 

extern double lambda;
extern int maxIter;
extern double randomWeightMin;
extern double randomWeightMax;
extern double minError;
string outputFile = "finalweights";



int train(Network &n, int numIterations, double** trainData, double truthVals[]);



/*  
 * Main method - runs the network defined in network.hpp
 * Currently, I have    my file I/O inside my main method, but I will 
 * try and generalize it as fast as possible. 
 * The main method reads weights and inputs from the file given by 
 * the command line argument (the file name default is inputs.txt)
 * @param argc the argument counter (number of arguments from command line + 1)
 * @param argv[] the list of arguments from the console - first one
 * is always the file name 
 */
int main(int argc, char* argv[])
{  
   
   /** 
    * File reading - when calling ./output, it MUST be followed by 
    * a file denoting the location of the training data + connectivity model
    * another command argument denoting the location of the hyperparameter values
    * can be given, but is optional. 
    */
   string file = "inputs";            //Creating the file - if a name was given
   string configFile = "configs";

   if (argc == 2)
   {
      file = argv[1];      //If a filename is given, then use that filename
   }
   else if(argc == 3)
   {
      file = argv[1];
      configFile = argv[2];

   }

   ifstream temp(configFile); 

   /**
    * If the file path is invalid, the reader resorts to the default
    * hyperparameters
    */ 
   if (temp)
   {
      temp.close();
      readConfigFile(configFile);
      
   }
   else
   {
      cout << "File not found! Resorting to default hyperparamter values..." << endl << endl;

   }

   
   
   /*
    * The format of the file starts with a line with a single number "n" representing the
    * number of run iterations. For each n, the file should contain a line containing two numbers
    * representing the number of inputs and the number of hidden layer nodes. A third number
    * should be either 0 or a non-zero number. Zero denotes that the user will NOT input weights
    * (currently, filling weights is non-functional) and any other number (standard is zero) represents that
    * the user WILL input weights.   
    * That should be followed by a line containing the specified number of inputs
    * and number of hidden layers + 1 lines containing the weights between the perceptron layers. 
    * There should not be empty line spaces between anything. 
    * 
    * 
    */  
   ifstream fin(file);     // Setting up the file input stream and 
   int numIter = 0;        // reading in the number of iterations 
   int hasWeights = 1; 
   
   fin >> numIter >> hasWeights;  

   /**
    * General data structure declarations that store the information read
    * from the file. 
    */
   int nLayers = 3;

   std::vector<std::vector<std::vector<double> > > weights;
   int* layerSizes = new int[nLayers];   //Stores the size of each layer in the perceptron
   int numInputs, numHidden; 
   int numHiddenLayers = 1; 
   int numOutputs = 1; 
   

   /**
    * Reads in the network size parameters
    */
   fin >> numInputs >> numHidden >> numOutputs; 

   double** inputs; 
   
   inputs = new double*[numIter];
   for (int i = 0; i < numIter; i++)
   {
      inputs[i] = new double[numInputs];
   }
   double truths[numIter];

   /**
    * Memory allocation for the weights array 
    * resize() does not change the size of the array; rather, it
    * allocates new memory for the weights array to hold. 
    */
   weights.resize(nLayers - 1);             
   weights[0].resize(numInputs);
   weights[1].resize(numHidden);
   for (int j = 0; j < numHidden; j++)
   {
      weights[1][j].resize(numOutputs);
   }

   for (int k = 0; k < numInputs; k++)
   {
      weights[0][k].resize(numHidden);

   }
   
   /**
    * If the user does not supply weights to input into the network, 
    * the file reader does not read the input and skips this weights reading. 
    * The reader knows the user does not want to supply weights if the
    * hasWeights flag (the second number in the file) = 0. If it is
    * any other number, the reader will attempt to read a set of weights in
    * depending on the network configuration. 
    */
   if(hasWeights != 0)
   {
      
      for (int k = 0; k < numInputs; k++)    //Iterating over the input layer
      {
         for (int j = 0; j < numHidden; j++) //Iterating over the hidden layer
         {
            /**
             * Sets the current weight to the current number
             * read in by the file 
             */
            double currentWeight; 
            fin >> currentWeight; 
            cout << " weight 0" << k << j << " is " << currentWeight << endl; 
            weights[0][k][j] = currentWeight;
         }
      }
      
      for (int j = 0; j < numHidden; j++)    //Iterating over the hidden layer
      {
         /**
          * Reads in the hidden-output weights. 
          */ 
         double currentWeight; 
         fin >> currentWeight; 
         cout << " weight 1" << j << "0" << " is " << currentWeight << endl; 
         
         weights[1][j][0] = currentWeight;
         

      }
      cout << "Read weights successfully!" << endl << endl; 

   }
   
   /**
    * This for loop resizes every array in the hidden layer weight std::vector to the number of
    * outputs for each node in the hidden layer - currently, there is only one output, so 
    * each std::vector of weights for hidden nodes are just singular elements
    */
   for (int j = 0; j < weights[1].size(); j++)   
   {
      weights[1][j].resize(numOutputs);

   }
   layerSizes[0] = numInputs;       //The sizes of the layers in the network are held
   layerSizes[1] = numHidden;       //in this std::vector
   layerSizes[nLayers-1] = numOutputs;      //Only has one output element for AB-1 Network                                             
   
   //cout << numHiddenLayers << " " << numInputs << " " << layerSizes[2] << " " << weights.size() << endl; 

   
   Network net = Network(numHiddenLayers, numInputs, layerSizes, hasWeights, weights); //Creating the network object
   
   /**
    * The network is run on as many input data given by the user - the first line of the file
    * contains the number of iterations n to run the network followed by n sets of 
    * input and weights. 
    */
   for (int iter = 0; iter < numIter; iter++)       //Iterates over the given number of execution iterations
   {
      
      /**
       * All the inputs are on one line of the input file - once the numInputs
       * value has been read, the next numInputs values in the file can be read
       * as inputs. 
       */
      for (int i = 0; i < numInputs; i++) 
      {
         double inputI; 
         fin >> inputI; 
         
         inputs[iter][i] = inputI;

      }      
      double truth; 
      fin >> truth;                          //Reading in the truth as the last file input for the network
      truths[iter] = truth; 

   }  //Reading in inputs
   
   
   int successful = train(net, numIter, inputs, truths);

   /**
    * Printing finishing messages whether the maximum number of iterations
    * is left or if the error drops below the threshold. 
    */ 
   if (successful == 1)
   {
      std::cout << "Training cut short - error went below " << minError << endl << endl; 

   }
   else
   {
      std::cout << "Training finished. " << maxIter << " iterations complete. " << endl << endl;

   }

   std::cout << "TRAINING SUMMARY" << endl << endl; 
   /**
    * Debugging output - prints all the configurable hyperparameters
    * and network results. 
    */
   std::cout<< "Lambda: " << lambda << endl; 
   std::cout<< "Max number of iterations: " << maxIter << endl;
   std::cout << "Weight range: " << randomWeightMin << " to " << randomWeightMax << endl << endl;
   
   exportWeights(net.getWeights(), outputFile);
   std::cout << "Final weights saved to output file with name \"" << outputFile << "\"" << endl << endl;


   /**
    * Printing the results ofs the network for each test case
    */
   for (int i = 0; i < numIter; i++)
   {
      std::cout << "Test Output for " << inputs[i][0] << " and " << inputs[i][1]; 
      std::cout << ": " << net.run(inputs[i]) << endl; 

   }
   
   fin.close();   //Closing statements - closes the input stream, 
   return 0;      //completes the program and properly exits. 
}



/**
 * Runs the training loop for the given network given the training data
 * and truth values (minimum error is a global variable)
 * 
 * @param n the network to train
 * @param numIterations the maximum number of iterations to train under
 * @param trainData the training data to train the network on
 * @param truth the truth values of the given training data
 * @return 1 if the training goes below the minimum error, 0 is the maximum
 * number of iterations is reached. 
 */
int train(Network &n, int numIterations, double** trainData, double truthVals[])
{
   bool errorReachedThreshold = false; 

   /**
    * Training the network - on each iteration, the network is run
    * on all the training data and update the weights after each
    * input is run. The total error is calculated and displayed
    * after each training iteration and breaks when the error
    * goes below the threshold or the maximum amount of iterations
    * is reached. 
    */ 
   for (int i = 0; i < maxIter && !errorReachedThreshold; i++)
   {
      double error = 0.0;
      for (int j = 0; j < numIterations; j++)
      {
         
         n.setTruth(truthVals[j]);
         double output = n.run(trainData[j]);
         //cout << output << endl; 
         //cout << inputs[j][0] << " " << inputs[j][1] << " goes to " << output << endl;
         n.updateWeights();
         error += n.error();     // The error displayed is the sum of each input's error    
         

      }
      cout << "Iteration " << i << " Error: " << error << endl;

      if (error < minError)        // Break if the error goes below the threshold
      {
         errorReachedThreshold = true; 
         return 1;
      }
      
      
      
   }
   return 0; 

}




