/*
 * Main method for training the network after reading
 * in the input files. Uses the Network class functionalities
 * for running multiple output network and updating weights via backpropagation. 
 * The network is trained on data given in a file also containing the desired network's
 * structure (layer sizes, number of layers, etc). Note - training works for n layers. 
 * 
 * Services: main method to run the network which calls helper train() method to train the 
 * network given parameters, training data, and iteration details. The training uses whatever
 * error metric and training algorithm present in the network. 
 * @author Kailash Ranganathan
 * @version March 20, 2020
 *
 */


#include <iostream>
#include <fstream> 
#include <string>
#include <stdlib.h>
#include "network.hpp"
#include "reader.hpp"


using namespace std; 

extern double lambda;
extern int maxIter;
extern double randomWeightMin;
extern double randomWeightMax;
extern double minError;
string outputFile = "finalweights";



int train(int nOut, Network &n, int numIterations, double** trainData, double** truthVals);
int test (int nOut, Network &n, double* testData);




/*  
 * Main method - runs the network defined in network.hpp
 * Calls upon the Reader functionalities to read in training data
 * and network structure and trains the network through the train() helper method
 * (results are echoed back at the end)
 * @param argc the argument counter (number of arguments from command line + 1)
 * @param argv[] the list of arguments from the console - first one
 * is always the file name 
 */
int main(int argc, char* argv[])
{  
   
   /* 
    * File reading - when calling ./output, it MUST be followed by 
    * a file denoting the location of the training data + connectivity model
    * another command argument denoting the location of the hyperparameter values
    * can be given, but is optional. 
    */
   string file = "inputs";            //Creating the file - if a name was given
   string configFile = "configs";
   string testFile = "testfile";
   
   if (argc == 2)
   {
      file = argv[1];      //If a filename is given, then use that filename
   }
   else if (argc == 3)
   {
      file = argv[1];
      configFile = argv[2];

   }
   else if (argc == 4)
   {
      file = argv[1];
      configFile = argv[2];
      testFile = argv[3];
   }

   ifstream temp(configFile);

   /*
    * If the file path is invalid, the reader resorts to the default
    * hyperparameters
    */ 
   if (temp)
   {
      temp.close();   
   }
   else
   {
      cout << "Config file not found! Resorting to default hyperparamter values..." << endl << endl;
      configFile = "\0";

   }
  
   /*
    * The reader takes in a properly formatted file containing network specifications and
    * the training data and organizes the input into their respective data structures to be
    * ready for use by the network. Thus, the main method serves as the "link" between the file I/O
    * frontend and the network backend. 
    */     
   Reader reader = Reader(file, configFile, testFile);
  
   /*
    * Getting the input data stored by the reader after reading the 
    * user's input file. 
    */
   vector<vector<vector<double> > > weights = reader.getWeights();
   int* layerSizes = reader.getLayerSizes();
   int* metadata = reader.getMetaData();
   double** inputs = reader.getTrainingData();
   double** truths = reader.getTruths();
   double* testSet = reader.getTest();



   int numIter = metadata[0];
   int hasWeights = metadata[1];
   int numLayers = metadata[2];
   int numOutputs = layerSizes[numLayers-1];
   int testOrTrain = metadata[3];
   
   Network net = Network(numLayers, layerSizes, hasWeights, weights); //Creating the network object
   
   
   /*
    * The network is trained using the train method
    * successful is an integer flag (0 for not, 1 for successful)
    * representing if the network converged or not
    * 
    */
   int successful = 2; 
   if (testOrTrain == 1)
   {
      successful = train(numOutputs, net, numIter, inputs, truths);

   }
   else
   {
      test(numOutputs, net, testSet);
     
   }
   
   

   /*
    * Printing finishing messages whether the maximum number of iterations
    * is left or if the error drops below the threshold. 
    */ 
   if (successful == 1)
   {
      std::cout << "Training cut short - error went below " << minError << endl << endl; 

   }
   else if (successful == 2)
   {
      std::cout << "Testing complete. " << endl; 
      
   }
   else
   {
      std::cout << "Training finished. " << maxIter << " iterations complete. " << endl << endl;

   }
   if (successful != 2)
   {
         std::cout << "TRAINING SUMMARY" << endl << endl; 

      /*
      * Debugging output - prints the test output for each training set
      * given by the network to make sure the results are somewhat accurate. 
      */
      for (int setNum = 0; setNum < numIter; setNum++)
      {
         double* outputs = net.run(inputs[setNum]);

         std::cout << "Test output for " << inputs[setNum][0] << " and " << inputs[setNum][1]; 
         std::cout << ": "; 

         for (int i = 0; i < numOutputs; i++)   //Prints out the "i" outputs
         {                                      //of the network
            std::cout << outputs[i] << " "; 

         }
         std::cout << endl; 

      }  //for (int setNum = 0; setNum < numIter; setNum++)
      std::cout << endl; 

      /*
      * Echoing back hyperparameter + debugging information after the network has trained
      */
      std::cout << "Lambda: " << lambda << endl; 
      std::cout << "Max number of iterations: " << maxIter << endl;
      std::cout << "Weight range: " << randomWeightMin << " to " << randomWeightMax << endl;
      std::cout << "Network configuration: "; 

      for (int n = 0; n < numLayers; n++)
      {
         std::cout << layerSizes[n] << " "; 
      }
      std::cout << endl << endl; 

      /*
      * Exporting weights out to a file
      */
      exportWeights(net.getWeights(), outputFile);
      std::cout << "Final weights saved to output file with name \"" << outputFile << "\"" << endl << endl;  
      
   

   }
   
   return 0;      //completes the program and properly exits. 

}  //int main()



/*
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
int train(int nOut, Network &n, int numIterations, double** trainData, double** truthVals)
{
   bool errorReachedThreshold = false; 
   int isSuccessful = 0; 
   /*
    * Training the network - on each iteration, the network is run
    * on all the training data and update the weights after each
    * input is run. The total error is calculated and displayed
    * after each training iteration and breaks when the error
    * goes below the threshold or the maximum amount of iterations
    * is reached. 
    */ 
   double error = 0.0;
   double previousError = 2000000.0; 
   for (int i = 0; i < maxIter && !errorReachedThreshold; i++)
   {
      
      
      for (int currentSet = 0; currentSet < numIterations; currentSet++)
      {
         /*
          * For each training set, the input values are forward propagated in the method
          * run() and the weights are updated using whatever algorithm written in the
          * network (currently backpropagation). Total iteration error is defined as
          * the sum of the individual training set errors. 
          */
         n.setTruth(truthVals[currentSet]);
         double* output = n.run(trainData[currentSet]);
         error += n.error();        // The error displayed is the sum of each training set's error   
         n.updateWeights();
          

      }
      error = error/(1.0*numIterations);
      if (error > previousError)
      {
         lambda *= 1;
      }
      else
      {
         lambda *= 1; 
      }
      
      cout << "Iteration " << i << " Error: " << error << endl;
      //cout << "Prev Iteration Error is " << previousError << endl; 
      previousError = error; 
      
      
      cout << "New lambda " << lambda << endl; 

      
      

      if (error < minError)        // Break if the error goes below the threshold
      {
         errorReachedThreshold = true; 
         isSuccessful = 1; 
      }
      
   }  //for (int i = 0; i < maxIter && !errorReachedThreshold; i++)

   return isSuccessful; 

}     //int train() method


int test (int nOut, Network &n,  double* testData)
{
   double* output; 
   
   output = n.run(testData);
   std::cout << "Test set output: "; 
   for (int j = 0; j < nOut; j++)
   {
      std::cout << output[j] << " ";
         
   }
   std::cout << endl; 
   
   return 0; 
}



