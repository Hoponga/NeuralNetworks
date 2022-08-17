/*
 * Implementation of reader functionalities - currently reads in 
 * network structure and training data from a properly formatted
 * input file as well as hyperparameters from an optional config file. 
 * Through this implementation, initial reading is independent of the network
 * and training functionalities
 * 
 * Reader class services: 
 * void readConfigFile(string config) updates hyperparameters given by config file
 * values (must be properly formatted), void readMetaData(ifstream& fileIn) reads in
 * Reader important values such as number of training sets, a weights existence flag, and
 * the number of layers in the network as well as the shape of the network layers, 
 * void readTrainingData(ifstream& fileIn) reads in training data given the number of 
 * training sets and input activations per set, void readWeights(ifstream& fileIn)
 * populates a weights array if the user has predefined values, void exportWeights(weights)
 * stores the given weights in an output
 * 
 * @author Kailash Ranganathan
 * @version 3/21/20
 */


#include <fstream>
#include <iostream>
#include <string>

#include "reader.hpp"

using namespace std; 


/*
 * Retrieving already defined values from other source files.
 * (global variables)
 */
extern double lambda;
extern int maxIter;
extern double randomWeightMin;
extern double randomWeightMax;
extern double minError;
extern string outputFile; 


/*
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

   while (!confstream.eof())     //Keep reading while the input stream
   {                             //is not finished
      /*
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
      if (currentArg.find("lambda") != string::npos)
      {
         
         lambda = val;
      }
      else if (currentArg.find("maxIter") != string::npos)
      {
         maxIter = val;
      }
      else if (currentArg.find("minWeight") != string::npos)
      {
         randomWeightMin = val;
      }
      else if (currentArg.find("maxWeight") != string::npos)
      {
         randomWeightMax = val;
      }
      else if (currentArg.find("minError") != string::npos)
      {
        
         minError = val;
      }
      
      
   }
   confstream.close(); //Closing the input stream
   return;

}                      //readConfig(string filename) method


/*
 * Constructor for the Reader class. Takes in a file name for the training data
 * and network parameters and configFile name (if it is empty, then just uses 
 * default values) - the constructor reads all of the input in 
 * from the file and stores them in their
 * respective data structures (weights array, input array, 
 * truths array, and network layer size array)
 * @param filename the name of the training data file
 * @param configFile the name of the optional config file (\0 if no file is provided)
 */
Reader::Reader(string filename, string configFile, string testFile)
{
   
   ifstream fileIn(filename);

   /*
    * If a valid filename is given for configurations/hyperparameters
    * use that instead and read from that file. 
    */ 
   if (configFile != "\0")
   {
      readConfigFile(configFile);
      
   }
   
   /*
    * Data at the top of the file (number of training sets, hasWeights flag,
    *  and number of layers for the network) helps the reader correctly read in inputs
    */
   readMetaData(fileIn);
   
   if (hasWeights == 1)       //Read weights if the user has written them
   {
      
      readWeights(fileIn);
      
      
   }
   if(testOrTrain == 1)
   {
      readTrainingData(fileIn);

   }
   else
   {
      readTestData(testFile);

   }
   
   
   fileIn.close();            //Closing the input stream 

}                             //Reader class constructor
/*
 * Reads the test data from the test directory
 * and prepares it to be executed by the network
 */
void Reader::readTestData(string testFileName)
{
   
   
   
   
   ifstream testFile(testFileName.c_str());
   test = new double[numInputs];
   for (int i = 0; i < numInputs; i++)
   {
      double current;
      testFile >> current; 
      test[i] = current/(255.0);
      

      
   }
   cout << "Evaluation of file " << "\"" <<  testFileName << "\"" << endl; 
   return; 
}

double* Reader::getTest()
{
   return test; 
}

/*
 * Reads the metadata at the top of a text file giving the
 * total number of training sets and whether the user would like to input weights or not
 * Stores these values in their respective variables
 * 
 * @param fileIn the file input stream passed by reference
 */
void Reader::readMetaData(ifstream& fileIn)
{
   fileIn >> numTrain >> hasWeights >> numLayers >> testOrTrain;  //Reading in training set amounts
   layerSizes = new int[numLayers];                
   cout << endl << endl; 
   /*
    * Reading in the layer sizes from the input file
    */
   cout << "Network structure: "; 
   for (int n = 0; n < numLayers; n++)
   {
      int currentLayerSize; 
      fileIn >> currentLayerSize; 
      
      layerSizes[n] = currentLayerSize; 
      
      cout << layerSizes[n] << " ";

   }
   cout << endl; 

   /*
    * Declared for readability in later sections
    */
   numInputs = layerSizes[0];
   numOutputs = layerSizes[numLayers-1];

   /*
    * Allocating memory space for the weights array
    * 
    */
   weightsRead.resize(numLayers - 1);       

   for (int n = 0; n < numLayers - 1; n++)      //Iterating over the number of layers
   {
      /*
       * Allocating memory for the current weights layer
       * Resizing the 2nd dimension to the source layer (current layer)
       * and iterating over that to resize the 3rd dimension to the destination layer
       *
       */
      weightsRead[n].resize(layerSizes[n]);
      for (int j = 0; j < layerSizes[n]; j++)       //Iterating over the current source node
      {
         weightsRead[n][j].resize(layerSizes[n+1]); //Allocating memory for the current node's weights

      }
   }  //for (int n = 0; n < numLayers - 1; n++)

   return; 

}     //void Reader::readMetaData(ifstream& fileIn)



/*
 * This method allocates memory for the truth and input arrays and reads the 
 * values in from the input file into these arrays. The arrays are held in the 
 * Reader instance.  
 * 
 */
void Reader::readTrainingData(ifstream& fileIn)
{
  
   
   
   inputs = new double*[numTrain];
   truths = new double*[numTrain];
    
   /*
    * Allocating memory for the second dimension of the 
    * truth values and input values
    */
   for (int i = 0; i < numTrain; i++)
   {
      
      truths[i] = new double[numOutputs];
      inputs[i] = new double[numInputs];
   }

   double currentInput = 5.5;
   double currentTruth = 5.5;
   string currentImage =  "wacko";
   string currentTruthPath = "truth"; 
   for (int i = 0; i < numTrain; ++i)         //Iterates over each training set to read it
   { 
     
      
      currentImage = "train/train";
      currentTruthPath =  "truth/truth"; 


      currentImage += to_string(i);
      currentTruthPath += to_string(i);
      cout << currentImage << endl; 
      cout << currentTruthPath << endl; 
      
      ifstream currentFile(currentImage);
      ifstream currentTruthFile(currentTruthPath);
      

      
      /*
       * Reading in the current input values
       * 
       */
      for (int j = 0; j < numInputs; j++)    //Reads in the appropriate number of inputs
      {
         currentFile >> currentInput; 
        
         inputs[i][j] = (1.0*currentInput)/(255.0);
         
      }

      /*
       * Reading in the current truth values
       * 
       */
      for (int j = 0; j < numOutputs; j++)   //Reads in the appropriate number of outputs(truth values)
      {
         currentTruthFile >> currentTruth; 
         cout << currentTruth << endl; 
         truths[i][j] = currentTruth;
      }
     
      currentInput = 0.0; 
      currentTruth = 0.0; 
      currentFile.close();
      currentTruthFile.close();
      


      


   } // for (int i = 0; i < numTrain; i++)

   
   //cout << numOutputs; 
   

   return; 

}    // void Reader::readTrainingData(ifstream& fileIn)


/*
 * If the user inputs weights as part of the file, this method
 * reads in those weights given by the dimensions of the weights array
 * @param fileIn the file input stream passed by reference
 */
void Reader::readWeights(ifstream& fileIn)
{
   string weightsFile; 
   string throwaway; 
   getline(fileIn, throwaway);
   getline(fileIn, weightsFile);
   
   std::string const weightsFileName = weightsFile + "\0"; 
   ifstream weightsFileIn;
   weightsFileIn.open("finalweights");
   cout << "Weights from from " << weightsFile << endl << endl; 

   for (int n = 0; n < numLayers - 1; n++)         //Iterating over the layers
   {
      for (int j = 0; j < layerSizes[n]; j++)      //Iterating over the source layer
      {
         for (int i = 0; i < layerSizes[n+1]; i++) //Iterating over the destination layer
         {
            weightsFileIn >> weightsRead[n][j][i];        //Reading in the current weight
            
         }

      }

   }
   weightsFileIn.close();


   return; 

}  //void Reader::readWeights(ifstream& fileIn)

/*
 * Returns a shallow copy of the metadata in a double array (currently)
 * of length 3. The first index has the number of training sets, the second index
 * has the hasWeights flag, and the third index gives the number of layers for the network
 */
int* Reader::getMetaData()
{
   static int metaData[4];    // Static so that it does not disappear
                              // at the end of this block
   metaData[0] = numTrain; 
   metaData[1] = hasWeights; 
   metaData[2] = numLayers; 
   metaData[3] = testOrTrain; 

   return metaData; 

}

/*
 * Returns the array containing the size of each layer of the network 
 */
int* Reader::getLayerSizes()
{
   return layerSizes;
}

/*
 * Returns the weights array shaped by the reader. If no weights were read in, 
 * hasWeights will be zero and the network MUST populate the weights randomly or else
 * it will a bunch of uninitialized double values. 
 */
vector<vector<vector<double> > > Reader::getWeights()
{
   return weightsRead; 
}

/*
 * Returns a copy of the training data
 */
double** Reader::getTrainingData()
{
   return inputs; 

}

/*
 * Returns a copy of the truth values
 * for each training set in a two dimensional pointer
 */
double** Reader::getTruths()
{
   return truths; 
}

/*
 * Exports the given weights to a file with the name of the parameter
 * @param weights the weights to export
 * @param filename the filename of the weights file. 
 */
void exportWeights(vector<vector<vector<double> > > weights, string fileName)
{
   ofstream fout(fileName);

   /*
    * Iterates over the weights array and outputs the weights
    * one by one - each "layer" of weights corresponds to a line
    * in the file. 
    */ 
   for (int n = 0; n < weights.size(); n++)             //Iterating over the layers
   {
      for (int j = 0; j < weights[n].size(); j++)       //Iterating over the source layer
      {
         for (int i = 0; i < weights[n][j].size(); i++) //Iterating over the destination layer
         {
            fout << weights[n][j][i] << " ";            //Exporting current weight to file

         }
         
      }
      fout << endl; 
   }
   fout.close();     //Closing the output stream 

   return; 

}                    //exportWeights method
