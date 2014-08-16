/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package hmm;

import DataStructures.TimeSeriesClassificationData;
import Util.MatrixDouble;
import static hmm.HMMModelTipes.*;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;

/**
 *
 * @author Пользователь
 */
public class HMM implements Serializable{

    //Variables for all the HMMs
    protected boolean trained = false;
    protected boolean useScaling = false;
    protected int numStates = 5;			//The number of states for each model
    protected int numSymbols = 10;		//The number of symbols for each model
    protected int numInputDimensions = 0;
//    protected int numOutputDimensions = 0;
//    protected int numTrainingIterationsToConverge;
    protected int numClasses;
    protected int predictedClassLabel;
    protected HMMModelTipes modelType = LEFTRIGHT;         //Set if the model is ERGODIC or LEFTRIGHT
    protected int delta = 1;				//The number of states a model can move to in a LeftRight model
    protected int maxNumIter = 100;		//The maximum number of iter allowed during the full training
    protected int numRandomTrainingIterations;
    protected double minImprovement = 1.0e-2;  //The minimum improvement value for each model during training
    protected boolean useNullRejection = false;

    protected double bestDistance;
    protected double maxLikelihood;
    protected double[] classLikelihoods = new double[0];
    protected double[] classDistances = new double[0];

    protected double[] nullRejectionThresholds;

    int[] classLabels = null;
    ArrayList<HiddenMarkovModel> models = new ArrayList<HiddenMarkovModel>();

    //static RegisterClassifierModule< HMM> registerModule;
    public boolean setNumStates(int numStates) {

        if (numStates > 0) {
            this.numStates = numStates;
            return true;
        }

        System.err.println("setNumStates( int numStates) - Num states must be greater than zero!");
        return false;
    }

    public boolean setNumSymbols(int numSymbols) {

        if (numSymbols > 0) {
            this.numSymbols = numSymbols;
            return true;
        }

        System.err.println("setNumSymbols( int numSymbols) - Num symbols must be greater than zero!");
        return false;
    }

    public boolean setModelType(HMMModelTipes modelType) {

        if (modelType == ERGODIC || modelType == LEFTRIGHT) {
            this.modelType = modelType;
            return true;
        }

        System.err.println("setModelType( int modelType) - Unknown model type!");
        return false;
    }

    public boolean setDelta(int delta) {

        if (delta > 0) {
            this.delta = delta;
            return true;
        }

        System.err.println("setDelta( int delta) - Delta must be greater than zero!");
        return false;
    }

    public boolean setMaxNumIterations(int maxNumIter) {

        if (maxNumIter > 0) {
            this.maxNumIter = maxNumIter;
            return true;
        }

        System.err.println("setMaxNumIterations( int maxNumIter) - The maximum number of iterations must be greater than zero!");
        return false;
    }

    public boolean setNumRandomTrainingIterations(int numRandomTrainingIterations) {

        if (numRandomTrainingIterations > 0) {
            this.numRandomTrainingIterations = numRandomTrainingIterations;
            return true;
        }

        System.err.println("setMaxNumIterations( int maxNumIter) - The number of random training iterations must be greater than zero!");
        return false;
    }

    public boolean setMinImprovement(double minImprovement) {

        if (minImprovement > 0) {
            this.minImprovement = minImprovement;
            return true;
        }

        System.err.println("setMinImprovement( double minImprovement) - Minimum improvement must be greater than zero!");
        return false;
    }

    public boolean train(TimeSeriesClassificationData trainingData) {

        clear();

        if (trainingData.getNumSamples() == 0) {
            System.err.println("train_(TimeSeriesClassificationData &trainingData) - There are no training samples to train the HMM classifer!");
            return false;
        }

        if (trainingData.getNumDimensions() != 1) {
            System.err.println("train_(TimeSeriesClassificationData &trainingData) - The number of dimensions in the training data must be 1. If your training data is not 1 dimensional then you must quantize the training data using one of the GRT quantization algorithms");
            return false;
        }

        //Reset the HMM
        numInputDimensions = trainingData.getNumDimensions();
        numClasses = trainingData.getNumClasses();
        models.ensureCapacity(numClasses);
        classLabels = new int[numClasses];

        //Init the models
        for (int k = 0; k < numClasses; k++) {
            models.add(k, new HiddenMarkovModel());
            models.get(k).resetModel(numStates, numSymbols, modelType, delta);
            models.get(k).maxNumIter = maxNumIter;
            models.get(k).minImprovement = minImprovement;
        }

        //Train each of the models
        for (int k = 0; k < numClasses; k++) {
            //Get the class ID of this gesture
            int classID = trainingData.getClassTracker().get(k).classLabel;
            classLabels[k] = classID;

            //Convert this classes training data into a list of observation sequences
            TimeSeriesClassificationData classData = trainingData.getClassData(classID);
            int[][] observationSequences = null;
            if ((observationSequences = convertDataToObservationSequence(classData)) == null) {
                return false;
            }

            //Train the model
            if (!models.get(k).train(observationSequences)) {
                System.err.println("train_(TimeSeriesClassificationData &trainingData) - Failed to train HMM for class " + classID);
                return false;
            }
        }

        //Compute the rejection thresholds
        nullRejectionThresholds = new double[numClasses];

        for (int k = 0; k < numClasses; k++) {
            //Get the class ID of this gesture
            int classID = trainingData.getClassTracker().get(k).classLabel;
            classLabels[k] = classID;

            //Convert this classes training data into a list of observation sequences
            TimeSeriesClassificationData classData = trainingData.getClassData(classID);
            int[][] observationSequences = null;
            if ((observationSequences = convertDataToObservationSequence(classData)) == null) {
                return false;
            }

            //Test the model
            double loglikelihood = 0;
            double avgLoglikelihood = 0;
            for (int i = 0; i < observationSequences.length; i++) {
                loglikelihood = models.get(k).predict(observationSequences[i]);
                avgLoglikelihood += Math.abs(loglikelihood);
            }
            nullRejectionThresholds[k] = -(avgLoglikelihood / (double) observationSequences.length);
        }

        //Flag that the model has been trained
        trained = true;

        return true;
    }

    public int[][] convertDataToObservationSequence(TimeSeriesClassificationData classData) {

        int[][] observationSequences = new int[classData.getNumSamples()][];

        for (int i = 0; i < classData.getNumSamples(); i++) {
            MatrixDouble timeseries = classData.get(i).getData();
            observationSequences[i] = new int[timeseries.getNumRows()];
            for (int j = 0; j < timeseries.getNumRows(); j++) {
                if (timeseries.get(j, 0) >= numSymbols) {
                    System.err.println("train(TimeSeriesClassificationData &trainingData) - Found an observation sequence with a value outside of the symbol range! Value: " + timeseries.get(j, 0));
                    return null;
                }
                observationSequences[i][j] = (int) timeseries.get(j, 0);
            }
        }

        return observationSequences;
    }

    private void clear() {
        models.clear();
    }

    public boolean predict(MatrixDouble timeseries) {
        if (timeseries.getNumCols() != 1) {
            System.err.println("predict_(MatrixDouble &timeseries) The number of columns in the input matrix must be 1. It is: " + timeseries.getNumCols());
            return false;
        }

        //Covert the matrix double to observations
        final int M = timeseries.getNumRows();
        int[] observationSequence = new int[M];

        for (int i = 0; i < M; i++) {
            observationSequence[i] = (int) timeseries.get(i, 0);

            if (observationSequence[i] >= numSymbols) {
                System.err.println("predict_(VectorDouble &inputVector) - The new observation is not a valid symbol! It should be in the range [0 numSymbols-1]");
                return false;
            }
        }

        if (classLikelihoods.length != numClasses) {
            classLikelihoods = new double[numClasses];
        }
        if (classDistances.length != numClasses) {
            classDistances = new double[numClasses];
        }

        bestDistance = -99e+99;
        int bestIndex = 0;
        double sum = 0;
        for (int k = 0; k < numClasses; k++) {
            classDistances[k] = models.get(k).predict(observationSequence);

            //Set the class likelihood as the antilog of the class distances
            classLikelihoods[k] = antilog(classDistances[k]);

            //The loglikelihood values are negative so we want the values closest to 0
            if (classDistances[k] > bestDistance) {
                bestDistance = classDistances[k];
                bestIndex = k;
            }

            sum += classLikelihoods[k];
        }

        //Turn the class distances into proper likelihoods
        for (int k = 0; k < numClasses; k++) {
            classLikelihoods[k] /= sum;
        }

        maxLikelihood = classLikelihoods[ bestIndex];
        predictedClassLabel = classLabels[ bestIndex];

        if (useNullRejection) {
            if (maxLikelihood > nullRejectionThresholds[ bestIndex]) {
                predictedClassLabel = classLabels[ bestIndex];
            } else {
                predictedClassLabel = 0;
            }
        }

        return true;
    }

    public int getPredictedClassLabel() {
        if (trained) {
            return predictedClassLabel;
        }
        return 0;
    }

    public double[] getClassLikelihoods() {
        if (trained) {
            return classLikelihoods;
        }
        return null;
    }

    public double[] getClassDistances() {
        if (trained) {
            return classDistances;
        }
        return null;
    }

    public double getMaximumLikelihood() {
        if (trained) {
            return maxLikelihood;
        }
        return 0;
    }

    private double antilog(double d) {
        return Math.exp(d);
    }

    public boolean loadModelFromFile(String file) throws IOException {
        clear();

        BufferedReader reader;

        try {
            reader = new BufferedReader(new FileReader(file));
        } catch (FileNotFoundException ex) {
            System.err.println("loadDatasetFromFile(String filename) - FILE NOT OPEN!");
            return false;
        }

        String word;
        double value;

        //Find the file type header
        word = reader.readLine();
        if (!word.contains("HMM_MODEL_FILE_V2.0")) {
            System.err.println("loadModelFromFile( fstream &file ) - Could not find Model File Header!");
            return false;
        }

        //Load the trained state
        word = reader.readLine();
        if (!word.contains("Trained:")) {
            System.err.println("loadBaseSettingsFromFile(fstream &file) - Failed to read Trained header!");
            return false;
        }

        String[] buf = word.split(" ");
        trained = (Integer.parseInt(buf[1]) == 1);

        //Load the scaling state
        word = reader.readLine();
        if (!word.contains("UseScaling:")) {
            System.err.println("loadBaseSettingsFromFile(fstream &file) - Failed to read UseScaling header!");
            return false;
        }
        buf = word.split(" ");
        useScaling = (Integer.parseInt(buf[1]) == 1);

        //Load the NumInputDimensions
        word = reader.readLine();
        if (!word.contains("NumInputDimensions:")) {
            System.err.println("loadBaseSettingsFromFile(fstream &file) - Failed to read NumInputDimensions header!");
            return false;
        }
        buf = word.split(" ");
        numInputDimensions = Integer.parseInt(buf[1]);

        //Load the NumOutputDimensions
        word = reader.readLine();
        if (!word.contains("NumOutputDimensions:")) {
            System.err.println("loadBaseSettingsFromFile(fstream &file) - Failed to read NumOutputDimensions header!");
            return false;
        }
//        buf = word.split(" ");
//        numOutputDimensions = Integer.parseInt(buf[1]);

        //Load the numTrainingIterationsToConverge
        word = reader.readLine();
        if (!word.contains("NumTrainingIterationsToConverge:")) {
            System.err.println("loadBaseSettingsFromFile(fstream &file) - Failed to read NumTrainingIterationsToConverge header!");
            return false;
        }

//        buf = word.split(" ");
//        numTrainingIterationsToConverge = Integer.parseInt(buf[1]);
        //Load the MinNumEpochs
        word = reader.readLine();
        if (!word.contains("MinNumEpochs:")) {
            System.err.println("loadBaseSettingsFromFile(fstream &file) - Failed to read MinNumEpochs header!");
            return false;
        }

//                buf = word.split(" ");
//        min = Integer.parseInt(buf[1]);
        //Load the maxNumEpochs
        word = reader.readLine();
        if (!word.contains("MaxNumEpochs:")) {
            System.err.println("loadBaseSettingsFromFile(fstream &file) - Failed to read MaxNumEpochs header!");
            return false;
        }
//                buf = word.split(" ");        numStates = Integer.parseInt(buf[1]);maxNumEpochs;

        //Load the ValidationSetSize
        word = reader.readLine();
        if (!word.contains("ValidationSetSize:")) {
            System.err.println("loadBaseSettingsFromFile(fstream &file) - Failed to read ValidationSetSize header!");
            return false;
        }
//                buf = word.split(" ");        numStates = Integer.parseInt(buf[1]);validationSetSize;

        //Load the LearningRate
        word = reader.readLine();
        if (!word.contains("LearningRate:")) {
            System.err.println("loadBaseSettingsFromFile(fstream &file) - Failed to read LearningRate header!");
            return false;
        }
//                buf = word.split(" ");        numStates = Integer.parseInt(buf[1]);learningRate;

        //Load the MinChange
        word = reader.readLine();
        if (!word.contains("MinChange:")) {
            System.err.println("loadBaseSettingsFromFile(fstream &file) - Failed to read MinChange header!");
            return false;
        }
//                buf = word.split(" ");        numStates = Integer.parseInt(buf[1]);minChange;

        //Load the UseValidationSet
        word = reader.readLine();
        if (!word.contains("UseValidationSet:")) {
            System.err.println("loadBaseSettingsFromFile(fstream &file) - Failed to read UseValidationSet header!");
            return false;
        }
//                buf = word.split(" ");        numStates = Integer.parseInt(buf[1]);useValidationSet;

        //Load the RandomiseTrainingOrder
        word = reader.readLine();
        if (!word.contains("RandomiseTrainingOrder:")) {
            System.err.println("loadBaseSettingsFromFile(fstream &file) - Failed to read RandomiseTrainingOrder header!");
            return false;
        }
//                buf = word.split(" ");        numStates = Integer.parseInt(buf[1]);randomiseTrainingOrder;

        //Load if the number of clusters
        word = reader.readLine();
        if (!word.contains("UseNullRejection:")) {
            System.err.println("loadBaseSettingsFromFile(fstream &file) - Failed to read UseNullRejection header!");
            clear();
            return false;
        }
        buf = word.split(" ");
        useNullRejection = (Integer.parseInt(buf[1]) == 1);

        //Load if the classifier mode
        word = reader.readLine();
        if (!word.contains("ClassifierMode:")) {
            System.err.println("loadBaseSettingsFromFile(fstream &file) - Failed to read ClassifierMode header!");
            clear();
            return false;
        }
//                buf = word.split(" ");        numStates = Integer.parseInt(buf[1]);classifierMode;
        //Load if the null rejection coeff
        word = reader.readLine();
        if (!word.contains("NullRejectionCoeff:")) {
            System.err.println("loadBaseSettingsFromFile(fstream &file) - Failed to read NullRejectionCoeff header!");
            clear();
            return false;
        }
//                buf = word.split(" ");        numStates = Integer.parseInt(buf[1]);nullRejectionCoeff;

        //If the model is trained then load the model settings
        if (trained) {
            //Load the number of classes
            word = reader.readLine();
            if (!word.contains("NumClasses:")) {
                System.err.println("loadBaseSettingsFromFile(fstream &file) - Failed to read NumClasses header!");
                clear();
                return false;
            }
            buf = word.split(" ");
            numClasses = Integer.parseInt(buf[1]);

            //Load the null rejection thresholds
            word = reader.readLine();
            if (!word.contains("NullRejectionThresholds:")) {
                System.err.println("loadBaseSettingsFromFile(fstream &file) - Failed to read NullRejectionThresholds header!");
                clear();
                return false;
            }
            nullRejectionThresholds = new double[numClasses];
            buf = word.split(" ");
            System.out.println(word);
            for (int i = 0; i < nullRejectionThresholds.length; i++) {

                nullRejectionThresholds[i] = Integer.parseInt(buf[i + 1]);
            }
            //Load the class labels
            word = reader.readLine();
            if (!word.contains("ClassLabels:")) {
                System.err.println("loadBaseSettingsFromFile(fstream &file) - Failed to read ClassLabels header!");
                clear();
                return false;
            }
            classLabels = new int[numClasses];
            buf = word.split(" ");
            for (int i = 0; i < classLabels.length; i++) {
                classLabels[i] = Integer.parseInt(buf[i + 1]);
            }

            if (useScaling) {
                //Load if the Ranges
                word = reader.readLine();
                if (!word.contains("Ranges:")) {
                    System.err.println("loadClustererSettingsFromFile(fstream &file) - Failed to read Ranges header!");
                    clear();
                    return false;
                }
//                ranges.resize(numInputDimensions);
//
//                for (int i = 0; i < ranges.size(); i++) {
//                            buf = word.split(" ");        numStates = Integer.parseInt(buf[1]);ranges[i].minValue;
//                            buf = word.split(" ");        numStates = Integer.parseInt(buf[1]);ranges[i].maxValue;
//                }
            }
        }
        word = reader.readLine();
        if (!word.contains("NumStates:")) {
            System.err.println("loadModelFromFile( fstream &file ) - Could not find NumStates.");
            return false;
        }
        buf = word.split(" ");
        numStates = Integer.parseInt(buf[1]);

        word = reader.readLine();
        if (!word.contains("NumSymbols:")) {
            System.err.println("loadModelFromFile( fstream &file ) - Could not find NumSymbols.");
            return false;
        }
        buf = word.split(" ");
        numSymbols = Integer.parseInt(buf[1]);

        word = reader.readLine();
        if (!word.contains("ModelType:")) {
            System.err.println("loadModelFromFile( fstream &file ) - Could not find ModelType.");
            return false;
        }
        buf = word.split(" ");
        modelType = Integer.parseInt(buf[1]) == 0 ? ERGODIC : LEFTRIGHT;

        word = reader.readLine();
        if (!word.contains("Delta:")) {
            System.err.println("loadModelFromFile( fstream &file ) - Could not find Delta.");
            return false;
        }
        buf = word.split(" ");
        delta = Integer.parseInt(buf[1]);

        word = reader.readLine();
        if (!word.contains("NumRandomTrainingIterations:")) {
            System.err.println("loadModelFromFile( fstream &file ) - Could not find NumRandomTrainingIterations.");
            return false;
        }
        buf = word.split(" ");
        numRandomTrainingIterations = Integer.parseInt(buf[1]);;

        //If the HMM has been trained then load the models
        if (trained) {

            //Resize the buffer
            models.ensureCapacity(numClasses);

            //Load each of the K classes
            for (int k = 0; k < numClasses; k++) {
                int modelID;
                word = reader.readLine();
                if (!word.contains("Model_ID:")) {
                    System.err.println("loadModelFromFile( fstream &file ) - Could not find model ID for the " + (k + 1) + "th model");
                    return false;
                }
                buf = word.split(" ");
                modelID = Integer.parseInt(buf[1]);

                if (modelID - 1 != k) {
                    System.err.println("loadModelFromFile( fstream &file ) - Model ID does not match the current class ID for the " + (k + 1) + "th model");
                    return false;
                }
                word = reader.readLine();
                if (!word.contains("NumStates:")) {
                    System.err.println("loadModelFromFile( fstream &file ) - Could not find the NumStates for the " + (k + 1) + "th model");
                    return false;
                }
                buf = word.split(" ");
                models.add(k, new HiddenMarkovModel());
                models.get(k).numStates = Integer.parseInt(buf[1]);

                System.out.println("Num states: " + numStates);
                
                word = reader.readLine();
                if (!word.contains("NumSymbols:")) {
                    System.err.println("loadModelFromFile( fstream &file ) - Could not find the NumSymbols for the " + (k + 1) + "th model");
                    return false;
                }
                buf = word.split(" ");
                models.get(k).numSymbols = Integer.parseInt(buf[1]);

                word = reader.readLine();
                if (!word.contains("ModelType:")) {
                    System.err.println("loadModelFromFile( fstream &file ) - Could not find the modelType for the " + (k + 1) + "th model");
                    return false;
                }
                buf = word.split(" ");
                models.get(k).modelType = Integer.parseInt(buf[1]) == 0 ? ERGODIC : LEFTRIGHT;

                word = reader.readLine();
                if (!word.contains("Delta:")) {
                    System.err.println("loadModelFromFile( fstream &file ) - Could not find the Delta for the " + (k + 1) + "th model");
                    return false;
                }
                buf = word.split(" ");
                models.get(k).delta = Integer.parseInt(buf[1]);

                word = reader.readLine();
                if (!word.contains("Threshold:")) {
                    System.err.println("loadModelFromFile( fstream &file ) - Could not find the Threshold for the " + (k + 1) + "th model");
                    return false;
                }
                buf = word.split(" ");
                models.get(k).cThreshold = Integer.parseInt(buf[1]);

                word = reader.readLine();
                if (!word.contains("NumRandomTrainingIterations:")) {
                    System.err.println("loadModelFromFile( fstream &file ) - Could not find the numRandomTrainingIterations for the " + (k + 1) + "th model.");
                    return false;
                }
                buf = word.split(" ");
                models.get(k).numRandomTrainingIterations = Integer.parseInt(buf[1]);

                word = reader.readLine();
                if (!word.contains("MaxNumIter:")) {
                    System.err.println("loadModelFromFile( fstream &file ) - Could not find the MaxNumIter for the " + (k + 1) + "th model.");
                    return false;
                }
                buf = word.split(" ");
                models.get(k).maxNumIter = Integer.parseInt(buf[1]);

                models.get(k).a.resize(models.get(k).numStates, models.get(k).numStates);
                models.get(k).b.resize(models.get(k).numStates, models.get(k).numSymbols);
                models.get(k).pi = new double[models.get(k).numStates];

                word = reader.readLine();
                //Load the A, B and Pi matrices
                if (!word.contains("A:")) {
                    System.err.println("loadModelFromFile( fstream &file ) - Could not find the A matrix for the " + (k + 1) + "th model.");
                    return false;
                }

                //Load A
                models.get(k).a.resize(models.get(k).numStates, models.get(k).numStates);
                for (int i = 0; i < models.get(k).numStates; i++) {
                    word = reader.readLine();
                    System.out.println(word);
                    buf = word.split("\t");
                    for (int j = 0; j < models.get(k).numStates; j++) {
                        value = Double.parseDouble(buf[j]);
                        models.get(k).a.set(value, i, j);
                    }
                }
                word = reader.readLine();
                if (!word.contains("B:")) {
                    System.err.println("loadModelFromFile( fstream &file ) - Could not find the B matrix for the " + (k + 1) + "th model.");
                    return false;
                }

                //Load B
  //              word = reader.readLine();
                models.get(k).a.resize(models.get(k).numStates, models.get(k).numStates);
                for (int i = 0; i < models.get(k).numStates; i++) {
                    word = reader.readLine();
                    buf = word.split("\t");
                    for (int j = 0; j < models.get(k).numSymbols; j++) {
                        value = Double.parseDouble(buf[j]);;
                        models.get(k).b.set(value, i, j);
                    }
                }
                
                word = reader.readLine();
                if (!word.contains("Pi:")) {
                    System.err.println("loadModelFromFile( fstream &file ) - Could not find the Pi matrix for the " + (k + 1) + "th model.");
                    return false;
                }

                //Load Pi
                word = reader.readLine();
                buf = word.split("\t");
                for (int i = 0; i < models.get(k).numStates; i++) {
                    value = Double.parseDouble(buf[i]);
                    models.get(k).pi[i] = value;
                }
            }

            maxLikelihood = 0;
            bestDistance = 0;
            classLikelihoods = new double[numClasses];
            classDistances = new double[numClasses];
        }
        return true;
    }
}
