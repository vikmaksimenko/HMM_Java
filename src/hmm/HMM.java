/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package hmm;

import DataStructures.TimeSeriesClassificationData;
import Util.MatrixDouble;
import static hmm.HMMModelTipes.*;
import java.util.ArrayList;

/**
 *
 * @author Пользователь
 */
public class HMM {

    //Variables for all the HMMs
    protected boolean trained = false;
    protected int numStates = 5;			//The number of states for each model
    protected int numSymbols = 10;		//The number of symbols for each model
    protected int numInputDimensions = 0;
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
}
