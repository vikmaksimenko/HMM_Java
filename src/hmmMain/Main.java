/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package hmmMain;

import DataStructures.TimeSeriesClassificationData;
import java.io.IOException;
import KMeans.KMeansQuantizer;
import Util.MatrixDouble;
import hmm.HMM;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.logging.Level;
import java.util.logging.Logger;
//import hmm.HMM;
//import Util.MatrixDouble;

/**
 *
 * @author Пользователь
 */
public class Main {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
//        //Load the training data
//        TimeSeriesClassificationData trainingData = new TimeSeriesClassificationData();
//
//        try {
//            if (!trainingData.loadDatasetFromFile("HMMTrainingDataACC1.txt")) {
//                System.err.println("ERROR: Failed to load training data!");
//                trainingData.clear();
//                return;
//            }
//        } catch (IOException ex) {
//            System.err.println(ex);
//        }

        //trainingData.printStats();
        //Remove 20% of the training data to use as test data
        TimeSeriesClassificationData testData = new TimeSeriesClassificationData();//=  trainingData.partition( 80 );
        try {
            if (!testData.loadDatasetFromFile("HMMTrainingDataACC1.txt")) {
                System.err.println("ERROR: Failed to load test data!");
                return;
            }
        } catch (IOException ex) {
            System.err.println("ERROR: Failed to load training data!");
        }
        //The input to the HMM must be a quantized discrete value
        //We therefore use a KMeansQuantizer to covert the N-dimensional continuous data into 1-dimensional discrete data
        final int NUM_SYMBOLS = 20;    // 10 - default
        KMeansQuantizer quantizer = new KMeansQuantizer(NUM_SYMBOLS);

//        //Train the quantizer using the training data
//        if (!quantizer.train(trainingData.getDataAsMatrixDouble())) {
//            System.err.println("ERROR: Failed to train quantizer!");
//            return;
//        }
//
//                /* TEST! */
//        try {
//            ObjectOutputStream os = new ObjectOutputStream(new FileOutputStream("HMMQuantizer.ser"));
//            os.writeObject(quantizer);
//        } catch (Exception ex) {
//            System.out.println(ex);
//        }
        try {
            ObjectInputStream is = new ObjectInputStream(new FileInputStream("HMMQuantizer.ser"));
            quantizer = (KMeansQuantizer)is.readObject();
        } catch (Exception ex) {
            System.out.println(ex);
        }
        
//        //Quantize the training data
//        TimeSeriesClassificationData quantizedTrainingData = new TimeSeriesClassificationData(1);
//
//        for (int i = 0; i < trainingData.getNumSamples(); i++) {
//
//            int classLabel = trainingData.get(i).getClassLabel();
//            MatrixDouble quantizedSample = new MatrixDouble();
//
//            for (int j = 0; j < trainingData.get(i).getLength(); j++) {
//
//                quantizer.quantize(trainingData.get(i).getData().getRowVector(j));
//                quantizedSample.push_back(quantizer.getFeatureVector());
//            }
//
//            if (!quantizedTrainingData.addSample(classLabel, quantizedSample)) {
//                System.out.println("ERROR: Failed to quantize training data!");
//                return;
//            }
//
//        }

//        quantizedTrainingData.printStats();
        //Create a new HMM instance
        HMM hmm = new HMM();

//        //Set the number of states in each model
//        hmm.setNumStates(4); // default 4
//
//        //Set the number of symbols in each model, this must match the number of symbols in the quantizer
//        hmm.setNumSymbols(NUM_SYMBOLS);
//
//        //Set the HMM model type to LEFTRIGHT with a delta of 1
//        hmm.setModelType(LEFTRIGHT);
//        hmm.setDelta(1);
//
//        //Set the training parameters
//        hmm.setMinImprovement(1.0e-5);
//        hmm.setMaxNumIterations(100); // default 100
//        hmm.setNumRandomTrainingIterations(20);
//
//        //Train the HMM model
//        if (!hmm.train(quantizedTrainingData)) {
//            System.out.println("ERROR: Failed to train the HMM model!");
//            return;
//        }
//    //Save the HMM quantiazer to a file
//    if( !quantizer.saveModelToFile( "HMMQuantizer.txt" ) ){
//        System.out.println("ERROR: Failed to save the quantizer to a file!");
//        return;
//    }
//    //Load the HMM quantizer from a file
//    if( !quantizer.loadModelFromFile( "HMMQuantizer.txt" ) ){
//        System.out.println("ERROR: Failed to load the quantizer from a file!");
//        return;
//    }
        //Save the HMM model to a file
//        if (!hmm.saveModelToFile("HMMModel.txt")) {
//            System.out.println("ERROR: Failed to save the model to a file!");
//            return;
//        }
//
        //Load the HMM model from a file
        try {
            System.out.println("In try");
            hmm.loadModelFromFile("HMMModel.txt");
        } catch (IOException e) {
            System.err.println("ERROR: Failed to load the model from a file! " + e);
            return;
        }

        /* TEST! */
        try {
            ObjectOutputStream os = new ObjectOutputStream(new FileOutputStream("HMMModel.ser"));
            os.writeObject(hmm);
        } catch (Exception ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        }
        try {
            ObjectInputStream is = new ObjectInputStream(new FileInputStream("HMMModel.ser"));
            hmm = (HMM)is.readObject();
        } catch (Exception ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        }
            
        
        //Quantize the test data
        TimeSeriesClassificationData quantizedTestData = new TimeSeriesClassificationData(1);

        for (int i = 0; i < testData.getNumSamples(); i++) {

            int classLabel = testData.get(i).getClassLabel();
            MatrixDouble quantizedSample = new MatrixDouble();

            for (int j = 0; j < testData.get(i).getLength(); j++) {
                quantizer.quantize(testData.get(i).getData().getRowVector(j));
                quantizedSample.push_back(quantizer.getFeatureVector());
            }

            if (!quantizedTestData.addSample(classLabel, quantizedSample)) {
                System.out.println("ERROR: Failed to quantize training data!");
                return;
            }
        }

        //Compute the accuracy of the HMM models using the test data
        double numCorrect = 0;
        double numTests = 0;
        for (int i = 0; i < quantizedTestData.getNumSamples(); i++) {

            int classLabel = quantizedTestData.get(i).getClassLabel();
            hmm.predict(quantizedTestData.get(i).getData());

            if (classLabel == hmm.getPredictedClassLabel()) {
                numCorrect++;
            }
            numTests++;

            double[] classLikelihoods = hmm.getClassLikelihoods();
            double[] classDistances = hmm.getClassDistances();
            System.out.println("ClassLabel: " + classLabel);
            System.out.println(" PredictedClassLabel: " + hmm.getPredictedClassLabel());
            System.out.println(" MaxLikelihood: " + hmm.getMaximumLikelihood());
            System.out.println("  ClassLikelihoods: ");
            for (int k = 0; k < classLikelihoods.length; k++) {
                System.out.println(classLikelihoods[k]);// "\t";
            }
            System.out.println("ClassDistances: ");
            for (int k = 0; k < classDistances.length; k++) {
                System.out.println(classDistances[k]); //+ "\t");
            }
        }
        System.out.println("Test Accuracy: " + (numCorrect / numTests * 100.0));
    }
}
