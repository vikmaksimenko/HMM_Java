package hmmMain;

import DataStructures.TimeSeriesClassificationData;
import java.io.IOException;
import KMeans.KMeansQuantizer;
import Util.MatrixDouble;
import hmm.HMM;
import java.io.FileInputStream;
import java.io.ObjectInputStream;

public class Main {

    public static void main(String[] args) {

        //The input to the HMM must be a quantized discrete value
        //We therefore use a KMeansQuantizer to covert the N-dimensional continuous data into 1-dimensional discrete data
        final int NUM_SYMBOLS = 20;    // 10 - default
        KMeansQuantizer quantizer = new KMeansQuantizer(NUM_SYMBOLS);

        // Load quantizer from serialized file
        try {
            ObjectInputStream is = new ObjectInputStream(new FileInputStream("HMMQuantizer.ser"));
            quantizer = (KMeansQuantizer) is.readObject();
        } catch (Exception ex) {
            System.err.println("ERROR: Failed to load quantizer! " + ex);
        }

        //Create a new HMM instance
        HMM hmm = new HMM();

        //Load the HMM model from a file
        try {
            ObjectInputStream is = new ObjectInputStream(new FileInputStream("HMMModel.ser"));
            hmm = (HMM) is.readObject();
        } catch (Exception ex) {
            System.err.println("ERROR: Failed to load hmm! " + ex);
        }

        //Load recognition data
        TimeSeriesClassificationData testData = new TimeSeriesClassificationData();
        try {
            if (!testData.loadDatasetFromFile("HMMTrainingDataACC1.txt")) {
                System.err.println("ERROR: Failed to load test data!");
                return;
            }
        } catch (IOException ex) {
            System.err.println("ERROR: Failed to load training data! " + ex);
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

        // Recognizing
        for (int i = 0; i < quantizedTestData.getNumSamples(); i++) {
            hmm.predict(quantizedTestData.get(i).getData());
            printLabel(hmm.getPredictedClassLabel());
        }
    }

    //Converting class id into label
    static void printLabel(int val) {
        System.out.print("Figure is ");
        switch (val) {
            case 1:
                System.out.print("circle");
                break;
            case 2:
                System.out.print("clock");
                break;
            case 3:
                System.out.print("pi");
                break;
            case 4:
                System.out.print("shield");
                break;
            case 5:
                System.out.print("triangle");
                break;
            case 6:
                System.out.print("v");
                break;
            case 7:
                System.out.print("z");
                break;
        }
    }
}
