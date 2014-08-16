package KMeans;

import DataStructures.TimeSeriesClassificationData;
import Util.MatrixDouble;
import java.io.Serializable;
import java.util.ArrayList;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
/**
 *
 * @author Пользователь
 */
public class KMeansQuantizer implements Serializable {//extends FeatureExtraction{

    protected boolean trained = false;
    protected boolean featureDataReady = false;
    protected boolean initialized;
    protected int numClusters;
    protected int minNumEpochs = 0;
    protected int maxNumEpochs = 100;
    protected int numInputDimensions = 0;
    protected int numOutputDimensions = 0;
    protected double minChange = 1.0e-5;
    protected MatrixDouble clusters = new MatrixDouble();
    protected String featureExtractionType;
    protected String classType = "";
    protected ArrayList<Double> featureVector = new ArrayList<Double>();
    protected ArrayList<Double> quantizationDistances = new ArrayList<Double>();

    //protected VectorDouble quantizationDistances;
    //static RegisterFeatureExtractionModule< KMeansQuantizer > registerModule;
    /**
     * Default constructor. Initalizes the KMeansQuantizer, setting the number
     * of input dimensions and the number of clusters to use in the quantization
     * model.
     *
     * @param int numClusters: the number of quantization clusters
     */
    public KMeansQuantizer(final int numClusters) {
        this.numClusters = numClusters;
        classType = "KMeansQuantizer";
        featureExtractionType = classType;

        featureVector.add(0.0);  // КОСТЫЛЬ!!!
//
//        debugLog.setProceedingText("[DEBUG KMeansQuantizer]");
//        errorLog.setProceedingText("[ERROR KMeansQuantizer]");
//        warningLog.setProceedingText("[WARNING KMeansQuantizer]");
    }

    public int quantize(ArrayList<Double> inputVector) {
        if (!trained) {
            System.err.println("computeFeatures(const VectorDouble &inputVector) - The quantizer has not been trained!");
            return 0;
        }

        if (inputVector.size() != numInputDimensions) {
            System.err.println("computeFeatures(const VectorDouble &inputVector) - The size of the inputVector (" + inputVector.size() + ") does not match that of the filter (" + numInputDimensions + ")!");
            return 0;
        }

        //Find the minimum cluster
        double minDist = Double.MAX_VALUE;
        int quantizedValue = 0;

        for (int k = 0; k < numClusters; k++) {

            //Compute the squared Euclidean distance
            quantizationDistances.add(k, 0.0);
            for (int i = 0; i < numInputDimensions; i++) {
                double val = quantizationDistances.get(k);
                val += Math.pow(inputVector.get(i) - clusters.get(k, i), 2);
                quantizationDistances.set(k, val);

            }

            if (quantizationDistances.get(k) < minDist) {
                minDist = quantizationDistances.get(k);
                quantizedValue = k;
            }
        }
        //System.out.println("featureVector size:  " + featureVector.size());
        featureVector.set(0, (double) quantizedValue);
        featureDataReady = true;

        return quantizedValue;
    }

    public boolean train(MatrixDouble trainingData) {
        //Clear any previous model
        clear();

        //Train the KMeans model
        KMeans kmeans = new KMeans();
        kmeans.setNumClusters(numClusters);
        kmeans.setComputeTheta(true);
        kmeans.setMinChange(minChange);
        kmeans.setMinNumEpochs(minNumEpochs);
        kmeans.setMaxNumEpochs(maxNumEpochs);

        /* DEBUG */
//    System.out.println("numClusters: " + numClusters );
//    System.out.println("minChange: " + minChange );
//    System.out.println("minNumEpochs: " + minNumEpochs );
//    System.out.println("maxNumEpochs: " + maxNumEpochs );
        if (!kmeans.train(trainingData)) {
            System.err.println("train_(MatrixDouble &trainingData) - Failed to train quantizer!");
            return false;
        }

        trained = true;
        initialized = true;
        numInputDimensions = trainingData.getNumCols();
        numOutputDimensions = 1; //This is always 1 for the KMeansQuantizer
        featureVector.ensureCapacity(numOutputDimensions);
        clusters = kmeans.getClusters();
        quantizationDistances.ensureCapacity(numClusters);

        return true;
    }

    private boolean clear() {
        clusters = null;//.clear();
        quantizationDistances.clear();

        return true;
    }

    public ArrayList<Double> getFeatureVector() {
        return featureVector;
    }

}
