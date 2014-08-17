package KMeans;

import Util.MatrixDouble;
import java.io.Serializable;
import java.util.ArrayList;

/**
 * The KMeansQuantizer module quantizes the N-dimensional input vector to a
 * 1-dimensional discrete value. This value will be between [0 K-1], where K is
 * the number of clusters used to create the quantization model. Before you use
 * the KMeansQuantizer, you need to train a quantization model. To do this, you
 * select the number of clusters you want your quantizer to have and then give
 * it any training data as the MatrixDouble
 */
public class KMeansQuantizer implements Serializable {

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
    }

    public ArrayList<Double> getFeatureVector() {
        return featureVector;
    }

    /**
     * Sets the FeatureExtraction clear function, overwriting the base
     * FeatureExtraction function.
     *
     * @return true if the instance was reset, false otherwise
     */
    public boolean clear() {
        clusters = null;
        quantizationDistances.clear();

        return true;
    }

    /**
     * Quantizes the input value using the quantization model. The quantization
     * model must be trained first before you call this function.
     *
     * @param const VectorDouble &inputVector: the vector you want to quantize
     * @return returns the quantized value
     */
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

    /**
     * Trains the quantization model using the training dataset.
     *
     * @param MatrixDouble &trainingData: the training dataset that will be used
     * to train the quantizer
     * @return returns true if the quantizer was trained successfully, false
     * otherwise
     */
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
}
