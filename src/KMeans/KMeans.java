package KMeans;

import Util.MatrixDouble;
import Util.MinMax;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Random;

/**
 * This class implements the KMeans clustering algorithm.
 */
class KMeans implements Serializable {

    protected boolean trained = false;
    protected boolean computeTheta = true;
    protected int numClusters = 10;
    protected int minNumEpochs = 5;
    protected int maxNumEpochs = 1000;
    protected int numTrainingSamples = 0;            ///<Number of training examples
    protected int nchg = 0;                          ///<Number of values changes
    protected double finalTheta = 0;
    protected double minChange = 1.0e-5;
    protected MatrixDouble clusters = new MatrixDouble();

    protected ArrayList<Double> thetaTracker = new ArrayList<Double>();
    protected ArrayList<MinMax> ranges = new ArrayList<MinMax>();

    protected int[] assign; // = new ArrayList<Integer>();
    protected int[] count; //= new ArrayList<Integer>();

    protected boolean useScaling;
    protected boolean converged;
    protected int numInputDimensions;
    protected int numTrainingIterationsToConverge;

    public KMeans(int numClusters, int minNumEpochs, int maxNumEpochs, double minChange, boolean computeTheta) {
        this.numClusters = numClusters;
        this.minNumEpochs = minNumEpochs;
        this.maxNumEpochs = maxNumEpochs;
        this.minChange = minChange;
        this.computeTheta = computeTheta;
    }

    public KMeans() {
    }

    MatrixDouble getClusters() {
        return clusters;
    }

    public void setComputeTheta(boolean computeTheta) {
        this.computeTheta = computeTheta;
    }

    public void setNumClusters(int numClusters) {
        this.numClusters = numClusters;
    }

    public void setMinNumEpochs(int minNumEpochs) {
        this.minNumEpochs = minNumEpochs;
    }

    public void setMaxNumEpochs(int maxNumEpochs) {
        this.maxNumEpochs = maxNumEpochs;
    }

    public void setMinChange(double minChange) {
        this.minChange = minChange;
    }

    boolean train(MatrixDouble data) {
        trained = false;

        if (numClusters == 0) {
            System.err.println("train_(MatrixDouble &data) - Failed to train model. NumClusters is zero!");
            return false;
        }

        if (data.getNumRows() == 0 || data.getNumCols() == 0) {
            System.err.println("train_(MatrixDouble &data) - The number of rows or columns in the data is zero!");
            return false;
        }

        numTrainingSamples = data.getNumRows();
        numInputDimensions = data.getNumCols();

        clusters.resize(numClusters, numInputDimensions);
        assign = new int[numTrainingSamples];
        count = new int[numClusters];

        //Randomly pick k data points as the starting clusters
        int[] randIndexs = new int[numTrainingSamples];
        for (int i = 0; i < numTrainingSamples; i++) {
            randIndexs[i] = i;
        }

        Random rand = new Random();

        for (int i = 0; i < numTrainingSamples; i++) {
            int change = i + rand.nextInt(numTrainingSamples - i);

            int temp = randIndexs[i];
            randIndexs[i] = randIndexs[change];
            randIndexs[change] = temp;
        }

        //Copy the clusters
        for (int k = 0; k < numClusters; k++) {
            for (int j = 0; j < numInputDimensions; j++) {

                double val = data.get(randIndexs[k], j);
                clusters.set(val, k, j);
            }
        }

        try {
            clusters.loadDataTxt("clusters.txt");
        } catch (Exception ex) {
            System.err.println(ex);
        }

        return trainModel(data);
    }

    /**
     * This is the main training algorithm for training a KMeans model. You
     * should only call this function if you have manually set the clusters,
     * otherwise you should use any of the train or train_ in functions.
     *
     * @param MatrixDouble &trainingData: the training data that will be used to
     * train the ML model
     * @return returns true if the model was successfully trained, false
     * otherwise
     */
    public boolean trainModel(MatrixDouble data) {

        if (numClusters == 0) {
            System.err.println("trainModel(MatrixDouble &data) - Failed to train model. NumClusters is zero!");
            return false;
        }

        if (clusters.getNumRows() != numClusters) {
            System.err.println("trainModel(MatrixDouble &data) - Failed to train model. The number of rows in the cluster matrix does not match the number of clusters! You should need to initalize the clusters matrix first before calling this function!");
            return false;
        }

        if (clusters.getNumCols() != numInputDimensions) {
            System.err.println("trainModel(MatrixDouble &data) - Failed to train model. The number of columns in the cluster matrix does not match the number of input dimensions! You should need to initalize the clusters matrix first before calling this function!");
            return false;
        }

        int currentIter = 0;
        int numChanged = 0;
        boolean keepTraining = true;
        double theta = 0;
        double lastTheta = 0;
        double delta = 0;
        double startTime = 0;
        thetaTracker.clear();
        finalTheta = 0;
        numTrainingIterationsToConverge = 0;
        trained = false;
        converged = false;

        //Scale the data if needed
        ranges = data.getRanges();
        if (useScaling) {
            data.scale(0.0, 1.0);
        }

        //Init the assign and count vectors
        //Assign is set to K+1 so that the nChanged values in the eStep at the first iteration will be updated correctly
        for (int m = 0; m < numTrainingSamples; m++) {
            assign[m] = numClusters + 1;
        }
        for (int k = 0; k < numClusters; k++) {
            count[k] = 0;
        }

        //Run the training loop
        while (keepTraining) {
            startTime = System.currentTimeMillis();

            //Compute the E step
            numChanged = estep(data);

            //Compute the M step
            mstep(data);

            //Update the iteration counter
            currentIter++;

            //Compute theta if needed
            if (computeTheta) {
                theta = calculateTheta(data);
                delta = lastTheta - theta;
                lastTheta = theta;
            } else {
                theta = delta = 0;
            }

            //Check convergance
            if (numChanged == 0 && currentIter > minNumEpochs) {
                converged = true;
                keepTraining = false;
            }
            if (currentIter >= maxNumEpochs) {
                keepTraining = false;
            }

            if (Math.abs(delta) < minChange && computeTheta && currentIter > minNumEpochs) {
                converged = true;
                keepTraining = false;
            }
            if (computeTheta) {
                thetaTracker.add(theta);
            }
            System.out.println("Epoch: " + currentIter + "/" + maxNumEpochs);
            System.out.println(" Epoch time: " + (System.currentTimeMillis() - startTime) / 1000.0 + " seconds");
            System.out.println(" Theta: " + theta + " Delta: " + delta);
        }
        System.out.println("Model Trained at epoch: " + currentIter + " with a theta value of: " + theta);

        finalTheta = theta;
        numTrainingIterationsToConverge = currentIter;
        trained = true;

        return true;
    }

    protected int estep(MatrixDouble data) {
        int k, m, n, kmin;
        double dmin, d;
        nchg = 0;
        kmin = 0;

        //Reset Count
        for (k = 0; k < numClusters; k++) {
            count[k] = 0;
        }

        //Search for the closest center and reasign if needed
        for (m = 0; m < numTrainingSamples; m++) {
            dmin = 9.99e+99; //Set dmin to a really big value
            for (k = 0; k < numClusters; k++) {
                d = 0.0;
                for (n = 0; n < numInputDimensions; n++) {
                    d += Math.pow(data.get(m, n) - clusters.get(k, n), 2);
                }
                if (d <= dmin) {
                    dmin = d;
                    kmin = k;
                }
            }
            if (kmin != assign[m]) {
                nchg++;
                assign[m] = kmin;
            }
            count[kmin]++;
        }
        return nchg;
    }

    protected void mstep(MatrixDouble data) {
        int n, k, m;

        //Reset means to zero
        for (k = 0; k < numClusters; k++) {
            for (n = 0; n < numInputDimensions; n++) {
                clusters.set(0.0, k, n);
            }
        }

        //Get new mean by adding assigned data points and dividing by the number of values in each cluster
        for (m = 0; m < numTrainingSamples; m++) {
            for (n = 0; n < numInputDimensions; n++) {
                double val = clusters.get(assign[m], n);
                val += data.get(m, n);
                clusters.set(val, assign[m], n);
            }
        }

        for (k = 0; k < numClusters; k++) {
            if (count[k] > 0) {
                for (n = 0; n < numInputDimensions; n++) {
                    double val = clusters.get(k, n);
                    val /= (double) count[k];
                    clusters.set(val, k, n);
                }
            }
        }
    }

    protected double calculateTheta(MatrixDouble data) {
        double theta = 0;
        double sum = 0;
        int m, n, k = 0;
        for (m = 0; m < numTrainingSamples; m++) {
            k = assign[m];
            sum = 0;
            for (n = 0; n < numInputDimensions; n++) {
                sum += Math.pow(clusters.get(k, n) - data.get(m, n), 2);
            }
            theta += Math.sqrt(sum);
        }
        theta /= numTrainingSamples;

        return theta;
    }
}
