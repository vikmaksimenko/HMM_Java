/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package DataStructures;

import Util.ClassTracker;
import Util.MatrixDouble;
import Util.MinMax;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author Пользователь
 */
public class TimeSeriesClassificationData {

    String datasetName = "NOT_SET";                      ///< The name of the dataset
    String infoText = "";                             ///< Some infoText about the dataset
    int numDimensions = 0;			   ///< The number of dimensions in the dataset

    int totalNumSamples = 0;                               ///< The total number of samples in the dataset
    int kFoldValue;                                    ///< The number of folds the dataset has been spilt into for cross valiation
    boolean crossValidationSetup = false;                          ///< A flag to show if the dataset is ready for cross validation
    boolean useExternalRanges = false;                             ///< A flag to show if the dataset should be scaled using the externalRanges values
    boolean allowNullGestureClass = false;                         ///< A flag that enables/disables a user from adding new samples with a class label matching the default null gesture label

    ArrayList<MinMax> externalRanges
            = new ArrayList<MinMax>();                      ///< A ArrayList containing a set of externalRanges set by the user
    ArrayList<ClassTracker> classTracker
            = new ArrayList<ClassTracker>();                  ///< A ArrayList of ClassTracker, which keeps track of the number of samples of each class
    ArrayList<TimeSeriesClassificationSample> data
            = new ArrayList<TimeSeriesClassificationSample>();        ///< The labelled time series classification data
    // ArrayList<ArrayList<int>> crossValidationIndexs;         ///< A ArrayList to hold the indexs of the dataset for the cross validation

//    DebugLog debugLog;                                      ///< Default debugging log
//    ErrorLog errorLog;                                      ///< Default error log
//    WarningLog warningLog;                                  ///< Default warning log
    /**
     * Constructor, sets the name of the dataset and the number of dimensions of
     * the training data. The name of the dataset should not contain any spaces.
     *
     * @param int numDimensions: the number of dimensions of the training data,
     * should be an unsigned integer greater than 0
     * @param String datasetName: the name of the dataset, should not contain
     * any spaces
     * @param String infoText: some info about the data in this dataset, this
     * can contain spaces
     */
    //const int numDimensions=0,const String datasetName = "NOT_SET",const String infoText = "")
    public TimeSeriesClassificationData(final int numDimensions, final String datasetName, final String infoText) {
//        debugLog.setProceedingText("[DEBUG TSCD]");
//        errorLog.setProceedingText("[ERROR TSCD]");
//        warningLog.setProceedingText("[WARNING TSCD]");
        this.numDimensions = numDimensions;
        this.datasetName = datasetName;
        this.infoText = infoText;
    }

    public TimeSeriesClassificationData(int numDimensions) {
        //TimeSeriesClassificationData(numDimensions, "NOT_SET", "");

        this.numDimensions = numDimensions;
    }

    public TimeSeriesClassificationData() {

    }

    /**
     * Array Subscript Operator, returns the TimeSeriesClassificationSample at
     * index i. It is up to the user to ensure that i is within the range of [0
     * totalNumSamples-1]
     *
     * @param const int &i: the index of the training sample you want to access.
     * Must be within the range of [0 totalNumSamples-1]
     * @return a reference to the i'th TimeSeriesClassificationSample
     */
    public TimeSeriesClassificationSample get(final int i) {
        return data.get(i);
    }

    /**
     * Gets the number of samples in the classification data across all the
     * classes.
     *
     * @return an int representing the total number of samples in the
     * classification data
     */
    public int getNumSamples() {
        return totalNumSamples;
    }

    /**
     * Gets the number of classes.
     *
     * @return an int representing the number of classes
     */
    public int getNumClasses() {
        return classTracker.size();
    }

    public MatrixDouble getDataAsMatrixDouble() {

        //Count how many samples are in the entire dataset
        int M = 0;
        int index = 0;
        for (int x = 0; x < totalNumSamples; x++) {
            M += data.get(x).getLength();
        }

        // I don`t understand, what`s for is this code, but let it be...
        if (M == 0) {
            new MatrixDouble();
        }

        //Get all the data and concatenate it into 1 matrix
        MatrixDouble matrixData = new MatrixDouble(M, numDimensions);
        for (int x = 0; x < totalNumSamples; x++) {
            for (int i = 0; i < data.get(x).getLength(); i++) {
                for (int j = 0; j < numDimensions; j++) {
                    double val = data.get(x).getData(i, j);
                    matrixData.set(val, index, j);
                }
                index++;
            }
        }
        return matrixData;
    }

    private ArrayList<MinMax> getRanges() {
        if (useExternalRanges) {
            return externalRanges;
        }

        ArrayList<MinMax> ranges = new ArrayList<MinMax>(numDimensions);
        for (int i = 0; i < numDimensions; i++) {
            ranges.add(new MinMax());
        }

        if (totalNumSamples > 0) {
            for (int j = 0; j < numDimensions; j++) {
                ranges.get(j).minValue = data.get(0).getData(0, 0);
                ranges.get(j).maxValue = data.get(0).getData(0, 0);

                //ranges.add(new MinMax(data.get(0).getData(0, 0), data.get(0).getData(0, 0)));
                for (int x = 0; x < totalNumSamples; x++) {
                    for (int i = 0; i < data.get(x).getLength(); i++) {
                        if (data.get(x).getData(i, j) < ranges.get(j).minValue) {
                            ranges.get(j).minValue = data.get(x).getData(i, j);
                        } //Search for the min value
                        else if (data.get(x).getData(i, j) > ranges.get(j).maxValue) {
                            ranges.get(j).maxValue = data.get(x).getData(i, j);
                        }	//Search for the max value
                    }
                }
            }
        }
        return ranges;
    }

    /**
     * Loads the labelled timeseries classification data from a custom file
     * format.
     *
     * @param filename: the name of the file the data will be loaded from
     * @return true if the data was loaded successfully, false otherwise
     */
    public boolean loadDatasetFromFile(final String filename) throws IOException {

        int numClasses = 0;
        clear();

        BufferedReader reader;

        try {
            reader = new BufferedReader(new FileReader(filename));
        } catch (FileNotFoundException ex) {
            System.err.println("loadDatasetFromFile(String filename) - FILE NOT OPEN!");
            return false;
        }

        String word;

        //Check to make sure this is a file with the Training File Format
        word = reader.readLine();
        if (!word.equals("GRT_LABELLED_TIME_SERIES_CLASSIFICATION_DATA_FILE_V1.0")) {
            System.err.println("loadDatasetFromFile(String filename) - Failed to find file header!");
            reader.close();
            return false;
        }

        //Get the name of the dataset
        word = reader.readLine();
        if (!word.contains("DatasetName:")) {
            System.err.println("loadDatasetFromFile(String filename) - failed to find DatasetName!");
            reader.close();
            return false;
        }
        datasetName = word.split(" ")[1];

        word = reader.readLine();
        if (!word.contains("InfoText:")) {
            System.err.println("loadDatasetFromFile(String filename) - failed to find InfoText!");
            reader.close();
            return false;
        }
        infoText = word.split(" ")[1];

        word = reader.readLine();
        //Get the number of dimensions in the training data
        if (!word.contains("NumDimensions:")) {
            System.err.println("loadDatasetFromFile(String filename) - Failed to find NumDimensions!");
            reader.close();
            return false;
        }
        numDimensions = Integer.parseInt(word.split(" ")[1]);

        //Get the total number of training examples in the training data
        word = reader.readLine();
        if (!word.contains("TotalNumTrainingExamples:")) {
            System.err.println("loadDatasetFromFile(String filename) - Failed to find TotalNumTrainingExamples!");
            reader.close();
            return false;
        }
        totalNumSamples = Integer.parseInt(word.split(" ")[1]);

        //Get the total number of classes in the training data
        word = reader.readLine();
        if (!word.contains("NumberOfClasses:")) {
            System.err.println("loadDatasetFromFile(String filename) - Failed to find NumberOfClasses!");
            reader.close();
            return false;
        }
        numClasses = Integer.parseInt(word.split(" ")[1]);

        //Resize the class counter buffer and load the counters
        classTracker.ensureCapacity(numClasses);

        //Get the total number of classes in the training data
        word = reader.readLine();
        if (!word.contains("ClassIDsAndCounters:")) {
            System.err.println("loadDatasetFromFile(String filename) - Failed to find ClassIDsAndCounters!");
            reader.close();
            return false;
        }

        for (int i = 0; i < numClasses; i++) {
            word = reader.readLine();
            classTracker.add(new ClassTracker(
                    Integer.parseInt((word.split("\t")[0])),
                    Integer.parseInt((word.split("\t")[1])),
                    "NOT_SET"));
        }

        //Get the UseExternalRanges
        word = reader.readLine();
        if (!word.contains("UseExternalRanges:")) {
            System.err.println("loadDatasetFromFile(String filename) - Failed to find UseExternalRanges!");
            reader.close();
            return false;
        }
        if (!word.split(" ")[1].equals("0")) {
            useExternalRanges = true;
            externalRanges.ensureCapacity(numDimensions);
            String[] buf = word.split(" ");
            for (int i = 0; i < externalRanges.size(); i++) {
                externalRanges.get(i).minValue = Double.parseDouble(buf[i * 2 + 1]);
                externalRanges.get(i).maxValue = Double.parseDouble(buf[i * 2 + 2]);
            }
        }

        //Get the main training data
        word = reader.readLine();
        if (!word.contains("LabelledTimeSeriesTrainingData:")) {
            System.err.println("loadDatasetFromFile(String filename) - Failed to find LabelledTimeSeriesTrainingData!");
            reader.close();
            return false;
        }

        //Reset the memory
        data.ensureCapacity(totalNumSamples);

        //Load each of the time series
        for (int x = 0; x < totalNumSamples; x++) {
            int classLabel = 0;
            int timeSeriesLength = 0;
            word = reader.readLine();
            if (!word.contains("************TIME_SERIES************")) {
                System.err.println("loadDatasetFromFile(String filename) - Failed to find TimeSeries Header! ");
                reader.close();
                return false;
            }
            word = reader.readLine();
            if (!word.contains("ClassID:")) {
                System.err.println("loadDatasetFromFile(String filename) - Failed to find ClassID!");
                reader.close();
                return false;
            }
            classLabel = Integer.parseInt(word.split(" ")[1]);
            word = reader.readLine();
            if (!word.contains("TimeSeriesLength:")) {
                System.err.println("loadDatasetFromFile(String filename) - Failed to find TimeSeriesLength!");
                reader.close();
                return false;
            }
            timeSeriesLength = Integer.parseInt(word.split(" ")[1]);

            //System.out.println(timeSeriesLength);
            word = reader.readLine();
            if (!word.contains("TimeSeriesData: ")) {
                System.err.println("loadDatasetFromFile(String filename) - Failed to find TimeSeriesData!");
                reader.close();
                return false;
            }

            //Load the time series data
            MatrixDouble trainingExample = new MatrixDouble(timeSeriesLength, numDimensions);
            for (int i = 0; i < timeSeriesLength; i++) {
                word = reader.readLine();
                String[] strData = word.split(" ");
                for (int j = 0; j < numDimensions; j++) {
                    trainingExample.set(Double.parseDouble(strData[j]), i, j);
                }
            }

            data.add(new TimeSeriesClassificationSample());
            data.get(x).setTrainingSample(classLabel, trainingExample);
        }

        //System.out.println("In TimeSeries loadDattaSet.\n DataSet: " + this);
        reader.close();
        return true;
    }

    public boolean printStats() {
        System.out.println("DatasetName:\t" + datasetName);
        System.out.println("DatasetInfo:\t" + infoText);
        System.out.println("Number of Dimensions:\t" + numDimensions);
        System.out.println("Number of Samples:\t" + totalNumSamples);
        System.out.println("Number of Classes:\t" + getNumClasses());
        System.out.println("ClassStats:");

        for (int k = 0; k < getNumClasses(); k++) {
            System.out.println("ClassLabel:\t" + classTracker.get(k).classLabel);
            System.out.println("Number of Samples:\t" + classTracker.get(k).counter);
            System.out.println("ClassName:\t" + classTracker.get(k).className);
        }

        ArrayList< MinMax> ranges = getRanges();
        System.out.println("Dataset Ranges:");
        for (int j = 0; j < ranges.size(); j++) {
            System.out.println("[" + j + 1 + "] Min:\t" + ranges.get(j).minValue + "\tMax: " + ranges.get(j).maxValue);
        }
        System.out.println("Timeseries Lengths:");
        int M = (int) data.size();
        for (int j = 0; j < M; j++) {
            System.out.println("ClassLabel: " + data.get(j).getClassLabel() + " Length:\t" + data.get(j).getLength());

            MatrixDouble timeseriesData = data.get(j).getData();
            timeseriesData.print();
        }

        return true;
    }

    public void clear() {
        totalNumSamples = 0;
        data.clear();
        //classTracker.clear();
    }

    public boolean setNumDimensions(final int numDimensions) {
        if (numDimensions > 0) {
            //Clear any previous training data
            clear();

            //Set the dimensionality of the training data
            this.numDimensions = numDimensions;

            useExternalRanges = false;
            externalRanges.clear();

            return true;
        }

        System.err.println("setNumDimensions(int numDimensions) - The number of dimensions of the dataset must be greater than zero!");
        return false;
    }

    /**
     * Adds a new labelled timeseries sample to the dataset. The dimensionality
     * of the sample should match the number of dimensions in the dataset. The
     * class label should be greater than zero (as zero is used as the default
     * null rejection class label).
     *
     * @param int classLabel: the class label of the corresponding sample
     * @param MatrixDouble trainingSample: the new sample you want to add to the
     * dataset. The dimensionality of this sample (i.e. Matrix columns) should
     * match the number of dimensions in the dataset, the rows of the Matrix
     * represent time and do not have to be any specific length
     * @return true if the sample was correctly added to the dataset, false
     * otherwise
     */
    public boolean addSample(int classLabel, MatrixDouble trainingSample) {
        if (trainingSample.getNumCols() != numDimensions) {
            System.err.println("addSample(int classLabel, MatrixDouble trainingSample) - The dimensionality of the training sample (" + trainingSample.getNumCols() + ") does not match that of the dataset (" + numDimensions + ")");
            return false;
        }

        //The class label must be greater than zero (as zero is used for the null rejection class label
        if (classLabel == 0 && !allowNullGestureClass) {
            System.err.println("addSample(int classLabel, MatrixDouble sample) - the class label can not be 0!");
            return false;
        }

        TimeSeriesClassificationSample newSample = new TimeSeriesClassificationSample(classLabel, trainingSample);
        data.add(newSample);
        totalNumSamples++;

        if (classTracker.size() == 0) {
            ClassTracker tracker = new ClassTracker(classLabel, 1);
            classTracker.add(tracker);
        } else {
            boolean labelFound = false;
            for (int i = 0; i < classTracker.size(); i++) {
                if (classLabel == classTracker.get(i).classLabel) {
                    classTracker.get(i).counter++;
                    labelFound = true;
                    break;
                }
            }
            if (!labelFound) {
                ClassTracker tracker = new ClassTracker(classLabel, 1);
                classTracker.add(tracker);
            }
        }
        return true;
    }

    public int getNumDimensions() {
        return numDimensions;
    }

    public ArrayList<ClassTracker> getClassTracker() {
        return classTracker;
    }

    public TimeSeriesClassificationData getClassData(int classLabel) {
        TimeSeriesClassificationData classData = new TimeSeriesClassificationData(numDimensions);
        for (int x = 0; x < totalNumSamples; x++) {
            if (data.get(x).getClassLabel() == classLabel) {
                classData.addSample(classLabel, data.get(x).getData());
            }
        }
        return classData;
    }

}
