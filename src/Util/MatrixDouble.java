/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Util;


import java.io.BufferedReader;
import java.io.FileReader;
import java.io.Serializable;
import java.util.ArrayList;

/**
 *
 * @author Пользователь
 */
public class MatrixDouble implements Serializable{

    protected int rows;                ///< The number of rows in the Matrix
    protected int cols;                ///< The number of columns in the Matrix
    protected int capacity;            ///< The actual capacity of the Matrix, this will be the number of rows, not the actual memory size
    protected double[][] dataPtr;             ///< A pointer to the data

    /**
     * Default Constructor
     */
    public MatrixDouble() {
        rows = 0;
        cols = 0;
        capacity = 0;
        dataPtr = null;
    }

    /**
     * Constructor, sets the size of the matrix to [rows cols]
     *
     * @param rows: sets the number of rows in the matrix, must be a value
     * greater than zero
     * @param cols: sets the number of columns in the matrix, must be a value
     * greater than zero
     */
    public MatrixDouble(int rows, int cols) {
        dataPtr = null;
        resize(rows, cols);
    }

    public boolean resize(int r, int c) {
        //If the rows and cols are unchanged then do not resize the data
        if (r == rows && c == cols) {
            return true;
        }

        if (r > 0 && c > 0) {
            rows = r;
            cols = c;
            capacity = r;
            dataPtr = new double[rows][];

            //Check to see if the memory was created correctly
            for (int i = 0; i < rows; i++) {
                dataPtr[i] = new double[cols];
            }
            return true;
        }
        return false;
    }

    public double get(int i, int j) {
        return dataPtr[i][j];
    }

    public void set(double val, int i, int j) {
        dataPtr[i][j] = val;
    }

    public int getNumRows() {
        return rows;
    }

    public ArrayList<Double> getRowVector(int r) {
        ArrayList<Double> rowVector = new ArrayList<Double>(cols);
        for (int c = 0; c < cols; c++) {
            rowVector.add(c, dataPtr[r][c]);
        }
        return rowVector;
    }

    public boolean print() {
        if (dataPtr == null) {
            return false;
        }

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                System.out.print(dataPtr[i][j] + "\t");
            }
            System.out.println();
        }

        return true;
    }

    public int getNumCols() {
        return cols;
    }

    public ArrayList<MinMax> getRanges() {
        if (rows == 0) {
            return new ArrayList<MinMax>();
        }
        ArrayList< MinMax> ranges = new ArrayList<MinMax>(cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                ranges.add(new MinMax());
                ranges.get(j).updateMinMax(dataPtr[i][j]);
            }
        }
        return ranges;
    }

    public boolean scale(double minTarget, double maxTarget) {

        if (dataPtr == null) {
            return false;
        }

        ArrayList< MinMax> ranges = getRanges();

        return scale(ranges, minTarget, maxTarget);
    }

    public boolean scale(ArrayList<MinMax> ranges, double minTarget, double maxTarget) {
        if (dataPtr == null) {
            return false;
        }

        if (ranges.size() != cols) {
            return false;
        }

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                dataPtr[i][j] = utilScale(dataPtr[i][j], ranges.get(j).minValue, ranges.get(j).maxValue, minTarget, maxTarget, false);
            }
        }

        return true;
    }

    private double utilScale(double x, double minSource, double maxSource, double minTarget, double maxTarget, boolean rain) {

        if (rain) {
            if (x <= minSource) {
                return minTarget;
            }
            if (x >= maxSource) {
                return maxTarget;
            }
        }
        if (minSource == maxSource) {
            return minTarget;
        }
        return (((x - minSource) * (maxTarget - minTarget)) / (maxSource - minSource)) + minTarget;
    }

    public void loadDataTxt(String fileName) throws Exception {
        BufferedReader reader = new BufferedReader(new FileReader(fileName));

        rows = Integer.parseInt(reader.readLine());
        cols = Integer.parseInt(reader.readLine());

        dataPtr = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            String word = reader.readLine();
            System.out.println(word);
            String[] strData = word.split(" ");
            for (int j = 0; j < cols; j++) {
                dataPtr[i][j] = Double.parseDouble(strData[j]);
            }
        }
    }

    public void clear() {
        rows = 0;
        cols = 0;
        dataPtr = null;
    }

    /**
     * Adds the input sample to the end of the Matrix, extending the number of
     * rows by 1. The number of columns in the sample must match the number of
     * columns in the Matrix, unless the Matrix size has not been set, in which
     * case the new sample size will define the number of columns in the Matrix.
     *
     * @param std::vector<T> &sample: the new column vector you want to add to
     * the end of the Matrix. Its size should match the number of columns in the
     * Matrix
     * @return returns true or false, indicating if the push was successful
     */
    public boolean push_back(ArrayList<Double> sample) {
        //If there is no data, but we know how many cols are in a sample then we simply create a new buffer of size 1 and add the sample
        if (dataPtr == null) {
            cols = (int) sample.size();
            if (!resize(1, cols)) {
                clear();
                return false;
            }
            for (int j = 0; j < cols; j++) {
                dataPtr[0][j] = sample.get(j);
            }
            return true;
        }

        //If there is data and the sample size does not match the number of columns then return false
        if (sample.size() != cols) {
            return false;
        }

        //Check to see if we have reached the capacity, if not then simply add the new data
        if (rows < capacity) {
            //Add the new sample at the end
            for (int j = 0; j < cols; j++) {
                dataPtr[rows][j] = sample.get(j);
            }

        } else { //Otherwise we copy the existing data from the data ptr into a new buffer of size (rows+1) and add the sample at the end
            double[][] tempDataPtr = new double[rows + 1][cols];

            //Copy the original data
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    tempDataPtr[i][j] = dataPtr[i][j];
                }
            }

            //Add the new sample at the end
            for (int j = 0; j < cols; j++) {
                tempDataPtr[rows][j] = sample.get(j);
            }

            dataPtr = tempDataPtr;

            //Increment the capacity so it matches the number of rows
            capacity++;
        }

        //Increment the number of rows
        rows++;

        //Finally return true to signal that the data was added correctly
        return true;
    }
}
