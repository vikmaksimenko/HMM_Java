/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package DataStructures;

import Util.MatrixDouble;

/**
 *
 * @author Пользователь
 */
public class TimeSeriesClassificationSample {

    protected int classLabel;
    protected MatrixDouble data;

    public TimeSeriesClassificationSample() {
        classLabel = 0;
        data = null;
    }

    public TimeSeriesClassificationSample(int classLabel, MatrixDouble data) {
        this.classLabel = classLabel;
        this.data = data;
    }
//
//	public double operator[] (const UINT &n){
//		return data[n];
//	}
//    
//    inline const double* operator[] (const UINT &n) const {
//		return data[n];
//	}
//
//	void clear();

    public double getData(int i, int j) {
        return data.get(i, j);
    }

    public void setTrainingSample(int classLabel, MatrixDouble data) {
        this.classLabel = classLabel;
        this.data = data;
    }

    public int getLength() {
        return data.getNumRows();
    }
//    inline UINT getNumDimensions() const { return data.getNumCols(); }

    public int getClassLabel() {
        return classLabel;
    }
//    MatrixDouble &getData(){ return data; }

    public final MatrixDouble getData() {
        return data;
    }
//

//};
}
