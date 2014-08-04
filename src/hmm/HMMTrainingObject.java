/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package hmm;

import Util.MatrixDouble;

/**
 *
 * @author Пользователь
 */
class HMMTrainingObject {

    MatrixDouble alpha;     //The forward estimate matrix
    MatrixDouble beta;      //The backward estimate matrix
    double[] c;         //The scaling coefficient vector
    double pk = 0.0;			//P( O | Model )    
}
