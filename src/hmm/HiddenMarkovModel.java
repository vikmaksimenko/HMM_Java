package hmm;

import Util.MatrixDouble;
import java.util.ArrayList;
import static hmm.HMMModelTipes.*;
import java.io.Serializable;

/**
 * This class implements a discrete Hidden Markov Model.
 */
public class HiddenMarkovModel implements Serializable {

    boolean modelTrained = false;

    int numStates = 0;                          //The number of states for this model
    int numSymbols = 0;                         //The number of symbols for this model
    int delta = 1;				//The number of states a model can move to in a LeftRight model
    int numRandomTrainingIterations = 5;	//The number of training loops to find the best starting values
    int maxNumIter = 100;                       //The maximum number of iter allowed during the full training

    double logLikelihood = 0.0;                 //The log likelihood of an observation sequence given the modal, calculated by the forward method
    double cThreshold = -1000;                  //The classification threshold for this model
    double minImprovement = 1.0e-5;             //The minimum improvement value for the training loop

    int[] observationSequence = new int[0];
    int[] estimatedStates = new int[0];

    double[] pi;                                //The state start probability vector

    HMMModelTipes modelType = ERGODIC;

    ArrayList<Double> trainingIterationLog
            = new ArrayList<Double>();          //Stores the loglikelihood at each iteration the BaumWelch algorithm

    MatrixDouble a = new MatrixDouble();        //The transitions probability matrix
    MatrixDouble b = new MatrixDouble();        //The emissions probability matrix

    private int currentIter;
    private double newLoglikelihood;

    public HiddenMarkovModel() {
    }

    private double getRandomNumberUniform(double minRange, double maxRange) {
        return (Math.random() * (maxRange - minRange)) + minRange;
    }

    public void printMatrices() {
        System.out.println("A: ");
        for (int i = 0; i < a.getNumRows(); i++) {
            for (int j = 0; j < a.getNumCols(); j++) {
                System.out.print(a.get(i, j) + "\t");
            }
            System.out.println();
        }

        System.out.println("B: ");
        for (int i = 0; i < b.getNumRows(); i++) {
            for (int j = 0; j < b.getNumCols(); j++) {
                System.out.print(b.get(i, j) + "\t");
            }
            System.out.println();
        }

        System.out.print("Pi: ");
        for (int i = 0; i < pi.length; i++) {
            System.out.print(pi[i] + "\t");
        }
        System.out.println();

        //Check the weights all sum to 1
        if (true) {
            double sum = 0.0;
            for (int i = 0; i < a.getNumRows(); i++) {
                sum = 0.0;
                for (int j = 0; j < a.getNumCols(); j++) {
                    sum += a.get(i, j);
                }
                if (sum <= 0.99 || sum >= 1.01) {
                    System.err.println("WARNING: A Row " + i + " Sum: " + sum);
                }
            }

            for (int i = 0; i < b.getNumRows(); i++) {
                sum = 0.0;
                for (int j = 0; j < b.getNumCols(); j++) {
                    sum += b.get(i, j);
                }
                if (sum <= 0.99 || sum >= 1.01) {
                    System.err.println("WARNING: B Row " + i + " Sum: " + sum);
                }
            }
        }
    }

    public boolean resetModel(int numStates, int numSymbols, HMMModelTipes modelType, int delta) {
        this.numStates = numStates;
        this.numSymbols = numSymbols;
        this.modelType = modelType;
        this.delta = delta;
        return randomizeMatrices(numStates, numSymbols);
    }

    double predict(int[] obs) {
        final int N = numStates;
        final int T = obs.length;
        int t, i, j = 0;
        MatrixDouble alpha = new MatrixDouble(T, numStates);
        double[] c = new double[T];

        ////////////////// Run the forward algorithm ////////////////////////
        //Step 1: Init at t=0
        t = 0;
        c[t] = 0.0;
        for (i = 0; i < N; i++) {
            double val = pi[i] * b.get(i, obs[t]);
            alpha.set(val, t, i);
            c[t] += val;
        }

        //Set the inital scaling coeff
        c[t] = 1.0 / c[t];

        //Scale alpha
        for (i = 0; i < N; i++) {
            double val = alpha.get(t, i);
            val *= c[t];
            alpha.set(val, t, i);
        }

        //Step 2: Induction
        for (t = 1; t < T; t++) {
            c[t] = 0.0;
            for (j = 0; j < N; j++) {
                alpha.set(0.0, t, j);
                for (i = 0; i < N; i++) {
                    double val = alpha.get(t, j);
                    val += alpha.get(t - 1, i) * a.get(i, j);
                    alpha.set(val, t, j);

                }
                double val = alpha.get(t, j);
                val *= b.get(j, obs[t]);
                alpha.set(val, t, j);
                c[t] += alpha.get(t, j);
            }

            //Set the scaling coeff
            c[t] = 1.0 / c[t];

            //Scale Alpha
            for (j = 0; j < N; j++) {
                double val = alpha.get(t, j);
                val *= c[t];
                alpha.set(val, t, j);
            }
        }

        if (estimatedStates.length != T) {
            estimatedStates = new int[T];
        }
        for (t = 0; t < T; t++) {
            double maxValue = 0;
            for (i = 0; i < N; i++) {
                if (alpha.get(t, i) > maxValue) {
                    maxValue = alpha.get(t, i);
                    estimatedStates[t] = i;
                }
            }
        }

        //Termination
        double loglikelihood = 0.0;
        for (t = 0; t < T; t++) {
            loglikelihood += Math.log(c[t]);
        }
        return -loglikelihood; //Return the negative log likelihood
    }

    boolean train(int[][] trainingData) {
        //Clear any previous models
        modelTrained = false;
        observationSequence = null;
        estimatedStates = null;
        trainingIterationLog = null;

        int n, bestIndex = 0;
        double bestLogValue = 0;

//        currentIter = 0;
//        newLoglikelihood = 0;
        if (numRandomTrainingIterations > 1) {

            //A buffer to keep track each AB matrix
            ArrayList< MatrixDouble> aTracker = new ArrayList<MatrixDouble>(numRandomTrainingIterations);
            ArrayList< MatrixDouble> bTracker = new ArrayList<MatrixDouble>(numRandomTrainingIterations);
            double[] loglikelihoodTracker = new double[numRandomTrainingIterations];

            int maxNumTestIter = maxNumIter > 10 ? 10 : maxNumIter;

            //Try and find the best starting point
            for (n = 0; n < numRandomTrainingIterations; n++) {
                //Reset the model to a new random starting values
                randomizeMatrices(numStates, numSymbols);

                if (!train_(trainingData, maxNumTestIter)) {
                    return false;
                }

                aTracker.add(n, a);
                bTracker.add(n, b);
                loglikelihoodTracker[n] = newLoglikelihood;
            }

            //Get the best result and set it as the a and b starting values
            bestIndex = 0;
            bestLogValue = loglikelihoodTracker[0];
            for (n = 1; n < numRandomTrainingIterations; n++) {
                if (bestLogValue < loglikelihoodTracker[n]) {
                    bestLogValue = loglikelihoodTracker[n];
                    bestIndex = n;
                }
            }

            //Set a and b
            a = aTracker.get(bestIndex);
            b = bTracker.get(bestIndex);

        } else {
            randomizeMatrices(numStates, numSymbols);
        }

        //Perform the actual training
        if (!train_(trainingData, maxNumIter)) {
            return false;
        }
        //Calculate the observationSequence buffer length

        final int numObs = trainingData.length;
        int k = 0;
        int averageObsLength = 0;
        for (k = 0; k < numObs; k++) {
            final int T = trainingData[k].length;
            averageObsLength += T;
        }

        averageObsLength = (int) Math.floor(averageObsLength / (double) numObs);
        observationSequence = new int[averageObsLength];
        estimatedStates = new int[averageObsLength];

        //Finally, flag that the model was trained
        modelTrained = true;
        return true;
    }

    boolean train_(int[][] obs, int maxIter) {

        int numObs = obs.length;
        int i, j, k, t = 0;
        double num, denom, oldLoglikelihood = 0;
        boolean keepTraining = true;
        trainingIterationLog = new ArrayList<Double>();

        //Create the array to hold the data for each training instance
        HMMTrainingObject[] hmms = new HMMTrainingObject[numObs];

        //Create epislon and gamma to hold the re-estimation variables
        MatrixDouble[][] epsilon = new MatrixDouble[numObs][];
        MatrixDouble[] gamma = new MatrixDouble[numObs];

        //Resize the hmms, epsilon and gamma matrices so they are ready to be filled
        for (k = 0; k < numObs; k++) {
            final int T = obs[k].length;
            gamma[k] = new MatrixDouble(T, numStates);
            epsilon[k] = new MatrixDouble[T];
            for (t = 0; t < T; t++) {
                epsilon[k][t] = new MatrixDouble(numStates, numStates);
            }

            //Resize alpha, beta and phi
            hmms[k] = new HMMTrainingObject();
            hmms[k].alpha = new MatrixDouble(T, numStates);
            hmms[k].beta = new MatrixDouble(T, numStates);
            hmms[k].c = new double[T];
        }

        //For each training seq, run one pass of the forward backward
        //algorithm then reestimate a and b using the Baum-Welch
        oldLoglikelihood = 0;
        newLoglikelihood = 0;
        currentIter = 0;

        do {
            newLoglikelihood = 0.0;

            //Run the forwardbackward algorithm for each training example
            for (k = 0; k < numObs; k++) {
                if (!forwardBackward(hmms[k], obs[k])) {
                    return false;
                }
                newLoglikelihood += hmms[k].pk;
            }

            //Set the new log likelihood as the average of the observations
            newLoglikelihood /= numObs;

            trainingIterationLog.add(newLoglikelihood);

            if (++currentIter >= maxIter) {
                keepTraining = false;
                System.out.println("Max Iter Reached! Stopping Training");
            }
            if (Math.abs(newLoglikelihood - oldLoglikelihood) < minImprovement && currentIter > 1) {
                keepTraining = false;
                System.out.println("Min Improvement Reached! Stopping Training");
            }

            System.out.println("Iter: " + currentIter + " logLikelihood: " + newLoglikelihood + " change: " + (oldLoglikelihood - newLoglikelihood));

            printMatrices();

            oldLoglikelihood = newLoglikelihood;

            //Only update A, B, and Pi if needed
            if (keepTraining) {

                //Re-estimate A
                for (i = 0; i < numStates; i++) {

                    //Compute the denominator of A (which is independent of j)
                    denom = 0;
                    for (k = 0; k < numObs; k++) {
                        for (t = 0; t < obs[k].length - 1; t++) {
                            denom += hmms[k].alpha.get(t, i) * hmms[k].beta.get(t, i) / hmms[k].c[t];
                        }
                    }

                    //Compute the numerator and also update a[i][j]
                    if (denom > 0) {
                        for (j = 0; j < numStates; j++) {
                            num = 0;
                            for (k = 0; k < numObs; k++) {
                                for (t = 0; t < obs[k].length - 1; t++) {
                                    num += hmms[k].alpha.get(t, i) * a.get(i, j) * b.get(j, obs[k][t + 1]) * hmms[k].beta.get(t + 1, j);
                                }
                            }

                            //Update a[i][j]
                            a.set(num / denom, i, j);
                        }
                    } else {
                        System.err.println("Denom is zero for A!");
                        return false;
                    }
                }

                //Re-estimate B
                boolean renormB = false;
                for (i = 0; i < numStates; i++) {
                    for (j = 0; j < numSymbols; j++) {
                        num = 0.0;
                        denom = 0.0;
                        for (k = 0; k < numObs; k++) {
                            final int T = obs[k].length;
                            for (t = 0; t < T; t++) {
                                if (obs[k][t] == j) {
                                    num += hmms[k].alpha.get(t, i) * hmms[k].beta.get(t, i) / hmms[k].c[t];
                                }
                                denom += hmms[k].alpha.get(t, i) * hmms[k].beta.get(t, i) / hmms[k].c[t];
                            }
                        }

                        if (denom == 0) {
                            System.err.println("Denominator is zero for B!");
                            return false;
                        }
                        //Update b[i][j]
                        //If there are no observations at all for a state then the probabilities will be zero which is bad
                        //So instead we flag that B needs to be renormalized later
                        if (num > 0) {
                            double val = denom > 0 ? num / denom : 1.0e-5;
                            b.set(val, i, j);
                        } else {
                            b.set(0.0, i, j);
                            renormB = true;
                        }
                    }
                }

                if (renormB) {
                    double sum;
                    for (i = 0; i < numStates; i++) {
                        sum = 0.0;
                        for (k = 0; k < numSymbols; k++) {
                            double val = b.get(i, k);
                            val += 1.0 / numSymbols; //Add a small value to B to make sure the value will not be zero
                            b.set(val, i, k);
                            sum += val;
                        }
                        for (k = 0; k < numSymbols; k++) {
                            double val = b.get(i, k);
                            val /= sum;
                            b.set(val, i, k);
                        }
                    }
                }

                //Re-estimate Pi - only if the model type is ERGODIC, otherwise Pi[0] == 1 and everything else is 0
                if (modelType == ERGODIC) {
                    for (k = 0; k < numObs; k++) {
                        final int T = obs[k].length;
                        //Compute epsilon
                        for (t = 0; t < T - 1; t++) {
                            denom = 0.0;
                            for (i = 0; i < numStates; i++) {
                                for (j = 0; j < numStates; j++) {
                                    double val = hmms[k].alpha.get(t, i) * a.get(i, j) * b.get(j, obs[k][t + 1]) * hmms[k].beta.get(t + 1, j);
                                    epsilon[k][t].set(val, i, j);
                                    denom += val;
                                }
                            }
                            //Normalize Epsilon
                            for (i = 0; i < numStates; i++) {
                                for (j = 0; j < numStates; j++) {
                                    if (denom != 0) {
                                        double val = epsilon[k][t].get(i, j);
                                        val /= denom;
                                        epsilon[k][t].set(val, i, j);
                                    } else {
                                        epsilon[k][t].set(0, i, j);
                                    }
                                }
                            }
                        }

                        //Compute gamma
                        for (t = 0; t < T - 1; t++) {
                            for (i = 0; i < numStates; i++) {
                                gamma[k].set(0.0, t, i);
                                for (j = 0; j < numStates; j++) {
                                    double val = gamma[k].get(t, i);
                                    val += epsilon[k][t].get(i, j);
                                    gamma[k].set(val, t, i);
                                }
                            }
                        }
                    }

                    double sum = 0;
                    for (i = 0; i < numStates; i++) {
                        sum = 0.0;
                        for (k = 0; k < numObs; k++) {
                            sum += gamma[k].get(0, i);
                        }
                        pi[i] = sum / numObs;
                    }
                }
            }

        } while (keepTraining);

        return true;
    }

    private boolean forwardBackward(HMMTrainingObject hmm, int[] obs) {

        final int N = numStates;
        final int T = obs.length;
        int t, i, j = 0;

        ////////////////// Run the forward algorithm ////////////////////////
        //Step 1: Init at t=0
        t = 0;
        hmm.c[t] = 0.0;
        for (i = 0; i < N; i++) {
            double val = hmm.alpha.get(t, i);
            val = pi[i] * b.get(i, obs[t]);
            hmm.alpha.set(val, t, i);
            hmm.c[t] += val;
        }

        //Set the inital scaling coeff
        hmm.c[t] = 1.0 / hmm.c[t];

        //Scale alpha
        for (i = 0; i < N; i++) {
            double val = hmm.alpha.get(t, i);
            val *= hmm.c[t];
            hmm.alpha.set(val, t, i);
        }

        //Step 2: Induction
        for (t = 1; t < T; t++) {
            hmm.c[t] = 0.0;
            for (j = 0; j < N; j++) {
                hmm.alpha.set(0.0, t, j);
                for (i = 0; i < N; i++) {
                    double val = hmm.alpha.get(t, j);
                    val += hmm.alpha.get(t - 1, i) * a.get(i, j);
                    hmm.alpha.set(val, t, j);
                }
                double val = hmm.alpha.get(t, j);
                val *= b.get(j, obs[t]);
                hmm.alpha.set(val, t, j);
                hmm.c[t] += val;
            }

            //Set the scaling coeff
            hmm.c[t] = 1.0 / hmm.c[t];

            //Scale Alpha
            for (j = 0; j < N; j++) {
                double val = hmm.alpha.get(t, j);
                val *= hmm.c[t];
                hmm.alpha.set(val, t, j);
            }
        }

        //Termination
        hmm.pk = 0.0;
        for (t = 0; t < T; t++) {
            hmm.pk += Math.log(hmm.c[t]);

        }

        if (Double.isInfinite(hmm.pk)) {
            return false;
        }

        ////////////////// Run the backward algorithm ////////////////////////
        //Step 1: Init at time t=T (T-1 as everything is zero based)
        t = T - 1;
        for (i = 0; i < N; i++) {
            hmm.beta.set(1.0, t, i);
        }

        //Scale beta, using the same coeff as A
        for (i = 0; i < N; i++) {
            double val = hmm.beta.get(t, i);
            val *= hmm.c[t];
            hmm.beta.set(val, t, i);
        }

        //Step 2: Induction, from T-1 until 1 (T-2 until 0 as everything is zero based)
        for (t = T - 2; t >= 0; t--) {
            for (i = 0; i < N; i++) {
                //Calculate the backward step for t, using the scaled beta
                hmm.beta.set(0.0, t, i);
                for (j = 0; j < N; j++) {
                    double val = hmm.beta.get(t, i);
                    val += a.get(i, j) * b.get(j, obs[t]) * hmm.beta.get(t + 1, j);
                    hmm.beta.set(val, t, i);
                }

                //Scale B using the same coeff as A
                double val = hmm.beta.get(t, i);
                val *= hmm.c[t];
                hmm.beta.set(val, t, i);
            }
        }

        return true;

    }

    private boolean randomizeMatrices(int numStates, int numSymbols) {
        //Set the model as untrained as everything will now be reset
        modelTrained = false;
        logLikelihood = 0.0;

        //Set the new state and symbol size
        this.numStates = numStates;
        this.numSymbols = numSymbols;
        a.resize(numStates, numStates);
        b.resize(numStates, numSymbols);
        pi = new double[numStates];

        //Fill Transition and Symbol Matrices randomly
        //It's best to choose values in the range [0.9 1.1] rather than [0 1]
        //That way, no single value will get too large or too small a weight when the values are normalized
        for (int i = 0; i < a.getNumRows(); i++) {
            for (int j = 0; j < a.getNumCols(); j++) {
                a.set(getRandomNumberUniform(0.9, 1), i, j);
            }
        }

        for (int i = 0; i < b.getNumRows(); i++) {
            for (int j = 0; j < b.getNumCols(); j++) {
                b.set(getRandomNumberUniform(0.9, 1), i, j);
            }
        }

        //Randomise pi
        for (int i = 0; i < numStates; i++) {
            pi[i] = getRandomNumberUniform(0.9, 1);
        }

        //Set any raints on the model
        switch (modelType) {
            case ERGODIC:
                //Don't need todo anything
                break;
            case LEFTRIGHT:
                //Set the state transitions raints
                for (int i = 0; i < numStates; i++) {
                    for (int j = 0; j < numStates; j++) {
                        if ((j < i) || (j > i + delta)) {
                            a.set(0.0, i, j);
                        }
                    }
                }

                //Set pi to start in state 0
                for (int i = 0; i < numStates; i++) {
                    pi[i] = i == 0 ? 1 : 0;
                }
                break;
            default:
                System.err.println("HMM_ERROR: Unkown model type!");
                return false;
        }

        //Normalize the matrices
        double sum = 0.0;
        for (int i = 0; i < numStates; i++) {
            sum = 0.0;
            for (int j = 0; j < numStates; j++) {
                sum += a.get(i, j);
            }
            for (int j = 0; j < numStates; j++) {
                double val = a.get(i, j);
                val /= sum;
                a.set(val, i, j);
            }
        }
        for (int i = 0; i < numStates; i++) {
            sum = 0.0;
            for (int k = 0; k < numSymbols; k++) {
                sum += b.get(i, k);
            }
            for (int k = 0; k < numSymbols; k++) {
                double val = b.get(i, k);
                val /= sum;
                b.set(val, i, k);
            }
        }

        //Normalise pi
        sum = 0.0;
        for (int i = 0; i < numStates; i++) {
            sum += pi[i];
        }
        for (int i = 0; i < numStates; i++) {
            pi[i] /= sum;
        }

        return true;
    }
}
