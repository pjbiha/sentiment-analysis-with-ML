package NaiveBayes;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Vector;

public class NaiveBayesClassifier {

    private Map<Integer, Double> probWordGivenPos; //pithanotita P(word_j=1 | pos)
    private Map<Integer, Double> probWordGivenNeg; //pithanotita P(word_j=1 | neg)
    private double probPos;                       //pithanotita P(pos)
    private double probNeg;                       //pithanotita P(neg)
    private int vocabSize;                        //megethos tou lexilogiou

    public NaiveBayesClassifier(int vocabularySize) {
        this.vocabSize = vocabularySize;
        probWordGivenPos = new HashMap<>();
        probWordGivenNeg = new HashMap<>();
    }

    //ekpaideuei to Bernoulli Naive Bayes me Laplace smoothing
    //ta trainingVectors einai binary dianysmata (0/1) mikous = vocabSize
    //ta labels einai 0 h 1
    public void train(List<Vector<Integer>> trainingVectors, List<Integer> labels) {
        int numDocs = trainingVectors.size();
        if (numDocs == 0) {
            System.err.println("No training data provided!");
            return;
        }

        //metraei posa thetika kai arnitika paradeigmata exoume
        int countPos = 0, countNeg = 0;
        int[] posOnes = new int[vocabSize];//poses fores to word_j=1 sta thetika paradeigmata
        int[] negOnes = new int[vocabSize];//poses fores to word_j=1 sta arnitika paradeigmata

        //pernaw ta paradeigmata ena pros ena
        for (int i = 0; i < numDocs; i++) {
            Vector<Integer> vec = trainingVectors.get(i);
            int label = labels.get(i);

            if (label == 1) {
                countPos++;
                //gia Bernoulli: auksanoume to posOnes[j] an vector[j] == 1
                for (int j = 0; j < vocabSize; j++) {
                    if (vec.get(j) == 1) {
                        posOnes[j]++;
                    }
                }
            } else {
                countNeg++;
                for (int j = 0; j < vocabSize; j++) {
                    if (vec.get(j) == 1) {
                        negOnes[j]++;
                    }
                }
            }
        }

        //ypologizei tis priori pithanotites
        probPos = (double) countPos / numDocs;
        probNeg = (double) countNeg / numDocs;

        ///Laplace smoothing gia Bernoulli:
        //P(word_j=1 | pos) = (posOnes[j] + 1) / (countPos + 2)
        //To 2 sto paranomasti einai giati to word_j borei na einai 0 i 1
        for (int j = 0; j < vocabSize; j++) {
            double p1pos = (posOnes[j] + 1.0) / (countPos + 2.0);
            double p1neg = (negOnes[j] + 1.0) / (countNeg + 2.0);

            probWordGivenPos.put(j, p1pos);
            probWordGivenNeg.put(j, p1neg);
        }
    }

    //kanei provlepsi tou label (0 i 1) gia ena neo binary vector
    public int predict(Vector<Integer> vec) {
        //xekinaei me log(P(pos)) kai log(P(neg))
        double logProbPos = Math.log(probPos);
        double logProbNeg = Math.log(probNeg);

        //pernaw ola ta xaraktiristika ena pros ena
        for (int j = 0; j < vocabSize; j++) {
            double p1pos = probWordGivenPos.get(j);
            double p1neg = probWordGivenNeg.get(j);

            if (vec.get(j) == 1) {
                logProbPos += Math.log(p1pos);
                logProbNeg += Math.log(p1neg);
            } else {
                //an to word_j = 0
                logProbPos += Math.log(1.0 - p1pos);
                logProbNeg += Math.log(1.0 - p1neg);
            }
        }

        //epistrofi tou pithanoterou class
        if (logProbPos > logProbNeg) {
            return 1; //to keimeno einai thetiko
        } else {
            return 0; //to keimeno einai arnitiko
        }
    }
}
