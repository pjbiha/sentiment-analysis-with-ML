package LogisticRegression;

import java.util.HashMap;
import java.util.List;
import java.util.Random;

public class LogisticRegression {
    private double[] weights;
    private double bias;
    private int epochs = 80;
    private double h = 0.015;
    private double lambda = 0.015;
    //arxikopoiisi tou montelou me tuxaious varous
    public LogisticRegression(int m) {
        this.weights = new double[m];
        this.bias = 0;
        Random rand = new Random();
        for (int i = 0; i < m; i++) {
            weights[i] = rand.nextDouble() - 0.5;//arxikopoihsh varwn metaksi -0.5 kai 0.5
        }
    }

    //xrisimopoioume HashMap gia sparse xaraktiristika (feature index -> value = 1 an uparxei)
    public int classify(HashMap<Integer, Integer> features) {
        return ProbabilityOfCategoryP(features) >= 0.5 ? 1 : 0; //epistrefei 1 an pithanotita >= 0.5, alliws 0
    }

    ///ekpaideusi me sparse feature vectors
    public void train(List<HashMap<Integer, Integer>> vExamples, List<Integer> labels) {
        double h_initial = h; //kratame to arxiko learning rate
        double decayRate = 0.05; //taxitita meiwshs tou learning rate
    
        for (int c = 1; c <= epochs; c++) {
            //efarmogi meiwshs tou learning rate se kathe epoch
            h = h_initial / (1 + decayRate * c);
    
            for (int i = 0; i < vExamples.size(); i++) {
                HashMap<Integer, Integer> featureVect = vExamples.get(i);
                int label = labels.get(i);
    
                //ananeosi varwn gia ta xaraktiristika pou uparxoun sto paradeigma
                for (int j : featureVect.keySet()) {
                    weights[j] += h * gradient(featureVect, label, j);
                }
                bias += h * (label - ProbabilityOfCategoryP(featureVect));
            }
        }
    }

    //upologismos paragogou gia sparse xaraktiristika
    public double gradient(HashMap<Integer, Integer> features, int label, int weightIndex) {
        double prediction = ProbabilityOfCategoryP(features);
        double error = label - prediction;
        return error * features.get(weightIndex) - 2 * lambda * weights[weightIndex];
    }

    //upologismos pithanotitas gia mia klash me sparse xaraktiristika
    public double ProbabilityOfCategoryP(HashMap<Integer, Integer> features) {
        double wx = calculateSumOfWeightedFeatures(features);
        return logisticFunction(wx);
    }

    //upologismos tou athroismatos varwn kai xaraktiristikwn (sparse)
    public double calculateSumOfWeightedFeatures(HashMap<Integer, Integer> features) {
        double sum = bias;
        for (int j : features.keySet()) {
            sum += weights[j] * features.get(j); //features.get(j) einai 1 gia ola ta kleidia
        }
        return sum;
    }
    //h logistiki synartisi (sigmoid function)
    public double logisticFunction(double wx) {
        return 1 / (1 + Math.exp(-wx));
    }
}