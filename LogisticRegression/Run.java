package LogisticRegression;

import java.util.List;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import Utils.EvaluationMetrics;
import Utils.VectorImporter;

public class Run {
    public static void main(String[] args) {
        String trainVectorsPath = "C:\\Users\\PJ\\Desktop\\PartA\\train_vectors.txt";
        String trainLabelsPath = "C:\\Users\\PJ\\Desktop\\PartA\\train_labels.txt";
        String testVectorsPath = "C:\\Users\\PJ\\Desktop\\PartA\\test_vectors.txt";
        String testLabelsPath = "C:\\Users\\PJ\\Desktop\\PartA\\test_labels.txt";

        ///fortosi twn sparse vectors
        List<HashMap<Integer, Integer>> allTrainVectors = VectorImporter.importSparseVectors(trainVectorsPath);
        List<Integer> allTrainLabels = VectorImporter.importLabels(trainLabelsPath,Integer.class);
        List<HashMap<Integer, Integer>> testVectors = VectorImporter.importSparseVectors(testVectorsPath);
        List<Integer> testLabels = VectorImporter.importLabels(testLabelsPath,Integer.class);

        //elegxos gia isotimia metaksu vectors kai labels
        if (allTrainVectors.size() != allTrainLabels.size() || testVectors.size() != testLabels.size()) {
            System.err.println("Mismatch between vectors and labels.");
            return;
        }

        int totalTrainExamples = allTrainVectors.size();
        int totalTestExamples = testVectors.size();
        System.out.println("Total training examples: " + totalTrainExamples);
        System.out.println("Total test examples: " + totalTestExamples);

        if (totalTrainExamples == 0 || totalTestExamples == 0) {
            System.err.println("No data found. Exiting...");
            return;
        }

        //anakatema twn dedomenwn ekpaideusis
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < totalTrainExamples; i++) indices.add(i);
        Collections.shuffle(indices);

        //dimiourgia neas listas me ta anakatemena dedomena
        List<HashMap<Integer, Integer>> shuffledTrainVectors = new ArrayList<>(totalTrainExamples);
        List<Integer> shuffledTrainLabels = new ArrayList<>(totalTrainExamples);

        for (int idx : indices) {
            shuffledTrainVectors.add(allTrainVectors.get(idx));
            shuffledTrainLabels.add(allTrainLabels.get(idx));
        }

        //dianomi se 80% train kai 20% development
        int trainSize = (int) (0.8 * totalTrainExamples);
        List<HashMap<Integer, Integer>> trainVectors = shuffledTrainVectors.subList(0, trainSize);
        List<Integer> trainLabels = shuffledTrainLabels.subList(0, trainSize);
        List<HashMap<Integer, Integer>> devVectors = shuffledTrainVectors.subList(trainSize, totalTrainExamples);
        List<Integer> devLabels = shuffledTrainLabels.subList(trainSize, totalTrainExamples);

        System.out.println("Train size: " + trainVectors.size());
        System.out.println("Dev size: " + devVectors.size());

        //dimiourgia learning curve
        String learningCurvePath = "C:\\Users\\ernes\\Desktop\\PartA\\learning_curve_logistic.csv";
        int stepSize = 2000;
        LearningCurveGenerator.generateCurve(trainVectors, trainLabels, devVectors, devLabels, stepSize, learningCurvePath);

        //ekpaideusi tou telikou montelou me ola ta dedomena ekpaideusis kai axiologisi sto test set
        System.out.println("\n=== Training Final Model on Full Train Set ===");
        
        //ypologismos megethous lexilogiou apo ta pragmatika dedomena
        int vocabSize = getVocabularySize(trainVectors);
        LogisticRegression classifier = new LogisticRegression(vocabSize);
        classifier.train(trainVectors, trainLabels);

        //axiologisi tou telikou montelou sto test set
        System.out.println("Evaluating on Test Data...");
        evaluateModel(classifier, testVectors, testLabels);
    }
    //voithitiki methodos gia ton ypologismo tou megethous tou lexilogiou
    private static int getVocabularySize(List<HashMap<Integer, Integer>> vectors) {
        int maxIndex = -1;
        for (HashMap<Integer, Integer> vec : vectors) {
            for (int key : vec.keySet()) {
                if (key > maxIndex) maxIndex = key;
            }
        }
        return maxIndex + 1; //ypothesi oti h arithmisi twn xaraktiristikwn ksekinaei apo 0
    }
    //aksiologisi tou montelou kai upologismos metrikwn
    private static void evaluateModel(LogisticRegression classifier,List<HashMap<Integer, Integer>> vectors,List<Integer> labels) {
        int TP = 0, FP = 0, TN = 0, FN = 0;
        for (int i = 0; i < vectors.size(); i++) {
            int predicted = classifier.classify(vectors.get(i));
            int actual = labels.get(i);

            if (predicted == 1 && actual == 1) TP++;
            else if (predicted == 1 && actual == 0) FP++;
            else if (predicted == 0 && actual == 0) TN++;
            else if (predicted == 0 && actual == 1) FN++;
        }

        double precision_pos = EvaluationMetrics.precision(TP, FP);
        double recall_pos = EvaluationMetrics.recall(TP, FN);
        double f1_pos = EvaluationMetrics.f1Score(precision_pos, recall_pos);

        double precision_neg = EvaluationMetrics.precision(TN, FN);
        double recall_neg = EvaluationMetrics.recall(TN, FP);
        double f1_neg = EvaluationMetrics.f1Score(precision_neg, recall_neg);

        double macro_precision = (precision_pos + precision_neg) / 2.0;
        double macro_recall = (recall_pos + recall_neg) / 2.0;
        double macro_f1 = (f1_pos + f1_neg) / 2.0;

        //micro-averaged metrikwn (sunoliko athroisma metrikwn metaksi twn klasewn)
        int totalTP = TP + TN;
        int totalFP = FP + FN;
        int totalFN = FN + FP;

        double micro_precision = EvaluationMetrics.precision(totalTP, totalFP);
        double micro_recall = EvaluationMetrics.recall(totalTP, totalFN);
        double micro_f1 = EvaluationMetrics.f1Score(micro_precision, micro_recall);
        
        //ektuposi twn apotelesmatwn
        System.out.printf("\n=== Final Results on Test Set ===\n");
        System.out.printf("Precision (Positive Class): %.4f\n", precision_pos);
        System.out.printf("Recall (Positive Class):    %.4f\n", recall_pos);
        System.out.printf("F1 (Positive Class):        %.4f\n", f1_pos);

        System.out.println();
        System.out.printf("Precision (Negative Class): %.4f\n", precision_neg);
        System.out.printf("Recall (Negative Class):    %.4f\n", recall_neg);
        System.out.printf("F1 (Negative Class):        %.4f\n", f1_neg);

        System.out.println();
        System.out.printf("Macro-Averaged Precision: %.4f\n", macro_precision);
        System.out.printf("Macro-Averaged Recall:    %.4f\n", macro_recall);
        System.out.printf("Macro-Averaged F1:        %.4f\n", macro_f1);

        System.out.println();
        System.out.printf("Micro-Averaged Precision: %.4f\n", micro_precision);
        System.out.printf("Micro-Averaged Recall:    %.4f\n", micro_recall);
        System.out.printf("Micro-Averaged F1:        %.4f\n", micro_f1);
    }
}