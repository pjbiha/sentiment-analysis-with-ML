package NaiveBayes;
import Utils.VectorImporter;
import java.util.*;
import Utils.EvaluationMetrics;

public class Run {
    public static void main(String[] args) {
        //orismos path gia ta training & testing vectors
        String trainVectorsPath = "C:\\Users\\ernes\\Desktop\\PartA\\train_vectors.txt";
        String trainLabelsPath  = "C:\\Users\\ernes\\Desktop\\PartA\\train_labels.txt";
        String testVectorsPath  = "C:\\Users\\ernes\\Desktop\\PartA\\test_vectors.txt";
        String testLabelsPath   = "C:\\Users\\ernes\\Desktop\\PartA\\test_labels.txt";

        //fortwsi dedomenwn ekpaideusi
        List<Vector<Integer>> allTrainVectors = VectorImporter.importVectors(trainVectorsPath,Integer.class);
        List<Integer> allTrainLabels = VectorImporter.importLabels(trainLabelsPath,Integer.class);

        //fortwsi dedomenwn elegxou (olokliro to test set, 25,000 reviews)
        List<Vector<Integer>> testVectors = VectorImporter.importVectors(testVectorsPath,Integer.class);
        List<Integer> testLabels = VectorImporter.importLabels(testLabelsPath,Integer.class);

        //basikoi elegxoi
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

        //Shuffle ta training data
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < totalTrainExamples; i++) indices.add(i);
        Collections.shuffle(indices);

        //dhmiourgeia newn shuffled lists
        List<Vector<Integer>> shuffledTrainVectors = new ArrayList<>(totalTrainExamples);
        List<Integer> shuffledTrainLabels = new ArrayList<>(totalTrainExamples);

        for (int idx : indices) {
            shuffledTrainVectors.add(allTrainVectors.get(idx));
            shuffledTrainLabels.add(allTrainLabels.get(idx));
        }

        //dianomi se 80% ekpaideusi kai 20% development (apo to training set)
        int trainSize = (int) (0.8 * totalTrainExamples);
        List<Vector<Integer>> trainVectors = shuffledTrainVectors.subList(0, trainSize);
        List<Integer> trainLabels = shuffledTrainLabels.subList(0, trainSize);

        List<Vector<Integer>> devVectors = shuffledTrainVectors.subList(trainSize, totalTrainExamples);
        List<Integer> devLabels = shuffledTrainLabels.subList(trainSize, totalTrainExamples);

        System.out.println("Train size: " + trainVectors.size());
        System.out.println("Dev size: " + devVectors.size());

        //dimiourgia learning curve
        String learningCurvePath = "C:\\Users\\ernes\\Desktop\\PartA\\learning_curve.csv";
        int stepSize = 2000;  //afksisi megethous ekpaideusis me vima 2000

        LearningCurveGenerator.generateCurve(
            trainVectors, trainLabels,
            devVectors, devLabels,
            stepSize,
            learningCurvePath
        );

        //ekpaideusi sto 80% tou training set kai axiologisi sto test set
        System.out.println("\n=== Training Final Model on Full Train Set ===");
        int vocabSize = trainVectors.get(0).size();
        NaiveBayesClassifier classifier = new NaiveBayesClassifier(vocabSize);
        classifier.train(trainVectors, trainLabels);

        //axiologisi tou telikou montelou sto test set
        System.out.println("Evaluating on Test Data...");
        evaluateModel(classifier, testVectors, testLabels);
    }

    //methodos axiologisis me ypologismo precision, recall kai f1 gia kathe kathgoria kai mesous orous
    private static void evaluateModel(NaiveBayesClassifier classifier,List<Vector<Integer>> vectors,List<Integer> labels){
        // 1. dhmiourgia 2x2 confusion matrix: TP, FP, TN, FN
        int TP = 0; // predicted=1, actual=1
        int FP = 0; // predicted=1, actual=0
        int TN = 0; // predicted=0, actual=0
        int FN = 0; // predicted=0, actual=1

        for (int i = 0; i < vectors.size(); i++) {
            Vector<Integer> v = vectors.get(i);
            int actual = labels.get(i);
            int predicted = classifier.predict(v);

            if (predicted == 1 && actual == 1) {
                TP++;
            } else if (predicted == 1 && actual == 0) {
                FP++;
            } else if (predicted == 0 && actual == 0) {
                TN++;
            } else if (predicted == 0 && actual == 1) {
                FN++;
            }
        }

        // 2.ypologismos metrikon gia tin thetiki klasi (label = 1)
        double precision_pos = EvaluationMetrics.precision(TP, FP);
        double recall_pos    = EvaluationMetrics.recall(TP, FN);
        double f1_pos        = EvaluationMetrics.f1Score(precision_pos, recall_pos);

        // 3.ypologismos metrikon gia tin arnhtiki klasi (label = 0)
        // metatroph tis arnhtikis klasis ws ena diaforetiko "one-vs-all":
        //TP_neg = TN, FP_neg = FN, FN_neg = FP, TN_neg = TP
        int TP_neg = TN;
        int FP_neg = FN;
        int FN_neg = FP;
        int TN_neg = TP;
        double precision_neg = EvaluationMetrics.precision(TP_neg, FP_neg);
        double recall_neg    = EvaluationMetrics.recall(TP_neg, FN_neg);
        double f1_neg        = EvaluationMetrics.f1Score(precision_neg, recall_neg);

        // 4.ypologismos macro-averaged metrikon
        double macro_precision = (precision_pos + precision_neg) / 2.0;
        double macro_recall    = (recall_pos + recall_neg)       / 2.0;
        double macro_f1        = (f1_pos + f1_neg)               / 2.0;

        // 5. ypologismos micro-averaged metrikon
        // se mia two class periptosi, to micro-score isoutai me tin sunoliki akriveia
        // gia precision & recall. Prosthetoume tis metriseis gia tis klaseis:

        int totalTP = TP + TN; //apo tis dyo klaseis
        int totalFP = FP + FN;
        int totalFN = FN + FP;
        double micro_precision = EvaluationMetrics.precision(totalTP, totalFP);
        double micro_recall    = EvaluationMetrics.recall(totalTP, totalFN);
        double micro_f1        = EvaluationMetrics.f1Score(micro_precision, micro_recall);

        // 6.ektypwsi apotelesmatwn
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
