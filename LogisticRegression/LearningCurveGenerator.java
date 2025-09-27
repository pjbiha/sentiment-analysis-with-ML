package LogisticRegression;
import Utils.EvaluationMetrics;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;
import java.util.ArrayList;
import java.util.HashMap;

public class LearningCurveGenerator {
    //dimiourgei tin kampuli mathisis gia to Logistic Regression
    public static void generateCurve(
        List<HashMap<Integer, Integer>> trainVectors,
        List<Integer> trainLabels,
        List<HashMap<Integer, Integer>> devVectors,
        List<Integer> devLabels,
        int stepSize,
        String csvPath
    ) {
        //listes gia apothikeusi twn metrikwn
        List<Integer> trainingSizes = new ArrayList<>();
        List<Double> trainPrecisions = new ArrayList<>();
        List<Double> trainRecalls = new ArrayList<>();
        List<Double> trainF1s = new ArrayList<>();
        List<Double> devPrecisions = new ArrayList<>();
        List<Double> devRecalls = new ArrayList<>();
        List<Double> devF1s = new ArrayList<>();

        System.out.println("\n=== Learning Curve ===");
        System.out.println("TrainSize | TrainPrec | TrainRec  | TrainF1   || DevPrec   | DevRec    | DevF1");

        int maxSize = trainVectors.size();
        //auksanoume to megethos tou training set se vhmata (stepSize)
        for (int size = stepSize; size <= maxSize; size += stepSize) {
            //epilogi enos uposynolou tou training set
            List<HashMap<Integer, Integer>> subsetTrainVectors = trainVectors.subList(0, size);
            List<Integer> subsetTrainLabels = trainLabels.subList(0, size);

            //ypologizei to megethos tou lexilogiou (arithmos xaraktiristikwn)
            int vocabSize = getVocabularySize(trainVectors);
            LogisticRegression classifier = new LogisticRegression(vocabSize);
            classifier.train(subsetTrainVectors, subsetTrainLabels);

            //axiologisi sto training set
            int TP_train = 0, FP_train = 0, FN_train = 0;
            for (int i = 0; i < subsetTrainVectors.size(); i++) {
                int predicted = classifier.classify(subsetTrainVectors.get(i));
                int actual = subsetTrainLabels.get(i);

                if (predicted == 1 && actual == 1) TP_train++;
                if (predicted == 1 && actual == 0) FP_train++;
                if (predicted == 0 && actual == 1) FN_train++;
            }
            double trainPrecision = EvaluationMetrics.precision(TP_train, FP_train);
            double trainRecall = EvaluationMetrics.recall(TP_train, FN_train);
            double trainF1 = EvaluationMetrics.f1Score(trainPrecision, trainRecall);

            //axiologisi sto validation set
            int TP_dev = 0, FP_dev = 0, FN_dev = 0;
            for (int i = 0; i < devVectors.size(); i++) {
                int predicted = classifier.classify(devVectors.get(i));
                int actual = devLabels.get(i);

                if (predicted == 1 && actual == 1) TP_dev++;
                if (predicted == 1 && actual == 0) FP_dev++;
                if (predicted == 0 && actual == 1) FN_dev++;
            }
            double devPrecision = EvaluationMetrics.precision(TP_dev, FP_dev);
            double devRecall = EvaluationMetrics.recall(TP_dev, FN_dev);
            double devF1 = EvaluationMetrics.f1Score(devPrecision, devRecall);
            //ektupwsi apotelesmatwn
            System.out.printf("%9d | %9.4f | %9.4f | %9.4f || %9.4f | %9.4f | %9.4f\n",
                    size, trainPrecision, trainRecall, trainF1, devPrecision, devRecall, devF1);

            trainingSizes.add(size);
            trainPrecisions.add(trainPrecision);
            trainRecalls.add(trainRecall);
            trainF1s.add(trainF1);
            devPrecisions.add(devPrecision);
            devRecalls.add(devRecall);
            devF1s.add(devF1);
        }
        //eksagogi twn apotelesmatwn se CSV
        exportLearningCurve(trainingSizes, trainPrecisions, trainRecalls, trainF1s, 
                          devPrecisions, devRecalls, devF1s, csvPath);
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
    //eksagei ta apotelesmata tis kampulis mathisis se arxeio CSV
    private static void exportLearningCurve(
            List<Integer> sizes,
            List<Double> trainPrec, List<Double> trainRec, List<Double> trainF1,
            List<Double> devPrec, List<Double> devRec, List<Double> devF1,
            String filePath
    ) {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath))) {
            writer.write("TrainingSize,TrainPrecision,TrainRecall,TrainF1,DevPrecision,DevRecall,DevF1\n");
            for (int i = 0; i < sizes.size(); i++) {
                writer.write(
                        sizes.get(i) + ","
                                + trainPrec.get(i) + ","
                                + trainRec.get(i) + ","
                                + trainF1.get(i) + ","
                                + devPrec.get(i) + ","
                                + devRec.get(i) + ","
                                + devF1.get(i) + "\n"
                );
            }
            System.out.println("Learning curve exported to: " + filePath);
        } catch (IOException e) {
            System.err.println("Error writing learning curve file: " + e.getMessage());
        }
    }
}