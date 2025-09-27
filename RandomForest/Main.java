package RandomForest;
import Utils.EvaluationMetrics;
import java.io.IOException;
import java.util.*;

public class Main {
    public static void main(String[] args) {
        //diavasma twn dedomenwn apo ta arxeia
        try {
            List<Vector<Integer>> vExamples = DataParser.parseVectors("C:\\Users\\PJ\\Desktop\\PartA\\vectors.txt");
            List<Integer> labels = DataParser.parseLabels("C:\\Users\\PJ\\Desktop\\PartA\\labels.txt");
            List<Pair<Integer,String>> features = DataParser.parseFeatures("C:\\Users\\PJ\\Desktop\\PartA\\features.txt");
			List<Vector<Integer>> testVExamples = DataParser.parseVectors("C:\\Users\\PJ\\Desktop\\PartA\\test_vectors.txt");
            List<Integer> testLabels = DataParser.parseLabels("C:\\Users\\PJ\\Desktop\\PartA\\test_labels.txt"); 
			
		    //anakateuei ta dedomena gia tuxaia taksinomisi
			List<Integer> indices = new ArrayList<>();
			for (int i = 0; i < vExamples.size(); i++) {
				indices.add(i);
			}
			Collections.shuffle(indices);

			//dhmiourgia shuffled datasets
			List<Vector<Integer>> shuffledVectors = new ArrayList<>();
			List<Integer> shuffledLabels = new ArrayList<>();

			for (int i = 0; i < vExamples.size(); i++) {
				shuffledVectors.add(vExamples.get(indices.get(i)));
				shuffledLabels.add(labels.get(indices.get(i)));
			}
		

		
		
			//85% ekpaideusi, 15% axiologisi (development set)
			int trainSize = (int) (0.85 * vExamples.size());
			List<Vector<Integer>> trainingVExamples = new ArrayList<>();
			List<Integer> trainingLabels = new ArrayList<>();
			List<Vector<Integer>>developmentVExamples = new ArrayList<>();
			List<Integer> developmentLabels  = new ArrayList<>();
			
		    //dianomi twn dedomenwn sta sets ekpaideusis kai axiologisis
			for(int i=0;i<vExamples.size();i++){
				if(i<trainSize){
					trainingVExamples.add(shuffledVectors.get(i));
					trainingLabels.add(shuffledLabels.get(i));
				}else{
					developmentVExamples.add(shuffledVectors.get(i));
					developmentLabels.add(shuffledLabels.get(i));
				}
			}
            

		
            //ektupwsi megethous kathe set
			System.out.println(" Train size: " + trainingVExamples.size());
			System.out.println(" Dev size: " + developmentVExamples.size());
			System.out.println(" Test size: " + testVExamples.size());
            //metrisi twn thetikwn kai arnitikwn paradeigmatwn sto development set
			long onesD = developmentLabels.stream().filter(x -> x.equals(1)).count();
			long zerosD = developmentLabels.stream().filter(x -> x.equals(0)).count();
            System.out.println("Dev positives: " + onesD  + ", Dev negatives: " + (developmentLabels.size() - onesD));
                  
			long onesTr = trainingLabels.stream().filter(x -> x.equals(1)).count();
			long zerosTr = trainingLabels.stream().filter(x -> x.equals(0)).count();
            System.out.println("Dev positives: " + onesTr  + ", Dev negatives: " + (trainingLabels.size() - onesTr));
            //ektelesi tou LearningCurveGenerator gia na dimiourgisoume kampuli mathisis
            String exportpath = "C:\\Users\\PJ\\Desktop\\PartA\\learning_curve_RandomForest.csv";
		    RandomForest forest  = LearningCurveGenerator.generateCurve(trainingVExamples,trainingLabels,developmentVExamples,developmentLabels,features,2000,exportpath);
			//axiologisi tou montelou sto test set
			evaluateModelV2(forest,testVExamples,testLabels);	
			
		} catch (IOException e) {
            System.err.println("Error reading input files: " + e.getMessage());
        }
    }
	
	
	//axiologisi enos montelou me confusion matrix kai upologismos metrikwn
	private static void evaluateModelV2(RandomForest classifier,List<Vector<Integer>> vectors,List<Integer> labels){
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

        //ypologismos metrikwn gia tin thetiki klasi (label = 1)
        double precision_pos = EvaluationMetrics.precision(TP, FP);
        double recall_pos    = EvaluationMetrics.recall(TP, FN);
        double f1_pos        = EvaluationMetrics.f1Score(precision_pos, recall_pos);

        //ypologismos metrikwn gia tin arnitiki klasi (label = 0)
        int TP_neg = TN;
        int FP_neg = FN;
        int FN_neg = FP;
        int TN_neg = TP;
        double precision_neg = EvaluationMetrics.precision(TP_neg, FP_neg);
        double recall_neg    = EvaluationMetrics.recall(TP_neg, FN_neg);
        double f1_neg        = EvaluationMetrics.f1Score(precision_neg, recall_neg);

        //ypologismos macro-averaged metrikwn
        double macro_precision = (precision_pos + precision_neg) / 2.0;
        double macro_recall    = (recall_pos + recall_neg)       / 2.0;
        double macro_f1        = (f1_pos + f1_neg)               / 2.0;

        //ypologismos micro-averaged metrikwn
        int totalTP = TP + TN; // from both "views"
        int totalFP = FP + FN;
        int totalFN = FN + FP;
        double micro_precision = EvaluationMetrics.precision(totalTP, totalFP);
        double micro_recall    = EvaluationMetrics.recall(totalTP, totalFN);
        double micro_f1        = EvaluationMetrics.f1Score(micro_precision, micro_recall);

        //ektupwsi apotelesmatwn
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