package Utils;

public class EvaluationMetrics {
    //ypologizei to precision (akriveia) twn provlepsewn
    public static double precision(int TP, int FP) {
        if (TP + FP == 0) return 0.0;
        return (double) TP / (TP + FP);
    }
    //ypologizei to recall (anaklisi) twn provlepsewn
    public static double recall(int TP, int FN) {
        if (TP + FN == 0) return 0.0;
        return (double) TP / (TP + FN);
    }
    //ypologizei to f1-score pou einai h zygismeni mesi timi tou precision kai recall
    public static double f1Score(double precision, double recall) {
        if (precision + recall == 0) return 0.0;
        return 2.0 * (precision * recall) / (precision + recall);
    }
}