package RandomForest;

import java.io.*;
import java.util.*;

public class DataParser{
    //diavazei ta dianysmata apo ena arxeio kai ta epistrefei ws lista apo vectors
    public static List<Vector<Integer>> parseVectors(String fileName) throws IOException {
        List<Vector<Integer>> vectors = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(fileName))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] parts = line.trim().split(",");
                Vector<Integer> vector = new Vector<>();
                for (String part : parts) {
                    vector.add(Integer.parseInt(part.trim())); //metatrepei kathe timi se akeraio
                }
                vectors.add(vector);
            }
        }
        return vectors;
    }
    //diavazei ta labels apo ena arxeio kai ta epistrefei ws lista apo akeraious
    public static  List<Integer> parseLabels(String fileName) throws IOException {
        List<Integer> labels = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(fileName))) {
            String line;
            while ((line = br.readLine()) != null) {
                labels.add(Integer.parseInt(line.trim()));//metatrepei tin grammh se akerai
            }
        }
        return labels;
    }
    //diavazei ta xaraktiristika (features) apo ena arxeio kai ta epistrefei ws lista apo Pair<Integer,String>
    public static List<Pair<Integer,String>> parseFeatures(String fileName) throws IOException {
        List<Pair<Integer,String>> features = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(fileName))) {
            String line;
            boolean isFirstLine = true;
            while ((line = br.readLine()) != null) {
                if (isFirstLine) {
                    isFirstLine = false; //paraleipoume tin prwti grammh (header)
                    continue;
                }
                //dianoume tin grammh me to prwto ":"
                String[] parts = line.split(":", 2);
                if (parts.length == 2) {
                    features.add(new Pair<>(Integer.parseInt(parts[0].trim()),parts[1].trim()));
                }
            }
        }
        return features;
    }
    
}


