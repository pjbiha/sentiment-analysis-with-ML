package Utils;

import java.io.*;
import java.util.*;

public class VectorImporter<T extends Number> {

    /**
     //diavazei to lexilogio (feature lekseis) apo ena arxeio (mia ana grammh)
     */
    public static List<String> importFeatureWordsFromFile(String filePath) {
        List<String> vocabulary = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            System.out.println("Importing feature words from: " + filePath);
            String line;
            while ((line = reader.readLine()) != null) {
                vocabulary.add(line.trim());//prosthetei kathe leksi sti lista
            }
            System.out.println("Feature words imported: " + vocabulary.size());
        } catch (IOException e) {
            System.err.println("Error reading feature words file: " + e.getMessage());
        }
        return vocabulary;
    }

    //diavazei mia lista me dianysmata apo ena arxeio (gia naive bayes)
    //kathe grammh exei times diaxwrismenes me komma, px "1,0,1,1,0,..."
    public static <T extends Number> List<Vector<T>> importVectors(String filePath, Class<T> type) {
        List<Vector<T>> vectors = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            System.out.println("Importing dense vectors from: " + filePath);
            String line;
            while ((line = reader.readLine()) != null) {
                String[] parts = line.split(",");
                Vector<T> vector = new Vector<>();
                for (String p : parts) {
                    vector.add(parseNumber(p.trim(), type));//metatrepei kathe stoixeio sto swsto typou
                }
                vectors.add(vector);
            }
            System.out.println("Dense vectors imported: " + vectors.size());
        } catch (IOException e) {
            System.err.println("Error reading dense vectors file: " + e.getMessage());
        }
        return vectors;
    }

    //diavazei sparse vectors apo ena arxeio kai ta apothikeuei ws hashmap (gia logistic regression)
    //kathe grammh exei times diaxwrismenes me komma, alla kratame mono ta "1"
    public static List<HashMap<Integer, Integer>> importSparseVectors(String filePath) {
        List<HashMap<Integer, Integer>> vectors = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            System.out.println("Importing sparse vectors from: " + filePath);
            String line;
            while ((line = reader.readLine()) != null) {
                String[] parts = line.split(",");
                HashMap<Integer, Integer> vector = new HashMap<>();
                for (int j = 0; j < parts.length; j++) {
                    int value = Integer.parseInt(parts[j].trim());
                    if (value == 1) {
                        vector.put(j, 1); //kratame mono ta mi-midenika stoixeia
                    }
                }
                vectors.add(vector);
            }
            System.out.println("Sparse vectors imported: " + vectors.size());
        } catch (IOException e) {
            System.err.println("Error reading sparse vectors file: " + e.getMessage());
        }
        return vectors;
    }

    //diavazei ta labels apo ena arxeio, mia mia grammh
    public static <T extends Number> List<T> importLabels(String filePath, Class<T> type) {
        List<T> labels = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            System.out.println("Importing labels from: " + filePath);
            String line;
            while ((line = reader.readLine()) != null) {
                labels.add(parseNumber(line.trim(), type));
            }
            System.out.println("Labels imported: " + labels.size());
        } catch (IOException e) {
            System.err.println("Error reading labels file: " + e.getMessage());
        }
        return labels;
    }

    //voithitiki methodos gia metatropi string se arithmitiko typo
    private static <T extends Number> T parseNumber(String value, Class<T> type) {
        if (type == Integer.class) {
            return type.cast(Integer.parseInt(value));
        } else if (type == Double.class) {
            return type.cast(Double.parseDouble(value));
        } else if (type == Float.class) {
            return type.cast(Float.parseFloat(value));
        } else if (type == Long.class) {
            return type.cast(Long.parseLong(value));
        } else {
            throw new IllegalArgumentException("Unsupported numeric type: " + type.getSimpleName());
        }
    }
}
