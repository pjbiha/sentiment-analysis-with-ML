package RandomForest;

import java.util.*;

public class ID3 {

    private Node root;
    private int maxFeaturesPerSplit; //arithmos twn xaraktiristikwn pou tha eks examinedei se kathe node

    //constructor gia na oristei to plithos twn xaraktiristikwn pou prepei na eksetastoun se kathe node
    public ID3(int maxFeaturesPerSplit) {
        this.maxFeaturesPerSplit = maxFeaturesPerSplit;
    }

    public Node getRoot() {
        return this.root;
    }

    //ekpaideuei to dentro me oles ta diathesima features
    public void train(List<Vector<Integer>> vExamples,
                      List<Integer> labels,
                      List<Pair<Integer,String>> featuresPairs) {
        //xrisimopoioume HashSet anti gia lista gia taxyteri anazitisi twn idi eksetazomenwn fetaures
        this.root = buildTree(vExamples, labels, featuresPairs, new HashSet<>());
    }

    public int predict(Vector<Integer> input){
        return predictRecursive(root, input);
    }

    private int predictRecursive(Node node, Vector<Integer> input){
        if(node.isLeaf()){
            return node.getLabel();//an ftasoume se fyllo, epistrefoume to label
        }
        //an i timi tou feature einai 0, pame aristera, alliws dexia
        if(input.get(node.getFeatureIndex()) == 0){
            return predictRecursive(node.getLeftChild(), input);
        } else {
            return predictRecursive(node.getRightChild(), input);
        }
    }

     //anakdromiki domisi tou dentrou
    private Node buildTree(
            List<Vector<Integer>> vExamples,
            List<Integer> labels,
            List<Pair<Integer,String>> featuresPairs,
            Set<Integer> seenFeaturesIndexes
    ){
        //early stopping an i purity enos node einai 95%
        int count1 = 0;
        int count0 = 0;
        for(int label : labels){
            if(label == 1) {
                count1++;
            } else {
                count0++;
            }
        }
        int maxCount = Math.max(count1, count0);
        // an to 95% h parapanw twn samples auotu tou node exoun to idio label, kanto fullo
        if((0.90 <= (double) maxCount / labels.size())) {
            return new Node(maxCount == count1 ? 1 : 0);
        }

        //an den uparxoun diaxwrismena features, epistrefoume to pleionotiko label
        if(seenFeaturesIndexes.size() == featuresPairs.size()){
            return new Node(majority(labels));
        }

        //an h lista einai adeia, epistrepse thn pleiopshfia
        if(featuresPairs.isEmpty()){
            return new Node(majority(labels));
        }

        //epilogi tou kaluterou features apo ena tixaio uposynolo twn non seen xaraktiristikwn
        int bestIndexInFeaturesList = bestFeatureIndex(vExamples, labels, seenFeaturesIndexes, featuresPairs);
        if (bestIndexInFeaturesList == -1) {
            return new Node(majority(labels));
        }

       //epilogi feature pou tha xrisimopoiithei gia to split
        String bestFeatureName = featuresPairs.get(bestIndexInFeaturesList).getSecond();
        int realIndexInVector  = featuresPairs.get(bestIndexInFeaturesList).getFirst();

        //dimiourgia esoterikou kombou
        Node node = new Node(bestFeatureName, realIndexInVector);

        //dimiourgia twn paidikwn kombwn analoga me tin timi tis xaraktiristikis
        List<Vector<Integer>> leftChildData  = new ArrayList<>();
        List<Integer>         leftLabels     = new ArrayList<>();
        List<Vector<Integer>> rightChildData = new ArrayList<>();
        List<Integer>         rightLabels    = new ArrayList<>();

        for(int i=0; i<vExamples.size(); i++){
            Vector<Integer> example = vExamples.get(i);
            if(example.get(realIndexInVector) == 0){
                leftChildData.add(example);
                leftLabels.add(labels.get(i));
            } else {
                rightChildData.add(example);
                rightLabels.add(labels.get(i));
            }
        }

        //antigrafi tou set twn eks examinedwn xaraktiristikwn gia kathe paidi
        Set<Integer> leftSeenFeatures  = new HashSet<>(seenFeaturesIndexes);
        Set<Integer> rightSeenFeatures = new HashSet<>(seenFeaturesIndexes);

        leftSeenFeatures.add(bestIndexInFeaturesList);
        rightSeenFeatures.add(bestIndexInFeaturesList);

        //anakdromiki dimiourgia twn paidikwn kombwn
        node.setLeftChild(buildTree(leftChildData, leftLabels, featuresPairs, leftSeenFeatures));
        node.setRightChild(buildTree(rightChildData, rightLabels, featuresPairs, rightSeenFeatures));

        return node;
    }

    //epistrefei to pleionotiko label apo mia lista labels
    private int majority(List<Integer> labels){
        int count1=0;
        for(int label : labels){
            if(label == 1) {
                count1++;
            }
        }
        int count0 = labels.size() - count1;
        return (count1 > count0) ? 1 : 0;
    }

    //epilogi tou kaluterou xaraktiristikou apo ena tixaio uposynolo twn non seen xaraktiristikwn
    private int bestFeatureIndex(List<Vector<Integer>> vExamples,List<Integer> labels,Set<Integer> seenFeaturesIndexes,List<Pair<Integer,String>> features)
    {
        //anagnorizei tous deiktes twn xaraktiristikwn pou den exoun akoma eksetastei
        List<Integer> unseen = new ArrayList<>();
        for(int i = 0; i < features.size(); i++){
            if(!seenFeaturesIndexes.contains(i)){
                unseen.add(i);
            }
        }
        if(unseen.isEmpty()) {
            return -1;
        }

        //anakateuei kai epilegei tuxaia mexri 'maxFeaturesPerSplit' apo ta non seen xaraktiristika
        Collections.shuffle(unseen);
        int subsetSize = Math.min(maxFeaturesPerSplit, unseen.size());
        List<Integer> candidateFeatures = unseen.subList(0, subsetSize);

         //metaksi twn epilogon, epilegei to xaraktiristiko me to megalutero Information Gain (IG)
        double baseEntropy = entropy(labels);
        double maxIg = -1.0;
        int bestIndex = -1;

        for(int index : candidateFeatures){
            int featureIndexInVector = features.get(index).getFirst();
            double ig = baseEntropy - conditionalEntropy(vExamples, labels, featureIndexInVector);
            if(ig > maxIg){
                maxIg = ig;
                bestIndex = index; // index in 'features'
            }
        }

        return bestIndex; 
    }
    //ypologismos synthitikou entropy gia ena sygkekrimeno xaraktiristiko
    private double conditionalEntropy(List<Vector<Integer>> vExamples,List<Integer> labels,int featureIndex)
    {
        List<Integer> labels0 = new ArrayList<>();
        List<Integer> labels1 = new ArrayList<>();

        for(int i=0; i<vExamples.size(); i++){
            if(vExamples.get(i).get(featureIndex) == 0){
                labels0.add(labels.get(i));
            } else {
                labels1.add(labels.get(i));
            }
        }
        double pF0 = (double) labels0.size() / labels.size();
        double pF1 = (double) labels1.size() / labels.size();

        return pF0 * entropy(labels0) + pF1 * entropy(labels1);
    }
    //ypologismos entropy gia mia lista labels
    private double entropy(List<Integer> labels){
        //Laplace smoothing
        int totalCount = labels.size() + 2;
        int posCount   = 1;
        int negCount   = 1;

        for(int label: labels){
            if(label == 1){
                posCount++;
            } else {
                negCount++;
            }
        }

        double pPos = (double) posCount / totalCount;
        double pNeg = (double) negCount / totalCount;

        return -(pPos * log2(pPos) + pNeg * log2(pNeg));
    }

    private static double log2(double x) {
        return Math.log(x) / Math.log(2);
    }
}
