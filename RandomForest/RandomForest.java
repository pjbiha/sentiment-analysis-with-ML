package RandomForest;
import java.util.*;

public class RandomForest {
    
    private List<ID3> trees = new ArrayList<>();
    private int numberOfTrees;
    private int maxFeaturesPerNode;  //posa xaraktiristika tha eks examinedontai se kathe node


    public RandomForest(List<Vector<Integer>> vExamples, List<Integer> labels, List<Pair<Integer,String>> features)
    {
        //arithmos dentrwn pou tha xrisimopoiisoume sto forest
        this.numberOfTrees = 150;

        //arithmos xaraktiristikwn pou tha eksetazontai se kathe node
        //typika gia classification: m = sqrt(#features)
        this.maxFeaturesPerNode = (int) Math.sqrt(features.size());

        //dimiourgia twn dentrwn
        init(vExamples, labels, features);
    }
    
    private void init(List<Vector<Integer>> vExamples,List<Integer> labels,List<Pair<Integer,String>> features)
    {
        Random rand = new Random();

        for(int i = 0; i < numberOfTrees; i++){
            //bagging: epilogi dedomenwn me epanalipseis
            List<Vector<Integer>> treeVExamples = new ArrayList<>();
            List<Integer> treeLabels = new ArrayList<>();
            for(int j = 0; j < vExamples.size(); j++){
                int randomIndex = rand.nextInt(vExamples.size());
                treeVExamples.add(vExamples.get(randomIndex));
                treeLabels.add(labels.get(randomIndex));
            }

            //dimiourgia enos ID3 dentrou pou tha epilegei tuxaia subsets xaraktiristikwn
            ID3 tree = new ID3(maxFeaturesPerNode);

            //ekpaideusi tou dentrou me ola ta xaraktiristika
            tree.train(treeVExamples, treeLabels, features);

            trees.add(tree);
        }
    }

    public int predict(Vector<Integer> input){
        int count1 = 0;
        int count0 = 0;
        int majority = (trees.size() + 1) / 2;  //orizoume orio gia na termatisoume noris

        for(ID3 tree : trees){
            if(tree.predict(input) == 1){
                count1++;
            } else {
                count0++;
            }
            //an exoume idi pleiopsifia gia 1 i 0, termatizoume
            if(count1 >= majority){
                return 1;
            }
            if(count0 >= majority){
                return 0;
            }
        }

        //an den exei ginei early stopping, epistrefoume tin pleiopsifia twn provlepsewn
        return (count1 >= count0) ? 1 : 0;
    }
}
