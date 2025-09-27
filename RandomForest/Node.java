package RandomForest;


public class Node{

	private int label;
		
	private String feature;
	private int featureIndex;
	private Node leftChild;  //to aristero paidi (gia times 0)
	private Node rightChild;  //to deksi paidi (gia times 1)

	
	
	//constructor gia eswteriko node (den einai fyllo)
	public Node(String feature,int featureIndex){	
		this.feature=feature;
		this.label=-999;//to -999 deixnei oti den einai fyllo
		this.featureIndex=featureIndex;
	}
	
	public int getFeatureIndex(){
		return this.featureIndex;
	}
	// constructor gia leaf node
	public Node(int label){
		this.label=label;
		this.feature=null;
	}
	
	public String getFeature(){
		return this.feature;
	}
	
	public Node getLeftChild(){
		return this.leftChild;
	}
	
	public Node getRightChild(){
		return this.rightChild;
	}
	
	public void setLeftChild(Node child){
		this.leftChild=child;
	}
	
	public void setRightChild(Node child){
		this.rightChild=child;
	}
	
	public boolean isLeaf(){
		return label!=-999;
	}
	
	public int getLabel(){
		return this.label;
	}

}