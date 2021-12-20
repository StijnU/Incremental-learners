/**
 * Copyright (c) DTAI - KU Leuven – All rights reserved. Proprietary, do not copy or distribute
 * without permission. Written by Pieter Robberechts, 2021
 */
import java.io.*;
import java.lang.reflect.Array;
import java.nio.charset.StandardCharsets;
import java.util.*;


/** This class is a stub for VFDT. */
public class Vfdt extends IncrementalLearner<Integer> {

  private int[] nbFeatureValues;
  private double delta;
  private double tau;
  private double nmin;


  private VfdtNode root;
  private ArrayList<VfdtNode> leaves;


  /**
   * Vfdt constructor
   *
   * <p>THIS METHOD IS REQUIRED
   *
   * @param nbFeatureValues are nb of values of each feature. e.g. nbFeatureValues[3]=5 means that
   *     feature 3 can have values 0,1,2,3 and 4.
   * @param delta is the parameter used for the Hoeffding bound
   * @param tau is the parameter that is used to deal with ties
   * @param nmin is the parameter that is used to limit the G computations
   */
  public Vfdt(int[] nbFeatureValues, double delta, double tau, int nmin) {
    this.nbFeatureValues = nbFeatureValues;
    this.delta = delta;
    this.tau = tau;
    this.nmin = nmin;
    this.parameters = new double[]{delta, tau, nmin};


    nbExamplesProcessed = 0;
    int[] possibleFeatures = new int[nbFeatureValues.length];
    for (int i = 0; i < nbFeatureValues.length; i++) possibleFeatures[i] = i;
    this.root = new VfdtNode(nbFeatureValues, possibleFeatures);

    // first there is only one leaf which is the root
    this.leaves = new ArrayList<VfdtNode>();
    this.leaves.add(root);
  }

  /**
   * This method will update the parameters of your model using the given example.
   *
   * <p>THIS METHOD IS REQUIRED
   *
   * @param example is a training example
   */
  @Override
  public void update(Example<Integer> example) {
    super.update(example);
    VfdtNode leafNode = findLeafNode(this.root, example);
    leafNode.addExample(example);

    ArrayList<VfdtNode> splittedLeaves = new ArrayList<VfdtNode>();
    ArrayList<VfdtNode[]> newLeaves = new ArrayList<VfdtNode[]>();

    for (VfdtNode leaf : leaves) {
      // first check if leaf node has enough instances
      if (leaf.getNbExamples() >= nmin) {
        HashMap<Integer, Double> igList = new HashMap<Integer, Double>();

        for (int splitFeature : leaf.getPossibleSplitFeatures()) {
          double ig = leaf.splitEval(splitFeature);
          igList.put(splitFeature, ig);
        }
        double highestIg = 0;
        int bestSplitFeature = -1;

        double secondHighestIg = 0;
        int secondBestSplitFeature = -1;

        for (int splitFeature : igList.keySet()) {
          double currIg = igList.get(splitFeature);

          if (currIg > highestIg) {
            secondHighestIg = highestIg;
            secondBestSplitFeature = bestSplitFeature;

            highestIg = currIg;
            bestSplitFeature = splitFeature;
          } else {
            if (currIg > secondHighestIg) {
              secondHighestIg = currIg;
              secondBestSplitFeature = splitFeature;
            }
          }
        }
        if (bestSplitFeature != -1 && secondBestSplitFeature != -1) {
          double deltaG = highestIg - secondHighestIg;

          double root = (1 * Math.log(2 / delta)) / (2 * leaf.getNbExamples());
          double hoeffding = Math.sqrt(root);
          if (deltaG > hoeffding || deltaG < tau) {
            VfdtNode[] newLeavesArr = leaf.split(bestSplitFeature, nbFeatureValues);
            newLeaves.add(newLeavesArr);
            splittedLeaves.add(leaf);
          }
        }
      }
    }
    updateLeaves(splittedLeaves, newLeaves);
  }


  private void updateLeaves(ArrayList<VfdtNode> lastLeafs, ArrayList<VfdtNode[]> newLeavesList){
    for (VfdtNode lastLeaf : lastLeafs) {
      leaves.remove(leaves.indexOf(lastLeaf));
    }
    for (VfdtNode[] newLeaves : newLeavesList) {
      for (VfdtNode newLeaf : newLeaves) {
        leaves.add(newLeaf);
      }
    }
  }
  /**
   *  Finds all leaf nodes
   */
  private VfdtNode[] findAllLeafNodes(VfdtNode root){
    VfdtNode children[] = root.getChildren();
    VfdtNode leafNodes[] = {};

    if (children == null){
      VfdtNode arr[] = {root};
      return arr;
    }
    for (VfdtNode child : children){
      leafNodes = concatWithArrayCopy(leafNodes, findAllLeafNodes(child));
    }

    return leafNodes;
  }

  /**
   *  Finds the leaf nodes of a tree according to example
   */
  private VfdtNode findLeafNode(VfdtNode root, Example<Integer> example) {
    VfdtNode[] children = root.getChildren();
    int splitFeature = root.getSplitFeature();

    if (children == null) {
      return root;
    }
    return findLeafNode(children[example.attributeValues[splitFeature]], example);
  }

  private static <Object> Object[] concatWithArrayCopy(Object[] array1, Object[] array2) {
    Object[] result = Arrays.copyOf(array1, array1.length + array2.length);
    System.arraycopy(array2, 0, result, array1.length, array2.length);
    return result;
}
  


  /**
   * Uses the current model to calculate the probability that an attributeValues belongs to class
   * "1";
   *
   * <p>THIS METHOD IS REQUIRED
   *
   * @param example is a the test instance to classify
   * @return the probability that attributeValues belongs to class "1"
   */
  @Override
  public double makePrediction(Integer[] example) {

    double prediction = 0;
    VfdtNode node = this.root;

    // find the leaf node of the example

    while (node.getChildren() != null){
      VfdtNode child = node.getChildren()[example[node.getSplitFeature()]];

      // check if there are enough instances in child node, if not use the parent node for prediction
      int instanceCount = child.getNbExamples();

      // only use node for prediction if it has atleast nmin counts in total -> this doesnt mean has nmin examples
      // every parent node will have this because a split will only happen in a node when nmin examples are seen in this node
      // using nmin for this, could use other variable as this is not the true intention of nmin
      if (instanceCount > 50) {
        // go to next node and iterate through while loop
        // while loop also stops when there are no childs thus if a child has enough instance counts -> child is used for prediction
        node = child;
      }
      else {
        // stop going to next node because child has to few counts -> use parent node for prediction
        break;
      }

      /**
      for (VfdtNode child : childs){
        int nijk[][][] = child.getInstances();
        if (nijk[0][0][node.getSplitFeature()] == example[node.getSplitFeature()]){
          node = child;
        }
      }
       **/
    }

    // node is now the leaf node of the example
    int nijk[][][] = node.getInstances();

    // prediction is 0.5 when no examples
    if (nijk.length == 0){
      return 0.5;
    }
    double examplePositiveSum = 0;
    double totalPositiveSum = 0;
    for (int i = 0; i < nijk.length; i++) {
      try {
        examplePositiveSum += nijk[i][example[i]][1];
        /**
        for (int[] featureValues : nijk[i]) {
          totalPositiveSum += featureValues[1];
        }
         */
        totalPositiveSum += nijk[i][example[i]][0];
      }
      catch (NullPointerException e){
        // not in nijk
      }
      if (totalPositiveSum > 0) {
        prediction = examplePositiveSum / (totalPositiveSum + examplePositiveSum);
      }
      else {
        prediction = 0;
      }
    }

    return prediction;
  }

  /**
   * Writes the current model to a file.
   *
   * <p>The written file can be read in with readModel.
   *
   * <p>THIS METHOD IS REQUIRED
   *
   * @param path the path to the file
   * @throws IOException
   */
  @Override
  public void writeModel(String path) throws IOException {
    VfdtNode node = this.root;
    File modelFile = new File(path);
    modelFile.createNewFile();
    FileWriter writer = new FileWriter(modelFile);

    // first calculate node amount ant print in the first line
    int nodeAmount = countNodes(node, 0);
    writer.write(nodeAmount + "\n");

    // write all the nodes with their respective node ID
    int id = setLeafNodeIds(this.root, 0);
    setDecisionNodeIds(this.root, id);
    writeLeafNodes(this.root, writer);
    writeDecisionNodes(this.root, writer);
    writer.close();

  }

  private int setLeafNodeIds(VfdtNode node, int id){
    if (node.getChildren() == null){
      node.setId(id);
      return id + 1;
    }
    else{
      VfdtNode children[] = node.getChildren();
      for (VfdtNode child : children){
        id = setLeafNodeIds(child, id);
      }
    }
    return id;
  }

  private int setDecisionNodeIds(VfdtNode node, int id){
    if (node.getChildren() == null){
      return id;
    }
    else{
      VfdtNode children[] = node.getChildren();
      for (VfdtNode child : children){
        id = setDecisionNodeIds(child, id);
      }
      node.setId(id);
    }
    return id + 1;
  }

  private void writeLeafNodes(VfdtNode node, FileWriter writer) throws IOException{
    if (node.getChildren() == null){
      // leaf nodes have null children, leaf nodes are then printed 
      // the node ID is actually the total amount of nodes - the given nodeID this makes us print the nodes bottom up
      writer.write(node.getId()+ " L pf:" + intArrayToString(node.getPossibleSplitFeatures()) + " nijk:" + nijkToString(node.getInstances()) + "\n");
    }else{
      VfdtNode children[] = node.getChildren();
      for (VfdtNode child : children){
        writeLeafNodes(child, writer);
      }
    }
  }

  private void writeDecisionNodes(VfdtNode node, FileWriter writer) throws IOException {
    if (node.getChildren() != null){
      VfdtNode children[] = node.getChildren();
      for (VfdtNode child : children){
        writeDecisionNodes(child, writer);
      }
      writer.write(node.getId() + " D f:" + node.getSplitFeature() + " ch:" + nodesToString(children) + "\n");
    }
  }

  private String intArrayToString(int[] arr){
    String str = "[";
    for (int val : arr){
      String valStr = String.valueOf(val);
      str += valStr;
      str += ",";
    }
    return str + "]";
  }

  private String nijkToString(int[][][] nijk){
    String nijkStrings = "[";
    for (int featureId = 0; featureId < nijk.length; featureId++){
      if (nijk[featureId] != null){
        int featureVal = 0;
        for (int[] counts : nijk[featureId]){
          for (int classVal = 0; classVal < 2; classVal++) {
            if (counts[classVal] > 0) {
              nijkStrings += featureId + ":" + featureVal + ":" + classVal + ":" + counts[classVal] + ",";
            }
          }
          featureVal++;
        }
      }
    }
    return nijkStrings + "]";
  }

  private String nodesToString(VfdtNode[] nodes){
    String nodeString = "[";
    for (VfdtNode node : nodes){
      nodeString += node.getId() + ",";
    }
    return nodeString + "]";
  }

  private static int countNodes(VfdtNode node, int amount){
    int totalAmount = 0;
    if (node.getChildren() == null){
      return 1;
    }
    for (VfdtNode child : node.getChildren()){
      totalAmount += countNodes(child, amount);
    }
    return totalAmount + 1;
  }


  /**
   * Reads in the model in the file and sets it as the current model. Sets the number of examples
   * processed.
   *
   * <p>THIS METHOD IS REQUIRED
   *
   * @param path the path to the model file
   * @param nbExamplesProcessed the nb of examples that were processed to get to the model in the
   *     file.
   * @throws IOException
   */
  @Override
  public void readModel(String path, int nbExamplesProcessed) throws IOException {
    super.readModel(path, nbExamplesProcessed);

    /* FILL IN HERE */
    // init empty tree
    try {
      File modelFile = new File(path);
      Scanner reader = new Scanner(modelFile);

  
      // get first value of text file the node amount
      int nodeAmount = Integer.parseInt(reader.nextLine());
      ArrayList<String[]> nodeStrings = new ArrayList<>();

      // parse all next lines of the text file
      while (reader.hasNextLine()){
        String nextLine = reader.nextLine();
        String parsedLine[] = parseNodes(nextLine);
        nodeStrings.add(parsedLine);
      }
      reader.close();

      // dont know this is correct, feature values aren't given
      int[] nbFeatureValues = readAllFeatureValues(nodeStrings);

      createTree(nodeStrings, nbFeatureValues);

      
    }
    finally{
      // maybe print error message with exception
    }
  }

  private int[] readAllFeatureValues(ArrayList<String[]> nodeStrings){
    HashMap<Integer, Integer> featureAmounts = new HashMap<Integer, Integer>();

    for (String[] nodeString : nodeStrings){
      if (nodeString[2].charAt(0) == 'f') {
        int nbValues = nodeString[3].substring(4, nodeString[3].length() - 2).split(",").length;
        int featureId = Integer.parseInt(nodeString[2].substring(2,3));
        featureAmounts.put(featureId, nbValues);
      }
      else {
        // add feature amounts from nijk
        // extract nijks
        if (!nodeString[3].equals("nijk:[]")) { // check if there is an actual value for nijk
          String[] nijks = nodeString[3].substring(6, nodeString[3].length() - 2).split(",");
          for (String nijk : nijks) {
            int featureId = Integer.parseInt(nijk.substring(0,1));
            if (featureAmounts.keySet().contains(featureId)) {
              int newAmount = featureAmounts.get(featureId) + 1;
              featureAmounts.put(featureId, newAmount);
            } else {
              featureAmounts.put(featureId, 1);
            }
          }
        }
      }
    }

    // all feature amounts are put into hashmap, now we need to extract them and put them in order in an array

    // init array
    int[] nbFeatureValues = new int[featureAmounts.size()];
    for (int id : featureAmounts.keySet()){
      nbFeatureValues[id] = featureAmounts.get(id);
    }
    return nbFeatureValues;
  }

  private String[] parseNodes(String node){
    String parsedString[] = node.split(" ");
    return parsedString;
  }

  private void createTree(ArrayList<String[]> parsedTreeStrings, int[] nbFeatureValues){
    // get the last parsed line which is the root node
    String[] rootNode = parsedTreeStrings.get(parsedTreeStrings.size() - 1);

    // get values for root node
    int[] possibleSplitFeatures = {};
      
    if (rootNode[2].charAt(0) == 'p'){
        String[] possibleSplitFeaturesStrings = rootNode[2].substring(4, rootNode[2].length() - 2).split(",");
        possibleSplitFeatures = new int[possibleSplitFeaturesStrings.length];
        for(int i = 0; i < possibleSplitFeaturesStrings.length; i++){
            possibleSplitFeatures[i] = Integer.parseInt(possibleSplitFeaturesStrings[i]);
        }
      }else{
        possibleSplitFeatures = new int[]{Integer.parseInt(rootNode[2].substring(2))};
      }

    // create root node
    this.root = new VfdtNode(nbFeatureValues, possibleSplitFeatures);
    addToTree(this.root, rootNode, parsedTreeStrings, nbFeatureValues);
  }

  private void addToTree(VfdtNode node, String[] nodeString, ArrayList<String[]> ParsedTreeStrings, int[] nbFeatureValues){
    String childString = nodeString[nodeString.length - 1];
    if (childString.charAt(0) != 'n'){
      childString = childString.substring(4, childString.length() - 2);
      String childIds[] = childString.split(",");

      int splitFeature = Integer.parseInt(nodeString[2].substring(2));

      ArrayList<VfdtNode> childs = new ArrayList<VfdtNode>();
      HashMap<VfdtNode, String[]> childsWithString = new HashMap<VfdtNode, String[]>();

      for (String childId : childIds){
        for (String[] parsedNodeString : ParsedTreeStrings){
          if (parsedNodeString[0].equals(childId)){
            // get values for node
            int[] possibleSplitFeatures = {};

            if (parsedNodeString[2].charAt(0) == 'p'){
              String[] possibleSplitFeaturesStrings = parsedNodeString[2].substring(4, parsedNodeString[2].length() - 2).split(",");
              possibleSplitFeatures = new int[possibleSplitFeaturesStrings.length];
              for(int i = 0; i < possibleSplitFeaturesStrings.length; i++){
                  possibleSplitFeatures[i] = Integer.parseInt(possibleSplitFeaturesStrings[i]);
                }
            }else{
              possibleSplitFeatures = new int[]{Integer.parseInt(parsedNodeString[2].substring(2))};
            }

            // create node
            VfdtNode childNode = new VfdtNode(nbFeatureValues, possibleSplitFeatures);
            childs.add(childNode);
            childsWithString.put(childNode, parsedNodeString);
          }   
        }
      }

      VfdtNode[] nodes = new VfdtNode[childs.size()];
      nodes = childs.toArray(nodes);

      node.addChildren(splitFeature, nodes);

      for (VfdtNode child : nodes){
        addToTree(child, childsWithString.get(child), ParsedTreeStrings, nbFeatureValues);
      }
    }
    else {
      // add nijk to node
      int[][][] nijk = stringToNijk(childString);
      if (nijk != null) {
        node.setInstances(nijk);
      }
    }
  }

  private int[][][] stringToNijk(String nijkString){
    try {
      String[] parsedString = nijkString.substring(6, nijkString.length() - 1).split(",");


    // hashmap with for each featureId its corresponding feature values
    HashMap<Integer, ArrayList<Integer>> featureIds = new HashMap<Integer, ArrayList<Integer>>();



    // loop to add all features and features values
    for (String nijk : parsedString){
      String[] parsedNijk = nijk.split(":");

      // extract values from parsed nijk
      int featureId = Integer.parseInt(parsedNijk[0]);
      int featureValue = Integer.parseInt(parsedNijk[1]);


      if (!featureIds.containsKey(featureId)){
        featureIds.put(featureId, new ArrayList<Integer>());
        featureIds.get(featureId).add(featureValue);

      }
      else{
        if (!featureIds.get(featureId).contains(featureValue)){
            featureIds.get(featureId).add(featureValue);
        }
      }
    }

    // sort arraylist just to make sure but not really needed normally
    for (int key : featureIds.keySet()){
      Collections.sort(featureIds.get(key));
    }

    // init featureValuecounts
    // this is a hashmap with for each feature id the different count for each feature value
    HashMap<Integer, HashMap<Integer, Integer[]>> featureValueCounts = new HashMap<Integer, HashMap<Integer, Integer[]>>();
    for (int key : featureIds.keySet()){
      featureValueCounts.put(key, new HashMap<Integer, Integer[]>());
      for (int featureVal : featureIds.get(key)){
        Integer[] counts = new Integer[2];
        counts[0] = 0; counts[1] = 0;
        featureValueCounts.get(key).put(featureVal, counts);
      }
    }

    // loop to count all class value counts for each feature value
    for (String nijk : parsedString){
      String[] parsedNijk = nijk.split(":");

      // extract values from parsed nijk
      int featureId = Integer.parseInt(parsedNijk[0]);
      int featureValue = Integer.parseInt(parsedNijk[1]);
      int classValue = Integer.parseInt(parsedNijk[2]);
      int count = Integer.parseInt(parsedNijk[3]);

      featureValueCounts.get(featureId).get(featureValue)[classValue] = count;
    }

    // construct nijk
    ArrayList<ArrayList<Integer[]>> nijkList = new ArrayList<ArrayList<Integer[]>>();
    ArrayList<Integer> keys = new ArrayList<Integer>(featureValueCounts.keySet());

    // again sort keys just to make sure it's in right order
    Collections.sort(keys);

    for (int key : keys){
      // generate list for each feature to add in nijk
      ArrayList<Integer[]> featureValuesList = new ArrayList<Integer[]>();
      int lastKey = 0;
      for (int featureValueKey : featureValueCounts.get(key).keySet()) {
        while (featureValueKey > lastKey){
          featureValuesList.add(new Integer[2]);
          lastKey+= 1;
        }
        featureValuesList.add(featureValueCounts.get(key).get(featureValueKey));
        lastKey = featureValueKey;
        if (lastKey == 0){
          lastKey += 1;
        }
      }
      nijkList.add(featureValuesList);
    }

    // init nijk with all features
    int[][][] nijk = new int[keys.get(keys.size() - 1) + 1][][];
    for (int key : featureIds.keySet()){
      nijk[key] = new int[featureIds.get(key).size() + 1][2];
    }

    // add all counts to nijk array
    int i = 0;
    for (ArrayList<Integer[]> featureList : nijkList){
      int j = 0;
      for (Integer[] classCount : featureList){
        int[] actualClassCount = new int[2];
        try {
          actualClassCount[0] = classCount[0];
          actualClassCount[1] = classCount[1];
          nijk[keys.get(i)][j] = actualClassCount;
        }
        catch (NullPointerException e){
          // the values of classcount are null because there is no count for these feature values
        }

        j++;
      }
      i++;
    }
    return nijk;
    }
    catch (NumberFormatException e){
      return null;
    }
  }


  /**
   * Return the visualization of the tree.
   *
   * <p>DO NOT CHANGE THIS METHOD.
   *
   * @return Visualization of the tree
   */
  public String getVisualization() {
    return root.getVisualization("");
  }


  /**
   * This runs your code to generate the required output for the assignment.
   *
   * <p>DO NOT CHANGE THIS METHOD.
   */
  public static void main(String[] args) {
    if (args.length < 7) {
      System.err.println(
          "Usage: java Vfdt <delta> <tau> <nmin> <data set> <nbFeatureValues> <output file>"
              + " <reportingPeriod> [-writeOutAllPredictions]");
      throw new Error("Expected 7 or 8 arguments, got " + args.length + ".");
    }
    try {
      // parse input
      double delta = Double.parseDouble(args[0]);
      double tau = Double.parseDouble(args[1]);
      int nmin = Integer.parseInt(args[2]);
      Data<Integer> data = new IntData(args[3], ",");
      int[] nbFeatureValues = parseNbFeatureValues(args[4]);
      String out = args[5];
      int reportingPeriod = Integer.parseInt(args[6]);
      boolean writeOutAllPredictions =
          args.length > 7 && args[7].contains("writeOutAllPredictions");

      // initialize learner
      Vfdt vfdt = new Vfdt(nbFeatureValues, delta, tau, nmin);
      // generate output for the learning curve
      vfdt.makeLearningCurve(data, 0.5, out + ".vfdt", reportingPeriod, writeOutAllPredictions);
    } catch (IOException e) {
      System.err.println(e.toString());
    }
  }

  /**
   * This method parses the file that specifies the nb of possible values for each feature.
   *
   * <p>DO NOT CHANGE THIS METHOD.
   */
  private static int[] parseNbFeatureValues(String path) throws IOException {
    BufferedReader reader = new BufferedReader(new FileReader(path));
    reader.readLine(); // skip header
    String[] splitLine = reader.readLine().split(",");
    int[] nbFeatureValues = new int[splitLine.length];

    for (int i = 0; i < nbFeatureValues.length; i++) {
      nbFeatureValues[i] = Integer.parseInt(splitLine[i]);
    }
    reader.close();
    return nbFeatureValues;
  }
}
/**
 * This class implements Data for Integers
 *
 * <p>DO NOT CHANGE THIS CLASS
 */
class IntData extends Data<Integer> {

  public IntData(String dataDir, String sep) throws FileNotFoundException {
    super(dataDir, sep);
  }

  @Override
  protected Integer parseAttribute(String attrString) {
    return Integer.parseInt(attrString);
  }

  @Override
  protected Integer[] emptyAttributes(int i) {
    return new Integer[i];
  }

  public static void main(String[] args) {
    if (args.length < 3) {
      throw new Error("Expected 2 arguments, got " + args.length + ".");
    }

    try {
      Data<Integer> d = new IntData(args[0], args[1]);
      d.print();
    } catch (FileNotFoundException e) {
      System.err.print(e.toString());
    }
  }
}
