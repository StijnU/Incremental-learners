import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

/**
 * Copyright (c) DTAI - KU Leuven – All rights reserved. Proprietary, do not copy or distribute
 * without permission. Written by Pieter Robberechts, 2021
 */

/** This class is a stub for VFDT. */
public class VfdtNode {

  private VfdtNode[] children; /* child children (null if node is a leaf) */

  private final int[] possibleSplitFeatures; /* The features that this node can split on */

  private int splitFeature; /* splitting feature */

  private int[][][] nijk; /* instance counts (see paper) */

  /* FILL IN HERE */

  private int id; /* only used for writing models */

  private int nbExamples;

  /**
   * Create and initialize a leaf node.
   *
   * <p>THIS METHOD IS REQUIRED.
   *
   * @param nbFeatureValues are the nb of values for each feature in this node. If a feature has k
   *     values, then the values are [0:k-1].
   */
  public VfdtNode(int[] nbFeatureValues, int[] possibleSplitFeatures) {
    this.possibleSplitFeatures = possibleSplitFeatures;
    this.id = -1;

    // add for each feature a list to nijk
    this.nijk = new int[nbFeatureValues.length][][];
    for (int featureCounts = 0; featureCounts < nbFeatureValues.length; featureCounts++){
      // add only for splitfeatures the class lists
      for (int splitFeature : possibleSplitFeatures){
        if (featureCounts == splitFeature){
          this.nijk[featureCounts] = new int[nbFeatureValues[featureCounts]][];
          for (int classValueCounts = 0; classValueCounts < nbFeatureValues[featureCounts]; classValueCounts++) {
            // add for each possible class value a count (positive and negative)
            this.nijk[featureCounts][classValueCounts] = new int[2];
            // initialize both on 0
            this.nijk[featureCounts][classValueCounts][0] = 0;
            this.nijk[featureCounts][classValueCounts][1] = 0;
          }
        }
      }
    }
    this.children = null;
  }

  /**
   * Add and example to the node
   */
  public void addExample(Example<Integer> example) {
    for (int splitFeature : possibleSplitFeatures){
      nijk[splitFeature][example.attributeValues[splitFeature]][example.classValue] += 1;
    }
    nbExamples += 1;
  }

  public int getNbExamples(){
    return nbExamples;
  }

  /**
   * Split on feature value
   */
  public void split(int splitFeature, int[] nbFeatureValues){
    ArrayList<VfdtNode> childs = new ArrayList<VfdtNode>();

    // create new possible split features list
    int[] newPossibleSplitFeatures = new int[possibleSplitFeatures.length - 1];
    for (int i = 0, j = 0; i < newPossibleSplitFeatures.length; i++){
      if (possibleSplitFeatures[i] != splitFeature){
        newPossibleSplitFeatures[j] = possibleSplitFeatures[i];
        j++;
      }
    }

    // create childs
    for (int i = 0; i < nbFeatureValues[splitFeature]; i++){
      childs.add(new VfdtNode(nbFeatureValues, newPossibleSplitFeatures));
    }
    addChildren(splitFeature, childs.toArray(new VfdtNode[nbFeatureValues[splitFeature]]));
  }

  /**
   * Turn a leaf node into a internal node.
   *
   * <p>THIS METHOD IS REQUIRED.
   *
   * @param splitFeature is the feature to test on this node.
   * @param nodes are the children (the index of the node is the value of the splitFeature).
   */
  public void addChildren(int splitFeature, VfdtNode[] nodes) {
    if (nodes == null) throw new IllegalArgumentException("null children");
    this.children = nodes;
    this.splitFeature = splitFeature;
    //nbSplits++;
  }

  public int getSplitFeature(){
    return this.splitFeature;
  }
  /**
   * 
   * @return
   */
  public VfdtNode[] getChildren(){
    return this.children;
  }

  /**
   *
   * @return
   */
  public int[] getPossibleSplitFeatures(){
    return this.possibleSplitFeatures;
  }

  /**
   *
   * @return
   */
  public int[][][] getInstances(){
    return this.nijk;
  }

  /**
   * Add instance to node
   * @param nijk
   * @return
   */
  public void setInstances(int[][][] nijk){
    this.nijk = nijk;
  }

  /**
   * Returns the leaf node corresponding to the test attributeValues.
   *
   * <p>THIS METHOD IS REQUIRED.
   *
   * @param example is the test attributeValues to sort.
   */
  public VfdtNode sortExample(Integer[] example) {
    VfdtNode leaf = this;
    while (leaf.getChildren() != null){
      leaf = leaf.getChildren()[example[leaf.splitFeature]];
    }
    return leaf;
  }

  /**
   * Split evaluation method (function G in the paper)
   *
   * <p>Compute a splitting score for the feature featureId. For now, we'll use information gain,
   * but this may be changed. You can test your code with other split evaluations, but be sure to
   * change it back to information gain in the submitted code and for the experiments with default
   * values.
   *
   * @param featureId is the feature to be considered.
   */
  public double splitEval(int featureId) {
    return informationGain(featureId, nijk);
  }

  /**
   * Compute the information gain of a feature for this leaf node.
   *
   * <p>THIS METHOD IS REQUIRED.
   *
   * @param featureId is the feature to be considered.
   * @param nijk are the instance counts.
   */
  public static double informationGain(int featureId, int[][][] nijk) {
    // calculate the entropy before splitting
    double priorEntropy = entropy(nijk[featureId]);

    // calculate entropy after splitting
    double postEntropy = entropy(nijk, featureId);

    return priorEntropy - postEntropy;
  }

  private static double entropy(int[][][] nijk, int featureId) {
    double entropy = 0;

    // loop to count instances for each feature value
    for (int i = 0; i < nijk[featureId].length; i++) {
      if (nijk[featureId][i][0] + nijk[featureId][i][1] != 0) {
        double p = (double)((nijk[featureId][i][0]) / (nijk[featureId][i][0] + nijk[featureId][i][1]));
        entropy += calculateEntropy(p);
      }
    }
    return entropy;
  }

  private static double entropy(int[][] njk){
    double h = 0;
    double classCount = 0;
    double totalInstances = 0;

    // loop to count total amount of instances
    for (int[] featureValCounts : njk) {
      totalInstances += (featureValCounts[0] + featureValCounts[1]);
      classCount += featureValCounts[1];
    }
    if (totalInstances == 0){
      return 0;
    }

    return calculateEntropy(classCount/totalInstances);
  }

  /**
   *  Calculates entropy of binary variable x with chance p to be 1
   * @return
   */
  private static double calculateEntropy(double p){
    double entropy = 0;
    double b = 1 - p;

    if (p == 0 || b == 0) {
      entropy = 0;
    } else {
      entropy = -(p * Math.log(p) / Math.log(2) + b * Math.log(b) / Math.log(2));
    }
    return entropy;
  }

  public void setId(int id){
    this.id = id;
  }

  public int getId(){
    return id;
  }

  /**
   * Return the visualization of the tree.
   *
   * <p>DO NOT CHANGE THIS METHOD.
   *
   * @return Visualization of the tree
   */
  public String getVisualization(String indent) {
    if (children == null) {
      return indent + "Leaf\n";
    } else {
      String visualization = "";
      for (int v = 0; v < children.length; v++) {
        visualization += indent + splitFeature + "=" + v + ":\n";
        visualization += children[v].getVisualization(indent + "| ");
      }
      return visualization;
    }
  }


}
