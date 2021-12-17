/**
 * Copyright (c) DTAI - KU Leuven – All rights reserved. Proprietary, do not copy or distribute
 * without permission. Written by Pieter Robberechts, 2021
 */
import java.io.*;
import java.util.Scanner;

/** This class is a stub for incrementally building a Perceptron model. */
public class Perceptron extends IncrementalLearner<Double> {

  private double learningRate;
  private double[] weights;
  private double bias;


  /**
   * Perceptron constructor.
   *
   * <p>THIS METHOD IS REQUIRED
   *
   * @param numFeatures is the number of features.
   * @param learningRate is the learning rate
   */
  public Perceptron(int numFeatures, double learningRate) {
    this.nbExamplesProcessed = 0;
    this.learningRate = learningRate;


    /*
      FILL IN HERE
      You will need other data structures, initialize them here
    */

    this.weights = new double[numFeatures];
    this.bias = 0.0;
  }

  /**
   * This method will update the parameters of you model using the given example.
   *
   * <p>THIS METHOD IS REQUIRED
   *
   * @param example is a training example
   */
  @Override
  public void update(Example<Double> example) {
    super.update(example);
    /*
      FILL IN HERE
      Update the parameters given the new data to improve J(weights)
    */
    double[] gradients = computeGradients(example);
    updateWeightsAndBias(gradients);
  }

  private double[] computeGradients(Example<Double> example){
    // init gradients array (all weights + bias)
    double gradients[] = new double[weights.length + 1];

    // compute output of current model
    double output = makePrediction(example.attributeValues);

    // transform class value in proper value for perceptron
    double expected = example.classValue == 0 ? -1 : 1;

    // calculate error
    double error = (output - expected);

    // calculate gradients for bias (bias acts as an input of 1)
    gradients[0] = -error * 1;

    // calculate the gradients for weights
    for (int i = 1; i < gradients.length; i++){
      gradients[i] = -error * example.attributeValues[i - 1];
    }
    return gradients;
  }

  private void updateWeightsAndBias(double[] gradients){
    // update the bias
    bias += learningRate * gradients[0];

    // update the weights
    for (int i = 1; i < gradients.length; i++){
      weights[i - 1] += learningRate * gradients[i];
    }
  }

  /**
   * Uses the current model to calculate the likelihood that an attributeValues belongs to class
   * "1";
   *
   * <p>This method gives the output of the perceptron, before it is passed through the threshold
   * function.
   *
   * <p>THIS METHOD IS REQUIRED
   *
   * @param example is a test attributeValues
   * @return the likelihood that attributeValues belongs to class "1"
   */
  @Override
  public double makePrediction(Double[] example) {
    double pr = bias;
    /* FILL IN HERE */
    for (int i = 0; i < example.length; i++){
      pr += weights[i] * example[i];
    }

    return pr;
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

    /* FILL IN HERE */
    File modelFile = new File(path);
    modelFile.createNewFile();
    FileWriter writer = new FileWriter(modelFile);

    // first print bias
    writer.write(String.valueOf(bias));
    writer.write(" ");
    // print weights
    for (double weight : weights){
      writer.write(String.valueOf(weight));
      writer.write(" ");
    }
    writer.close();

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
    File modelFile = new File(path);
    Scanner reader = new Scanner(modelFile);


    // read first line, this is the line with the weights
    String string = reader.nextLine();
    String parsedString[] = string.split(" ");
    double parsedWeights[] = new double[parsedString.length - 1];
    for (int i = 0; i < parsedString.length; i++) {
      if (i == 0){
        bias = Double.parseDouble(parsedString[i]);
      }
      else {
        parsedWeights[i - 1] = Double.parseDouble(parsedString[i]);
      }
    }

    weights = parsedWeights;

    reader.close();

  }

  /**
   * This runs your code to generate the required output for the assignment.
   *
   * <p>DO NOT CHANGE THIS METHOD
   */
  public static void main(String[] args) {
    if (args.length < 4) {
      System.err.println(
          "Usage: java Perceptron <learningRate> <data set> <output file> <reportingPeriod>"
              + " [-writeOutAllPredictions]");
      throw new Error("Expected 4 or 5 arguments, got " + args.length + ".");
    }
    try {
      // parse input
      double learningRate = Double.parseDouble(args[0]);
      DoubleData data = new DoubleData(args[1], ",");
      String out = args[2];
      int reportingPeriod = Integer.parseInt(args[3]);
      boolean writeOutAllPredictions =
          args.length > 4 && args[4].contains("writeOutAllPredictions");

      // initialize learner
      Perceptron perceptron = new Perceptron(data.getNbFeatures(), learningRate);

      // generate output for the learning curve
      perceptron.makeLearningCurve(data, 0, out + ".pc", reportingPeriod, writeOutAllPredictions);

    } catch (FileNotFoundException e) {
      System.err.println(e.toString());
    }
  }
}

/**
 * This class implements Data for Doubles
 *
 * <p>DO NOT CHANGE THIS CLASS
 */
class DoubleData extends Data<Double> {

  public DoubleData(String dataDir, String sep) throws FileNotFoundException {
    super(dataDir, sep);
  }

  @Override
  protected Double parseAttribute(String attrString) {
    return Double.parseDouble(attrString);
  }

  @Override
  protected Double[] emptyAttributes(int i) {
    return new Double[i];
  }
}
