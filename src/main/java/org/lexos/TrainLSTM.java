package org.lexos;

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.util.*;

public class TrainLSTM {
    public static void main(String[] args) throws Exception {

        // --------------------------------------------------
        // 1) Data: We have 3 training examples
        // --------------------------------------------------
        List<int[]> inputSequences = Arrays.asList(
                new int[]{1, 2, 3},    // "Deep learning is"
                new int[]{5, 6, 7},    // "I enjoy reading"
                new int[]{9, 3, 10}    // "Java is a"
        );

        List<int[]> outputSequences = Arrays.asList(
                new int[]{4},   // "powerful"
                new int[]{8},   // "books"
                new int[]{11}   // "language"
        );

        // --------------------------------------------------
        // 2) Define Data Parameters
        // --------------------------------------------------
        int batchSize    = inputSequences.size(); // 3
        int timeSteps    = 3;  // 3 words per input sequence
        int inputSize    = 1;  // each word is a single integer
        int hiddenSize   = 100;
        int vocabSize    = 20; // total words in vocabulary

        // --------------------------------------------------
        // 3) Define LSTM Model
        // --------------------------------------------------
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.001))
                .list()
                .layer(0, new LSTM.Builder()
                        .nIn(inputSize)
                        .nOut(hiddenSize)
                        .activation(Activation.TANH)
                        .build())
                .layer(1, new RnnOutputLayer.Builder()
                        .nIn(hiddenSize)
                        .nOut(vocabSize)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunction.MCXENT)
                        .build())
                .build();

        // Initialize the network
        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();

        // --------------------------------------------------
        // 4) Create Input & Label Arrays
        // --------------------------------------------------
        INDArray input = Nd4j.zeros(batchSize, inputSize, timeSteps);  // shape = [3, 1, 3]
        INDArray labels = Nd4j.zeros(batchSize, vocabSize, timeSteps); // shape = [3, 20, 3]

        for (int i = 0; i < batchSize; i++) {
            int[] seq = inputSequences.get(i);
            for(int t = 0; t < timeSteps; t++) {
                input.putScalar(new int[]{i, 0, t}, seq[t]);  // Fill input
            }

            int predictedWordIndex = outputSequences.get(i)[0]; // Only last step is labeled
            labels.putScalar(new int[]{i, predictedWordIndex, timeSteps - 1}, 1.0);
        }

        // Wrap into a DataSet
        DataSet dataSet = new DataSet(input, labels);

        // --------------------------------------------------
        // 5) Create Label Mask (Ensures Only Final Time Step Is Used)
        // --------------------------------------------------
        INDArray labelMask = Nd4j.zeros(batchSize, timeSteps);
        for (int i = 0; i < batchSize; i++) {
            labelMask.putScalar(i, timeSteps - 1, 1.0);  // Only last time step matters
        }
        dataSet.setLabelsMaskArray(labelMask);  // This is enough when using ListDataSetIterator

        // --------------------------------------------------
        // 6) Train Model
        // --------------------------------------------------
        DataSetIterator iterator = new ListDataSetIterator<>(Collections.singletonList(dataSet), batchSize);
        model.fit(iterator, 2000);  // Train for 2000 epochs

        // --------------------------------------------------
        // 7) Save Model
        // --------------------------------------------------
        File modelFile = new File("lstm_model.zip");
        ModelSerializer.writeModel(model, modelFile, true);

        System.out.println("✅ Training Complete and Model Saved!");
    }
}
