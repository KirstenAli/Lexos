package org.lexos;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

public class PredictNextWord {
    public static void main(String[] args) throws Exception {
        // 1) Load trained model
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(new File("lstm_model.zip"));

        // 2) Tokenized input sentence: "Deep learning is" -> [1, 2, 3]
        int[] inputSentence = {1, 2, 3};
        int timeSteps = inputSentence.length;    // 3
        int inputSize = 1;                      // 1 feature per time step (the token ID)

        // 3) Create a 3D input of shape [batchSize=1, inputSize=1, timeSteps=3]
        INDArray input = Nd4j.create(1, inputSize, timeSteps); // shape: [1,1,3]
        // Fill input along the time dimension
        for (int t = 0; t < timeSteps; t++) {
            input.putScalar(0, 0, t, inputSentence[t]);
        }

        // 4) Get the RNN output
        //    Output shape is [batchSize, nOut (vocabSize), timeSteps].
        //    We only need the final time step to predict the "next word."
        INDArray output3D = model.output(input); // shape: [1, 20, 3] if vocabSize=20

        // 5) Extract the final time step's predictions
        //    finalTimeStep = [1, 20, 1]
        INDArray finalTimeStep = output3D.get(
                NDArrayIndex.point(0),              // batch=0
                NDArrayIndex.all(),                 // all vocab indices
                NDArrayIndex.point(timeSteps - 1)   // final time step (index = 2)
        );

        // 6) Argmax across vocabulary dimension to find predicted word index
        int predictedIndex = finalTimeStep.argMax(0).getInt(0);

        // 7) Word Index Mapping
        Map<Integer, String> reverseWordIndex = new HashMap<>();
        reverseWordIndex.put(1, "Deep");
        reverseWordIndex.put(2, "learning");
        reverseWordIndex.put(3, "is");
        reverseWordIndex.put(4, "powerful");
        reverseWordIndex.put(5, "I");
        reverseWordIndex.put(6, "enjoy");
        reverseWordIndex.put(7, "reading");
        reverseWordIndex.put(8, "books");
        reverseWordIndex.put(9, "Java");
        reverseWordIndex.put(10, "a");
        reverseWordIndex.put(11, "language");
        reverseWordIndex.put(12, "The");
        reverseWordIndex.put(13, "cat");
        reverseWordIndex.put(14, "sat");
        reverseWordIndex.put(15, "on");
        reverseWordIndex.put(16, "He");
        reverseWordIndex.put(17, "very");
        reverseWordIndex.put(18, "smart");

        // 8) Print Prediction
        String predictedWord = reverseWordIndex.get(predictedIndex);
        System.out.println("Input Sentence: \"Deep learning is\"");
        System.out.println("Predicted Next Word: " + predictedWord);
    }
}
