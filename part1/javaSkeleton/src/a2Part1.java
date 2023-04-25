import java.util.Arrays;
import java.util.List;

public class a2Part1 {

    public static void main(String[] _ignored) {
        List<String[]> lines = Util.getLines("/Users/rohan/Documents/COMP307/assignment2_data/part1/javaSkeleton/penguins307-train.csv");
        String[] header = lines.remove(0);
        String[] labels = Util.getLabels(lines);
        double[][] instances = Util.getData(lines);

        // scale features to [0,1] to improve training
        Rescaler rescaler = new Rescaler(instances);
        rescaler.rescaleData(instances);
        System.out.println(Arrays.deepToString(instances));

        // We can"t use strings as labels directly in the network, so need to do some transformations
        LabelEncoder label_encoder = new LabelEncoder(labels);
        // encode "Adelie" as 1, "Chinstrap" as 2, "Gentoo" as 3
        int[] integer_encoded = label_encoder.intEncode(labels);

        // encode 1 as [1, 0, 0], 2 as [0, 1, 0], and 3 as [0, 0, 1] (to fit with our network outputs!)
        int[][] onehot_encoded = label_encoder.oneHotEncode(labels);

        // Parameters. As per the handout.
        int n_in = 4, n_hidden = 2, n_out = 3;
        double learning_rate = 0.2;

        double[][] initial_hidden_layer_weights =
                new double[][]{{-0.28, -0.22}, {0.08, 0.20}, {-0.30, 0.32}, {0.10, 0.01}, {-0.02, -0.2}};
        double[][] initial_output_layer_weights = new double[][]{{-0.29, 0.03, 0.21}, {0.08, 0.13, -0.36}, {-0.33, 0.26, 0.06}};

        NeuralNetwork nn = new NeuralNetwork(n_in, n_hidden, n_out, initial_hidden_layer_weights, initial_output_layer_weights, learning_rate);

        System.out.printf("First instance has label %s, which is %d as an integer, and %s as a list of outputs.\n",
                labels[0], integer_encoded[0], Arrays.toString(onehot_encoded[0]));

        // need to wrap it into a 2D array
        int[] instance1_prediction = nn.predict(new double[][]{instances[0]});
        String instance1_predicted_label;
        if (instance1_prediction[0] == -1) {
            // This should never happen once you have implemented the feedforward.
            instance1_predicted_label = "???";
        } else {
            instance1_predicted_label = label_encoder.inverse_transform(instance1_prediction[0]);
        }
        System.out.println("Predicted label for the first instance is: " + instance1_predicted_label);

        // TODO: Perform a single backpropagation pass using the first instance only. (In other words, train with 1
        //  instance for 1 epoch!). Hint: you will need to first get the weights from a forward pass.
        //double[][] first_instance = nn.forward_pass(instances[0]);

        double[][] singleInstance = new double[1][];
        singleInstance[0] = instances[0];

        int[] singleInstanceInteger = new int[1];
        singleInstanceInteger[0] = integer_encoded[0];

        nn.train(singleInstance, singleInstanceInteger, 1);

        System.out.println("Weights after performing BP for first instance only:");
        System.out.println("Hidden layer weights:\n" + Arrays.deepToString(nn.hidden_layer_weights));
        System.out.println("Output layer weights:\n" + Arrays.deepToString(nn.output_layer_weights));

        // TODO: Train for 100 epochs, on all instances.

        nn.train(instances, integer_encoded, 100);
        System.out.println("\nAfter training:");
        //System.out.println("Hidden layer weights:\n" + Arrays.deepToString(nn.hidden_layer_weights));
        //System.out.println("Output layer weights:\n" + Arrays.deepToString(nn.output_layer_weights));

        List<String[]> lines_test = Util.getLines("/Users/rohan/Documents/COMP307/assignment2_data/part1/javaSkeleton/penguins307-test.csv");
        String[] header_test = lines_test.remove(0);
        String[] labels_test = Util.getLabels(lines_test);
        double[][] instances_test = Util.getData(lines_test);

        // scale the test according to our training data.
        rescaler.rescaleData(instances_test);

        // TODO: Compute and print the test accuracy
        int[] testPredictions = nn.predict(instances_test);
        int[] desired_outputs = label_encoder.intEncode(labels_test);

        double accuracy = nn.calculateAccuracy(testPredictions, desired_outputs);

        // int correct = 0;
        // for (int i = 0; i < testPredictions.length; i++) {
        //     if (testPredictions[i] == integer_encoded[i]) {
        //         correct++;
        //     }
        // }
        // double accuracy = (double) correct / testPredictions.length;
        System.out.println("Test accuracy: " + accuracy);
        System.out.println("Finished!");


        System.out.println("Final weights");
        System.out.println("Hidden layer weights:\n" + Arrays.deepToString(nn.hidden_layer_weights));
        System.out.println("Output layer weights:\n" + Arrays.deepToString(nn.output_layer_weights));
    }
}


