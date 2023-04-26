import java.util.Arrays;

public class NeuralNetwork {
    public final double[][] hidden_layer_weights;
    public final double[][] output_layer_weights;
    private final int num_inputs;
    private final int num_hidden;
    private final int num_outputs;
    private final double learning_rate;

    public NeuralNetwork(int num_inputs, int num_hidden, int num_outputs, double[][] initial_hidden_layer_weights,
            double[][] initial_output_layer_weights, double learning_rate) {
        // Initialise the network
        this.num_inputs = num_inputs;
        this.num_hidden = num_hidden;
        this.num_outputs = num_outputs;

        this.hidden_layer_weights = initial_hidden_layer_weights;
        this.output_layer_weights = initial_output_layer_weights;

        this.learning_rate = learning_rate;
    }

    // Calculate neuron activation for an input
    public double sigmoid(double input) {
        double output = 1.0 / (1.0 + Math.exp(-input));
        return output;
    }

    // Feed forward pass input to a network output
    public double[][] forward_pass(double[] inputs) {
        double[] hidden_layer_outputs = new double[num_hidden];
        for (int i = 0; i < num_hidden; i++) {
            double weighted_sum = 0.0;
            for (int j = 0; j < num_inputs; j++) {
                weighted_sum += inputs[j] * hidden_layer_weights[j][i];
            }

            // Comment out to disable bias
            weighted_sum += 1 * hidden_layer_weights[num_inputs][i]; // Add the bias

            double output = sigmoid(weighted_sum);
            hidden_layer_outputs[i] = output;
        }

        double[] output_layer_outputs = new double[num_outputs];
        for (int i = 0; i < num_outputs; i++) {
            double weighted_sum = 0.0;
            for (int j = 0; j < num_hidden; j++) {
                weighted_sum += hidden_layer_outputs[j] * output_layer_weights[j][i];
            }

            // Comment out to disable bias
            weighted_sum += 1 * output_layer_weights[num_hidden][i]; // Add the bias

            double output = sigmoid(weighted_sum);
            output_layer_outputs[i] = output;
        }
        return new double[][] { hidden_layer_outputs, output_layer_outputs };
    }

    public double[][][] backward_propagate_error(double[] inputs, double[] hidden_layer_outputs,
            double[] output_layer_outputs, int desired_outputs) {

        // Desired output values for classification
        double[] desired_output_values = new double[num_outputs];
        if (desired_outputs == 0) {
            desired_output_values[0] = 1;
            desired_output_values[1] = 0;
            desired_output_values[2] = 0;
        } else if (desired_outputs == 1) {
            desired_output_values[0] = 0;
            desired_output_values[1] = 1;
            desired_output_values[2] = 0;
        } else if (desired_outputs == 2) {
            desired_output_values[0] = 0;
            desired_output_values[1] = 0;
            desired_output_values[2] = 1;
        } else {
            System.out.println("Error: desired output is not 0, 1 or 2");
        }

        // Output layer betas
        double[] output_layer_betas = new double[num_outputs];
        for (int i = 0; i < output_layer_outputs.length; i++) {
            output_layer_betas[i] = desired_output_values[i] - output_layer_outputs[i];
        }
        //System.out.println("OL betas: " + Arrays.toString(output_layer_betas));

        // Hidden layer betas
        double[] hidden_layer_betas = new double[num_hidden];
        for (int j = 0; j < hidden_layer_outputs.length; j++) {
            double sum = 0;
            for (int k = 0; k < output_layer_betas.length; k++) {
                sum += output_layer_weights[j][k] * (output_layer_outputs[k] * (1 - output_layer_outputs[k]))
                        * output_layer_betas[k];
            }
            hidden_layer_betas[j] = sum;
        }
        //System.out.println("HL betas: " + Arrays.toString(hidden_layer_betas));

        // This is a HxO array (H hidden nodes, O outputs)
        // Output layer weight changes
        double[][] delta_output_layer_weights = new double[num_hidden + 1][num_outputs];
        for (int i = 0; i <= num_hidden; i++) {
            for (int j = 0; j < num_outputs; j++) {

                double prevLayerOutput = i == num_hidden ? 1 : hidden_layer_outputs[i];

                delta_output_layer_weights[i][j] = learning_rate * prevLayerOutput
                        * (output_layer_outputs[j] * (1 - output_layer_outputs[j])) * output_layer_betas[j];
            }
        }

        // This is a IxH array (I inputs, H hidden nodes)
        // Hidden layer weight changes
        double[][] delta_hidden_layer_weights = new double[num_inputs + 1][num_hidden];
        for (int i = 0; i <= num_inputs; i++) {
            for (int j = 0; j < num_hidden; j++) {
                double prevLayerOutput = i == num_inputs ? 1 : inputs[i];
                delta_hidden_layer_weights[i][j] = learning_rate * prevLayerOutput
                        * (hidden_layer_outputs[j] * (1 - hidden_layer_outputs[j])) * hidden_layer_betas[j];
            }
        }
        // Return the weights we calculated, so they can be used to update all the weights.
        return new double[][][] { delta_output_layer_weights, delta_hidden_layer_weights };
    }

    public void update_weights(double[][] delta_output_layer_weights, double[][] delta_hidden_layer_weights) {
        // Print out the initial weights before back propogation for each epoch
        /*
        System.out.println("Initial Hidden layer weights:");
            for (double[] row : hidden_layer_weights) {
                System.out.println(Arrays.toString(row));
            }
            System.out.println();
            System.out.println("Initial Output layer weights:");
            for (double[] row : output_layer_weights) {
                System.out.println(Arrays.toString(row));
            }
            System.out.println();
        */

        // Output layer weight update
        for (int i = 0; i <= num_hidden; i++) {
            for (int j = 0; j < num_outputs; j++) {
                output_layer_weights[i][j] += delta_output_layer_weights[i][j];
            }
        }
        // Hidden layer weight update
        for (int i = 0; i <= num_inputs; i++) {
            for (int j = 0; j < num_hidden; j++) {
                hidden_layer_weights[i][j] += delta_hidden_layer_weights[i][j];
            }
        }
    }

    // Calculates accuracy for the given instances and desired outputs
    public double calculateAccuracy(int[] predictions, int[] desired_outputs) {
        double correct = 0;
        for (int i = 0; i < predictions.length; i++) {
            if (predictions[i] == desired_outputs[i]) {
                correct++;
            }
        }
        return correct / predictions.length;
    }

    public void train(double[][] instances, int[] desired_outputs, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            System.out.println("epoch = " + epoch);
            int[] predictions = new int[instances.length];
            for (int i = 0; i < instances.length; i++) {
                double[] instance = instances[i];
                double[][] outputs = forward_pass(instance);
                double[][][] delta_weights = backward_propagate_error(instance, outputs[0], outputs[1],
                        desired_outputs[i]);
                int predicted_class = predictClassForOutput(outputs[1]);
                predictions[i] = predicted_class;

                // We use online learning, i.e. update the weights after every instance.
                update_weights(delta_weights[0], delta_weights[1]);
            }

            // Print weight updates after each epoch
            System.out.println("Hidden layer weights:");
            for (double[] row : hidden_layer_weights) {
                System.out.println(Arrays.toString(row));
            }
            System.out.println();
            System.out.println("Output layer weights:");
            for (double[] row : output_layer_weights) {
                System.out.println(Arrays.toString(row));
            }
            System.out.println();

            // Print accuracy achieved over this epoch
            double acc = calculateAccuracy(predictions, desired_outputs);
            System.out.println("Training accuracy = " + acc + "\n");
        }
    }

    // Converts the outputs to a predicted class label
    private int predictClassForOutput(double[] output) {
        if (output[0] > output[1] && output[0] > output[2]) {
            return 0;
        }
        if (output[1] > output[0] && output[1] > output[2]) {
            return 1;
        } else {
            return 2;
        }
    }

    public int[] predict(double[][] instances) {
        int[] predictions = new int[instances.length];
        for (int i = 0; i < instances.length; i++) {
            double[] instance = instances[i];
            double[][] outputs = forward_pass(instance);
            int predicted_class = predictClassForOutput(outputs[1]);
            predictions[i] = predicted_class;
        }
        return predictions;
    }
}