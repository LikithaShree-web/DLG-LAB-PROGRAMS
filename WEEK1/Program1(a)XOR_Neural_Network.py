import numpy as np
def sigmoid(x):
  return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
  return x*(1-x)

inputs=np.array([[0,0],[0,1],[1,0],[1,1]])
expected_output=np.array([[0],[1],[1],[0]])

inputLayerNeurons,hiddenLayerNeurons,outputNeurons=2,2,1
hidden_weights=np.random.uniform(size=(inputLayerNeurons,hiddenLayerNeurons))
hidden_bias=np.random.uniform(size=(1,hiddenLayerNeurons))
output_weights=np.random.uniform(size=(hiddenLayerNeurons,outputNeurons))
output_bias=np.random.uniform(size=(1,outputNeurons))

lr=0.1

epochs=10000

for _ in range(epochs):
  hidden_layer_activation=np.dot(inputs,hidden_weights)+hidden_bias
  hidden_layer_output=sigmoid(hidden_layer_activation)

  output_layer_activation=np.dot(hidden_layer_output,output_weights)+output_bias
  predicted_output=sigmoid(output_layer_activation)

  error=expected_output-predicted_output
  d_predicted_output=error*sigmoid_derivative(predicted_output)

  error_hidden_layer=d_predicted_output.dot(output_weights.T)
  d_hidden_layer=error_hidden_layer*sigmoid_derivative(hidden_layer_output)

  output_weights+=hidden_layer_output.T.dot(d_predicted_output)*lr
  output_bias+=np.sum(d_predicted_output,axis=0,keepdims=True)*lr
  hidden_weights+=inputs.T.dot(d_hidden_layer)*lr
  hidden_bias+=np.sum(d_hidden_layer,axis=0,keepdims=True)*lr

print("Predicted Output:")
print(predicted_output)