-- Load the Torch libraries
require 'torch'
require 'nn'

-- Define the network architecture
local inputSize = 2
local hiddenSize = 4
local outputSize = 1

-- Create the neural network model
local model = nn.Sequential()

-- Add a fully connected (Linear) layer with ReLU activation
model:add(nn.Linear(inputSize, hiddenSize))
model:add(nn.ReLU())

-- Add another fully connected (Linear) layer with Sigmoid activation
model:add(nn.Linear(hiddenSize, outputSize))
model:add(nn.Sigmoid())

-- Define the loss function (Binary Cross-Entropy)
local criterion = nn.BCECriterion()

-- Define training data
local inputs = torch.Tensor({
    {0, 0},
    {0, 1},
    {1, 0},
    {1, 1}
})

local targets = torch.Tensor({
    {0},
    {1},
    {1},
    {0}
})

-- Training parameters
local learningRate = 0.1
local epochs = 10000

-- Training loop
for epoch = 1, epochs do
    -- Zero the gradients
    model:zeroGradParameters()

    -- Forward pass
    local output = model:forward(inputs)
    local loss = criterion:forward(output, targets)
    
    -- Backward pass
    local gradOutput = criterion:backward(output, targets)
    model:backward(inputs, gradOutput)
    
    -- Update parameters
    model:updateParameters(learningRate)
    
    -- Print the loss every 1000 epochs
    if epoch % 1000 == 0 then
        print(string.format("Epoch %d: Loss = %.4f", epoch, loss))
    end
end

-- Testing the network
local function test(input)
    local output = model:forward(input)
    return output
end

-- Test the network
for i = 1, inputs:size(1) do
    local input = inputs[i]
    local output = test(input:view(1, -1))
    print(string.format("Input: %s => Output: %.4f", input, output[1][1]))
end
