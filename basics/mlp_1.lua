require 'nn'
toy = require 'toy'

-- take command line arguments to control parameters
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train MLP')
cmd:text()
cmd:text('Options')
cmd:option('-seed', os.time(), 'initial random seed (defult: current time)')
cmd:option('-hidden', 25, 'hidden state size')
cmd:option('-batch', 5, 'batch size')
cmd:option('-rate', 0.03, 'learn rate')
cmd:option('-iterations', 100, 'maximum number of iterations of SGD')
cmd:option('-trained', 'trained_mlp_1.t7', 'filename for saved trained model')
cmd:option('-grid', 'grid_predictions_mlp_1.csv', 'file name for saved grid predictions')
cmd:text()

-- parse input parameters
params = cmd:parse(arg)

-- set 'random seed' to make the result reproduceable
torch.manualSeed(params.seed)

--[[
  read data
    N: number of rows of data
    n_inputs: number of input variables (all columns in data - target column)
  ]]--
d = torch.load('fixed_width_3.t7')
N = d:size(1)
n_inputs = d:size(2) - 1

-- separate data into inputs (x) and targets (y)
x = d:narrow(2, 1, n_inputs)
y = d:narrow(2, n_inputs + 1, 1)

-- train/test split sizes
test_frac = 0.3
n_test = torch.floor(N * test_frac)
n_train = N - n_test

-- train/test splits
x_train = x:narrow(1, 1, n_train)
y_train = y:narrow(1, 1, n_train)

x_test = x:narrow(1, n_train + 1, n_test)
y_test = y:narrow(1, n_train + 1, n_test)

-- normalize training inputs
norm_mean = x_train:mean()
norm_std = x_train:std()
x_train_n = (x_train - norm_mean) / norm_std

-- normalize test inputs based on training data normalization values
x_test_n = (x_test - norm_mean) / norm_std

-- the nn SGD trainer needs a data structure where examples can be accessed via
-- the index operator, [], and should have a size() method
dataset = {}
function dataset:size()
    return torch.floor(n_train / params.batch)
end
for i = 1, dataset:size() do
    local start = (i - 1) * params.batch + 1
    dataset[i] = {x_train_n:narrow(1, start, params.batch),
                  y_train:narrow(1, start, params.batch)}
end

-- set up the neural net
n_hidden = params.hidden
mlp = nn.Sequential()
mlp:add(nn.Linear(n_inputs, n_hidden))
mlp:add(nn.Sigmoid())
mlp:add(nn.Linear(n_hidden, 1))

-- get all parameters packaged into a vector
mlp_params = mlp:getParameters()

-- we need our model to learn to predict real values target, so setting
-- least-square loss as the objective function
criterion = nn.MSECriterion()

-- set up trainer to use SGD - Stochastic Gradient Descent
trainer = nn.StochasticGradient(mlp, criterion)
trainer.maxIteration = params.iterations
trainer.learningRate = params.rate
function trainer:hookIteration(iteration)
    print('# test error = ' .. criterion:forward(mlp:forward(x_test_n), y_test))
end

-- train the model, after randomly initializing the parameters and clearing out
-- any existing gradient.
mlp_params:uniform(-0.1, 0.1)
mlp:zeroGradParameters()
print("parameter count: " .. mlp_params:size(1))
print("initial error before training = " .. criterion:forward(mlp:forward(x_test_n), y_test))
trainer:train(dataset)

-- save the trained model
torch.save(params.trained, {mlp = mlp, params = mlp_params})

-- Output predictions along a grid so we can see how well it learned the
-- function. We'll generate inputs without noise so we can see how well it does
-- in the absence of noise, which will give us a sense of whether it's learned
-- the true underlying function.
grid_size = 200
target_grid = torch.linspace(0, toy.max_target, grid_size):view(grid_size,1)
inputs_grid = toy.target_to_inputs(target_grid, 0)
inputs_grid_n = (inputs_grid - norm_mean) / norm_std
predictions = mlp:forward(inputs_grid_n)

-- Use penlight to write the data
pldata = require 'pl.data'
pred_d = pldata.new(predictions:totable())
pred_d:write(params.grid)
