local BuildChain, parent = torch.class('lstm.BuildChain', 'nn.Module')

function BuildChain:__init(inputSize, hiddenSize, maxLength)
    parent.__init(self)
    self.inputSize = inputSize
    self.hiddenSize = hiddenSize
    self.maxLength = maxLength
    self.gradInput = nil
    self.lstms = {}

    -- make enough lstm cells for the longest sequence
    for i=1,maxLength do
        self.lstms[i] = lstm.BuildCell(inputSize, hiddenSize)
        if i == 1 then
            self.lstm_params, self.lstm_grad_params = self.lstms[1]:parameters()
        else
            -- share parameters
            local clone_params, clone_grad_params = self.lstms[i]:parameters()
            for k=1,#clone_params do
                clone_params[k]:set(self.lstm_params[k])
                clone_grad_params[k]:set(self.lstm_grad_params[k])
            end
        end
    end
end

function BuildChain:parameters()
    return self.lstm_params, self.lstm_grad_params
end

function BuildChain:updateOutput(input)
    local h = torch.zeros(1, self.hiddenSize)
    local m = torch.zeros(1, self.hiddenSize)
    self.hidden_states = {[0] = h}
    self.memory_states = {[0] = m}
    local len = input:size(1)
    for i=1,len do
        local x = input[i]:view(1,-1)
        self.lstms[i]:forward({x, h, m})
        h, m = unpack(self.lstms[i].output)
        self.hidden_states[i] = h
        self.memory_states[i] = m
    end
    self.output = self.lstms[len].output
    return self.output
end

function BuildChain:updateGradInput(input, gradOutput)
    local h, m
    local len = input:size(1)
    self.gradInput = torch.Tensor(len, self.inputSize)
    for i=len,1,-1 do
        local x = input[i]
        h = self.hidden_states[i-1]
        m = self.memory_states[i-1]

        self.lstms[i]:backward({x, h, m}, gradOutput)

        gradOutput = self.lstms[i].gradInput
        -- Only h and m propagate back to the next cell, the gradient with
        -- respect to x gets stored in gradInput
        self.gradInput[i]:copy(gradOutput[1])
        gradOutput = {gradOutput[2], gradOutput[3]}
    end

    return self.gradInput
end

function BuildChain:accGradParameters(input, gradOutput)
end

function BuildChain:reset(stdv)
    local params = self.parameters()
    for i=1,#params do
        params[i]:uniform(-stdv, stdv)
    end
end

return BuildChain
