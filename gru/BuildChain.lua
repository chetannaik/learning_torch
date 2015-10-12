local BuildChain, parent = torch.class('gru.BuildChain', 'nn.Module')

function BuildChain:__init(inputSize, hiddenSize, maxLength)
    parent.__init(self)
    self.inputSize = inputSize
    self.hiddenSize = hiddenSize
    self.maxLength = maxLength
    self.gradInput = nil
    self.grus = {}

    -- make enough gru cells for the longest sequence
    for i=1,maxLength do
        self.grus[i] = gru.BuildCell(inputSize, hiddenSize)
        if i == 1 then
            self.gru_params, self.gru_grad_params = self.grus[1]:parameters()
        else
            -- share parameters
            local clone_params, clone_grad_params = self.grus[i]:parameters()
            for k=1,#clone_params do
                clone_params[k]:set(self.gru_params[k])
                clone_grad_params[k]:set(self.gru_grad_params[k])
            end
        end
    end
end

function BuildChain:parameters()
    return self.gru_params, self.gru_grad_params
end

function BuildChain:updateOutput(input)
    local h = torch.zeros(1, self.hiddenSize)
    self.hidden_states = {[0] = h}
    local len = input:size(1)
    for i=1,len do
        local x = input[i]:view(1,-1)
        self.grus[i]:forward({x, h})
        h = self.grus[i].output
        self.hidden_states[i] = h
    end
    self.output = {self.grus[len].output}
    return self.output
end

function BuildChain:updateGradInput(input, gradOutput)
    local h
    local len = input:size(1)
    self.gradInput = torch.Tensor(len, self.inputSize)
    for i=len,1,-1 do
        local x = input[i]
        h = self.hidden_states[i-1]

        self.grus[i]:backward({x, h}, gradOutput[1])

        gradOutput = self.grus[i].gradInput
        -- Only h propagates back to the next cell, the gradient with respect
        -- to x gets stored in gradInput.
        self.gradInput[i]:copy(gradOutput[1])
        gradOutput = gradOutput[2]
    end

    return self.gradInput
end

return BuildChain
