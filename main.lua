require 'torch'
require 'batcher'
require 'encoding'

local text = load_text()
local fractions = torch.Tensor{0.25, 0.75}
local alphabet, batch_iterators = make_batch_iterators(text, fractions, 2, 2)

for i, batch_iterator in pairs(batch_iterators) do
    print("> Batch Iterator: " .. i)
    print(batch_iterator())
end
