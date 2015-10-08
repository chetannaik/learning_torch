require 'table'
require 'torch'

local encoding = {}

-- function to create character based vocalulary with unique id for every
-- character and return the vocabulary along with encoded dataset based on the
-- ids.
function char_to_ints(text)
    local alphabet = {}
    local encoded = torch.Tensor(#text)

    for i = 1, #text do
        local c = text:sub(i, i)
        if alphabet[c] == nil then
            alphabet[#alphabet + 1] = c
            alphabet[c] = #alphabet
        end
        encoded[i] = alphabet[c]
    end

    return alphabet, encoded
end

-- function for one hot encoding
function ints_to_one_hot(ints, width)
    local height = ints:size()[1]
    local zeros = torch.zeros(height, width)
    local indices = ints:view(-1, 1):long()
    local one_hot = zeros:scatter(2, indices, 1)

    return one_hot
end

-- export the following functions globally
encoding.ints_to_one_hot = ints_to_one_hot
encoding.char_to_ints = char_to_ints

return encoding
