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

function invert_alphabet(alphabet)
    local inverted = {}
    for char, code in pairs(alphabet) do
        inverted[code] = char
    end

    return inverted
end

function ints_to_chars(alphabet, ints)
    -- with the current code, there is no need to invert because the alphabet
    -- table already contains inverted key value pair.
    local decoder = invert_alphabet(alphabet)
    local decoded = {}
    for i = 1, ints:size(1) do
        decoded[i] = decoder[ints[i]]
    end
end

-- function for one hot encoding
function ints_to_one_hot(ints, width)
    local height = ints:size()[1]
    local zeros = torch.zeros(height, width)
    local indices = ints:view(-1, 1):long()
    local one_hot = zeros:scatter(2, indices, 1)

    return one_hot
end

-- function from one hot encoding to ints
function one_hot_to_ints(ont_hot)
    -- y,i=torch.max(x,1) returns the largest element in each column (across
    -- rows) of x, and a tensor i of their corresponding indices in x.
    local _, ints = torch.max(one_hot:t(), 1)
    return ints
end

-- export the following functions globally
encoding.ints_to_one_hot = ints_to_one_hot
encoding.char_to_ints = char_to_ints

return encoding
