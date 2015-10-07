local table = require 'table'
local math = require 'math'
local torch = require 'torch'

-- function to read the text file
function load_text()
    local f = io.open('input.txt')
    local text = f:read('*a')
    f:close()

    return text
end

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

function make_chunk_iterator(encoded_text, indices, chunk_size, n_symbols)
    function co()
        for i=1, indices:size(1) do
            local index = indices[i]
            local lower = (index - 1)*chunk_size
            local upper = lower + chunk_size - 1
            local chunk = ints_to_one_hot(encoded_text[{{lower, upper}}], n_symbols)
            coroutine.yield(chunk)
        end
    end

    return coroutine.wrap(co)
end

function split_indices(indices, split_fractions)
    local split_sizes = (split_fractions*indices:size(1)):long()
    local split_points = torch.cat(torch.LongTensor{0}, split_sizes:cumsum())
    local splits = {}
    for i = 1, split_points:size(1) - 1 do
        local lower, upper = split_points[i] + 1, split_points[i + 1]
        splits[i] = indices[{{lower, upper}}]
    end

    return splits
end

function make_chunk_iterators(text, split_fractions, chunk_size)
    local alphabet, encoded_text = char_to_ints(text)
    local n_chunks = math.floor(#text/chunk_size)

--     local indices = torch.range(1, n_chunks)
    local indices = torch.randperm(n_chunks)
    local splits = split_indices(indices, split_fractions)

    local iterators = {}
    for _, split in pairs(splits) do
        iterator = make_chunk_iterator(encoded_text, split, chunk_size, #alphabet)
        iterators[#iterators + 1] = iterator
    end

    return alphabet, iterators
end

local text = load_text()
local fractions = torch.Tensor{0.25, 0.75}
local alphabet, iterators = make_chunk_iterators(text, fractions, 2)

for i, iterator in pairs(iterators) do
    print(iterator())
end
