require 'table'
require 'math'
require 'torch'
require 'encoding'

local batcher = {}

-- function to read the text file
function load_text()
    local f = io.open('input.txt')
    local text = f:read('*a')
    f:close()

    return text
end

function make_chunk_iterator(encoded_text, indices, chunk_size, n_symbols)
    function co()
        for i=1, indices:size(1) do
            local index = indices[i]
            local lower = (index - 1) * chunk_size + 1
            local upper = lower + chunk_size - 1
            local chunk = ints_to_one_hot(encoded_text[{{lower, upper}}], n_symbols)
            coroutine.yield(chunk)
        end
    end

    return coroutine.wrap(co)
end

function split_indices(indices, split_fractions)
    local split_sizes = (split_fractions*indices:size(1)):long()
    -- x=torch.cat(x_1,x_2,[dimension]) returns a tensor x which is the
    -- concatenation of tensors x_1 and x_2 along dimension `dimension`.
    local split_points = torch.cat(torch.LongTensor{0}, split_sizes:cumsum())
    local splits = {}
    for i = 1, split_points:size(1) - 1 do
        local lower, upper = split_points[i] + 1, split_points[i + 1]
        splits[i] = indices[{{lower, upper}}]
        -- can use indices:narrow(dim, index, size) to create split here
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
    for i, split in pairs(splits) do
        iterator = make_chunk_iterator(encoded_text, split, chunk_size, #alphabet)
        iterators[#iterators + 1] = iterator
    end

    return alphabet, iterators
end

function stack(tensors)
    local shape = torch.totable(tensors[1]:size())
    table.insert(shape, 1, #tensors)

    local result = torch.Tensor(torch.LongStorage(shape))
    for i = 1, #tensors do
        result[i] = tensors[i]
    end

    return result
end

function make_batch_iterator(chunk_iterator, batch_size)
    function co()
        local batch = {}
        while true do
            local chunk = chunk_iterator()
            if chunk then
                batch[#batch + 1] = chunk
            else
                break
            end

            if #batch == batch_size then
                coroutine.yield(stack(batch))
                batch = {}
            end

        end
    end

    return coroutine.wrap(co)
end

function make_batch_iterators(text, split_fractions, chunk_size, batch_size)
    local alphabet, chunk_iterators = make_chunk_iterators(text, split_fractions, chunk_size)

    local batch_iterators = {}
    for _, chunk_iterator in pairs(chunk_iterators) do
        batch_iterators[#batch_iterators + 1] = make_batch_iterator(chunk_iterator, batch_size)
    end

    return alphabet, batch_iterators
end

-- export the following functions globally
batcher.make_batch_iterators = make_batch_iterators
batcher.make_batch_iterator = make_batch_iterator
batcher.stack = stack
batcher.make_chunk_iterators = make_chunk_iterators
batcher.split_indices = split_indices
batcher.load_text = load_text
batcher.make_chunk_iterator = make_chunk_iterator

return batcher
