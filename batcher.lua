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

-- function to generate chunks of encoded data based on chunk size
function generate_chunks(encoded_text, chunk_size)
    local n_chunks = math.floor(encoded_text:size()[1]/chunk_size)
    local indices = torch.randperm(n_chunks)
--     local indices = torch.range(1, n_chunks)

    function co()
        for i=1, n_chunks do
            local index = indices[i]
            local lower = (index - 1)*chunk_size
            local upper = index*chunk_size
            print("index: " .. index)
            print("lower: " .. lower)
            print("upper: " .. upper)
            chunk = encoded_text[{{lower, upper}}]
            coroutine.yield(chunk)
        end
    end

    return coroutine.create(co)
end

local text = load_text()
local alphabet, encoded = char_to_ints(text)
local chunk_generator = generate_chunks(encoded, 10)
print(coroutine.resume(chunk_generator))
print(coroutine.resume(chunk_generator))
