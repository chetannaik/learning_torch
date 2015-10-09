-- METHOD 1
-- read csv with numbers to create a tensor
local file = "test1.csv"
local separator = ","

require 'torch'

csvFile = io.open(file, 'r')
header = csvFile:read():split(separator)

local i = 0
for line in io.lines(file) do
    if i == 0 then
        COLS = #line:split(',')
    end
    i = i + 1
end

local ROWS = i - 1  -- Minus 1 because of header

local data = torch.Tensor(ROWS, COLS)

local i = 0
for line in csvFile:lines('*l') do
    i = i + 1
    local l = line:split(separator)
    for key, val in ipairs(l) do
        data[i][key] = val
    end
end

csvFile:close()


-- METHOD 2
-- use library to do the same. But note that here, order of columns is not
-- maintained.`
local file = "test1.csv"

local csv2tensor = require 'csv2tensor'

f = csv2tensor.load(file)


-- METHOD 3
-- read csv which contain any general data
local file = "test2.csv"

require 'csvigo'

f = csvigo.load{ path=file }


-- CHECK: http://lua-users.org/wiki/CsvUtils
