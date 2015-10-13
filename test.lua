require 'nngraph'
gru = require 'gru'
lstm = require 'lstm'

local num_input = 10
local num_hidden = 64
local max_length = 32

-- GRU Cell Network
local gru_cell = gru.BuildCell(num_input, num_hidden)
graph.dot(gru_cell.fg, 'GRU Cell', 'gru/graph/gru_cell')

-- GRU Chain Network
local x = nn.Identity()()
local gru_chain = gru.BuildChain(num_input, num_hidden, max_length)
local lastHidden = nn.SelectTable(1)(gru_chain(x))
local gru_chain_module = nn.gModule({x}, {lastHidden})
graph.dot(gru_chain_module.fg, 'GRU Chain', 'gru/graph/gru_chain')


-- LSTM Cell Network
local lstm_cell = lstm.BuildCell(num_input, num_hidden)
graph.dot(lstm_cell.fg, 'LSTM Cell', 'lstm/graph/lstm_cell')

-- LSTM Chain Network
local x = nn.Identity()()
local lstm_chain = lstm.BuildChain(num_input, num_hidden, max_length)
local lastHidden = nn.SelectTable(1)(lstm_chain(x))
local lstm_chain_module = nn.gModule({x}, {lastHidden})
graph.dot(lstm_chain_module.fg, 'LSTM Chain', 'lstm/graph/lstm_chain')
