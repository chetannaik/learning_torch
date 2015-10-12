require 'torch'
require 'nngraph'

--[[
Make a GRU graph node.
Paper reference: http://arxiv.org/pdf/1412.3555v1.pdf

GRU Architecture:
1. (Update gate) : u_gate = sigm( W_z * x + U_z * h_prev )
2. (Reset gate)  : r_gate = sigm( W_r * x + U_r * h_prev )
3. (New memory)  : h_temp = tanh( W_t * x + r_gate ◦ U_t * h_prev )
4. (Hidden state): h_next = ((1 − u_gate) ◦ h_temp) + (u_gate ◦ h_prev)

where, ◦ - elementwise product


For a batch size (B), input size (I) and hidden size (H), the sizes should be
as follows

    x: BxI
    prev_h: BxH

Returns a nn Module output from nngraph gModule()
]]--

local function gru_cell(input_size, hidden_size)

    -- input placeholders
    local x = nn.Identity()()
    local h_prev = nn.Identity()()

    -- there are two sets of weights for each of x and h_prev.
    local x2h = nn.Linear(input_size, hidden_size*2)(x)
    local h2h = nn.Linear(hidden_size, hidden_size*2)(h_prev)

    local xh2h = nn.CAddTable()({x2h, h2h})

    -- data flowing through xh2h is size Bx2H. We reshape this to Bx2xH so that
    -- that we can separate the data into two separate BxH streams. Thus when
    -- we split, we split on the second dimension to split into two separate
    -- streams.
    local xh2h_reshaped = nn.Reshape(2, hidden_size, true)(xh2h)
    local xh2h_split_by_gate = nn.SplitTable(2)(xh2h_reshaped)

    -- separate out the split tables
    local xh2h_u_gate = nn.SelectTable(1)(xh2h_split_by_gate)
    local xh2h_r_gate = nn.SelectTable(2)(xh2h_split_by_gate)

    -- update and reset gates
    local u_gate = nn.Sigmoid()(xh2h_u_gate)
    local r_gate = nn.Sigmoid()(xh2h_r_gate)

    -- compute temporary/candidate hidden state
    local h_reset = nn.CMulTable()({r_gate, h_prev})
    local h_transformed = nn.Linear(hidden_size, hidden_size)(h_reset)
    local x_transformed = nn.Linear(input_size, hidden_size)(x)
    local h_temp = nn.Tanh()(nn.CAddTable()({x_transformed, h_transformed}))

    -- compute new hidden state
    local u_gate_compliment = nn.AddConstant(1, false)(nn.MulConstant(-1, false)(u_gate))
    local h_updated = nn.CMulTable()({u_gate_compliment, h_prev})
    local h_temp_updated = nn.CMulTable()({u_gate, h_temp})
    local h_next = nn.CAddTable()({h_updated, h_temp_updated})

    -- make a module by combining all the above submodules into a graph
    local module = nn.gModule({x, h_prev}, {h_next})

    return module
end

return gru_cell
