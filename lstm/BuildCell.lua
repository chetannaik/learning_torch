require 'torch'
require 'nngraph'

--[[
Make a LSTM graph node.

[Implemented]
LSTM Architecture (simple):
1. (Input gate)        : i_gate = sigm(W_i * x + U_i * h_prev)
2. (Forget gate)       : f_gate = sigm(W_f * x + U_f * h_prev)
3. (Output gate)       : o_gate = sigm(W_o * x + U_o * h_prev)
4. (New memory cell)   : m_temp = tanh(W_c * x + U_c * h_prev)
5. (Final memory cell) : m_next = f_gate ◦ m_prev + i_gate ◦ m_temp
6. (Hidden state)      : h_next = o_gate ◦ tanh(c(t))
Reference:
    http://cs224d.stanford.edu/lecture_notes/LectureNotes4.pdf
    http://colah.github.io/posts/2015-08-Understanding-LSTMs/

[Not Implemented]
LSTM Architecture (with peephole connections):
i_gate = sigm(W_i * x + U_i * h_prev + V_i * m_prev)
f_gate = sigm(W_f * x + U_f * h_prev + V_f * m_prev)
m_temp = tanh(W_c * x + U_c * h_prev)
m_next = f_gate * m_prev + i_gate * m_temp
o_gate = sigm(W_o * x + U_o * h_prev + V_o * m_next)
h_next = o_gate * tanh(m_next)
Paper reference:
    http://arxiv.org/pdf/1308.0850v5.pdf
    http://arxiv.org/pdf/1503.04069.pdf

where, ◦ - elementwise product

For a batch size (B), input size (I) and hidden size (H), the sizes should be
as follows

    x: BxI
    h_prev: BxH
    m_prev: BxH

Returns a nn Module output from nngraph gModule()
]]--

local function BuildCell(input_size, hidden_size)

    -- input placeholders
    local x = nn.Identity()()
    local h_prev = nn.Identity()()
    local m_prev = nn.Identity()()

    -- there are four sets of weights for each of x and h_prev.
    local x2h = nn.Linear(input_size, hidden_size*4)(x)
    local h2h = nn.Linear(hidden_size, hidden_size*4)(h_prev)

    local xh2h = nn.CAddTable()({x2h, h2h})

    -- data flowing through xh2h is size Bx4H. We reshape this to Bx4xH so that
    -- that we can separate the data into four separate BxH streams. Thus when
    -- we split, we split on the second dimension to split into four separate
    -- streams.
    local xh2h_reshaped = nn.Reshape(4, hidden_size, true)(xh2h)
    local xh2h_split_by_gate = nn.SplitTable(2)(xh2h_reshaped)

    -- separate out the split tables
    local xh2h_i_gate = nn.SelectTable(1)(xh2h_split_by_gate)
    local xh2h_f_gate = nn.SelectTable(2)(xh2h_split_by_gate)
    local xh2h_o_gate = nn.SelectTable(3)(xh2h_split_by_gate)
    local xh2h_m_temp = nn.SelectTable(4)(xh2h_split_by_gate)

    -- input, forget and output gates
    local i_gate = nn.Sigmoid()(xh2h_i_gate)
    local f_gate = nn.Sigmoid()(xh2h_f_gate)
    local o_gate = nn.Sigmoid()(xh2h_o_gate)

    -- compute temporary memory
    local m_temp = nn.Tanh()(xh2h_m_temp)

    -- compute new memory
    local m_prev_forget = nn.CMulTable()({f_gate, m_prev})
    local m_temp_input = nn.CMulTable()({i_gate, m_temp})
    local m_next = nn.CAddTable()({m_prev_forget, m_temp_input})

    -- compute new hidden state
    local h_next = nn.CMulTable()({o_gate, nn.Tanh()(m_next)})

    -- make a module by combining all the above submodules into a graph
    local module = nn.gModule({x, h_prev, m_prev}, {h_next, m_next})

    return module
end

return BuildCell
