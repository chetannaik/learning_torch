require 'nn'
require 'nngraph'

x1 = nn.Identity()():annotate{
       name = 'Input 1', descrption = 'Input 1 Node',
          graphAttributes = {color = 'deepskyblue2', fontcolor = 'dodgerblue4',
             style = 'filled', fillcolor = 'deepskyblue2'}
         }
x2 = nn.Identity()():annotate{
       name = 'Input 2', descrption = 'Input 2 Node',
          graphAttributes = {color = 'deepskyblue2', fontcolor = 'dodgerblue4',
             style = 'filled', fillcolor = 'deepskyblue2'}
         }
a = nn.CAddTable()({x1, x2}):annotate{
       name = 'Output', descrption = 'Output Node',
          graphAttributes = {color = 'darkolivegreen3', fontcolor = 'darkgreen',
             style = 'filled', fillcolor = 'darkolivegreen3'}
         }
m = nn.gModule({x1, x2}, {a})


--h1 = nn.Linear(20, 10)()
--h2 = nn.Linear(10, 1)(nn.Tanh()(nn.Linear(10, 10)(nn.Tanh()(h1))))
--mlp = nn.gModule({h1}, {h2})

--x = torch.rand(20)
--dx = torch.rand(1)
--mlp:updateOutput(x)
--mlp:updateGradInput(x, dx)
--mlp:accGradParameters(x, dx)

-- draw graph (the forward graph, '.fg')
graph.dot(m.fg, 'LIN', 'simple1_output')
