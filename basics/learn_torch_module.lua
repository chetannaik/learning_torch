require 'torch'

---------------------
-- TENSOR CREATION --
---------------------

-- construct a 5x3 matrix, uninitialized
t = torch.Tensor(5,3)

-- creation of a 4D-tensor 2x3x1x4
t = torch.Tensor(2,3,1,4)

-- creating tensors using LongStorage
ls = torch.LongStorage({3, 5, 1, 4, 6, 2})
t = torch.Tensor(ls)



-----------------------
-- TENSOR ATTRIBUTES --
-----------------------
-- dummy/placeholder tensor to check attributes
t = torch.Tensor(2,3,1,4)

-- check number of DIMENSIONS of a tensor
dim = t:nDimension()
-- or
dim = t:dim()

-- check the NUMBER OF ELEMENTS in a tensor
numElem = t:nElement()

-- retrieve SIZES of all the dimensions
size = t:size()

-- retrieve SIZE of a specific dimention
s2 = t:size(2)

-- retrieve TYPE of a tensor
ttype = t:type()

-- check if an object IS A TENSOR
isT = torch.isTensor(t)

-- a Tensor is a particular way of viewing a `Storage`.
-- access the tensor STORAGE
s = t:storage()

-- check if TWO TENSORS are of SAME SIZE?
x = torch.Tensor(2, 3)
y = torch.Tensor(2, 3)
sameSize = x:isSameSizeAs(y)


-- check is a TENSOR has the SAME SIZE as of aa STORAGE
x = torch.Tensor(2, 3)
y = torch.LongStorage({2, 3})
sameSize = x:isSize(y)



-------------------------
-- TENSOR CONSTRUCTION --
-------------------------

-- create a CUSTOM tensor by copying data from a TABLE
t = torch.Tensor({1, 2, 0.5, 8})
t = torch.Tensor({{1, 3}, {5, 9}})

-- create a tensor of ONES
t = torch.ones(3)
t = torch.ones(3, 4)

-- create a tensor of ZEROS
t = torch.zeros(3, 4)

-- create a IDENTITY matrix
t = torch.eye(3)

-- create a LINEAR SPACE: ONE DIMENSIONAL TENSOR of n equally spaced points
t = torch.linspace(1, 4, 5)

-- create a DIAGONAL matrix using input tensor as the diagonal element
td = torch.diag(t)

-- create a LOG SPACE: ONE DIMENSIONAL TENSOR of n logarithmically equally
-- spaced points between 10^(arg1) and 10^(arg2)
t = torch.logspace(1, 4, 5)

-- create a tensor using RANGE and size parameters
t = torch.range(1, 4, 0.9)

-- create a tensor using UNIFORM RANDOM distribution of numbers range (0, 1)
t = torch.rand(2, 3)

-- create a tensor using GAUSSIAN/NORMAL RANDOM distribution of numbers with
-- mean 0 and variance 1
t = torch.randn(2, 3)

-- create a tensor containing RANDOM PERMUTATION of numbers between 1 and n
t = torch.randperm(5)

-- ITERATE and fill elements into a tensor
t = torch.Tensor(2, 3)
for i = 1, 2 do
    for j = 1, 3 do
        t[i][j] = (i - 1)*3 + j
    end
end

-- construct tensor with content using APPLY FUNCTION
t = torch.Tensor(2, 3)
local i = 0
t:apply(function()
            i = i + 1
            return i
        end)

-- fill ZEROS into a tensor
t = torch.Tensor(2, 3)
t:zero()

-- FILL ANY VALUE into a tensor
t = torch.Tensor(2, 3)
t:fill(3.14)

-- by COPYING from an existing tensor
t1 = torch.Tensor(2, 3):random(1, 6)
t2 = torch.Tensor(6):copy(t1)

-- by CLONING existing tensor
t1 = torch.rand(2, 3) t2 = t1:clone()

-- clone if NON CONTIGUOUS else just return a reference
t1 = torch.Tensor(2, 3):fill(1)
--print(t1)
t2 = t1:contiguous():fill(2)
--print(t2)
t2[1][2] = 4
--print(t1)

-- create tensor based on MULTINOMIAL PROBABILITY DISTRIBUTION
-- y=torch.multinomial(p,n,[replacement]) returns a tensor y where each row
-- contains n indices sampled from the multinomial probability distribution
-- located in the corresponding row of tensor p.
p = torch.Tensor{1, 1, 0.5, 0}
t = torch.multinomial(p, 15, true)

-- create a LOWER DIAGONAL MATRIC from full matrix
full = torch.Tensor(4, 5):random(1, 8)
ld = torch.tril(full)

-- create a UPPER DIAGONAL MATRIC from full matrix
full = torch.Tensor(4, 5):random(1, 8)
ud = torch.triu(full)



-------------------------
-- TENSOR MANIPULATION --
-------------------------

-- RESHAPE tensor
t1 = torch.Tensor(2, 3):random(1, 8)
t2 = torch.reshape(t1, 3, 2)

-- RESIZE tensor
t = torch.range(1, 12):double():reshape(3, 4)

-- TRANSPOSE tensor
-- transpose(dim1, dim2)
-- Returns a tensor where dimensions dim1 and dim2 have been swapped. For 2D
-- tensors, the convenience method of t() is available.
t = torch.Tensor({{1, 2}, {3, 4}, {8, 9}})
tt = t:t()

-- transpose one set of dimensions to another = PERMUTE
t1 = torch.Tensor(2,4,6,3)
t2 = t1:permute(3, 4, 2, 1)

-- manipulating tensor VIEW
t = torch.Tensor(1, 9):random(1, 9)
tv = t:view(3, 3)
-- FLATTEN
t = tv:view(tv:nElement())

-- EXPAND tensor
-- expanding tensor does not allocate new memory
t = torch.rand(5, 1)
t2 = torch.expand(t, 5, 3)

-- REPEAT tensor
-- repeating tensor allocated new memory
t = torch.rand(4, 1)
t2 = torch.repeatTensor(t, 1, 3)

-- SQUEEZE tensor
-- squeeze([dim])
-- removes all singleton dimensions of the tensor. If dim is given, squeezes
-- only that particular dimension of the tensor.
t1 = torch.rand(2, 1, 2, 1, 2)
t2 = torch.squeeze(t1)
t3 = torch.squeeze(t1, 2)

-- UNFOLD tensor
t1 = torch.Tensor(5):random(1, 9)
t2 = t1:unfold(1, 3, 1)
t3 = t1:unfold(1, 3, 2)

-- SORT tensor
-- y,i=torch.sort(x) returns a tensor y where all entries are sorted along the
-- last dimension, in ascending order. It also returns a tensor i that provides
-- the corresponding indices from x.
t = torch.Tensor(4, 5):random(1, 9)
sorted = torch.sort(t)
sorted, indices = torch.sort(t, 1)


------------------
-- TYPE CASTING --
------------------

-- change TYPE
t = torch.Tensor(3):fill(3.14)
t:int()
-- other types: byte(), char(), short(), int(), long(), float(), double()


--------------------
-- ITEM SELECTION --
--------------------

t = torch.Tensor(4, 6):random(1, 9)
-- METHOD 1
t1 = t[1][3]
t2 = t[2][4]

-- METHOD 2
-- ROW/COLUMN SELECTION
t1 = t[{1, 3}] -- equivalent to t[1][3]
t2 = t[{{}, 3}] -- print 3rd column, all rows
t2 = t[{2, {3, 5}}] -- print 2nd row columns 3 to 5

-- SUBSET/SUB TENSOR SELECTION
-- select a subset and returns a REFERENCE to original tensor
t3 = t[{{2, 3}, {3, 5}}]
-- selects a subset, CLONES and returns a new tensor
t4 = t:sub(2, 3, 3, 5)
-- selects a subset and returns a REFERENCE to original tensor
t5 = t:narrow(1, 2, 2)
t6 = t:narrow(1, 2, 2):narrow(2, 3, 3)

-- can use select(dim, index) to perform tensor slicing

-- select using INDEX
-- index(dim, index)
-- Returns a new Tensor which indexes the original Tensor along dimension dim
-- using the entries in torch.LongTensor index. The returned Tensor does
-- not use the same storage as the original Tensor.
t = torch.Tensor(4, 5):random(1, 9)
t2 = t:index(1, torch.LongTensor{3,1})

-- select using GATHER
-- gather(dim, index)
-- Creates a new Tensor from the original tensor by gathering a number of
-- values from each "row", where the rows are along the dimension dim. The
-- values in a LongTensor, passed as index, specify which values to take from
-- each row.
t = torch.Tensor(5, 5):random(1, 9)
t2 = t:gather(1, torch.LongTensor{{1, 2, 3, 4, 5}, {2, 3, 4, 5, 1}})

-- select using SCATTER
-- scatter(dim, index, src|val)
-- Writes all values from tensor src or the scalar val into self at the
-- specified indices. The indices are specified with respect to the given
-- dimension, dim, in the manner described in gather.
t = torch.rand(2, 4)
t2 = torch.zeros(4, 4):scatter(1, torch.LongTensor{{1, 2, 4 ,3}, {3, 4 ,2, 1}}, t)

-- select using MASKED SELECT
-- maskedSelect(mask)
-- Returns a new Tensor which contains all elements aligned to a 1 in the
-- corresponding mask. This mask is a torch.ByteTensor of zeros and ones.
t = torch.range(1, 12):double():resize(3, 4)
mask = torch.ByteTensor(2, 6):bernoulli()
t2 = t:maskedSelect(mask)


----------------------
-- TENSOR FUNCTIONS --
----------------------
-- ref: https://github.com/torch/torch7/blob/master/doc/tensor.md
-- apply
-- map
-- map2
-- split
-- chunk
