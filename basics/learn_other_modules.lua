----------------
-- sys module --
----------------
require 'sys'

-- high precision clock
t = sys.clock()

-- measure runtime of something
sys.tic()
-- do something
t = sys.toc()

-- sleep 1.7 seconds
sys.sleep(1.7)

-- Always returns the path of the file in which this call is made.
path, fname = sys.fpath()


-----------------
-- path module --
-----------------

require 'paths'

-- iter dirs+files in a directory
for f in paths.files('.') do
    print(f)
end

-- iter files in a directory
for f in paths.iterfiles('.') do
    print(f)
end
