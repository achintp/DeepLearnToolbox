function test_example_DAG
load mnist_uint8;

train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);

% normalize
[train_x, mu, sigma] = zscore(train_x);
test_x = normalize(test_x, mu, sigma);

rand('state',0)
nn = daggensetup([784 500 100 10]);
opts.numepochs =  3;   %  Number of full sweeps through data
opts.batchsize = 100;  %  Take a mean gradient step over this many samples
[nn, L] = daggentrain(nn, train_x, train_y, opts);

[er, bad] = daggentest(nn, test_x, test_y);

assert(er < 0.08, 'Too big error');