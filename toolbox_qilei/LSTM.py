import cPickle

from toolbox_qilei.LSTM_functions import *
SEED = 123
numpy.random.seed(SEED)

class lstm:
    def __init__(self):
        self.options = lstm_configuration()
        self.tparams = lstm_init(self.options)
        self.data = lstm_data(self.options)

    def lstm_build(self):
        options = self.options
        tparams = self.tparams
        trng = RandomStreams(SEED)
        use_noise = theano.shared(numpy_floatX(0.))

        x = tensor.matrix('x', dtype='int64')
        y = tensor.vector('y', dtype='int64')

        n_timesteps = x.shape[0]
        n_samples = x.shape[1]
        proj = lstm_layer(tparams, x, options, options['encoder'])

        #Here, original author takes the average values on ntimesteps as the output of hidden layer
        #proj = proj.mean(axis=0)
        #In our code, the teacher is the position of joints in next timestep. Therefore, the code
        #should be changed as following.
        if options['use_dropout']:
            proj = dropout_layer(proj, use_noise, trng)

        pred = tensor.nnet.softmax(tensor.dot(proj, tparams['U']) + tparams['b'])

        f_pred_prob = theano.function([x], pred, name='f_pred_prob')
        f_pred = theano.function([x], pred.argmax(axis=1), name='f_pred')

        off = 1e-8
        if pred.dtype == 'float16':
            off = 1e-6
        
        cost = -tensor.log(pred[tensor.arange(n_samples), y] + off).mean()

        if options['decay_c'] > 0:
            decay_c = theano.shared(numpy_floatX(options['decay_c']), name='decay_c')
            weight_decay = 0
            weight_decay = (tparams['U'] ** 2).sum()
            weight_decay *= options['decay_c']
            cost += weight_decay
        f_cost = theano.function([x, y], cost, name='f_cost')
        grads = tensor.grad(cost, wrt=list(tparams.values()))

        lr = tensor.scalar(name='lr')
        f_grad_shared, f_update = options['optimizer'](lr, tparams, grads,
                                                       x, y, cost)
        funcs = OrderedDict()
        funcs['f_pred_prob'] = f_pred_prob
        funcs['f_pred'] = f_pred
        funcs['f_grad_shared'] = f_grad_shared
        funcs['f_update'] = f_update
        self.funcs = funcs
        self.use_noise = use_noise


    def lstm_train(self):
        #Firstly, we should load the parameters from options
        options = self.options
        tparams = self.tparams
        use_noise = self.use_noise
        funcs = self.funcs
        f_pred_prob = funcs['f_pred_prob']
        f_pred = funcs['f_pred']
        f_grad_shared = funcs['f_grad_shared']
        f_update = funcs['f_update']
        dim_proj = options['dim_proj'], # word embedding dimension and LSTM number of hidden units
        patience = options['patience'], # Number of epochs to wait before early stop if no progress
        max_epochs = options['max_epochs'], # the maxium number of epoch to run
        dispFreq = options['dispFreq'], #Display to stdout the training progress every N updates
        decay_c = options['decay_c'], #weught decay for the clssifier applied to the U weights
        lrate = options['lrate'], #learning rate for sgd (not used for adadelta and rmsprop)
        optimizer = options['optimizer'], #sgd, adadelta and rmsprop available, sgd very hard to use,
         #recommanded (probably need momentum and decaying learning rate).
        saveto = options['saveto'], #the best model will be saced there
        validFreq = options['validFreq'], #Compute the validation error after this number of updates.
        saveFreq = options['saveFreq'], #Save the parameters after every saveFreq updates
        maxlen = options['maxlen'], #Sequence longer then this get ignored
        batch_size = options['batch_size'], #the batch size during training
        valid_batch_size = options['valid_batch_size'], #the batch size used for validation/test set/
        dataset = options['dataset'],

        noise_std = options['noise_std'] =0.,
        use_dropout = options['use_dropout']=True,
        reload_model = options['reload_model']=None, #path to a saved model we want to start from
        test_size = options['test_size']=-1,# if > 0, we keep only this number of test example



        history_errs = []
        best_p = None
        bad_count = 0

        #prepare data
        train = self.data[0]
        test = self.data[1]
        valid = self.data[2]
        kf_valid = get_minibatches_idx(valid.shape[0], valid_batch_size)
        kf_test = get_minibatches_idx(test.shape[0], valid_batch_size)

        print("%d train examples" % train.shape[0])
        print("%d valid examples" % valid.shape[0])
        print("%d test examples" % test.shape[0])



        if validFreq == -1:
            validFreq = len(train[0]) // batch_size
        if saveFreq == -1:
            saveFreq = len(train[0]) // batch_size

        uidx = 0 #the number of update done
        estop = False #early stop

        start_time = time.time()

        try:
            for eidx in range(max_epochs):
                n_samples = 0
                kf = get_minibatches_idx(train.shape[0], batch_size, shuffle=True)

                for _, train_index in kf:
                    uidx += 1
                    use_noise.set_value(1.)

                    #select the random examples for this minibatch

                    y = [train[t, -1] for t in train_index]
                    x = [train[t, :-1] for t in train_index]

                    #get the data in numpy.ndarray foramt
                    #this swap the axis
                    #return something of shape(minibatch maxlen nsamples)

                    n_samples += x.shape[0]
                    cost  = f_grad_shared(x, y)
                    f_update(lrate)

                    if numpy.isnan(cost) or numpy.isinf(cost):
                        print('bad cost detected:', cost)
                        return 1., 1., 1.
                    if numpy.mod(uidx, dispFreq) == 0:
                        print('Epoch', eidx, 'Update', uidx, 'Cost', cost)

                    if saveto and numpy.mod(uidx, saveFreq) == 0:
                        print('Saving...')
                        if best_p is not None:
                            params = best_p
                        else:
                            params = unzip(tparams)

                    if numpy.mod(uidx, validFreq) == 0:
                        use_noise.set_value(0.)
                        train_err = pred_error(f_pred, train, kf)
                        valid_err = pred_error(f_pred, valid, kf_valid)
                        test_err = pred_error(f_pred, test, kf_test)
                        history_errs.append([valid_err, test_err])

                        if(best_p is None or valid_err <= numpy.array(history_errs)[:, 0].min()):
                            best_p = unzip(tparams)
                            bad_counter = 0
                        print(('Train ', train_err, 'Valid ', valid_err, 'Test ', test_err))

                        if(len(history_errs) > patience and
                                   valid_err >= numpy.array(history_errs)[:-patience, 0].min()):
                            bad_counter += 1
                            if bad_counter > patience:
                                print('Early Stop!')
                                estop = True
                                break
                print('Seen %d samples' % n_samples)

                if estop:
                    break
        except KeyboardInterrupt:
            print("Training interupted")

        end_time = time.time()
        if best_p is not None:
            zipp(best_p, tparams)
        else:
            best_p = unzip(unzip(tparams))
        use_noise.set_value(0.)
        kf_train_sorted = get_minibatches_idx(len(train[0]), batch_size)
        train_err = pred_error(f_pred, train, kf_train_sorted)
        valid_err = pred_error(f_pred, valid, kf_valid)
        test_error = pred_error(f_pred, test, kf_test)

        print('Train', train_err, 'Valid', valid_err, 'Test', test_err)

        if saveto:
            numpy.savez(saveto, train_err=train_err,
                        valid_err=valid_err, test_err=test_err,
                        history_errs=history_errs, **best_p)

        print('The code run for %d epochs, with %f sec/epochs' % (
                (eidx + 1), (end_time - start_time) / (1. * (eidx + 1))))
        print( ('Training took %.1fs' %
            (end_time - start_time)))
        return train_err, valid_err, test_err



def lstm_configuration():
    options=OrderedDict()
    options['dim_proj']= 32 # LSTM number of hidden units
    options['dim_input'] = 3 # Number of input units
    options['dim_output'] = 64 # Number of output units
    options['patience']=10 # Number of epochs to wait before early stop if no progress
    options['max_epochs']=5000 # the maxium number of epoch to run
    options['dispFreq']=10 #Display to stdout the training progress every N updates
    options['decay_c']=0 #weught decay for the clssifier applied to the U weights
    options['lrate']=0.0001 #learning rate for sgd (not used for adadelta and rmsprop)
    options['optimizer']=adadelta, #sgd, adadelta and rmsprop available, sgd very hard to use,
     #recommanded (probably need momentum and decaying learning rate).
    options['saveto']='lstm_model.npz', #the best model will be saced there
    options['validFreq']=370 #Compute the validation error after this number of updates.
    options['saveFreq']=1110 #Save the parameters after every saveFreq updates
    options['maxlen']=100 #Sequence longer then this get ignored
    options['batch_size']=16 #the batch size during training
    options['valid_batch_size']=64 #the batch size used for validation/test set/
    options['dataset']='imdb'

    options['noise_std'] =0.
    options['use_dropout']=True
    options['reload_model']=None #path to a saved model we want to start from
    options['test_size']=-1 # if > 0, we keep only this number of test example
    options['data_path'] = './data/'
    options['encoder'] = 'lstm'


    return options

def lstm_init(options):
    #Set the random number generators' seeds for consistency
    SEED = 123
    numpy.random.seed(SEED)

    params = OrderedDict()
    tparams = OrderedDict()

    #the following three parameters are for lstm
    randn = numpy.random.rand(options['dim_input'],
                              options['dim_proj'])
    randn = randn.astype(config.floatX)
    W = numpy.concatenate([randn, randn, randn, randn], axis=1)
    #W = numpy.concatenate([ortho_weight(options['dim_proj']),
    #                       ortho_weight(options['dim_proj']),
    #                      ortho_weight(options['dim_proj']),
    #                       ortho_weight(options['dim_proj'])], axis=1)
    params['lstm_W'] = W
    U = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params['lstm_U'] = U
    b = numpy.zeros((4 * options['dim_proj'],))
    params['lstm_b'] = b.astype(config.floatX)

    #the following two parameters are for classifier

    params['U'] = 0.01 * numpy.random.randn(options['dim_proj'],
                                            options['dim_output']).astype(config.floatX)
    params['b'] = numpy.zeros((options['dim_output'],)).astype(config.floatX)

    # for the calculation in theano, we must convert the parameters in numpy format
    # to thenano format

    for kk, pp in params.items():
        tparams[kk] = theano.shared(params[kk], name=kk)

    return tparams
def lstm_data(options):
    #this function is design to prepare the train. test, valid data
    data_path = options['data_path']
    train_data = cPickle.load(open(data_path + 'train_data_traj.save'))
    test_data = cPickle.load(open(data_path + 'test_data_traj.save'))
    valid_data = cPickle.load(open(data_path + 'valid_data_traj.save'))
    data = [0,0,0]
    data = [train_data, valid_data, test_data]
    return data
