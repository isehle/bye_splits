import h5py
import datetime
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
from tensorflow.keras.layers import Dense, Flatten, Conv1D

from data_processing import preprocess, postprocess
#from plotter import Plotter

DATAMAX = 5
DATAMIN = 0

def are_tensors_equal(t1, t2):
    assert t1.shape == t2.shape
    return tf.math.count_nonzero(tf.math.equal(t1,t2))==t1.shape
        
def tensorflow_assignment(tensor, mask, lambda_op):
    """
    Emulate assignment by creating a new tensor.
    The mask must be 1 where the assignment is intended, 0 everywhere else.
    """
    assert tensor.shape == mask.shape
    other = lambda_op(tensor)
    return tensor * (1 - mask) + other * mask

def tensorflow_wasserstein_1d_loss(indata, outdata):
    """Calculates the 1D earth-mover's distance."""
    loss = tf.cumsum(indata) - tf.cumsum(outdata)
    loss = tf.abs( loss )
    return tf.reduce_sum( loss )

# https://www.tensorflow.org/guide/keras/custom_layers_and_models#the_model_class
class Architecture(tf.keras.Model):
    """Neural network model definition."""
    def __init__(self, inshape):
        super().__init__()
        assert len(inshape)==1
        self.inshape = inshape
        
        self.dense1 = Dense( units=1000,
                             activation='sigmoid',
                             use_bias=True )
        self.dense2 = Dense( units=self.inshape[0],
                             activation='sigmoid',
                             use_bias=True )

    def __call__(self, x):
        x = tf.cast(x, dtype=tf.float32)
        x = tf.reshape(x, shape=(-1, x.shape[0]))
        x = self.dense1(x)
        x = self.dense2(x)
        x = tf.squeeze(x)
        return x

class TriggerCellDistributor(tf.Module):
    """Neural net workings"""
    def __init__( self, indata,
                  # inbins, bound_size,
                  # kernel_size, window_size,
                  # phibounds, nbinsphi, rzbounds, nbinsrz,
                  # pars
                 ):
        """
        Manages quantities related to the neural model being used.
        Args: 
              - kernel_size: Length of convolutional kernels
              - window_size: Number of bins considered for each variance calculation.
              Note this is not the same as number of trigger cells (each bin has
              multiple trigger cells).
        """        
        self.indata = indata
        # self.boundary_size = bound_size
        # self.kernel_size = kernel_size
        # self.boundary_width = window_size-1
        # self.phibounds, self.nbinsphi = phibounds, nbinsphi
        # self.rzbounds, self.nbinsrz = rzbounds, nbinsrz

        self.architecture = Architecture(self.indata.shape)

        # assert len(pars)==1
        # self.pars = pars

        # self.subtract_max = lambda x: x - (tf.math.reduce_max(x)+1)
        # # convert bin ids coming before 0 (shifted boundaries) to negative ones
        # self.inbins = tensorflow_assignment( tensor=inbins,
        #                                      mask=tf.concat((tf.ones(self.boundary_size),
        #                                                      tf.zeros(inbins.shape[0]-self.boundary_size)), axis=0),
        #                                      lambda_op=self.subtract_max,
        #                                     )

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')

    def calc_loss(self, originaldata, outdata):
        """
        Calculates the model's loss function. Receives slices in R/z as input.
        Each array value corresponds to a trigger cell.
        Args: -originaldata: similar to originaldata, but after postprocessing (denormalization).
                 To avoid confusions, `originaldata` should not be used in this method.
              -inbins: ordered (ascendent) original phi bins
              -bound_size: number of replicated trigger cells to satisfy boundary conditions
              -outdata: neural net postprocessed phi values output
        """
        # opt = {'summarize': 10} #number of entries to print if assertion fails
        # asserts = [ tf.debugging.Assert(originaldata.shape == outdata.shape,
        #                                 [originaldata.shape, outdata.shape], **opt) ]
        asserts = []
        with tf.control_dependencies(asserts):
            equality_loss = tf.reduce_sum( tf.math.square(outdata-originaldata) )
            # loss_pars = [ tf.Variable(x, trainable=False) for x in self.pars ]

        return ( { 'equality_loss': 1 * equality_loss,
                  }
                )

    def train(self):
        # Reset the metrics at the start of the next epoch
        self.train_loss.reset_states()

        # for n in tf.range(steps): # any future loop within one epoch is done here
        with tf.GradientTape() as tape:
            tape.watch(self.indata)
            # tape.watch(self.inbins)

            prediction = self.architecture(self.indata)
            #prediction = postprocess(prediction, self.phibounds)
            #prediction = postprocess(prediction, (DATAMIN,DATAMAX))
            #original_data = postprocess(self.indata, self.phibounds)
            #original_data = postprocess(self.indata, (DATAMIN,DATAMAX))
            original_data = self.indata

            losses = self.calc_loss(original_data, prediction)
            loss_sum = tf.reduce_sum(list(losses.values()))
            print('Orig: ', original_data)

        #print(self.architecture.trainable_variables)

        gradients = tape.gradient(loss_sum, self.architecture.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, self.architecture.trainable_variables))
        self.train_loss(loss_sum)
        return losses, prediction

    def save_architecture_diagram(self, name):
        """Plots the structure of the used architecture."""
        assert name.split('.')[1] == 'png'
        tf.keras.utils.plot_model(
            self.architecture,
            to_file=name,
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=True,
            dpi=96,
        )

def save_scalar_logs(writer, scalar_map, epoch):
    """
    Saves tensorflow info for Tensorboard visualization.
    `scalar_map` expects a dict of (scalar_name, scalar_value) pairs.
    """
    with writer.as_default():
        for k,v in scalar_map.items():
            tf.summary.scalar(k, v, step=epoch)

def optimization(algo, **kw):
    #store_in  = h5py.File(kw['OptimizationIn'],  mode='r')
    #plotter = Plotter(**optimization_kwargs)

    #assert len(store_in.keys()) == 1
    #train_data = [tf.constant([.6]) for _ in range(2)]#
    train_data = [tf.range(DATAMAX, dtype=tf.float32) for _ in range(2)]
    train_data[0] -= DATAMIN
    train_data[0] /= DATAMAX-DATAMIN

    #_, train_data, boundary_sizes = preprocess(
    #     data=store_in['data'],
    #     nbins_phi=kw['NbinsPhi'],
    #     phi_bounds=(kw['MinPhi'],kw['MaxPhi']),
    #     nbins_rz=kw['NbinsRz'],
    #     window_size=kw['WindowSize']
    # )

    #orig_data = postprocess(train_data[0][:,0], phi_bounds=(kw['MinPhi'],kw['MaxPhi']))
    #plotter.save_orig_data( orig_data )
    
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'

    for i,rzslice in enumerate(train_data):
        if i>0:        #look at the first R/z slice only
            break
        summary_writer = tf.summary.create_file_writer(train_log_dir)

        tcd = TriggerCellDistributor(
            # indata=tf.convert_to_tensor(rzslice[:,0], dtype=tf.float32),
            indata=rzslice,
            # inbins=tf.convert_to_tensor(rzslice[:,1], dtype=tf.float32),
            # bound_size=boundary_sizes[i],
            # kernel_size=kw['KernelSize'],
            # window_size=kw['WindowSize'],
            # phibounds=(kw['MinPhi'],kw['MaxPhi']),
            # nbinsphi=kw['NbinsPhi'],
            # rzbounds=(kw['MinROverZ'],kw['MaxROverZ']),
            # nbinsrz=kw['NbinsRz'],
            # pars=(1., 0., 0., 0.),
        )
        #tcd.save_architecture_diagram('model{}.png'.format(i))

        for epoch in range(kw['Epochs']):
            dictloss, outdata = tcd.train()
            #dictloss.update({'initial_variance': initial_variance})
            save_scalar_logs(
                writer=summary_writer,
                scalar_map=dictloss,
                epoch=epoch
            )
            print('Epoch {}: {}'.format(epoch+1, tcd.train_loss.result()))
            print('In: ', train_data)
            print('Out: ', outdata)
            print()


            #plotter.save_gen_data(outdata.numpy())

        #plotter.plot(show_html=True)

if __name__ == "__main__":
    from airflow.airflow_dag import optimization_kwargs
    for falgo in optimization_kwargs['FesAlgos']:
        optimization( falgo, **optimization_kwargs )
