import time
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_v2_behavior

from tqdm import tqdm

from libs.pgainutils import normalization, renormalization, rounding, binary_sampler, uniform_sampler, sample_batch_index

np.random.seed(42)
tf.random.set_seed(42)

def pgain(data_x, parameters):
    '''Impute missing values in data_x with pearson correlation
    Args:
    - data_x: original data with missing values
    - parameters: GAN Network parameters:
      - batch_size: Batch size
      - hint_rate: Hint rate
      - alpha: Hyperparameter
      - iterations: Iterations
    Returns:
    - imputed_data: imputed data
    '''

    # tf.reset_default_graph()
    tf.compat.v1.reset_default_graph()

    # System parameters
    batch_size = parameters['batch_size']
    hint_rate = parameters['hint_rate']
    alpha = parameters['alpha']
    iterations = parameters['iterations']

    # Other parameters
    no, dim = data_x.shape

    # Hidden state dimensions
    h_dim = int(dim)

    # Define mask matrix
    data_m = 1-np.isnan(data_x)

    # Normalization
    norm_data, norm_parameters = normalization(data_x)
    norm_data_x = np.nan_to_num(norm_data, 0)

    def generator(x, m):
        tf.compat.v1.disable_eager_execution()
        with tf.compat.v1.variable_scope('generator', reuse=tf.compat.v1.AUTO_REUSE):
            w_init = tf.initializers.GlorotUniform()
            # w_init = tf.Variable(xavier_init([dim*2, h_dim]))

            cat1 = tf.compat.v1.concat([x, m], 1)

            dense1 = tf.compat.v1.layers.dense(cat1, dim, kernel_initializer=w_init)
            relu1 = tf.compat.v1.nn.leaky_relu(dense1)

            dense2 = tf.compat.v1.layers.dense(relu1, h_dim, kernel_initializer=w_init)
            relu2 = tf.compat.v1.nn.leaky_relu(dense2)

            g_logit = tf.compat.v1.layers.dense(relu2, dim, kernel_initializer=w_init)
            g_prob = tf.compat.v1.nn.sigmoid(g_logit)

            return g_prob

    def discriminator(x, h, reuse=False):
        tf.compat.v1.disable_eager_execution()
        with tf.compat.v1.variable_scope('discriminator', reuse=reuse):#tf.compat.v1.AUTO_REUSE):
            w_init = tf.initializers.GlorotUniform()

            # w_init = tf.Variable(xavier_init([dim*2, h_dim]))

            cat1 = tf.compat.v1.concat([x, h], 1)

            dense1 = tf.compat.v1.layers.dense(cat1, dim, kernel_initializer=w_init)
            relu1 = tf.compat.v1.nn.leaky_relu(dense1)

            dense2 = tf.compat.v1.layers.dense(relu1, h_dim, kernel_initializer=w_init)
            relu2 = tf.compat.v1.nn.leaky_relu(dense2)

            d_logit = tf.compat.v1.layers.dense(relu2, dim, kernel_initializer=w_init)
            # d_logit = tf.layers.dense(relu2, 1, kernel_initializer=w_init)
            d_prob = tf.compat.v1.nn.sigmoid(d_logit)

            return d_prob, d_logit

    # variables : input
    tf.compat.v1.disable_eager_execution()
    x = tf.compat.v1.placeholder(tf.float32, shape=(None, dim))
    m = tf.compat.v1.placeholder(tf.float32, shape=(None, dim))
    h = tf.compat.v1.placeholder(tf.float32, shape=(None, dim))
    # z = tf.compat.v1.placeholder(tf.float32, shape=(None, 50))

    # networks : generator
    G_sample = generator(x, m)
    hat_x = x * m + G_sample * (1  - m)

    # networks : discriminator
    D_real, D_real_logits = discriminator(x, h)
    D_fake, D_fake_logits = discriminator(hat_x, h, reuse=True)

    # loss for each network
    # D_loss_real = tf.compat.v1.reduce_mean(tf.compat.v1.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.compat.v1.ones([batch_size, dim])))
    # D_loss_fake = tf.compat.v1.reduce_mean(tf.compat.v1.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.compat.v1.zeros([batch_size, dim])))
    D_loss_temp = -tf.compat.v1.reduce_mean(m * tf.compat.v1.math.log(D_fake + 1e-8) + (1 - m) * tf.compat.v1.math.log(1. - D_fake + 1e-8))

    # D_loss = D_loss_real + D_loss_fake + D_loss_temp
    D_loss = D_loss_temp

    # G_loss = tf.compat.v1.reduce_mean(tf.compat.v1.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones([batch_size, dim])))
    G_loss_temp = -tf.compat.v1.reduce_mean((1 - m) * tf.compat.v1.math.log(D_fake + 1e-8))
    MSE_loss = tf.compat.v1.reduce_mean((m * x - m * G_sample)**2) / tf.compat.v1.reduce_mean(m)

    G_loss = G_loss_temp + alpha * MSE_loss

    # trainable variables for each network
    T_vars = tf.compat.v1.trainable_variables()
    D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
    G_vars = [var for var in T_vars if var.name.startswith('generator')]

    # optimizer for each network
    with tf.compat.v1.control_dependencies(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)):
        D_optim = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001, beta1=0.5).minimize(D_loss, var_list=D_vars)
        G_optim = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001, beta1=0.5).minimize(G_loss, var_list=G_vars)

    # open session and initialize all variables
    sess = tf.compat.v1.InteractiveSession()
    tf.compat.v1.global_variables_initializer().run()

    train_hist = {}
    train_hist['D_losses'] = []
    train_hist['G_losses'] = []
    train_hist['per_epoch_ptimes'] = []
    train_hist['total_ptime'] = []

    # training-loop
    np.random.seed(int(time.time()))
    # print('training start!')
    start_time = time.time()

    for it in tqdm(range(iterations)):
        G_losses = []
        D_losses = []
        epoch_start_time = time.time()
        # for iter in range(len(train_set) // batch_size):

        batch_idx = sample_batch_index(no, batch_size=batch_size)

        # update discriminator
        x_mb = norm_data_x[batch_idx, :]
        m_mb = data_m[batch_idx, :]

        # random noise vector
        z_mb = uniform_sampler(0, 0.01, batch_size, dim)
        # hint vector
        h_mb_temp = binary_sampler(hint_rate, batch_size, dim)
        h_mb = m_mb * h_mb_temp

        # combine random noise with the observed x values
        x_mb = m_mb * x_mb + (1 - m_mb) * z_mb

        loss_d_, *_ = sess.run([D_loss, D_optim], {x: x_mb, h: h_mb, m: m_mb})
        D_losses.append(loss_d_)

        # update generator
        loss_g_, *_ = sess.run([G_loss, G_optim, MSE_loss], {x: x_mb, m:m_mb, h:h_mb})
        G_losses.append(loss_g_)

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time

        train_hist['D_losses'].append(np.mean(D_losses))
        train_hist['G_losses'].append(np.mean(G_losses))
        train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

    end_time = time.time()
    total_ptime = end_time - start_time
    train_hist['total_ptime'].append(total_ptime)

    # return imputed data
    z_mb = uniform_sampler(0, 0.01, no, dim)
    m_mb = data_m
    x_mb = norm_data_x
    x_mb = m_mb * x_mb + (1 - m_mb) * z_mb

    imputed_data = sess.run([G_sample], {x: x_mb, m:m_mb})[0]
    # imputed_data = data_m * norm_data_x + (1 - data_m) * imputed_data
    imputed_data = m_mb * norm_data_x + (1 - m_mb) * imputed_data

    # renormalization
    imputed_data = renormalization(imputed_data, norm_parameters)

    # rounding
    imputed_data = rounding(imputed_data, data_x)

    return imputed_data