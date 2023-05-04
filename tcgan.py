import tensorflow as tf
import numpy as np

def tcgan (odata, parameters):
  tf.compat.v1.reset_default_graph()
  no, seq_len, dim = np.asarray(odata).shape
  otime, max_seq_len = etime(odata)
  
  def MinMaxScaler(data):
    min_val = np.min(np.min(data, axis = 0), axis = 0)
    data = data - min_val
    max_val = np.max(np.max(data, axis = 0), axis = 0)
    norm_data = data / (max_val + 1e-7)
    return norm_data, min_val, max_val
  
  ori_data, min_val, max_val = MinMaxScaler(odata)
  hidden_dim = parameters['hidden_dim']
  num_layers = parameters['num_layer']
  iterations = parameters['iterations']
  batch_size = parameters['batch_size']
  module_name = parameters['module']
  z_dim = dim
  gamma = 1
    
  tf.compat.v1.disable_eager_execution()
  X = tf.compat.v1.placeholder(tf.float32, [None, max_seq_len, dim], name = "inputx")
  Z = tf.compat.v1.placeholder(tf.float32, [None, max_seq_len, z_dim], name = "inputz")
  T = tf.compat.v1.placeholder(tf.int32, [None], name = "inputt")
  
  def transverter (X, T):
    with tf.compat.v1.variable_scope("transverter", reuse = tf.compat.v1.AUTO_REUSE):
      e_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
      e_outputs, e_last_states = tf.compat.v1.nn.dynamic_rnn(e_cell, X, dtype=tf.float32, sequence_length = T)
      H = tf.contrib.layers.fully_connected(e_outputs, hidden_dim, activation_fn=tf.nn.sigmoid)
    return H
      
  def restorer (H, T):
    with tf.variable_scope("restorer", reuse = tf.AUTO_REUSE):
      r_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
      r_outputs, r_last_states = tf.nn.dynamic_rnn(r_cell, H, dtype=tf.float32, sequence_length = T)
      Y = tf.contrib.layers.fully_connected(r_outputs, dim, activation_fn=tf.nn.sigmoid)
    return Y
    
  def generator (Z, T):  
    with tf.variable_scope("generator", reuse = tf.AUTO_REUSE):
      e_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
      e_outputs, e_last_states = tf.nn.dynamic_rnn(e_cell, Z, dtype=tf.float32, sequence_length = T)
      E = tf.contrib.layers.fully_connected(e_outputs, hidden_dim, activation_fn=tf.nn.sigmoid)     
    return E

  def discrimination (H, T):
    with tf.variable_scope("discrimination", reuse = tf.AUTO_REUSE):
      e_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers-1)])
      e_outputs, e_last_states = tf.nn.dynamic_rnn(e_cell, H, dtype=tf.float32, sequence_length = T)
      S = tf.contrib.layers.fully_connected(e_outputs, hidden_dim, activation_fn=tf.nn.sigmoid)
    return S
  H = transverter(X, T)
  X_tilde = restorer(H, T)
  E_hat = generator(Z, T)
  H_hat = discrimination(E_hat, T)
  X_hat = restorer(H_hat, T)
  Y_fake = transverter(H_hat, T)
  Y_real = transverter(H, T)
  Y_fake_e = transverter(E_hat, T)
  t_vars = [v for v in tf.trainable_variables() if v.name.startswith('transverter')]
  r_vars = [v for v in tf.trainable_variables() if v.name.startswith('restorer')]
  g_vars = [v for v in tf.trainable_variables() if v.name.startswith('generator')]
  d_vars = [v for v in tf.trainable_variables() if v.name.startswith('discrimination')]
  T_loss_real = tf.losses.sigmoid_cross_entropy(tf.ones_like(Y_real), Y_real)
  T_loss_fake = tf.losses.sigmoid_cross_entropy(tf.zeros_like(Y_fake), Y_fake)
  T_loss_fake_e = tf.losses.sigmoid_cross_entropy(tf.zeros_like(Y_fake_e), Y_fake_e)
  T_loss = T_loss_real + T_loss_fake + gamma * T_loss_fake_e
  G_loss_U = tf.losses.sigmoid_cross_entropy(tf.ones_like(Y_fake), Y_fake)
  G_loss_U_e = tf.losses.sigmoid_cross_entropy(tf.ones_like(Y_fake_e), Y_fake_e)
  G_loss_S = tf.losses.mean_squared_error(H[:,1:,:], H_hat_supervise[:,:-1,:])
  G_loss_V1 = tf.reduce_mean(tf.abs(tf.sqrt(tf.nn.moments(X_hat,[0])[1] + 1e-6) - tf.sqrt(tf.nn.moments(X,[0])[1] + 1e-6)))
  G_loss_V2 = tf.reduce_mean(tf.abs((tf.nn.moments(X_hat,[0])[0]) - (tf.nn.moments(X,[0])[0])))
  G_loss_V = G_loss_V1 + G_loss_V2
  G_loss = G_loss_U + gamma * G_loss_U_e + 100 * tf.sqrt(G_loss_S) + 100*G_loss_V
  E_loss_T0 = tf.losses.mean_squared_error(X, X_tilde)
  E_loss0 = 10*tf.sqrt(E_loss_T0)
  E_loss = E_loss0  + 0.1*G_loss_S

  gdata = gdata * max_val
  gdata = gdata + min_val
    
  return gdata