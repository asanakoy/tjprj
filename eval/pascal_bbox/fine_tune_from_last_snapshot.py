import os.path
import tensorflow as tf
from tfext import network_spec
import tfext.alexnet
import tfext.stlnet
import tfext.utils
import eval.stl.eval_aux as eval_aux



def setup_network(**params):

    with tf.Graph().as_default():
        net = tfext.stlnet.Stlnet(random_init_type=tfext.stlnet.Stlnet.RandomInitType.XAVIER_GAUSSIAN, num_classes=500)
        # net.restore_from_snapshot(snapshot_path=params['snapshot_path'], num_layers=12)

        with tf.device(params['device_id']):
            # Loss for metric learning
            logits = net.fc_stl10
            loss = network_spec.loss(logits, net.y_gt)

            # train_op = network_spec.training_stl_eval(net, loss, base_lr=params['base_lr'])
            with tf.variable_scope('lr'):
                conv_lr_pl = tf.placeholder(tf.float32, tuple(), name='conv_lr')
            train_op = network_spec.training_convnet(net, loss, fc_lr=params['base_lr'], conv_lr=conv_lr_pl)


            # Add the Op to compare the logits to the labels during correct_classified_top1.
            eval_correct_top1 = network_spec.correct_classified_top1(logits, net.y_gt)
            accuracy = tf.cast(eval_correct_top1, tf.float32) / \
                       tf.constant(params['batch_size'], dtype=tf.float32)

        saver = tf.train.Saver()

        # Instantiate a SummaryWriter to output summaries and the Graph of the current sesion.
        summary_writer = tf.summary.FileWriter(params['output_dir'], net.sess.graph)
        summary = tf.summary.scalar(['loss', 'batch_accuracy'], [loss, accuracy])

        net.sess.run(tf.global_variables_initializer())
    return {'net': net, 'train_op': train_op, 'loss': loss, 'saver': saver, 'summary_writer': summary_writer,
            'summary': summary}


def run_training(**params):

    params_net = setup_network(**params)
    params.update(params_net)
    print("Starting training...")
    evaluator = eval_aux.supervised_evaluation(params['net'])

    checkpoint_file = os.path.join(params['output_dir'], 'checkpoint_before_evaluation')
    params['saver'].save(params['net'].sess, checkpoint_file, global_step=0)

    # Update the network of the evaluator and train last fc with supervision
    evaluator.update_net(params['net'])
    evaluator.train(n_iters_training=20000, loss=params['loss'], train_op=params['train_op'], output_dir=params['output_dir'],
                    saver=params['saver'], summary=params['summary'], summary_writter=params['summary_writer'], warmup_iters=params['warmup_iters'], conv_lr=params['base_lr'])

    # Compute accuracy and flush to tensorboard
    acc = evaluator.test()
    print acc


def main():
    dataset = 'STL10'
    category = 'STL10'
    output_dir = '/export/home/mbautist/tmp/tf_test/STL10_supervised_from_scratch'
    params = {
        'im_shape': (96, 96, 3),
        'batch_size': 128,
        'base_lr': 0.01,
        'warmup_iters': 1000,
        'fc_lr_mult': 1.0,
        'conv_lr_mult': 1.0,
        'num_layers_to_init': 7,
        'dataset': dataset,
        'category': category,
        'num_classes': None,
        'snapshot_iter': 2000,
        'max_iter': 2000,
        'indexing_1_based': 0,
        'indexfile_path': None,
        'seed': 1988,
        'test_step': 2000,
        'output_dir': output_dir,
        'init_model': None,
        'device_id': '/gpu:{}'.format(0),
        'snapshot_path': '/export/home/mbautist/tmp/tf_test/STL10/checkpoint_before_evaluation-600000',
        'gpu_memory_fraction': 0.4,
        'shuffle_every_epoch': False,
        'online_augmentations': False,
        'async_preload': False,
        'num_data_workers': 5,
        'batch_ldr': None,
        'augmenter_params': dict(hflip=True, vflip=False,
                                 scale_to_percent=(0.9, 1.1),
                                 scale_axis_equally=True,
                                 rotation_deg=10, shear_deg=7,
                                 translation_x_px=30, translation_y_px=30)
    }
    run_training(**params)

main()