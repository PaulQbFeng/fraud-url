import model
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--BUCKET',
        help='GCS path to data. We assume that data is in gs://BUCKET/babyweight/preproc/',
        required=False
    )
    parser.add_argument(
        '--NUM_EPOCH',
        help='Number of examples to compute gradient over.',
        type=int,
        default=20
    )
    parser.add_argument(
        '--BATCH_SIZE',
        help='Number of examples to compute gradient over.',
        type=int,
        default=256
    )
    parser.add_argument(
        '--SAVE_CKPT_STEPS',
        help='Number of set to commpute before check points',
        type=int,
        default=200
    )
    parser.add_argument(
        '--EVAL_INTERVAL_SEC',
        help='Positive number of steps for which to evaluate model',
        type=int,
        default=5
    )
    parser.add_argument(
        '--HIDDEN_UNITS',
        help='Hidden layer sizes to use for DNN feature columns -- provide space-separated layers',
        nargs='+',
        type=int,
        default=[256, 128, 64, 32]
    )
    parser.add_argument(
        '--MAX_STEPS',
        help='',
        type=int,
        default=12000
    )
    parser.add_argument(
        '--LEARNING_RATE_LINEAR',
        help='Linear model learning rate',
        type=float,
        default=0.008
    )
    parser.add_argument(
        '--LEARNING_RATE_DNN',
        help='dnn model learning rate',
        type=float,
        default=0.0008
    )
    parser.add_argument(
        '--model',
        help='GCS location to write checkpoints and export models',
        required=True
    )
    parser.add_argument(
        '--outdir',
        help='GCS location to write checkpoints and export models',
        required=True
    )

    # parse all arguments
    args = parser.parse_args()
    arguments = args.__dict__

    # unused args provided by service
    arguments.pop('job_dir', None)
    arguments.pop('job-dir', None)

    # assign the arguments to the model variables
    model.outdir = arguments.pop('outdir')
    model.BUCKET = arguments.pop('BUCKET')
    model.NUM_EPOCH = arguments.pop('NUM_EPOCH')
    model.BATCH_SIZE = arguments.pop('BATCH_SIZE')
    model.SAVE_CKPT_STEPS = arguments.pop('SAVE_CKPT_STEPS')
    model.EVAL_INTERVAL_SEC = arguments.pop('EVAL_INTERVAL_SEC')
    model.HIDDEN_UNITS = arguments.pop('HIDDEN_UNITS')
    model.MAX_STEPS = arguments.pop('MAX_STEPS')
    model.LEARNING_RATE_LINEAR = arguments.pop('LEARNING_RATE_LINEAR')
    model.LEARNING_RATE_DNN = arguments.pop('LEARNING_RATE_DNN')
    model.model = arguments.pop('model')
    # model.TRAIN_STEPS = (arguments.pop('train_examples') * 1000) / model.BATCH_SIZE
    # model.EVAL_STEPS = arguments.pop('eval_steps')
    print("Will train for {} steps using batch_size={}".format(model.MAX_STEPS, model.BATCH_SIZE))

    if model.model == "linear":
        print("Will use linear model")
    else:
        print("Will use DNN size of {}".format(model.HIDDEN_UNITS))

    # Append trial_id to path if we are doing hptuning

    # Run the training job
    model.train_and_evaluate(model.outdir, model.model)
