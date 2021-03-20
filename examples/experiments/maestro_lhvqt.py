# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from amt_models.models import OnsetsFrames2
from amt_models.datasets import MAESTRO_V1, MAPS
from amt_models.features import LHVQT

from amt_models import train, validate
from amt_models.transcribe import *
from amt_models.evaluate import *

import amt_models.tools as tools

import lhvqt as l

# Regular imports
from sacred.observers import FileStorageObserver
from torch.utils.data import DataLoader
from sacred import Experiment

import torch
import os

EX_NAME = '_'.join([OnsetsFrames2.model_name(),
                    MAESTRO_V1.dataset_name(),
                    LHVQT.features_name(), 'rand_10'])

ex = Experiment('Onsets & Frames 2 w/ LHVQT on MAESTRO')


@ex.config
def config():
    # Number of samples per second of audio
    sample_rate = 16000

    # Number of samples between frames
    hop_length = 512

    # Number of consecutive frames within each example fed to the model
    num_frames = 300

    # Number of training iterations to conduct
    iterations = 1000

    # How many equally spaced save/validation checkpoints - 0 to disable
    checkpoints = 50

    # Number of samples to gather for a batch
    batch_size = 8

    # The initial learning rate
    learning_rate = 5e-4

    # The id of the gpu to use, if available
    gpu_id = 0

    # Flag to re-acquire ground-truth data and re-calculate-features
    # This is useful if testing out different feature extraction parameters
    reset_data = False

    # The random seed for this experiment
    seed = 0

    # Create the root directory for the experiment to hold train/transcribe/evaluate materials
    expr_cache = os.path.join(tools.constants.HOME, 'Desktop', 'LHVQT', 'generated', 'experiments')
    root_dir = os.path.join(expr_cache, EX_NAME)
    os.makedirs(root_dir, exist_ok=True)

    # Add a file storage observer for the log directory
    ex.observers.append(FileStorageObserver(root_dir))


def visualize(model, i=None):
    """
    Run this code during each validation checkpoint to visualize the evolution of the filterbank.

    Parameters
    ----------
    model : OnsetsFramesLHVQT
      Model with a filterbank to visualize
    i : int
      Current iteration for directory organization
    """

    # Construct a path to the save directory
    save_dir = os.path.join('..', '..', 'generated', 'visualization', EX_NAME)

    if i is not None:
        # Add an additional folder for the checkpoint
        save_dir = os.path.join(save_dir, f'checkpoint-{i}')

    # Visualize the filterbank
    model.frontend.fb.visualize(save_dir,
                                fix_scale=True,
                                include_axis=True,
                                n_fft=2**18,
                                decibels=True,
                                include_negative=True,
                                sort_by_centroid=True)


class OnsetsFrames2LHVQT(OnsetsFrames2):
    """
    Implements the Onsets & Frames model (V2) with a learnable filterbank frontend.
    """

    def __init__(self, dim_in, profile, in_channels, lhvqt, model_complexity=2, detach_heads=True, device='cpu'):
        """
        Initialize the model and establish parameter defaults in function signature.

        Parameters
        ----------
        See OnsetsFrames2 class for others...

        lhvqt : LHVQT
          Filterbank to use as frontend
        """

        super().__init__(dim_in, profile, in_channels, model_complexity, detach_heads, device)

        # Append the filterbank learning module to the front of the model
        self.frontend.add_module('fb', lhvqt.lhvqt)
        self.frontend.add_module('rl', torch.nn.ReLU())

    def post_proc(self, batch):
        """
        Calculate KL-divergence for the 1D convolutional layer in the filterbank and append to the tracked loss.

        Parameters
        ----------
        batch : dict
          Dictionary including model output and potentially
          ground-truth for a group of tracks

        Returns
        ----------
        output : dict
          Dictionary containing multi pitch, onsets, and offsets output as well as loss
        """

        # Perform standard Onsets & Frames 2 steps
        output = super().post_proc(batch)

        # Check to see if loss is being tracked
        if tools.KEY_LOSS in output.keys():
            # Obtain a pointer to the filterbank module
            fb_module = self.frontend.fb.get_modules()[0]

            # Extract all of the losses
            loss = output[tools.KEY_LOSS]

            # Calculate the KL-divergence term
            loss[tools.KEY_LOSS_KLD] = fb_module.time_conv.kld()

            # Compute the total loss and add it back to the output dictionary
            #loss[tools.KEY_LOSS_TOTAL] += (max(min(self.iter - 250, 250), 0) / 500) * loss[tools.KEY_LOSS_KLD]
            loss[tools.KEY_LOSS_TOTAL] += loss[tools.KEY_LOSS_KLD]
            output[tools.KEY_LOSS] = loss

            #if self.iter == 250:
                # Turn on training mode for filterbank
            #    fb_module.update = True

        return output


@ex.automain
def onsets_frames_run(sample_rate, hop_length, num_frames, iterations, checkpoints,
                      batch_size, learning_rate, gpu_id, reset_data, seed, root_dir):
    # Seed everything with the same seed
    tools.seed_everything(seed)

    # Initialize the default piano profile
    profile = tools.PianoProfile()

    # Processing parameters
    dim_in = 60 * 8
    dim_out = profile.get_range_len()
    model_complexity = 3

    # Initialize learnable filterbank data processing module
    data_proc = LHVQT(sample_rate=sample_rate,
                      hop_length=hop_length,
                      lhvqt=l.lhvqt.LHVQT,
                      lvqt=l.lvqt_hilb.LVQT,
                      fmin=None,
                      harmonics=[1],
                      n_bins=dim_in,
                      bins_per_octave=60,
                      gamma=None,
                      max_p=1,
                      random=True,
                      update=True,
                      batch_norm=True,
                      var_drop=-10)

    # Initialize the estimation pipeline
    validation_estimator = ComboEstimator([NoteTranscriber(profile=profile),
                                           PitchListWrapper(profile=profile)])

    # Initialize the evaluation pipeline
    evaluators = {tools.KEY_LOSS : LossWrapper(),
                  tools.KEY_MULTIPITCH : MultipitchEvaluator(),
                  tools.KEY_NOTE_ON : NoteEvaluator(),
                  tools.KEY_NOTE_OFF : NoteEvaluator(0.2)}
    validation_evaluator = ComboEvaluator(evaluators, patterns=['loss', 'f1'])

    # Construct the MAESTRO splits
    train_split = ['train']
    val_split = ['validation']
    test_split = ['test']

    print('Loading training partition...')

    # Create a dataset corresponding to the training partition
    mstro_train = MAESTRO_V1(splits=train_split,
                             hop_length=hop_length,
                             sample_rate=sample_rate,
                             data_proc=data_proc,
                             profile=profile,
                             num_frames=num_frames,
                             reset_data=reset_data,
                             store_data=False)

    # Create a PyTorch data loader for the dataset
    train_loader = DataLoader(dataset=mstro_train,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0,
                              drop_last=True)

    print('Loading validation partition...')

    # Create a dataset corresponding to the validation partition
    mstro_val = MAESTRO_V1(splits=val_split,
                           hop_length=hop_length,
                           sample_rate=sample_rate,
                           data_proc=data_proc,
                           profile=profile,
                           num_frames=num_frames,
                           store_data=False)

    print('Loading testing partitions...')

    # Create a dataset corresponding to the MAESTRO testing partition
    mstro_test = MAESTRO_V1(splits=test_split,
                            hop_length=hop_length,
                            sample_rate=sample_rate,
                            data_proc=data_proc,
                            profile=profile,
                            store_data=False)

    # Initialize the MAPS testing splits as the real piano data
    test_splits = ['ENSTDkAm', 'ENSTDkCl']

    # Create a dataset corresponding to the MAPS testing partition
    # Need to reset due to HTK Mel-Spectrogram spacing
    maps_test = MAPS(splits=test_splits,
                     hop_length=hop_length,
                     sample_rate=sample_rate,
                     data_proc=data_proc,
                     profile=profile,
                     store_data=False,
                     reset_data=True)

    print('Initializing model...')

    # Initialize a new instance of the model
    onsetsframes = OnsetsFrames2LHVQT(dim_in, profile, data_proc.get_num_channels(), data_proc, model_complexity, True, gpu_id)
    onsetsframes.change_device()
    onsetsframes.train()

    # Initialize a new optimizer for the model parameters
    optimizer = torch.optim.Adam(onsetsframes.parameters(), learning_rate)

    print('Training classifier...')

    # Create a log directory for the training experiment
    model_dir = os.path.join(root_dir, 'models')

    # Visualize the filterbank before conducting any training
    visualize(onsetsframes, 0)

    # Train the model
    onsetsframes = train(model=onsetsframes,
                         train_loader=train_loader,
                         optimizer=optimizer,
                         iterations=iterations,
                         checkpoints=checkpoints,
                         log_dir=model_dir,
                         val_set=mstro_val,
                         estimator=validation_estimator,
                         evaluator=validation_evaluator,
                         vis_fnc=visualize)

    print('Transcribing and evaluating test partition...')

    # Add save directories to the estimators
    validation_estimator.set_save_dirs(os.path.join(root_dir, 'estimated', 'MAESTRO'), ['notes', 'pitch'])

    # Add a save directory to the evaluators and reset the patterns
    validation_evaluator.set_save_dir(os.path.join(root_dir, 'results', 'MAESTRO'))

    # Add a save directory to the evaluators and reset the patterns
    validation_evaluator.set_save_dir(os.path.join(root_dir, 'results'))
    validation_evaluator.set_patterns(None)

    # Get the average results for the MAESTRO testing partition
    results = validate(onsetsframes, mstro_test, evaluator=validation_evaluator, estimator=validation_estimator)

    # Log the average results in metrics.json
    ex.log_scalar('MAESTRO Results', results, 0)

    # Reset the evaluator
    validation_evaluator.reset_results()

    # Update save directories for the estimators
    validation_estimator.set_save_dirs(os.path.join(root_dir, 'estimated', 'MAPS'), ['notes', 'pitch'])

    # Update save directory for the evaluators
    validation_evaluator.set_save_dir(os.path.join(root_dir, 'results', 'MAPS'))

    # Get the average results for the MAPS testing partition
    results = validate(onsetsframes, maps_test, evaluator=validation_evaluator, estimator=validation_estimator)

    # Log the average results in metrics.json
    ex.log_scalar('MAPS Results', results, 0)
