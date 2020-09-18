import torch


def take_log(tfr, amin=(1e-5)**2, top_db=80.0, scale=False):
    """
    Take the log of a time-frequency representation using differentiable
    PyTorch functions, scaling values to be between 0 and 1. Adapted from
    the Librosa amplitude_to_db function.

    Expected input: batch x harmonic x frequency x time
                                  OR
                    batch x frequency x time
    """

    if len(tfr.size()) != 4 and len(tfr.size()) != 3:
        raise Exception('Expected input - B x H x F x T OR B x F x T')

    batch_size = tfr.size(0)

    if len(tfr.size()) == 4:
        # View input as B x F x T
        h_size = tfr.size(1)
        tfr = tfr.contiguous().view(batch_size * h_size, tfr.size(2), tfr.size(3))
    else:
        h_size = None

    # Convert amplitude to power spectrogram
    magnitude = torch.abs(tfr)
    power = magnitude * magnitude

    num_tfrs = tfr.size(0)
    # Get reference values (maximums) from power spectrogram for each tfr
    ref_value = torch.max(magnitude.view(num_tfrs, -1), dim = -1)[0] ** 2

    # Clamp power spectrogram at specified minimum - effectively max(amin, power)
    power[power < amin] = amin

    # Convert to dB
    log_tfr = 10.0 * torch.log10(power)

    # Make sure reference values are above minimum amplitude - effectively max(amin, ref_value)
    amin = torch.Tensor([amin] * num_tfrs).to(power.device)
    amin[amin < ref_value] = ref_value[amin < ref_value]
    maxes = amin.unsqueeze(-1).unsqueeze(-1).to(power.device) # Add dimensions to broadcast over tfr

    # Combined with previous log, we are performing 10 * log10(power / ref)
    log_tfr = log_tfr - 10.0 * torch.log10(maxes)

    # Clamp the dB values at the specified top - effectively max(log_tfr, log_tfr.max() - top_db)
    log_tfr = log_tfr.view(num_tfrs, -1) # Combine F and T dimension temporarily
    newVals = (torch.max(log_tfr, dim = -1)[0] - top_db).unsqueeze(-1).to(power.device)
    log_tfr[log_tfr < newVals] = newVals.repeat(1, log_tfr.size(-1))[log_tfr < newVals]

    # Scale values and offset to be between 0 and 1
    if scale:
        log_tfr = (log_tfr / top_db) + 1.0
    tfr = log_tfr.view(magnitude.size())

    if h_size is not None:
        # Match input dimensions to original HCQT
        tfr = tfr.contiguous().view(batch_size, h_size, tfr.size(1), tfr.size(2))

    return tfr
