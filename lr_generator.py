import math
import numpy as np


def get_lr(global_step, lr_init, lr_end, lr_max, warmup_epochs, total_epochs, steps_per_epoch, lr_decay_mode):
    """
    generate learning rate array

    Args:
       global_step(int): total steps of the training
       lr_init(float): init learning rate
       lr_end(float): end learning rate
       lr_max(float): max learning rate
       warmup_epochs(int): number of warmup epochs
       total_epochs(int): total epoch of training
       steps_per_epoch(int): steps of one epoch
       lr_decay_mode(string): learning rate decay mode, including steps, poly, cosine or default

    Returns:
       np.array, learning rate array
    """
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    warmup_steps = steps_per_epoch * warmup_epochs
    if lr_decay_mode == 'steps':
        decay_epoch_index = [0.3 * total_steps, 0.6 * total_steps, 0.8 * total_steps]
        for i in range(total_steps):
            if i < decay_epoch_index[0]:
                lr = lr_max
            elif i < decay_epoch_index[1]:
                lr = lr_max * 0.1
            elif i < decay_epoch_index[2]:
                lr = lr_max * 0.01
            else:
                lr = lr_max * 0.001
            lr_each_step.append(lr)
    elif lr_decay_mode == 'poly':
        if warmup_steps != 0:
            inc_each_step = (float(lr_max) - float(lr_init)) / float(warmup_steps)
        else:
            inc_each_step = 0
        for i in range(total_steps):
            if i < warmup_steps:
                lr = float(lr_init) + inc_each_step * float(i)
            else:
                base = (1.0 - (float(i) - float(warmup_steps)) / (float(total_steps) - float(warmup_steps)))
                lr = float(lr_max) * base * base
                if lr < 0.0:
                    lr = 0.0
            lr_each_step.append(lr)
    elif lr_decay_mode == 'cosine':
        decay_steps = total_steps - warmup_steps
        for i in range(total_steps):
            if i < warmup_steps:
                lr_inc = (float(lr_max) - float(lr_init)) / float(warmup_steps)
                lr = float(lr_init) + lr_inc * (i + 1)
            else:
                linear_decay = (total_steps - i) / decay_steps
                cosine_decay = 0.5 * (1 + math.cos(math.pi * 2 * 0.47 * i / decay_steps))
                decayed = linear_decay * cosine_decay + 0.00001
                lr = lr_max * decayed
            lr_each_step.append(lr)
    else:
        for i in range(total_steps):
            if i < warmup_steps:
                lr = lr_init + (lr_max - lr_init) * i / warmup_steps
            else:
                lr = lr_max - (lr_max - lr_end) * (i - warmup_steps) / (total_steps - warmup_steps)
            lr_each_step.append(lr)

    current_step = global_step
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    learning_rate = lr_each_step[current_step:]

    return learning_rate
