import tensorflow as tf
import tensorflow.keras.losses as losses
import tensorflow.keras.backend as tfK

def dice_coeff(y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f) #=truepositives
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth) #truepos+falseneg+truepos+false positives
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def bce_dice_loss(y_true, y_pred):
    loss = losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss

def bce(y_true,y_pred):
    loss = losses.binary_crossentropy(y_true, y_pred)
    return loss

def f1_score(y_true, y_pred):
    '''
    TP = tfK.sum(tfK.round(tfK.clip(y_true * y_pred, 0, 1)))
    TPTN = tfK.sum(tfK.round(tfK.clip(y_pred, 0, 1)))
    TPFN = tfK.sum(tfK.round(tfK.clip(y_true, 0, 1)))
    '''

    TP = tfK.sum(y_true * y_pred)
    TPFP = tfK.sum(y_pred)
    TPFN = tfK.sum(y_true)
    # F1=0 when there is no true sample
    if TPFN == 0:
        return 0

    # truepositives/truepositives+falsepositives
    precision = TP / TPFP

    # truepositives/truepositives+falsenegatives
    recall = TP / TPFN

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

def origf1_score(y_true, y_pred):

    TP = tfK.sum(tfK.round(tfK.clip(y_true * y_pred, 0, 1)))
    TPTN = tfK.sum(tfK.round(tfK.clip(y_pred, 0, 1)))
    TPFN = tfK.sum(tfK.round(tfK.clip(y_true, 0, 1)))

    # F1=0 when there is no true sample
    if TPFN == 0:
        return 0

    # truepositives/truepositives+falsepositives
    precision = TP / TPFP

    # truepositives/truepositives+falsenegatives
    recall = TP / TPFN

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


def f1_loss(y_true,y_pred):
    return 1-f1_score(y_true,y_pred)

def f1_score_noclip(y_true, y_pred):

    TP = tfK.sum(tfK.round(y_true * y_pred, 0, 1))
    TPTN = tfK.sum(tfK.round(y_pred, 0, 1))
    TPFN = tfK.sum(tfK.round(y_true, 0, 1))

    # F1=0 when there is no true sample
    if TPFN == 0:
        return 0

    # truepositives/truepositives+falsepositives
    precision = TP / TPFP

    # truepositives/truepositives+falsenegatives
    recall = TP / TPFN

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

