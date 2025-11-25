import copy
import numpy as np
import torch
from torch import distributions
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F


def cal_loss_and_accuracy(outputs, labels, device):
    """
    Calculate loss and accuracy with Cross-Entropy Loss.
    """
    logits = outputs.logits
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous().to(device)

    # Flatten the tokens
    # need to ignore mask:i, mask 0-5 id: 830
    # shift_labels
    # covert_maskid_to_padid
    shift_labels = shift_labels.masked_fill((shift_labels < 630), 0)

    loss_fct = CrossEntropyLoss(ignore_index=0, reduction='sum')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    _, preds = shift_logits.max(dim=-1)
    not_ignore = shift_labels.ne(0)
    num_targets = not_ignore.long().sum().item()

    correct = (shift_labels == preds) & not_ignore
    correct = correct.float().sum()

    accuracy = correct / num_targets
    loss = loss / num_targets

    return loss, accuracy


def get_all_normal_dis_pdf(voc_len=836, confs_num=629):
    """
    Calculate the probability density function

    Args:
        voc_len (int, optional): Length of vocabulary. Defaults to 836.
        confs_num (int, optional): Number of numerical tokens. Defaults to 629.
    """
    means = torch.arange(1, confs_num + 1)  # create x of normal distribution for conf num
    std_dev = 2.0
    normal_dist_list = [distributions.Normal(mean.float(), std_dev) for mean in means]

    # log PDF
    pdf_list = []
    zero_pdf = torch.zeros(voc_len)
    zero_pdf[0] = 1
    pdf_list.append(zero_pdf)
    for idx, normal_dist in enumerate(normal_dist_list):
        # if not confs num, make it as 0
        pdf = torch.zeros(voc_len)
        pdf[1:confs_num + 1] = normal_dist.log_prob(means.float()).exp().float()  # calculate PDF
        # rate of ground truth 50% default is set to 4
        pdf[idx + 1] = pdf[idx + 1] * 2
        # normalized pdf
        normalized_pdf = pdf / pdf.sum()
        pdf_list.append(normalized_pdf)

    return torch.stack(pdf_list)


def gce_loss_and_accuracy(outputs, labels, device, pdf_array=get_all_normal_dis_pdf()):
    """
    Calculate loss and accuracy with GCE loss.
    """
    logits = outputs.logits
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous().to(device)

    # Flatten the tokens
    # need to ignore mask:i, mask 0-5 id: 830 smiles
    # shift_labels
    # covert_maskid_to_padid
    shift_labels_copy = copy.deepcopy(shift_labels)
    shift_labels_copy = shift_labels_copy.masked_fill((shift_labels_copy != 829), 0)

    # only calculate GCE to numerical tokens
    shift_labels = shift_labels.masked_fill((shift_labels >= 630), 0)
    shift_labels_copy_copy = copy.deepcopy(shift_labels)
    shift_labels = shift_labels + shift_labels_copy
    one_hot = F.one_hot(shift_labels, num_classes=836).float()  # One-hot encoding to labels
    non_zero_indices = torch.nonzero(shift_labels_copy_copy)

    pdf_array = pdf_array.to(shift_labels.device).contiguous()
    rows, li_indices = non_zero_indices[:, 0], non_zero_indices[:, 1]
    poisson_one_hot = pdf_array[shift_labels[rows, li_indices]]
    one_hot[rows, li_indices] = poisson_one_hot

    # custom cross entropy loss
    logsoftmax = F.log_softmax(shift_logits, dim=-1)

    # mask 0
    not_ignore = shift_labels.ne(0)
    one_hot = not_ignore.unsqueeze(-1) * one_hot

    # / shift_labels.shape[0]
    loss = -torch.sum(one_hot * logsoftmax)

    _, preds = shift_logits.max(dim=-1)
    num_targets = not_ignore.long().sum().item()
    correct = (shift_labels == preds) & not_ignore
    correct = correct.float().sum()
    accuracy = correct / num_targets
    loss = loss / num_targets

    return loss, accuracy

def Variable(tensor):
    """Wrapper for torch.autograd.Variable that also accepts
       numpy arrays directly and automatically assigns it to
       the GPU. Be aware in case some operations are better
       left to the CPU."""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if torch.cuda.is_available():
        return torch.autograd.Variable(tensor).to(device)
    return torch.autograd.Variable(tensor)

def decrease_learning_rate(optimizer, decrease_by=0.01):
    """Multiplies the learning rate of the optimizer by 1 - decrease_by"""
    for param_group in optimizer.param_groups:
        param_group['lr'] *= (1 - decrease_by)

def seq_to_smiles(seqs, voc):
    """Takes an output sequence from the RNN and returns the
       corresponding SMILES."""
    smiles = []
    for seq in seqs.cpu().numpy():
        smiles.append(voc.decode(seq))
    return smiles

def fraction_valid_smiles(smiles):
    """Takes a list of SMILES and returns fraction valid."""
    i = 0
    for smile in smiles:
        if Chem.MolFromSmiles(smile):
            i += 1
    return i / len(smiles)

def unique(arr):
    # Finds unique rows in arr and return their indices
    arr = arr.cpu().numpy()
    arr_ = np.ascontiguousarray(arr).view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1])))
    _, idxs = np.unique(arr_, return_index=True)
    if torch.cuda.is_available():
        return torch.LongTensor(np.sort(idxs)).cuda()
    return torch.LongTensor(np.sort(idxs))
