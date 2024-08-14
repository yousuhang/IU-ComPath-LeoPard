from main_infererence_pipeline import save_dict_as_json
from resources.utils.utils import *
import os
from resources.dataset_modules import save_splits
from resources.models import MIL_fc, MIL_fc_mc, MIL_fc_reg, MIL_fc_reg_att, MIL_fc_reg_top_k_att
from resources.models import CLAM_MB, CLAM_SB
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _neg_partial_log(prediction, T, E):
    """
    calculate cox loss, Pytorch implementation by Huang, https://github.com/huangzhii/SALMON
    :param X: variables
    :param T: Time
    :param E: Status
    :return: neg log of the likelihood
    """

    current_batch_len = len(prediction)
    # print(current_batch_len)
    R_matrix_train = np.zeros([current_batch_len, current_batch_len], dtype=int)
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_matrix_train[i, j] = T[j] >= T[i]

    train_R = torch.FloatTensor(R_matrix_train)
    train_R = train_R.cuda()
    train_ystatus = E.type(torch.FloatTensor).cuda()

    theta = prediction.reshape(-1)
    exp_theta = torch.exp(theta)

    loss_nn = - torch.mean((theta - torch.log(torch.sum(exp_theta * train_R, dim=1))) * train_ystatus)

    return loss_nn

class _SVMLoss(nn.Module):

    def __init__(self, n_classes, alpha):

        assert isinstance(n_classes, int)

        assert n_classes > 0
        assert alpha is None or alpha >= 0

        super(_SVMLoss, self).__init__()
        self.alpha = alpha if alpha is not None else 1
        self.register_buffer('labels', torch.from_numpy(np.arange(n_classes)))
        self.n_classes = n_classes
        self._tau = None

    def forward(self, x, y):

        raise NotImplementedError("Forward needs to be re-implemented for each loss")

    @property
    def tau(self):
        return self._tau

    @tau.setter
    def tau(self, tau):
        if self._tau != tau:
            print("Setting tau to {}".format(tau))
            self._tau = float(tau)
            self.get_losses()

    def cuda(self, device=None):
        nn.Module.cuda(self, device)
        self.get_losses()
        return self

    def cpu(self):
        nn.Module.cpu()
        self.get_losses()
        return self

class SmoothTop1SVM(_SVMLoss):
    def __init__(self, n_classes, alpha=None, tau=1.):
        super(SmoothTop1SVM, self).__init__(n_classes=n_classes,
                                            alpha=alpha)
        self.tau = tau
        self.thresh = 1e3
        self.get_losses()

    def forward(self, x, y):
        smooth, hard = detect_large(x, 1, self.tau, self.thresh)

        loss = 0
        if smooth.data.sum():
            x_s, y_s = x[smooth], y[smooth]
            x_s = x_s.view(-1, x.size(1))
            loss += self.F_s(x_s, y_s).sum() / x.size(0)
        if hard.data.sum():
            x_h, y_h = x[hard], y[hard]
            x_h = x_h.view(-1, x.size(1))
            loss += self.F_h(x_h, y_h).sum() / x.size(0)

        return loss

    def get_losses(self):
        self.F_h = F.Top1_Hard_SVM(self.labels, self.alpha)
        self.F_s = F.Top1_Smooth_SVM(self.labels, self.tau, self.alpha)
class Accuracy_Logger(object):
    """Accuracy logger"""

    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]

    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)

    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()

    def get_summary(self, c):
        count = self.data[c]["count"]
        correct = self.data[c]["correct"]

        if count == 0:
            acc = None
        else:
            acc = float(correct) / count

        return acc, correct, count


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name='checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss


def train(datasets, cur, args):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'svm':
        # from topk.svm import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes=args.n_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    else:
        loss_fn = nn.CrossEntropyLoss()
    print('Done!')

    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out,
                  'n_classes': args.n_classes,
                  "embed_dim": args.embed_dim}

    if args.model_size is not None and args.model_type != 'mil':
        model_dict.update({"size_arg": args.model_size})

    if args.model_type in ['clam_sb', 'clam_mb']:
        if args.subtyping:
            model_dict.update({'subtyping': True})

        if args.B > 0:
            model_dict.update({'k_sample': args.B})

        if args.inst_loss == 'svm':
            # from topk.svm import SmoothTop1SVM
            instance_loss_fn = SmoothTop1SVM(n_classes=2)
            if device.type == 'cuda':
                instance_loss_fn = instance_loss_fn.cuda()
        else:
            weights = [0.25, 0.75]
            class_weights = torch.FloatTensor(weights).cuda()
            instance_loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        if args.model_type == 'clam_sb':
            model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
        elif args.model_type == 'clam_mb':
            model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
        else:
            raise NotImplementedError

    else:  # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)

    _ = model.to(device)
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')

    print('\nInit Loaders...', end=' ')
    # Here the testing param is not actually testing, it should be debugging
    train_loader = get_split_loader(train_split, training=True, testing=args.testing, weighted=args.weighted_sample)
    val_loader = get_split_loader(val_split, testing=args.testing)
    test_loader = get_split_loader(test_split, testing=args.testing)

    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience=20, stop_epoch=50, verbose=True)

    else:
        early_stopping = None
    print('Done!')

    for epoch in range(args.max_epochs):
        if args.model_type in ['clam_sb', 'clam_mb'] and not args.no_inst_cluster:
            train_loop_clam(epoch, model, train_loader, optimizer, args.n_classes, args.bag_weight, writer, loss_fn)
            stop = validate_clam(cur, epoch, model, val_loader, args.n_classes,
                                 early_stopping, writer, loss_fn, args.results_dir)

        else:
            train_loop(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn)
            stop = validate(cur, epoch, model, val_loader, args.n_classes,
                            early_stopping, writer, loss_fn, args.results_dir)

        if stop:
            break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    _, val_error, val_auc, _ = summary(model, val_loader, args.n_classes)
    print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

    results_dict, test_error, test_auc, acc_logger = summary(model, test_loader, args.n_classes)
    print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

        if writer:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

    if writer:
        writer.add_scalar('final/val_error', val_error, 0)
        writer.add_scalar('final/val_auc', val_auc, 0)
        writer.add_scalar('final/test_error', test_error, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)
        writer.close()
    return results_dict, test_auc, val_auc, 1 - test_error, 1 - val_error, model


def train_loop_clam(epoch, model, loader, optimizer, n_classes, bag_weight, writer=None, loss_fn=None):
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)

    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0

    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)

        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()

        instance_loss = instance_dict['instance_loss']
        inst_count += 1
        instance_loss_value = instance_loss.item()
        train_inst_loss += instance_loss_value

        total_loss = bag_weight * loss + (1 - bag_weight) * instance_loss

        inst_preds = instance_dict['inst_preds']
        inst_labels = instance_dict['inst_labels']
        inst_logger.log_batch(inst_preds, inst_labels)

        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, '.format(batch_idx, loss_value,
                                                                                                  instance_loss_value,
                                                                                                  total_loss.item()) +
                  'label: {}, bag_size: {}'.format(label.item(), data.size(0)))

        error = calculate_error(Y_hat, label)
        train_error += error

        # backward pass
        total_loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    if inst_count > 0:
        train_inst_loss /= inst_count
        print('\n')
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    print('Epoch: {}, train_loss: {:.4f}, train_clustering_loss:  {:.4f}, train_error: {:.4f}'.format(epoch, train_loss,
                                                                                                      train_inst_loss,
                                                                                                      train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer and acc is not None:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/clustering_loss', train_inst_loss, epoch)


def train_loop_clam_reg(epoch, model, loader, optimizer, n_classes, bag_weight, writer=None, loss_fn=None):
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)

    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0

    print('\n')
    for batch_idx, (data, label, event_time) in enumerate(loader):
        data, label, event_time = data.to(device), label.to(device), event_time.to(device)
        logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)

        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, event_time, label)
        loss_value = loss.item()

        instance_loss = instance_dict['instance_loss']
        inst_count += 1
        instance_loss_value = instance_loss.item()
        train_inst_loss += instance_loss_value

        total_loss = bag_weight * loss + (1 - bag_weight) * instance_loss

        inst_preds = instance_dict['inst_preds']
        inst_labels = instance_dict['inst_labels']
        inst_logger.log_batch(inst_preds, inst_labels)

        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, '.format(batch_idx, loss_value,
                                                                                                  instance_loss_value,
                                                                                                  total_loss.item()) +
                  'label: {}, bag_size: {}'.format(label.item(), data.size(0)))

        error = calculate_error(Y_hat, label)
        train_error += error

        # backward pass
        total_loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    if inst_count > 0:
        train_inst_loss /= inst_count
        print('\n')
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    print('Epoch: {}, train_loss: {:.4f}, train_clustering_loss:  {:.4f}, train_error: {:.4f}'.format(epoch, train_loss,
                                                                                                      train_inst_loss,
                                                                                                      train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer and acc is not None:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/clustering_loss', train_inst_loss, epoch)


def train_loop(epoch, model, loader, optimizer, n_classes, writer=None, loss_fn=None):
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.

    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)

        logits, Y_prob, Y_hat, _, _ = model(data)

        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()

        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx, loss_value, label.item(),
                                                                           data.size(0)))

        error = calculate_error(Y_hat, label)
        train_error += error

        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)


def train_loop_reg(epoch, model, loader, optimizer, writer=None, loss_fn=None, sampling_size=16):
    model.train()
    train_loss = 0.

    log_risk_pred_accu = []
    log_risk_pred_all = []

    occur_time_accu = []
    occur_time_all = []

    event_status_accu = []
    event_status_all = []

    iter_num = 0

    print('\n')
    for batch_idx, (data, label, event_time) in enumerate(loader):
        data, label, event_time = data.to(device), label.to(device), event_time.to(device)

        # logits, Y_prob, Y_hat, _, _ = model(data)
        top_log_risks_mean, top_risk_select_mean, _, _ = model(data)

        if batch_idx == 0:  # set the starting point of sample gathering

            log_risk_pred_all = top_log_risks_mean
            occur_time_all = event_time
            event_status_all = label
        if iter_num == 0:  # start to record the whole training batch for c-index calc
            log_risk_pred_accu = top_log_risks_mean
            occur_time_accu = event_time
            event_status_accu = label

        else:
            log_risk_pred_accu = torch.cat([log_risk_pred_accu, top_log_risks_mean])
            log_risk_pred_all = torch.cat([log_risk_pred_all, top_log_risks_mean])

            occur_time_accu = torch.cat([occur_time_accu, event_time])
            occur_time_all = torch.cat([occur_time_all, event_time])

            event_status_accu = torch.cat([event_status_accu, label])
            event_status_all = torch.cat([event_status_all, label])

        iter_num += 1

        if iter_num % sampling_size == 0 or batch_idx == len(loader) - 1:
            # when collecting enough samples :
            if torch.max(event_status_accu) < 1 or torch.min(event_status_accu > 0):
                # all the sample collected should not have the same label
                print("encounter uni-event batch, skip")
                log_risk_pred_accu = []
                occur_time_accu = []
                event_status_accu = []
                iter_num = 0
                continue
            optimizer.zero_grad()  # zero the gradient buffer
            loss_risk = loss_fn(log_risk_pred_accu, occur_time_accu, event_status_accu)

            # L1 regularization:
            l1_reg = None
            for W in model.parameters():
                if l1_reg is None:
                    l1_reg = torch.abs(W).sum()
                else:
                    l1_reg = l1_reg + torch.abs(W).sum()  # torch.abs(W).sum() is equivalent to W.norm(1)

            loss = loss_risk + 1e-5 * l1_reg
            loss_value = loss.item()
            train_loss += loss_value

            # backward pass
            loss.backward()
            # step
            optimizer.step()
            log_risk_pred_accu = []
            occur_time_accu = []
            event_status_accu = []
            iter_num = 0
        # if (batch_idx + 1) % (sampling_size*5) == 0:
        #     print(
        #         f'batch {batch_idx}, loss {loss_value:.4f}, event status: {label.item()}, event_time in years: {event_time.item():.4f}, bag_size: {data.size(0)}')
    # calculate mean train loss and C-index
    train_loss /= (len(loader) / sampling_size)
    train_c_index = CIndex_lifeline(log_risk_pred_all.cpu().detach().numpy(), event_status_all.cpu().detach().numpy(),
                                    occur_time_all.cpu().detach().numpy())
    train_pvalue_pred = cox_log_rank(log_risk_pred_all.cpu().detach().numpy(), event_status_all.cpu().detach().numpy(),
                                     occur_time_all.cpu().detach().numpy())
    print(
        f'Epoch: {epoch}, train_loss: {train_loss:.4f}, train_c_index: {train_c_index:.4f}, train_pvalue_pred: {train_pvalue_pred:.4f}')

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/c_index', train_c_index, epoch)
        writer.add_scalar('train/pvalue_pred', train_pvalue_pred, epoch)


def validate(cur, epoch, model, loader, n_classes, early_stopping=None, writer=None, loss_fn=None, results_dir=None):
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    # loader.dataset.update_mode(True)
    val_loss = 0.
    val_error = 0.

    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)

            logits, Y_prob, Y_hat, _, _ = model(data)

            acc_logger.log(Y_hat, label)

            loss = loss_fn(logits, label)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()

            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])

    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')

    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name=os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))

        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


def validate_clam(cur, epoch, model, loader, n_classes, early_stopping=None, writer=None, loss_fn=None,
                  results_dir=None):
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    val_inst_acc = 0.
    inst_count = 0

    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    sample_size = model.k_sample
    with torch.inference_mode():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)
            logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)
            acc_logger.log(Y_hat, label)

            loss = loss_fn(logits, label)

            val_loss += loss.item()

            instance_loss = instance_dict['instance_loss']

            inst_count += 1
            instance_loss_value = instance_loss.item()
            val_inst_loss += instance_loss_value

            inst_preds = instance_dict['inst_preds']
            inst_labels = instance_dict['inst_labels']
            inst_logger.log_batch(inst_preds, inst_labels)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()

            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    if inst_count > 0:
        val_inst_loss /= inst_count
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/inst_loss', val_inst_loss, epoch)

    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

        if writer and acc is not None:
            writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name=os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))

        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


def validate_reg(cur, epoch, model, loader, early_stopping=None, writer=None, loss_fn=None,
                 results_dir=None):
    model.eval()

    log_risk_pred_all = None
    occur_time_all = None
    event_status_all = None

    with torch.no_grad():
        # the built-in batch size is 1 , again we should accumulate the whole dataset
        for batch_idx, (data, label, event_time) in enumerate(loader):
            data, label, event_time = data.to(device, non_blocking=True), label.to(device,
                                                                                   non_blocking=True), event_time.to(
                device,
                non_blocking=True)

            #logits, Y_prob, Y_hat, _, _ = model(data)
            top_log_risks_mean, top_risk_select_mean, _, _ = model(data)

            if batch_idx == 0:
                log_risk_pred_all = top_log_risks_mean
                occur_time_all = event_time
                event_status_all = label
            else:
                log_risk_pred_all = torch.cat([log_risk_pred_all, top_log_risks_mean])
                occur_time_all = torch.cat([occur_time_all, event_time])
                event_status_all = torch.cat([event_status_all, label])

    loss_risk = loss_fn(log_risk_pred_all, occur_time_all, event_status_all)

    # L1 regularization:
    l1_reg = None
    for W in model.parameters():
        if l1_reg is None:
            l1_reg = torch.abs(W).sum()
        else:
            l1_reg = l1_reg + torch.abs(W).sum()  # torch.abs(W).sum() is equivalent to W.norm(1)

    loss = loss_risk + 1e-5 * l1_reg

    val_loss = loss.item()
    c_index = CIndex_lifeline(log_risk_pred_all.cpu().detach().numpy(), event_status_all.cpu().detach().numpy(),
                              occur_time_all.cpu().detach().numpy())
    pvalue_pred = cox_log_rank(log_risk_pred_all.cpu().detach().numpy(), event_status_all.cpu().detach().numpy(),
                               occur_time_all.cpu().detach().numpy())
    print(
        f'\nVal Set, val_loss: {val_loss:.4f}, val_c_index: {c_index:.4f}, val_pvalue_pred: {pvalue_pred:.4f}')

    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/c_index', c_index, epoch)
        writer.add_scalar('val/pvalue_pred', pvalue_pred, epoch)

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name=os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))

        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


def summary(model, loader, n_classes):
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.inference_mode():
            logits, Y_prob, Y_hat, _, _ = model(data)

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()

        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    return patient_results, test_error, auc, acc_logger


def summary_reg(model, loader):
    model.eval()

    slide_ids = loader.dataset.slide_data['slide_id']

    patient_results = {}
    log_risk_pred_all = None
    occur_time_all = None
    event_status_all = None
    # the built-in batch size is 1 , again we should accumulate the whole dataset
    for batch_idx, (data, label, event_time) in enumerate(loader):
        # time_start = time.time()
        data, label, event_time = data.to(device), label.to(device), event_time.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.inference_mode():
            # logits, Y_prob, Y_hat, _, _ = model(data)
            top_log_risks_mean, top_risk_select_mean, _, _ = model(data)

        if batch_idx == 0:
            log_risk_pred_all = top_log_risks_mean
            occur_time_all = event_time
            event_status_all = label
        else:
            log_risk_pred_all = torch.cat([log_risk_pred_all, top_log_risks_mean])
            occur_time_all = torch.cat([occur_time_all, event_time])
            event_status_all = torch.cat([event_status_all, label])
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'risk': top_risk_select_mean,
                                           'event': label.item(), 'follow_up_years': event_time}})
        # time_elapsed = time.time() - time_start
        # print(f'\nEvaluation for Slide {slide_id} took {time_elapsed} s')
    c_index = CIndex_lifeline(log_risk_pred_all.cpu().detach().numpy(), event_status_all.cpu().detach().numpy(),
                              occur_time_all.cpu().detach().numpy())
    pvalue_pred = cox_log_rank(log_risk_pred_all.cpu().detach().numpy(), event_status_all.cpu().detach().numpy(),
                               occur_time_all.cpu().detach().numpy())

    return patient_results, c_index, pvalue_pred


def reg_highest_index_selection(model, loader, percentage):
    model.eval()

    slide_ids = loader.dataset.slide_data['slide_id']

    index_result = {}
    # the built-in batch size is 1 , again we should accumulate the whole dataset
    for batch_idx, (data, label, event_time) in enumerate(loader):
        # time_start = time.time()
        data, label, event_time = data.to(device), label.to(device), event_time.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        # print(data.size()[0])
        sample_size = round(data.size()[0] * percentage)
        with torch.inference_mode():
            # logits, Y_prob, Y_hat, _, _ = model(data)
            _, _, _, log_risks_or_att = model(data)
        selected_index = torch.topk(log_risks_or_att[0, :], sample_size, dim=0)[1]
        index_result.update({f'{slide_id}': selected_index.cpu().numpy().tolist()})

    return index_result

def cls_highest_index_selection(model, loader, percentage, args):
    model.eval()

    slide_ids = loader.dataset.slide_data['slide_id']

    index_result = {}
    # the built-in batch size is 1 , again we should accumulate the whole dataset
    for batch_idx, (data, label) in enumerate(loader):
        # time_start = time.time()
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        # print(data.size()[0])
        sample_size = round(data.size()[0] * percentage)
        with torch.inference_mode():
            # logits, Y_prob, Y_hat, _, _ = model(data)
            _, _, _, prob_or_att,_ = model(data)
        if args.model_type == 'mil':
            prob_or_att_vector = prob_or_att[:,1]
        elif args.model_type == 'clam_sb':
            prob_or_att_vector = prob_or_att[0]
        else:
            return NotImplementedError
        selected_index = torch.topk(prob_or_att_vector[:], sample_size, dim=0)[1]
        index_result.update({f'{slide_id}': selected_index.cpu().numpy().tolist()})

    return index_result
def CIndex_lifeline(hazards, labels, survtime_all):  # all input should be numpy arrays
    labels = labels.reshape(-1)
    hazards = hazards.reshape(-1)
    label = []
    hazard = []
    surv_time = []
    for i in range(len(hazards)):
        if not np.isnan(hazards[i]):
            label.append(labels[i])
            hazard.append(hazards[i])
            surv_time.append(survtime_all[i])

    new_label = np.asarray(label)
    new_hazard = np.asarray(hazard)
    new_surv = np.asarray(surv_time)

    return (concordance_index(new_surv, -new_hazard, new_label))


def cox_log_rank(hazards, labels, survtime_all):  # all input should be numpy arrays
    hazardsdata = hazards.reshape(-1)
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    survtime_all = survtime_all.reshape(-1)
    idx = hazards_dichotomize == 0
    T1 = survtime_all[idx]
    T2 = survtime_all[~idx]
    E1 = labels[idx]
    E2 = labels[~idx]
    results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
    pvalue_pred = results.p_value
    return (pvalue_pred)


def train_reg(datasets, cur, args, all_dataset=None):
    """
        train for a single fold
    """
    if all_dataset == None:
        raise ValueError('Please insert the whole dataset to train_reg()')
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    loss_fn = _neg_partial_log

    # if args.bag_loss == 'svm':
    #     from topk.svm import SmoothTop1SVM
    #     loss_fn = SmoothTop1SVM(n_classes=args.n_classes)
    #     if device.type == 'cuda':
    #         loss_fn = loss_fn.cuda()
    #
    # else:
    #     loss_fn = nn.CrossEntropyLoss()
    print('Done!')

    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out,
                  'n_classes': args.n_classes,
                  "embed_dim": args.embed_dim}

    if args.model_size is not None and args.model_type != 'mil':
        model_dict.update({"size_arg": args.model_size})

    if args.model_type in ['clam_sb', 'clam_mb']:
        raise NotImplementedError
        # if args.subtyping:
        #     model_dict.update({'subtyping': True})
        #
        # if args.B > 0:
        #     model_dict.update({'k_sample': args.B})
        #
        # if args.inst_loss == 'svm':
        #     from topk.svm import SmoothTop1SVM
        #     instance_loss_fn = SmoothTop1SVM(n_classes=2)
        #     if device.type == 'cuda':
        #         instance_loss_fn = instance_loss_fn.cuda()
        # else:
        #     instance_loss_fn = nn.CrossEntropyLoss()
        #
        # if args.model_type == 'clam_sb':
        #     model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
        # elif args.model_type == 'clam_mb':
        #     model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
        # else:
        #     raise NotImplementedError

    elif args.model_type in ['mil_reg', 'mil_reg_att', 'mil_reg_topk_att']:  # args.model_type == 'mil_reg'
        if args.model_type == 'mil_reg':
            model_dict.update({'top_k': args.top_k})
            model = MIL_fc_reg(**model_dict)
        elif args.model_type == 'mil_reg_att':
            model_dict.update({'gate': args.use_gate_attention})
            model = MIL_fc_reg_att(**model_dict)
        elif args.model_type == 'mil_reg_topk_att':
            model_dict.update({'gate': args.use_gate_attention})
            model_dict.update({'top_k': args.top_k})
            model = MIL_fc_reg_top_k_att(**model_dict)
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError
        # if args.n_classes > 2:
        #     model = MIL_fc_mc(**model_dict)
        # else:
        #     model = MIL_fc(**model_dict)

    _ = model.to(device)
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')

    print('\nInit Loaders...', end=' ')
    # Here the testing param is not actually testing, it should be debugging
    train_loader = get_split_loader(train_split, training=True, testing=args.testing, weighted=args.weighted_sample,
                                    collate_func=collate_MIL_reg)
    val_loader = get_split_loader(val_split, testing=args.testing, collate_func=collate_MIL_reg)
    test_loader = get_split_loader(test_split, testing=args.testing, collate_func=collate_MIL_reg)
    all_loader = get_split_loader(all_dataset, testing=args.testing, collate_func=collate_MIL_reg)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience=20, stop_epoch=50, verbose=True)

    else:
        early_stopping = None
    print('Done!')

    for epoch in range(args.max_epochs):
        if args.model_type in ['clam_sb', 'clam_mb'] and not args.no_inst_cluster:
            raise NotImplementedError
            # train_loop_clam(epoch, model, train_loader, optimizer, args.n_classes, args.bag_weight, writer, loss_fn)
            # stop = validate_clam(cur, epoch, model, val_loader, args.n_classes,
            #                      early_stopping, writer, loss_fn, args.results_dir)

        else:
            train_loop_reg(epoch, model, train_loader, optimizer, writer, loss_fn, sampling_size=args.sampling_size)
            stop = validate_reg(cur, epoch, model, val_loader,
                                early_stopping, writer, loss_fn, args.results_dir)
            # train_loop(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn)
            # stop = validate(cur, epoch, model, val_loader, args.n_classes,
            #                 early_stopping, writer, loss_fn, args.results_dir)

        if stop:
            break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))
    _, val_c_index, val_pvalue_pred = summary_reg(model, val_loader)
    print(f'Val C-index: {val_c_index:.4f}, P value pred: {val_pvalue_pred:.4f}')

    test_results_dict, test_c_index, test_pvalue_pred = summary_reg(model, test_loader)

    print(f'Test C-index: {test_c_index:.4f}, P value pred: {test_pvalue_pred:.4f}')
    # for i in range(args.n_classes):
    #     acc, correct, count = acc_logger.get_summary(i)
    #     print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
    #
    #     if writer:
    #         writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)
    if args.select_top_feature_index:
        percentages = np.arange(0.05, 0.05 * (args.percentage_level + 1), 0.05)
        for percentage in percentages:
            # select_dict = {}
            #
            # train_select_dict = reg_highest_index_selection(model,train_loader_eval,percentage)
            # select_dict.update(train_select_dict)
            # val_select_dict = reg_highest_index_selection(model,val_loader,percentage)
            # select_dict.update(val_select_dict)
            # test_select_dict = reg_highest_index_selection(model,test_loader,percentage)
            # select_dict.update(test_select_dict)
            select_dict = reg_highest_index_selection(model, all_loader, percentage)
            save_path = f'{args.results_dir}/split_{args.current_fold}_{percentage:.2f}_select.json'
            save_dict_as_json(select_dict, save_path)

    if writer:
        writer.add_scalar('final/val_c_index', val_c_index, 0)
        writer.add_scalar('final/val_pvalue_pred', val_pvalue_pred, 0)
        writer.add_scalar('final/test_c_index', test_c_index, 0)
        writer.add_scalar('final/test_pvalue_pred', test_pvalue_pred, 0)
        writer.close()

    return test_results_dict, test_c_index, test_pvalue_pred, val_c_index, val_pvalue_pred
