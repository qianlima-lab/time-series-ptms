
from torch import nn
import torch


DTYPE = torch.float32


class GetAttr:
    "Inherit from this to have all attr accesses in `self._xtra` passed down to `self.default`"
    _default = 'default'

    def _component_attr_filter(self, k):
        if k.startswith('__') or k in ('_xtra', self._default): return False
        xtra = getattr(self, '_xtra', None)
        return xtra is None or k in xtra

    def _dir(self):
        return [k for k in dir(getattr(self, self._default)) if self._component_attr_filter(k)]

    def __getattr__(self, k):
        if self._component_attr_filter(k):
            attr = getattr(self, self._default, None)
            if attr is not None: return getattr(attr, k)
        # raise AttributeError(k)

    def __dir__(self):
        return custom_dir(self, self._dir())

    #     def __getstate__(self): return self.__dict__
    def __setstate__(self, data):
        self.__dict__.update(data)


def get_device(use_cuda=True, device_id=None, usage=5):
    "Return or set default device; `use_cuda`: None - CUDA if available; True - error if not available; False - CPU"
    if not torch.cuda.is_available():
        use_cuda = False
    else:
        if device_id is None:
            device_ids = get_available_cuda(usage=usage)
            device_id = device_ids[0]  # get the first available device
        torch.cuda.set_device(device_id)
    return torch.device(torch.cuda.current_device()) if use_cuda else torch.device('cpu')


def set_device(usage=5):
    "set the device that has usage < default usage  "
    device_ids = get_available_cuda(usage=usage)
    torch.cuda.set_device(device_ids[0])  # get the first available device


def default_device(use_cuda=True):
    "Return or set default device; `use_cuda`: None - CUDA if available; True - error if not available; False - CPU"
    if not torch.cuda.is_available():
        use_cuda = False
    return torch.device(torch.cuda.current_device()) if use_cuda else torch.device('cpu')


def get_available_cuda(usage=10):
    if not torch.cuda.is_available(): return
    # collect available cuda devices, only collect devices that has less that 'usage' percent
    device_ids = []
    for device in range(torch.cuda.device_count()):
        if torch.cuda.utilization(device) < usage: device_ids.append(device)
    return device_ids


def to_device(b, device=None, non_blocking=False):
    """
    Recursively put `b` on `device`
    components of b are torch tensors
    """
    if device is None:
        device = default_device(use_cuda=True)

    if isinstance(b, dict):
        return {key: to_device(val, device) for key, val in b.items()}

    if isinstance(b, (list, tuple)):
        return type(b)(to_device(o, device) for o in b)

    return b.to(device, non_blocking=non_blocking)


def to_numpy(b):
    """
    Components of b are torch tensors
    """
    if isinstance(b, dict):
        return {key: to_numpy(val) for key, val in b.items()}

    if isinstance(b, (list, tuple)):
        return type(b)(to_numpy(o) for o in b)

    return b.detach().cpu().numpy()


class Callback(GetAttr):
    _default = 'learner'


class SetupLearnerCB(Callback):
    def __init__(self):
        self.device = default_device(use_cuda=True)

    def before_batch_train(self):
        self._to_device()

    def before_batch_valid(self):
        self._to_device()

    def before_batch_predict(self):
        self._to_device()

    def before_batch_test(self):
        self._to_device()

    def _to_device(self):
        batch = to_device(self.batch, self.device)
        if self.n_inp > 1:
            xb, yb = batch
        else:
            xb, yb = batch, None
        self.learner.batch = xb, yb

    def before_fit(self):
        "Set model to cuda before training"
        self.learner.model.to(self.device)
        self.learner.device = self.device


class GetPredictionsCB(Callback):
    def __init__(self):
        super().__init__()

    def before_predict(self):
        self.preds = []

    def after_batch_predict(self):
        # append the prediction after each forward batch
        self.preds.append(self.pred)

    def after_predict(self):
        self.preds = torch.concat(self.preds)  # .detach().cpu().numpy()


class GetTestCB(Callback):
    def __init__(self):
        super().__init__()

    def before_test(self):
        self.preds, self.targets = [], []

    def after_batch_test(self):
        # append the prediction after each forward batch
        self.preds.append(self.pred)
        self.targets.append(self.yb)

    def after_test(self):
        self.preds = torch.concat(self.preds)  # .detach().cpu().numpy()
        self.targets = torch.concat(self.targets)  # .detach().cpu().numpy()


# Cell
class PatchCB(Callback):

    def __init__(self, patch_len, stride):
        """
        Callback used to perform patching on the batch input data
        Args:
            patch_len:        patch length
            stride:           stride
        """
        self.patch_len = patch_len
        self.stride = stride

    def before_forward(self): self.set_patch()

    def set_patch(self):
        """
        take xb from learner and convert to patch: [bs x seq_len x n_vars] -> [bs x num_patch x n_vars x patch_len]
        """
        xb_patch, num_patch = create_patch(self.xb, self.patch_len, self.stride)  # xb: [bs x seq_len x n_vars]
        # learner get the transformed input
        self.learner.xb = xb_patch  # xb_patch: [bs x num_patch x n_vars x patch_len]


class PatchMaskCB(Callback):
    def __init__(self, patch_len, stride, mask_ratio,
                 mask_when_pred: bool = False):
        """
        Callback used to perform the pretext task of reconstruct the original data after a binary mask has been applied.
        Args:
            patch_len:        patch length
            stride:           stride
            mask_ratio:       mask ratio
        """
        self.patch_len = patch_len
        self.stride = stride
        self.mask_ratio = mask_ratio

    def before_fit(self):
        # overwrite the predefined loss function
        self.learner.loss_func = self._loss
        device = self.learner.device

    def before_forward(self): self.patch_masking()

    def patch_masking(self):
        """
        xb: [bs x seq_len x n_vars] -> [bs x num_patch x n_vars x patch_len]
        """
        xb_patch, num_patch = create_patch(self.xb, self.patch_len,
                                           self.stride)  # xb_patch: [bs x num_patch x n_vars x patch_len]
        xb_mask, _, self.mask, _ = random_masking(xb_patch,
                                                  self.mask_ratio)  # xb_mask: [bs x num_patch x n_vars x patch_len]
        self.mask = self.mask.bool()  # mask: [bs x num_patch x n_vars]
        self.learner.xb = xb_mask  # learner.xb: masked 4D tensor
        self.learner.yb = xb_patch  # learner.yb: non-masked 4d tensor

    def _loss(self, preds, target):
        """
        preds:   [bs x num_patch x n_vars x patch_len]
        targets: [bs x num_patch x n_vars x patch_len]
        """
        loss = (preds - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * self.mask).sum() / self.mask.sum()
        return loss


def create_patch(xb, patch_len, stride):
    """
    xb: [bs x seq_len x n_vars]
    """
    seq_len = xb.shape[1]
    num_patch = (max(seq_len, patch_len) - patch_len) // stride + 1
    tgt_len = patch_len + stride * (num_patch - 1)
    s_begin = seq_len - tgt_len

    xb = xb[:, s_begin:, :]  # xb: [bs x tgt_len x nvars]
    xb = xb.unfold(dimension=1, size=patch_len, step=stride)  # xb: [bs x num_patch x n_vars x patch_len]
    return xb, num_patch


class Patch(nn.Module):
    def __init__(self, seq_len, patch_len, stride):
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.num_patch = (max(seq_len, patch_len) - patch_len) // stride + 1
        tgt_len = patch_len + stride * (self.num_patch - 1)
        self.s_begin = seq_len - tgt_len

    def forward(self, x):
        """
        x: [bs x seq_len x n_vars]
        """
        x = x[:, self.s_begin:, :]
        x = x.unfold(dimension=1, size=self.patch_len, step=self.stride)  # xb: [bs x num_patch x n_vars x patch_len]
        return x


def random_masking(xb, mask_ratio):
    # xb: [bs x num_patch x n_vars x patch_len]
    bs, L, nvars, D = xb.shape
    x = xb.clone()

    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(bs, L, nvars, device=xb.device)  # noise in [0, 1], bs x L x nvars

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)  # ids_restore: [bs x L x nvars]

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep, :]  # ids_keep: [bs x len_keep x nvars]
    x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, 1,
                                                                        D))  # x_kept: [bs x len_keep x nvars  x patch_len]

    # removed x
    x_removed = torch.zeros(bs, L - len_keep, nvars, D,
                            device=xb.device)  # x_removed: [bs x (L-len_keep) x nvars x patch_len]
    x_ = torch.cat([x_kept, x_removed], dim=1)  # x_: [bs x L x nvars x patch_len]

    # combine the kept part and the removed one
    x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, 1,
                                                                              D))  # x_masked: [bs x num_patch x nvars x patch_len]

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([bs, L, nvars], device=x.device)  # mask: [bs x num_patch x nvars]
    mask[:, :len_keep, :] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)  # [bs x num_patch x nvars]
    return x_masked, x_kept, mask, ids_restore


def random_masking_3D(xb, mask_ratio):
    # xb: [bs x num_patch x dim]
    bs, L, D = xb.shape
    x = xb.clone()

    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(bs, L, device=xb.device)  # noise in [0, 1], bs x L

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)  # ids_restore: [bs x L]

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]  # ids_keep: [bs x len_keep]
    x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))  # x_kept: [bs x len_keep x dim]

    # removed x
    x_removed = torch.zeros(bs, L - len_keep, D, device=xb.device)  # x_removed: [bs x (L-len_keep) x dim]
    x_ = torch.cat([x_kept, x_removed], dim=1)  # x_: [bs x L x dim]

    # combine the kept part and the removed one
    x_masked = torch.gather(x_, dim=1,
                            index=ids_restore.unsqueeze(-1).repeat(1, 1, D))  # x_masked: [bs x num_patch x dim]

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([bs, L], device=x.device)  # mask: [bs x num_patch]
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)  # [bs x num_patch]
    return x_masked, x_kept, mask, ids_restore


if __name__ == "__main__":
    bs, L, nvars, D = 2, 20, 4, 5
    xb = torch.randn(bs, L, nvars, D)
    xb_mask, mask, ids_restore = create_mask(xb, mask_ratio=0.5)
    breakpoint()


