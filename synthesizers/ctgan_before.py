import os.path
import warnings

import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
import torch
from packaging import version
from torch import optim
import torch.nn as nn
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential, functional, \
    Sigmoid, Softmax, PReLU
from .self_paced_ensemble import SelfPacedEnsemble
from .lrsample import LRSample
from .data_sampler import DataSampler
from .data_transformer import DataTransformer
from .base import BaseSynthesizer
from sklearn.ensemble import GradientBoostingClassifier
from torchmetrics import AUROC
from .auroc import idx_I0I1
from sklearn.metrics import roc_auc_score


class Discriminator(Module):

    def __init__(self, input_dim, discriminator_dim, pac=10):
        super(Discriminator, self).__init__()
        dim = input_dim * pac
        self.pac = pac
        self.pacdim = dim
        seq = []
        for item in list(discriminator_dim):
            seq += [Linear(dim, item), LeakyReLU(0.2), Dropout(0.5)]
            dim = item

        seq += [Linear(dim, 1)]
        self.seq = Sequential(*seq)

    def calc_gradient_penalty(self, real_data, fake_data, device='cpu', pac=10, lambda_=10):
        alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
        alpha = alpha.repeat(1, pac, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        disc_interpolates = self(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        gradient_penalty = ((
                                gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1
                            ) ** 2).mean() * lambda_

        return gradient_penalty

    def forward(self, input):
        assert input.size()[0] % self.pac == 0
        return self.seq(input.view(-1, self.pacdim))


class Residual(Module):

    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = Linear(i, o)
        self.bn = BatchNorm1d(o)
        self.relu = ReLU()

    def forward(self, input):
        out = self.fc(input)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input], dim=1)


class Generator(Module):

    def __init__(self, embedding_dim, generator_dim, data_dim):
        super(Generator, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(generator_dim):
            seq += [Residual(dim, item)]
            dim += item
        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)

    def forward(self, input):
        data = self.seq(input)
        return data


class Filter(Module):
    def __init__(self, input_dim):
        super(Filter, self).__init__()
        self.model = Sequential(
            Linear(input_dim, 64),
            ReLU(),
            Linear(64, 16),
            ReLU(),
            Linear(16, 2),
            Softmax()
        )

    def forward(self, input):
        return self.model(input)


class Classfier(Module):
    def __init__(self):
        super(Classfier, self).__init__()
        self.filter = Sequential(
            Linear(33, 16),
            LeakyReLU(),
            Linear(16, 2),
            Sigmoid()
        )
        self.model = Sequential(
            Linear(33, 50),
            ReLU(),
            Dropout(0.2),
            Linear(50, 100),
            PReLU(1),
            Linear(100, 1),
            Sigmoid()
        )

    def filt(self, input):
        active = self.filter(input)
        idx = active[:, 0] > active[:, 1]
        tmp = input[idx]
        if len(tmp) != 0:
            input = input[idx]
        return input

    def forward(self, input, target=None, flag=1):
        if flag == 0:
            active = self.filter(input)
            idx = active[:, 0] > active[:, 1]
            # in_idx = active[:, 1] > active[:, 0]
            # active = torch.Tensor([1, 0] * 1000).reshape(1000, -1).cuda()
            # inactive = torch.Tensor([0, 1] * 200).reshape(200, -1).cuda()
            # cat_ac = torch.cat((active, inactive), dim=0)
            input = torch.cat((input[:, :-2], active), dim=1)
            target = torch.cat((target, active), dim=1)
            # target_ac = target[idx]
            # target_iac = target[in_idx]
            # maj_tac = target_ac[target_ac[:, 0] > target_ac[:, 1]]
            # maj_tiac = target_iac[target_iac[:, 0] > target_iac[:, 1]]
            # min_tac = target_ac[target_ac[:, 0] < target_ac[:, 1]]
            # min_tiac = target_iac[target_iac[:, 0] < target_iac[:, 1]]
            # print("maj_active:", len(maj_tac))
            # print("maj_inactive:", len(maj_tiac))
            # print("min_active:", len(min_tac))
            # print("min_inactive:", len(min_tiac))
            # print()
            # idx = cat_ac[:, 0] > cat_ac[:, 1]
            input = input[idx]
            target = target[idx]
            # test = input[:, -2:]
            if len(input) != 0:
                return self.model(input), target, input
            return input, target, input
        return self.model(input), target


class CTGANSynthesizer(BaseSynthesizer):
    """Conditional Table GAN Synthesizer.

    This is the core class of the CTGAN project, where the different components
    are orchestrated together.
    For more details about the process, please check the [Modeling Tabular data using
    Conditional GAN](https://arxiv.org/abs/1907.00503) paper.
    Args:
        embedding_dim (int):
            Size of the random sample passed to the Generator. Defaults to 128.
        generator_dim (tuple or list of ints):
            Size of the output samples for each one of the Residuals. A Residual Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        discriminator_dim (tuple or list of ints):
            Size of the output samples for each one of the Discriminator Layers. A Linear Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        generator_lr (float):
            Learning rate for the generator. Defaults to 2e-4.
        generator_decay (float):
            Generator weight decay for the Adam Optimizer. Defaults to 1e-6.
        discriminator_lr (float):
            Learning rate for the discriminator. Defaults to 2e-4.
        discriminator_decay (float):
            Discriminator weight decay for the Adam Optimizer. Defaults to 1e-6.
        batch_size (int):
            Number of data samples to process in each step.
        discriminator_steps (int):
            Number of discriminator updates to do for each generator update.
            From the WGAN paper: https://arxiv.org/abs/1701.07875. WGAN paper
            default is 5. Default used is 1 to match original CTGAN implementation.
        log_frequency (boolean):
            Whether to use log frequency of categorical levels in conditional
            sampling. Defaults to ``True``.
        verbose (boolean):
            Whether to have print statements for progress results. Defaults to ``False``.
        pac (int):
            Number of samples to group together when applying the discriminator.
            Defaults to 10.
        cuda (bool):
            Whether to attempt to use cuda for GPU computation.
            If this is False or CUDA is not available, CPU will be used.
            Defaults to ``True``.
    """

    def __init__(self, embedding_dim=128, generator_dim=(256, 256), discriminator_dim=(256, 256),
                 generator_lr=2e-4, generator_decay=1e-6, discriminator_lr=2e-4,
                 discriminator_decay=1e-6, batch_size=500, discriminator_steps=1,
                 log_frequency=True, verbose=True, pac=10, cuda=True):

        self.optimizerC = None
        self.optimizerD = None
        self.optimizerG = None
        self.gen_step = False
        assert batch_size % 2 == 0

        self._embedding_dim = embedding_dim
        self._generator_dim = generator_dim
        self._discriminator_dim = discriminator_dim

        self._generator_lr = generator_lr
        self._generator_decay = generator_decay
        self._discriminator_lr = discriminator_lr
        self._discriminator_decay = discriminator_decay

        self._batch_size = batch_size
        self._discriminator_steps = discriminator_steps
        self._log_frequency = log_frequency
        self._verbose = verbose
        self.pac = pac

        if not cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = 'cuda'

        self._device = torch.device(device)

        self._transformer = None
        self._generator = None
        self._discriminator = None
        self._classfier = None
        self._filter = None
        self.train_data = None
        self.valid_data = None
        self._train_data_sampler = None
        self._valid_data_sampler = None
        self.criterion = torch.nn.BCEWithLogitsLoss()

    @staticmethod
    def _gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
        """Deals with the instability of the gumbel_softmax for older versions of torch.

        For more details about the issue:
        https://drive.google.com/file/d/1AA5wPfZ1kquaRtVruCd6BiYZGcDeNxyP/view?usp=sharing
        Args:
            logits:
                […, num_features] unnormalized log probabilities
            tau:
                non-negative scalar temperature
            hard:
                if True, the returned samples will be discretized as one-hot vectors,
                but will be differentiated as if it is the soft sample in autograd
            dim (int):
                a dimension along which softmax will be computed. Default: -1.
        Returns:
            Sampled tensor of same shape as logits from the Gumbel-Softmax distribution.
        """
        if version.parse(torch.__version__) < version.parse("1.2.0"):
            for i in range(10):
                transformed = functional.gumbel_softmax(logits, tau=tau, hard=hard,
                                                        eps=eps, dim=dim)
                if not torch.isnan(transformed).any():
                    return transformed
            raise ValueError("gumbel_softmax returning NaN.")

        return functional.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)

    def _apply_activate(self, data):
        """Apply proper activation function to the output of the generator."""
        data_t = []
        st = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if span_info.activation_fn == 'tanh':
                    ed = st + span_info.dim
                    data_t.append(torch.tanh(data[:, st:ed]))
                    st = ed
                elif span_info.activation_fn == 'softmax':
                    ed = st + span_info.dim
                    transformed = self._gumbel_softmax(data[:, st:ed], tau=0.2)
                    data_t.append(transformed)
                    st = ed
                else:
                    assert 0
        return torch.cat(data_t, dim=1)

    def _cond_loss(self, data, c, m):
        """Compute the cross entropy loss on the fixed discrete column."""
        loss = []
        st = 0
        st_c = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if len(column_info) != 1 or span_info.activation_fn != "softmax":
                    # not discrete column
                    st += span_info.dim
                else:
                    ed = st + span_info.dim
                    ed_c = st_c + span_info.dim
                    tmp = functional.cross_entropy(
                        data[:, st:ed],
                        torch.argmax(c[:, st_c:ed_c], dim=1),
                        reduction='none'
                    )
                    loss.append(tmp)
                    st = ed
                    st_c = ed_c

        loss = torch.stack(loss, dim=1)

        return (loss * m).sum() / data.size()[0]

    def _validate_discrete_columns(self, train_data, discrete_columns):
        """Check whether ``discrete_columns`` exists in ``train_data``.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        if isinstance(train_data, pd.DataFrame):
            invalid_columns = set(discrete_columns) - set(train_data.columns)
        elif isinstance(train_data, np.ndarray):
            invalid_columns = []
            for column in discrete_columns:
                if column < 0 or column >= train_data.shape[1]:
                    invalid_columns.append(column)
        else:
            raise TypeError('``train_data`` should be either pd.DataFrame or np.array.')

        if invalid_columns:
            raise ValueError('Invalid columns found: {}'.format(invalid_columns))

    def fit(self, train_data, epochs, discrete_columns=tuple()):
        """Fit the CTGAN Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        self.stable = False
        self._validate_discrete_columns(train_data, discrete_columns)

        self._transformer = DataTransformer()
        self._transformer.fit(train_data, discrete_columns)

        self.train_data = self._transformer.transform(train_data)
        np.random.shuffle(self.train_data)

        train_len = int(0.7 * len(self.train_data))
        self.valid_data = self.train_data[train_len:]
        self.train_data = self.train_data[:train_len]

        self._train_data_sampler = DataSampler(
            self.train_data,
            self._transformer.output_info_list,
            self._log_frequency)

        self._valid_data_sampler = DataSampler(
            self.valid_data,
            self._transformer.output_info_list,
            self._log_frequency)

        data_dim = self._transformer.output_dimensions

        self._generator = Generator(
            self._embedding_dim + self._train_data_sampler.dim_cond_vec(),
            self._generator_dim,
            data_dim
        ).to(self._device)

        self._discriminator = Discriminator(
            data_dim + self._train_data_sampler.dim_cond_vec(),
            self._discriminator_dim,
            pac=self.pac
        ).to(self._device)

        self._classfier = Classfier().to(self._device)

        self.optimizerG = optim.Adam(
            self._generator.parameters(),
            lr=self._generator_lr, betas=(0.5, 0.9),
            weight_decay=self._generator_decay
        )

        self.optimizerD = optim.Adam(
            self._discriminator.parameters(), lr=self._discriminator_lr,
            betas=(0.5, 0.9), weight_decay=self._discriminator_decay
        )

        self.optimizerC = optim.Adam(
            [{'params': self._classfier.parameters()}],
            lr=self._generator_lr, betas=(0.5, 0.9),
            weight_decay=self._generator_decay
        )

        steps_per_epoch = max(len(self.train_data) // self._batch_size, 1)
        for i in range(epochs):
            for id_ in range(steps_per_epoch):

                for n in range(self._discriminator_steps):
                    self.gen_step = False
                    mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
                    std = mean + 1

                    fakez = torch.normal(mean=mean, std=std)

                    condvec = self._train_data_sampler.sample_condvec(self._batch_size)
                    if condvec is None:
                        c1, m1, col, opt = None, None, None, None
                        real = self._train_data_sampler.sample_data(self._batch_size, col, opt)
                    else:
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self._device)
                        m1 = torch.from_numpy(m1).to(self._device)
                        fakez = torch.cat([fakez, c1], dim=1)

                        perm = np.arange(self._batch_size)
                        np.random.shuffle(perm)
                        real = self._train_data_sampler.sample_data(
                            self._batch_size, col[perm], opt[perm])
                        c2 = c1[perm]

                    fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake)

                    real = torch.from_numpy(real.astype('float32')).to(self._device)

                    if c1 is not None:
                        fake_cat = torch.cat([fakeact, c1], dim=1)
                        real_cat = torch.cat([real, c2], dim=1)
                    else:
                        real_cat = real
                        fake_cat = fakeact

                    y_fake = self._discriminator(fake_cat)
                    y_real = self._discriminator(real_cat)

                    pen = self._discriminator.calc_gradient_penalty(
                        real_cat, fake_cat, self._device, self.pac)
                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

                    self.optimizerD.zero_grad()
                    pen.backward(retain_graph=True)
                    loss_d.backward()
                    self.optimizerD.step()

                self.gen_step = True
                mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
                std = mean + 1

                fakez = torch.normal(mean=mean, std=std)
                condvec = self._train_data_sampler.sample_condvec(self._batch_size)

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    m1 = torch.from_numpy(m1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)

                perm = np.arange(self._batch_size)
                np.random.shuffle(perm)
                real = self._train_data_sampler.sample_data(
                    self._batch_size, col[perm], opt[perm])
                c2 = c1[perm]

                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)
                real = torch.from_numpy(real.astype('float32')).to(self._device)
                if c1 is not None:
                    y_fake = self._discriminator(torch.cat([fakeact, c1], dim=1))
                else:
                    y_fake = self._discriminator(fakeact)
                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = self._cond_loss(fake, c1, m1)

                loss_g = -torch.mean(y_fake) + cross_entropy
                self.optimizerG.zero_grad()
                loss_g.backward()
                self.optimizerG.step()

            if self._verbose:
                print(f"Epoch {i + 1}, Loss G: {loss_g.detach().cpu(): .4f}, "
                      f"Loss D: {loss_d.detach().cpu(): .4f}",
                      flush=True)
            if (i + 1) % 2500 == 0:
                self.save(os.path.join(os.path.join('./{}_stable_auc_model.pt'.format(i + 1))))

    def auc_fit(self, epochs, bs):
        self._batch_size = bs
        self.stable = True
        self.sampled = []
        self._classfier = Classfier().to(self._device)

        w0 = torch.empty(self._classfier.filter[0].weight.shape)
        nn.init.normal_(w0)
        self._classfier.filter[0].weight.data.copy_(w0)

        w2 = torch.empty(self._classfier.filter[2].weight.shape)
        nn.init.normal_(w2)
        self._classfier.filter[2].weight.data.copy_(w2)

        self.optimizerC = optim.Adam(
            [{'params': self._classfier.parameters()}],
            lr=0.001, betas=(0.9, 0.9999)
        )
        self.criterion = torch.nn.BCELoss()
        steps_per_epoch = max(len(self.train_data) // self._batch_size, 1)
        for i in range(epochs):
            for id_ in range(steps_per_epoch):
                for n in range(self._discriminator_steps):
                    self.gen_step = False
                    mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
                    std = mean + 1
                    fakez = torch.normal(mean=mean, std=std)

                    condvec = self._train_data_sampler.sample_condvec(self._batch_size)
                    if condvec is None:
                        c1, m1, col, opt = None, None, None, None
                        real = self._train_data_sampler.sample_data(self._batch_size, col, opt)
                    else:
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self._device)
                        m1 = torch.from_numpy(m1).to(self._device)
                        fakez = torch.cat([fakez, c1], dim=1)

                        perm = np.arange(self._batch_size)
                        np.random.shuffle(perm)
                        real = self._train_data_sampler.sample_data(
                            self._batch_size, col[perm], opt[perm])
                        c2 = c1[perm]

                    fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake)
                    real = torch.from_numpy(real.astype('float32')).to(self._device)

                    if c1 is not None:
                        fake_cat = torch.cat([fakeact, c1], dim=1)
                        real_cat = torch.cat([real, c2], dim=1)
                    else:
                        real_cat = real
                        fake_cat = fakeact

                    y_fake = self._discriminator(fake_cat)
                    y_real = self._discriminator(real_cat)

                    pen = self._discriminator.calc_gradient_penalty(
                        real_cat, fake_cat, self._device, self.pac)
                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

                    self.optimizerD.zero_grad()
                    pen.backward(retain_graph=True)
                    loss_d.backward()
                    self.optimizerD.step()

                self.gen_step = True
                mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
                std = mean + 1

                fakez = torch.normal(mean=mean, std=std)
                condvec = self._train_data_sampler.sample_condvec(self._batch_size)

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    m1 = torch.from_numpy(m1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)

                perm = np.arange(self._batch_size)
                np.random.shuffle(perm)
                real = self._train_data_sampler.sample_data(
                    self._batch_size, col[perm], opt[perm])
                c2 = c1[perm]

                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)
                real = torch.from_numpy(real.astype('float32')).to(self._device)

                if c1 is not None:
                    y_fake = self._discriminator(torch.cat([fakeact, c1], dim=1))
                else:
                    y_fake = self._discriminator(fakeact)
                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = self._cond_loss(fake, c1, m1)

                loss_g = -torch.mean(y_fake) + cross_entropy

                self.optimizerG.zero_grad()
                loss_g.backward()
                self.optimizerG.step()

                mean = torch.zeros(self._batch_size * 2, self._embedding_dim, device=self._device)
                std = mean + 1
                fakez = torch.normal(mean=mean, std=std)
                condvec = self._train_data_sampler.sample_condvec(self._batch_size * 2)
                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    m1 = torch.from_numpy(m1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)
                perm = np.arange(self._batch_size * 2)
                np.random.shuffle(perm)
                real = self._train_data_sampler.sample_data(
                    self._batch_size * 2, col[perm], opt[perm])
                c2 = c1[perm]
                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)
                real = torch.from_numpy(real.astype('float32')).to(self._device)
                cls_real = real[:, -2:]
                y_real_V = np.argmax(cls_real.cpu().detach().numpy(), axis=1)
                real_active = torch.Tensor([1, 0] * self._batch_size * 2).reshape(
                    self._batch_size * 2, -1).to(self._device)
                real = torch.cat((real[:, :-2], real_active), dim=1)

                # for name, param in self._classfier.filter.named_parameters():
                #     if param.requires_grad:
                #         print(name, param)

                # active = self._filter(fakeact)
                # fakeact = torch.cat((fakeact, active), dim=1)
                # active_idx = torch.where(active[:, 0] > active[:, 1])[0]
                # activate_idx = torch.randint(len(fakeact), (450, ))
                # if len(active_idx) != 0:
                #     fakeact = fakeact[active_idx]
                # fakeact_a = fakeact

                # sample_fr = torch.cat((fakeact, real), dim=0)
                # tmp = fakeact.detach().cpu().numpy()
                cls_fake = fakeact[:, -2:]
                y_fake_R = np.argmax(cls_fake.cpu().detach().numpy(), axis=1)
                # cls_fr = sample_fr[:, -2:]
                # y_fake_FR = np.argmax(cls_fr.cpu().detach().numpy(), axis=1)

                # idx0_R, idx1_R = idx_I0I1(y_fake_R)
                # if len(idx0_R) > len(idx1_R):
                #     idx_kk0 = np.random.choice(idx0_R, len(idx1_R), replace=False)
                #     idx_kk1 = idx1_R
                # else:
                #     idx_kk0 = idx0_R
                #     idx_kk1 = np.random.choice(idx1_R, len(idx0_R), replace=False)
                # dlogit = self._classfier(fakeact[idx_kk0]) - self._classfier(fakeact[idx_kk1])
                predict, cls_fake, sample = self._classfier(fakeact, cls_fake, 0)
                if len(predict) == 0:
                    break
                # loss_c = 1 - self.criterion(dlogit.flatten(), torch.ones(dlogit.shape[0]).cuda())
                loss_c = self.criterion(predict, cls_fake[:, 1].unsqueeze(1))
                # predic_real = self._classfier(real)
                # loss_c = functional.mse_loss(predic_real, cls_real[:, -1])
                self.optimizerC.zero_grad()
                loss_c.backward()
                self.optimizerC.step()

                pred_p, _ = self._classfier(real)
                auc_p = roc_auc_score(y_real_V, pred_p.cpu().detach().numpy())
                pred = pred_p.cpu().detach().numpy()
                pred = np.where(pred >= 0.5, 1, 0)
                auc = roc_auc_score(y_real_V, pred.squeeze())
                self.sampled.append(sample[:, :-2])

            if self._verbose:
                # epoch_sampled = torch.cat(self.sampled, dim=0)
                print(len(self.sampled[-1]))
                # samples_ori = self._transformer.inverse_transform(epoch_sampled.detach().cpu().numpy()).values
                # samples = samples_ori[:, :-1]
                # labels = samples_ori[:, -1]
                # samples_maj = samples[np.where(labels == 0)]
                # samples_min = samples[np.where(labels == 1)]
                # plt.scatter(samples_maj[:, 0], samples_maj[:, 1], s=.1)
                # plt.title("Samples-Maj")
                # plt.show()
                # plt.scatter(samples_min[:, 0], samples_min[:, 1], s=.1)
                # plt.title("Samples-Min")
                # plt.show()
                print(f"auc_Epoch {i + 1}, Loss G: {loss_g.detach().cpu()}, "
                      f"Loss D: {loss_d.detach().cpu()}",
                      f"Loss C: {loss_c.detach().cpu()}",
                      f"AUC_P: {auc_p}",
                      f"AUC: {auc}",
                      flush=True)
                if (i + 1) % 50 == 0:
                    self.save(os.path.join(os.path.join('./{}_auc_model.pt'.format(i + 1))))

    def sample(self, n, condition_column=None, condition_value=None, ratio=0):
        """Sample data similar to the training data.

        Choosing a condition_column and condition_value will increase the probability of the
        discrete condition_value happening in the condition_column.
        Args:
            n (int):
                Number of rows to sample.
            condition_column (string):
                Name of a discrete column.
            condition_value (string):
                Name of the category in the condition_column which we wish to increase the
                probability of happening.
        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        if condition_column is not None and condition_value is not None:
            condition_info = self._transformer.convert_column_name_value_to_id(
                condition_column, condition_value)
            global_condition_vec = self._train_data_sampler.generate_cond_from_condition_column_info(
                condition_info, self._batch_size)
        else:
            global_condition_vec = None

        steps = n // self._batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self._batch_size, self._embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self._device)

            if global_condition_vec is not None:
                condvec = global_condition_vec.copy()
            else:
                condvec = self._train_data_sampler.sample_original_condvec(self._batch_size)
            if condvec is None:
                pass
            else:
                c1 = condvec
                c1 = torch.from_numpy(c1).to(self._device)
                fakez = torch.cat([fakez, c1], dim=1)

            fake = self._generator(fakez)
            fakeact = self._apply_activate(fake)
            if self.stable:
                fakeact = self._classfier.filt(fakeact)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]
        # if self.stable:
        #     sampled = torch.cat(self.sampled, dim=0)
        #     samples_ori = self._transformer.inverse_transform(sampled.detach().cpu().numpy()).values
        #     samples = samples_ori[:, :-1]
        #     labels = samples_ori[:, -1]
        #     samples_maj = samples[np.where(labels == 0)]
        #     samples_min = samples[np.where(labels == 1)]
        #     plt.scatter(samples_maj[:, 0], samples_maj[:, 1], s=.1)
        #     plt.title("Samples-Maj")
        #     plt.show()
        #     plt.scatter(samples_min[:, 0], samples_min[:, 1], s=.1)
        #     plt.title("Samples-Min")
        #     plt.show()
        return self._transformer.inverse_transform(data)

    def set_device(self, device):
        self._device = device
        if self._generator is not None:
            self._generator.to(self._device)
