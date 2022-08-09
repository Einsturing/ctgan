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


# os.environ['CUDA_VISIBLE_DEVICES'] = '3'


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
    def __init__(self):
        super(Filter, self).__init__()
        self.model = Sequential(
            Linear(31, 64),
            ReLU(),
            Linear(64, 16),
            ReLU(),
            Linear(16, 2),
            Softmax()
        )

    def forward(self, input):
        labels = torch.zeros(len(input))
        incls = input[:, -2:]
        labels[torch.where(input[:, -2] > input[:, -1])] = 0
        active = self.model(input[:, :-2])
        # active = torch.Tensor([0.9, 0.1] * int(len(input))).reshape(int(len(input)), -1).cuda()
        input = torch.cat((input[:, :-2], active), dim=1)
        return input, labels, incls


class Composite(Module):
    def __init__(self):
        super(Composite, self).__init__()
        self.model = Sequential(
            Linear(4, 64),
            ReLU(),
            Linear(64, 16),
            ReLU(),
            Linear(16, 2),
            Softmax()
        )

    def forward(self, input, incls):
        composite = torch.cat((input[:, -2:], incls), dim=1)
        # active = torch.Tensor([0.9, 0.1] * int(len(input))).reshape(int(len(input)), -1).cuda()
        active = self.model(composite)
        input = torch.cat((input[:, :-2], active), dim=1)
        return input


class Classfier(Module):
    def __init__(self):
        super(Classfier, self).__init__()
        self.model = Sequential(
            Linear(33, 50),
            ReLU(),
            Dropout(0.2),
            Linear(50, 100),
            PReLU(1),
            Linear(100, 1),
            Sigmoid()
        )

    def forward(self, input):
        return self.model(input)


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
        self.optimizerF = None
        self.optimizerM = None
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
        self._verifyclf = None
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
                    transformed = self._gumbel_softmax(data[:, st:ed], tau=0.5)
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

    def _divide(self):
        orgdat = self.train_data
        samples_ori = self._transformer.inverse_transform(orgdat).values
        samples = samples_ori[:, :-1]
        labels = samples_ori[:, -1]
        samples_maj = samples[np.where(labels == 0)]
        samples_min = samples[np.where(labels == 1)]
        plt.scatter(samples_maj[:, 0], samples_maj[:, 1], s=.1)
        plt.title("Gendata-Maj " + str(len(samples_maj)))
        plt.show()
        plt.scatter(samples_min[:, 0], samples_min[:, 1], s=.1)
        plt.title("Gendata-Min " + str(len(samples_min)))
        plt.show()

        orgdat = self.valid_data
        samples_ori = self._transformer.inverse_transform(orgdat).values
        samples = samples_ori[:, :-1]
        labels = samples_ori[:, -1]
        samples_maj = samples[np.where(labels == 0)]
        samples_min = samples[np.where(labels == 1)]
        plt.scatter(samples_maj[:, 0], samples_maj[:, 1], s=.1)
        plt.title("Samples-Maj " + str(len(samples_maj)))
        plt.show()
        plt.scatter(samples_min[:, 0], samples_min[:, 1], s=.1)
        plt.title("Samples-Min " + str(len(samples_min)))
        plt.show()

        # # vdata = self._transformer.transform(vdata)
        # gtinput = torch.Tensor(vdata).reshape(np.shape(vdata)).cuda()
        # labcode = torch.Tensor([1, 0] * len(gtinput)).reshape(len(gtinput), -1).to(self._device)
        # gtdat = torch.cat((gtinput[:, :-2], labcode), dim=1)
        # gtlab = gtinput[:, -1].reshape(len(gtinput), 1).to(self._device)

        # tdata = self._transformer.transform(tdata)
        # np.random.shuffle(tdata)
        vinput = torch.Tensor(orgdat).reshape(np.shape(orgdat)).cuda()

        gtinput = vinput[:30000]
        labcode = torch.Tensor([1, 0] * len(gtinput)).reshape(len(gtinput), -1).to(self._device)
        gtdat = torch.cat((gtinput[:, :-2], labcode), dim=1)
        gtlab = gtinput[:, -1].reshape(len(gtinput), 1).to(self._device)
        vinput = vinput[30000:]
        return gtdat, gtlab, vinput

    def mutualinformation(self, input1, input2):
        '''
            input1: B, C, H, W
            input2: B, C, H, W
            return: scalar
        '''

        # Torch tensors for images between (0, 1)
        input1 = input1 * 255
        input2 = input2 * 255

        B, C, H, W = input1.shape
        assert ((input1.shape == input2.shape))

        x1 = input1.view(B, H * W, C)
        x2 = input2.view(B, H * W, C)

        pdf_x1, kernel_values1 = self.marginalPdf(x1)
        pdf_x2, kernel_values2 = self.marginalPdf(x2)
        pdf_x1x2 = self.jointPdf(kernel_values1, kernel_values2)

        H_x1 = -torch.sum(pdf_x1 * torch.log2(pdf_x1 + self.epsilon), dim=1)
        H_x2 = -torch.sum(pdf_x2 * torch.log2(pdf_x2 + self.epsilon), dim=1)
        H_x1x2 = -torch.sum(pdf_x1x2 * torch.log2(pdf_x1x2 + self.epsilon), dim=(1, 2))

        mutual_information = H_x1 + H_x2 - H_x1x2

        if self.normalize:
            mutual_information = 2 * mutual_information / (H_x1 + H_x2)

        return mutual_information

    def auc_fit(self, epochs, bs):
        self._batch_size = bs
        self.stable = True
        self.sampled = []
        self._filter = Filter().to(self._device)
        self._composite = Composite().to(self._device)
        self._classfier = Classfier().to(self._device)
        self._verifyclf = Classfier().to(self._device)

        w0 = torch.empty(self._filter.model[0].weight.shape)
        nn.init.normal_(w0)
        self._filter.model[0].weight.data.copy_(w0)

        w2 = torch.empty(self._filter.model[2].weight.shape)
        nn.init.normal_(w2)
        self._filter.model[2].weight.data.copy_(w2)

        self.optimizerM = optim.Adam(
            [{"params": self._filter.parameters()}, {"params": self._composite.parameters()}],
            lr=0.001, betas=(0.9, 0.9999))
        self.optimizerF = optim.Adam([{"params": self._filter.parameters()}], lr=0.001,
                                     betas=(0.9, 0.9999))
        self.optimizerC = optim.Adam(
            [{'params': self._classfier.parameters()}],
            lr=0.001, betas=(0.9, 0.9999)
        )
        self.optimizerV = optim.Adam(
            [{'params': self._verifyclf.parameters()}],
            lr=0.001, betas=(0.9, 0.9999)
        )
        self.criterion = torch.nn.BCELoss()
        self.criteriov = torch.nn.BCELoss()
        gtdat, gtlab, vinput = self._divide()
        # steps_per_epoch = max(len(self.train_data) // self._batch_size, 1)
        steps_per_epoch = max(len(vinput) // self._batch_size, 1)
        backround = 0
        gran = steps_per_epoch * 3

        for i in range(epochs):
            # history = torch.from_numpy(numpy.array([])).to(self._device)
            for id_ in range(steps_per_epoch):
                if backround % 1 == 0:
                    for n in range(self._discriminator_steps):
                        self.gen_step = False
                        mean = torch.zeros(self._batch_size, self._embedding_dim,
                                           device=self._device)
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

                    mean = torch.zeros(self._batch_size * 2, self._embedding_dim,
                                       device=self._device)
                    std = mean + 1
                    fakez = torch.normal(mean=mean, std=std)
                    condvec = self._valid_data_sampler.sample_condvec(self._batch_size * 2)
                    if condvec is None:
                        c1, m1, col, opt = None, None, None, None
                    else:
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self._device)
                        m1 = torch.from_numpy(m1).to(self._device)
                        fakez = torch.cat([fakez, c1], dim=1)
                    perm = np.arange(self._batch_size * 2)
                    np.random.shuffle(perm)
                    c2 = c1[perm]
                    fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake)

                    # real = self._valid_data_sampler.sample_data(self._batch_size * 2, col[perm],
                    #                                             opt[perm])
                    # real = torch.from_numpy(real.astype('float32')).to(self._device)
                    real = vinput[np.random.choice(np.arange(len(vinput)), self._batch_size * 2)]
                    cls_real = real[:, -2:]
                    y_real_V = np.argmax(cls_real.cpu().detach().numpy(), axis=1)
                    real_active = torch.Tensor([1, 0] * self._batch_size * 2).reshape(
                        self._batch_size * 2, -1).to(self._device)
                    real = torch.cat((real[:, :-2], real_active), dim=1)

                    # # reav = self._train_data_sampler.sample_data(self._batch_size * 2, col[perm],
                    # #                                             opt[perm])
                    # # reav = torch.from_numpy(reav.astype('float32')).to(self._device)
                    # reav = vinput[np.random.choice(np.arange(len(vinput)), self._batch_size * 2)]
                    # cls_reav = reav[:, -2:]
                    # y_reav_V = np.argmax(cls_reav.cpu().detach().numpy(), axis=1)
                    # # reav_active = torch.Tensor([1, 0] * self._batch_size * 2).reshape(
                    # #     self._batch_size * 2, -1).to(self._device)
                    # reav, faketcls1, incls1 = self._filter(reav)
                    # # reav = self._composite(reav, incls1)
                    # # reav = torch.cat((reav[:, :-2], reav_active), dim=1)

                    # print('upgraded', steps_per_epoch)

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
                tmp = fakeact.detach().cpu().numpy()
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

                # if backround % (gran / 10) == 0:
                #     print(self._filter.model[0].weight)
                fakeset, faketcls, incls = self._filter(fakeact)
                # fakeset = self._composite(fakeset, incls)
                predict = self._classfier(fakeset)

                # print(len(sample[sample[:, -2] > sample[:, -1]]),
                #       len(cls_fake[cls_fake[:, 0] > cls_fake[:, 1]]))

                idx = fakeset[:, -2] > fakeset[:, -1]
                in_idx = fakeset[:, -1] > fakeset[:, -2]
                target_ac = cls_fake[idx]
                target_iac = cls_fake[in_idx]
                maj_tac = target_ac[target_ac[:, 0] > target_ac[:, 1]]
                maj_tiac = target_iac[target_iac[:, 0] > target_iac[:, 1]]
                min_tac = target_ac[target_ac[:, 0] < target_ac[:, 1]]
                min_tiac = target_iac[target_iac[:, 0] < target_iac[:, 1]]
                maj_ac = len(maj_tac)
                maj_ic = len(maj_tiac)
                min_ac = len(min_tac)
                min_ic = len(min_tiac)

                # print([maj_ac, maj_ic, min_ac, min_ic], end=",")
                if len(predict) == 0:
                    break

                mark = ''
                #### update classification classifer with fake, filter: classifier = 1: gran - 1
                loss_f = torch.Tensor([torch.inf]).cuda()
                if (backround + 1) % (gran / 1) == 0:
                    mark = mark + '*'
                    # print(self._classfier.model[0].weight)
                    loss_f = self.criterion(predict, cls_fake[:, 1].unsqueeze(1)).mean()
                    # ce = self._cond_loss(fakeset, c1, m1)
                    # loss_f = loss_f + ce
                    self.optimizerM.zero_grad()
                    loss_f.backward()
                    self.optimizerM.step()
                    # print(self._classfier.model[0].weight)
                else:
                    mark = mark + '-'
                    loss_c = self.criterion(predict, cls_fake[:, 1].unsqueeze(1)).mean()
                    # loss_c = functional.mse_loss(predic_real, cls_real[:, -1])
                    self.optimizerC.zero_grad()
                    loss_c.backward()
                    self.optimizerC.step()

                #### update classification classifer with real, filter: classifier = 1: gran - 1
                pred_p = self._classfier(real)
                if (backround + 1) % (gran / 1) == 0:
                    mark = mark + '*'
                    # print(self._filter.model[0].weight)
                    # loss_f = self.criterion(pred_p, torch.Tensor(y_real_V).cuda().unsqueeze(1))
                    # self.optimizerM.zero_grad()
                    # loss_f.backward()
                    # self.optimizerM.step()
                    # print(self._filter.model[0].weight)
                else:
                    mark = mark + '-'
                    # #### Can be triggered for better performance with real dataset covering best learned ratio of maj/min
                    # # pred_p = self._classfier(reav)
                    # loss_c = self.criterion(pred_p, torch.Tensor(y_real_V).cuda().unsqueeze(1))
                    # # print(len(torch.where(reav[:, -2] >= reav[:, -1])[0]),
                    # #       len(torch.where(reav[:, -2] < reav[:, -1])[0]), len(reav))
                    # # loss_c = functional.mse_loss(predic_real, cls_real[:, -1])
                    # self.optimizerC.zero_grad()
                    # loss_c.backward()
                    # self.optimizerC.step()

                    #### update verification classifer
                    vdat = real
                    vlab = y_real_V
                    # idat = torch.cat([vdat, fakeset], dim=0)
                    # ilab = torch.cat(
                    #     [torch.Tensor(vlab).cuda().unsqueeze(1), cls_fake[:, 1].unsqueeze(1)], dim=0)
                    pred_v = self._verifyclf(vdat)
                    # loss_v = self.criterion(pred_v, ilab)
                    loss_v = self.criterion(pred_v, torch.Tensor(y_real_V).cuda().unsqueeze(1))
                    self.optimizerV.zero_grad()
                    loss_v.backward()
                    self.optimizerV.step()

                    self.sampled.append(fakeact[idx])

                if (backround + 1) % (gran * 10) == 0:
                    print(backround, i, len(self.sampled[-1]), end=', ')
                    epoch_sampled = torch.cat(self.sampled, dim=0)
                    # epoch_sampled = self.sampled[-1]
                    samples_ori = self._transformer.inverse_transform(
                        epoch_sampled.detach().cpu().numpy()).values
                    samples = samples_ori[:, :-1]
                    labels = samples_ori[:, -1]
                    samples_maj = samples[np.where(labels == 0)]
                    samples_min = samples[np.where(labels == 1)]
                    plt.scatter(samples_maj[:, 0], samples_maj[:, 1], s=.1)
                    plt.title("Samples-Maj " + str(len(samples_maj)))
                    plt.show()
                    plt.scatter(samples_min[:, 0], samples_min[:, 1], s=.1)
                    plt.title("Samples-Min " + str(len(samples_min)))
                    plt.show()

                backround = backround + 1
            self.sampled.clear()
            if self._verbose:
                print(mark, end=" ")
                y_real_cuda = torch.Tensor(y_real_V).cuda()
                print(backround, len(torch.where(y_real_cuda >= 0.5)[0]),
                      len(torch.where(y_real_cuda < 0.5)[0]),
                      len(torch.where(cls_fake[:, 1] >= 0.5)[0]),
                      len(torch.where(cls_fake[:, 1] < 0.5)[0]),
                      len(torch.where(vinput[:, -1] >= 0.5)[0]),
                      len(torch.where(vinput[:, -1] < 0.5)[0]),
                      len(torch.where(gtlab >= 0.5)[0]),
                      len(torch.where(gtlab < 0.5)[0]), end=', ')

                #### verify classification classifer AUC
                pred_p = self._classfier(real)
                auc_p = roc_auc_score(y_real_V, pred_p.cpu().detach().numpy())
                pred = pred_p.cpu().detach().numpy()
                pred = np.where(pred >= 0.5, 1, 0)
                auc = roc_auc_score(y_real_V, pred.squeeze())

                #### details of AUC
                tfn = len([y_true for y_true in y_real_V if y_true == 1])
                tpn = len([y_pred for y_pred in pred.squeeze() if y_pred == 1])
                y_merg = pred.squeeze() * y_real_V
                tp = len([y_merg for y_merg in y_merg if y_merg == 1])
                tfpn = len(y_merg)
                auc_v = roc_auc_score(y_real_V, pred.squeeze())

                #### verify verification classifier AVC
                pred_v = self._verifyclf(vdat)
                avc_p = roc_auc_score(vlab, pred_v.cpu().detach().numpy())
                pred = pred_v.cpu().detach().numpy()
                pred = np.where(pred >= 0.5, 1, 0)
                avc = roc_auc_score(vlab, pred.squeeze())

                #### groudtruth on classification classifier GUC
                pred_p = self._classfier(gtdat)
                pred = pred_p.cpu().detach().numpy()
                pred = np.where(pred >= 0.5, 1, 0)
                auc_gt = roc_auc_score(gtlab.cpu().detach().numpy(), pred.squeeze())

                #### groudtruth on verification classifier GVC
                pred_p = self._verifyclf(gtdat)
                pred = pred_p.cpu().detach().numpy()
                pred = np.where(pred >= 0.5, 1, 0)
                avc_gt = roc_auc_score(gtlab.cpu().detach().numpy(), pred.squeeze())
                print(f"auc_Epoch {i + 1}, ",
                      f"AUC_P: {auc_p}",
                      f"AVC_P: {avc_p}",
                      f"AUC: {auc}",
                      f"AVC: {avc}",
                      f"GUC: {auc_gt}",
                      f"GVC: {avc_gt}",
                      f"maic: {maj_ac}, {maj_ic}, {min_ac}, {min_ic}",
                      f"tpfn: {tfn}, {tpn}, {tp}, {tfpn}, {auc_v}",
                      f"Loss G: {loss_g.detach().cpu()}, "
                      f"Loss D: {loss_d.detach().cpu()}",
                      f"Loss C: {loss_c.detach().cpu()}",
                      f"Loss F: {loss_f.detach().cpu()}",
                      flush=True)

                if (i + 1) % 1000 == 0:
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
        # torch.autograd.set_detect_anomaly(True)
        self._device = device
        if self._generator is not None:
            self._generator.to(self._device)
