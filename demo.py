import torch
import numpy as np
import os
from PIL import Image
from models.dynamic_channel import set_uniform_channel_ratio, reset_generator
import models
import copy
import time
from matplotlib import pyplot as plt
# from tools.train_gan import finetune_prune


import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from torch import nn
from models.ops import StyledConv, EqualLinear, ModulatedConv2d, ToRGB, ConvLayer, ResBlock, EqualConv2d

# settings
# for attributes to use, modify the load_assets() function
config = 'anycost-ffhq-config-f'
assets_dir = 'assets/demo'
n_style_to_change = 12
device = 'cpu'

def fine_grained_prune(tensor: torch.Tensor, sparsity : float) -> torch.Tensor:
    """
    magnitude-based pruning for single tensor
    :param tensor: torch.(cuda.)Tensor, weight of conv/fc layer
    :param sparsity: float, pruning sparsity
        sparsity = #zeros / #elements = 1 - #nonzeros / #elements
    :return:
        torch.(cuda.)Tensor, mask for zeros
    """
    sparsity = min(max(0.0, sparsity), 1.0)
    if sparsity == 1.0:
        tensor.zero_()
        return torch.zeros_like(tensor)
    elif sparsity == 0.0:
        return torch.ones_like(tensor)

    num_elements = tensor.numel()

    ##################### YOUR CODE STARTS HERE #####################
    # Step 1: calculate the #zeros (please use round())
    num_zeros = round(sparsity * num_elements)
    # Step 2: calculate the importance of weight
    importance = torch.abs(tensor)
    # Step 3: calculate the pruning threshold
    threshold,s = torch.kthvalue(importance.view(-1), num_zeros)
    # Step 4: get binary mask (1 for nonzeros, 0 for zeros)
    mask = torch.gt(importance, threshold)
    ##################### YOUR CODE ENDS HERE #######################

    # Step 5: apply mask to prune the tensor
    tensor.mul_(mask)

    return mask

class FineGrainedPruner:
    def __init__(self, model, sparsity_dict):
        self.masks = FineGrainedPruner.prune(model, sparsity_dict)

    @torch.no_grad()
    def apply(self, model):
        for name, param in model.named_parameters():
            if name in self.masks:
                param *= self.masks[name]

    @staticmethod
    @torch.no_grad()
    def prune(model, sparsity_dict):
        masks = dict()
        for name, param in model.named_parameters():
            if param.dim() > 1 and param.numel() > 10 and name in sparsity_dict.keys(): # we only prune conv and fc weights
                print(name)
                masks[name] = fine_grained_prune(param, sparsity_dict[name])
        return masks

def get_num_channels_to_keep(channels: int, prune_ratio: float) -> int:
    """A function to calculate the number of layers to PRESERVE after pruning
    Note that preserve_rate = 1. - prune_ratio
    """
    ##################### YOUR CODE STARTS HERE #####################
    return int(round(channels * (1 - prune_ratio)))
    ##################### YOUR CODE ENDS HERE #####################

@torch.no_grad()
def channel_prune(model: nn.Module,
                  prune_ratio):
    """Apply channel pruning to each of the conv layer in the backbone
    Note that for prune_ratio, we can either provide a floating-point number,
    indicating that we use a uniform pruning rate for all layers, or a list of
    numbers to indicate per-layer pruning rate.
    """
    # sanity check of provided prune_ratio
    assert isinstance(prune_ratio, (float, list))
    n_conv = len([m for m in model.convs])
    # note that for the ratios, it affects the previous conv output and next
    # conv input, i.e., conv0 - ratio0 - conv1 - ratio1-...
    if isinstance(prune_ratio, list):
        assert len(prune_ratio) == n_conv - 1
    else:  # convert float to list
        prune_ratio = [prune_ratio] * (n_conv - 1)
    # we prune the convs in the backbone with a uniform ratio
    model = copy.deepcopy(model)  # prevent overwrite
    # we only apply pruning to the backbone features
    # all_convs = [m.conv for m in zip(model.convs, model.to_rgbs)]
    all_convs = [m.conv for m in model.convs]
    # all_bns = [m for m in model.backbone if isinstance(m, nn.BatchNorm2d)]
    # apply pruning. we naively keep the first k channels
    # assert len(all_convs) == len(all_bns)
    for i_ratio, p_ratio in enumerate(prune_ratio):
        prev_conv = all_convs[i_ratio]
        # prev_bn = all_bns[i_ratio]
        next_conv = all_convs[i_ratio + 1]
        original_channels = prev_conv.out_channel  # same as next_conv.in_channels
        n_keep = get_num_channels_to_keep(original_channels, p_ratio)
        # prune the output of the previous conv and bn
        prev_conv.weight.set_(prev_conv.weight.detach()[:n_keep])
        # prev_bn.weight.set_(prev_bn.weight.detach()[:n_keep])
        # prev_bn.bias.set_(prev_bn.bias.detach()[:n_keep])
        # prev_bn.running_mean.set_(prev_bn.running_mean.detach()[:n_keep])
        # prev_bn.running_var.set_(prev_bn.running_var.detach()[:n_keep])

        # prune the input of the next conv (hint: just one line of code)
        ##################### YOUR CODE STARTS HERE #####################
        next_conv.weight.set_(next_conv.weight.detach()[:, :n_keep])
        ##################### YOUR CODE ENDS HERE #####################

    return model

from torch.nn import parameter
from fast_pytorch_kmeans import KMeans
from collections import namedtuple

Codebook = namedtuple('Codebook', ['centroids', 'labels'])
def k_means_quantize(fp32_tensor: torch.Tensor, bitwidth=4, codebook=None):
    """
    quantize tensor using k-means clustering
    :param fp32_tensor:
    :param bitwidth: [int] quantization bit width, default=4
    :param codebook: [Codebook] (the cluster centroids, the cluster label tensor)
    :return:
        [Codebook = (centroids, labels)]
            centroids: [torch.(cuda.)FloatTensor] the cluster centroids
            labels: [torch.(cuda.)LongTensor] cluster label tensor
    """
    if codebook is None:
        ############### YOUR CODE STARTS HERE ###############
        # get number of clusters based on the quantization precision
        # hint: one line of code
        n_clusters = 2**bitwidth
        ############### YOUR CODE ENDS HERE #################
        # use k-means to get the quantization centroids
        kmeans = KMeans(n_clusters=n_clusters, mode='euclidean', verbose=0)
        labels = kmeans.fit_predict(fp32_tensor.view(-1, 1)).to(torch.long)
        centroids = kmeans.centroids.to(torch.float).view(-1)
        codebook = Codebook(centroids, labels)
    ############### YOUR CODE STARTS HERE ###############
    # decode the codebook into k-means quantized tensor for inference
    # hint: one line of code
    quantized_tensor = codebook.centroids[codebook.labels].view(fp32_tensor.size())

    ############### YOUR CODE ENDS HERE #################
    fp32_tensor.set_(quantized_tensor.view_as(fp32_tensor))
    return codebook

class KMeansQuantizer:
    def __init__(self, model : nn.Module, bitwidth=4):
        self.codebook = KMeansQuantizer.quantize(model, bitwidth)

    @torch.no_grad()
    def apply(self, model, update_centroids):
        for name, param in model.named_parameters():
            if name in self.codebook:
                if update_centroids:
                    update_codebook(param, codebook=self.codebook[name])
                self.codebook[name] = k_means_quantize(
                    param, codebook=self.codebook[name])

    @staticmethod
    @torch.no_grad()
    def quantize(model: nn.Module, bitwidth=4):
        codebook = dict()
        if isinstance(bitwidth, dict):
            for name, param in model.named_parameters():
                if name in bitwidth:
                    codebook[name] = k_means_quantize(param, bitwidth=bitwidth[name])
        else:
            for name, param in model.named_parameters():
                if param.dim() > 1:
                    codebook[name] = k_means_quantize(param, bitwidth=bitwidth)
        return codebook


def get_num_parameters(model: nn.Module, count_nonzero_only=False) -> int:
    """
    calculate the total number of parameters of model
    :param count_nonzero_only: only count nonzero weights
    """
    num_counted_elements = 0
    for param in model.parameters():
        if count_nonzero_only:
            num_counted_elements += param.count_nonzero()
        else:
            num_counted_elements += param.numel()
    return num_counted_elements


def get_model_size(model: nn.Module, data_width=32, count_nonzero_only=False) -> int:
    """
    calculate the model size in bits
    :param data_width: #bits per element
    :param count_nonzero_only: only count nonzero weights
    """
    return get_num_parameters(model, count_nonzero_only) * data_width


class WorkerSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)


class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        t1 = time.time()
        ret = self.fn(*self.args, **self.kwargs)
        t2 = time.time()
        self.signals.result.emit((ret, t2 - t1))


class FaceEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        # load assets
        self.load_assets()
        # title
        self.setWindowTitle('Face Editing with Anycost GAN')
        # window size
        # self.setGeometry(50, 50, 1000, 800)  # x, y, w, h
        self.setFixedSize(1000, 800)
        # background color
        # p = self.palette()
        # p.setColor(self.backgroundRole(), Qt.white)
        # self.setPalette(p)

        # plot the original image
        self.original_image = QLabel(self)
        self.set_img_location(self.original_image, 100, 72, 360, 360)
        pixmap = self.np2pixmap(self.org_image_list[0])
        self.original_image.setPixmap(pixmap)
        self.original_image_label = QLabel(self)
        self.original_image_label.setText('original')
        self.set_text_format(self.original_image_label)
        self.original_image_label.move(230, 42)

        # display the edited image
        self.edited_image = QLabel(self)
        self.set_img_location(self.edited_image, 540, 72, 360, 360)
        self.projected_image = self.generate_image()
        self.edited_image.setPixmap(self.projected_image)
        self.edited_image_label = QLabel(self)
        self.edited_image_label.setText('projected')
        self.set_text_format(self.edited_image_label)
        self.edited_image_label.move(670, 42)

        # build the sample list
        drop_list = QComboBox(self)
        drop_list.addItems(self.file_names)
        drop_list.currentIndexChanged.connect(self.select_image)
        drop_list.setGeometry(100, 490, 200, 30)
        drop_list.setCurrentIndex(0)
        drop_list_label = QLabel(self)
        drop_list_label.setText('* select sample:')
        self.set_text_format(drop_list_label, 'left', 15)
        drop_list_label.setGeometry(100, 470, 200, 30)

        # build editing sliders
        self.attr_sliders = dict()
        for i_slider, key in enumerate(self.direction_dict.keys()):
            tick_label = QLabel(self)
            tick_label.setText('|')
            self.set_text_format(tick_label, 'center', 10)
            tick_label.setGeometry(520 + 175, 470 + i_slider * 40 + 9, 50, 20)

            this_slider = QSlider(Qt.Horizontal, self)
            this_slider.setGeometry(520, 470 + i_slider * 40, 400, 30)
            this_slider.sliderReleased.connect(self.slider_update)
            this_slider.setMinimum(-100)
            this_slider.setMaximum(100)
            this_slider.setValue(0)
            self.attr_sliders[key] = this_slider

            attr_label = QLabel(self)
            attr_label.setText(key)
            self.set_text_format(attr_label, 'right', 15)
            attr_label.move(520 - 110, 470 + i_slider * 40 + 2)

        # build models sliders
        base_h = 560
        channel_label = QLabel(self)
        channel_label.setText('channel:')
        self.set_text_format(channel_label, 'left', 15)
        channel_label.setGeometry(100, base_h + 5, 100, 30)

        self.channel_slider = QSlider(Qt.Horizontal, self)
        self.channel_slider.setGeometry(180, base_h, 210, 30)
        self.channel_slider.sliderReleased.connect(self.model_update)
        self.channel_slider.setMinimum(0)
        self.channel_slider.setMaximum(3)
        self.channel_slider.setValue(3)
        for i, text in enumerate(['1/4', '1/2', '3/4', '1']):
            channel_label = QLabel(self)
            channel_label.setText(text)
            self.set_text_format(channel_label, 'center', 15)
            channel_label.setGeometry(180 + i * 63 - 50 // 2 + 10, base_h + 20, 50, 20)

        resolution_label = QLabel(self)
        resolution_label.setText('resolution:')
        self.set_text_format(resolution_label, 'left', 15)
        resolution_label.setGeometry(100, base_h + 55, 100, 30)

        self.resolution_slider = QSlider(Qt.Horizontal, self)
        self.resolution_slider.setGeometry(180, base_h + 50, 210, 30)
        self.resolution_slider.sliderReleased.connect(self.model_update)
        self.resolution_slider.setMinimum(0)
        self.resolution_slider.setMaximum(3)
        self.resolution_slider.setValue(3)
        for i, text in enumerate(['128', '256', '512', '1024']):
            resolution_label = QLabel(self)
            resolution_label.setText(text)
            self.set_text_format(resolution_label, 'center', 15)
            resolution_label.setGeometry(180 + i * 63 - 50 // 2 + 10, base_h + 70, 50, 20)

        # build button slider
        self.reset_button = QPushButton('Reset', self)
        self.reset_button.move(100, 700)
        self.reset_button.clicked.connect(self.reset_clicked)

        # build button slider
        self.finalize_button = QPushButton('Finalize', self)
        self.finalize_button.move(280, 700)
        from functools import partial
        self.finalize_button.clicked.connect(partial(self.slider_update, force_full_g=True))

        # add loading gif
        # create label
        self.loading_label = QLabel(self)
        self.loading_label.setGeometry(500 - 25, 240, 50, 50)

        self.loading_label.setObjectName("label")
        self.movie = QMovie(os.path.join(assets_dir, "loading.gif"))
        self.loading_label.setMovie(self.movie)
        self.movie.start()
        self.movie.setScaledSize(QSize(50, 50))
        self.loading_label.setVisible(False)

        # extra time stat
        self.time_label = QLabel(self)
        self.time_label.setText('')
        self.set_text_format(self.time_label, 'center', 18)
        self.time_label.setGeometry(500 - 25, 240, 50, 50)

        # status bar
        self.statusBar().showMessage('Ready.')

        # multi-thread
        self.thread_pool = QThreadPool()

        self.show()

    def load_assets(self):
        self.anycost_channel = 1.0
        self.anycost_resolution = 1024

        # build the generator
        self.generator = models.get_pretrained('generator', config).to(device)
        print(get_model_size(self.generator, count_nonzero_only=True))
        sparsity_dict = {
        'conv1.conv.weight': 0.2,
        'convs.0.conv.weight': 0.2,
        'convs.1.conv.weight': 0.2,
        'convs.2.conv.weight': 0.2,
        'convs.3.conv.weight': 0.2,
        'convs.4.conv.weight': 0.2,
        'convs.5.conv.weight': 0.2,
        'convs.6.conv.weight': 0.2,
        'convs.7.conv.weight': 0.2,
        'convs.8.conv.weight': 0.2,
        'convs.9.conv.weight': 0.2,
        'convs.10.conv.weight': 0.2,
        'convs.11.conv.weight': 0.2,
        'convs.12.conv.weight': 0.2,
        'convs.13.conv.weight': 0.2,
        'convs.14.conv.weight': 0.2,
        'convs.15.conv.weight': 0.2,
        }


        # def plot_num_parameters_distribution(model):
        #     num_parameters = dict()
        #     for name, param in model.named_parameters():
        #         if param.dim() > 1 and param.numel() > 10:
        #             num_parameters[name] = param.numel()
        #     fig = plt.figure(figsize=(8, 6))
        #     plt.grid(axis='y')
        #     plt.bar(list(num_parameters.keys()), list(num_parameters.values()))
        #     plt.title('#Parameter Distribution')
        #     plt.ylabel('Number of Parameters')
        #     plt.xticks(rotation=60)
        #     plt.tight_layout()
        #     plt.show()

        # plot_num_parameters_distribution(self.generator)
        # def plot_weight_distribution(model, bins=256, count_nonzero_only=False):
        #     fig, axes = plt.subplots(3,3, figsize=(10, 6))
        #     axes = axes.ravel()
        #     plot_index = 0
        #     for name, param in model.named_parameters():
        #         if param.dim() > 1 and name in sparsity_dict.keys() and plot_index < 9:
        #             ax = axes[plot_index]
        #             if count_nonzero_only:
        #                 param_cpu = param.detach().view(-1).cpu()
        #                 param_cpu = param_cpu[param_cpu != 0].view(-1)
        #                 ax.hist(param_cpu, bins=bins, density=True,
        #                         color = 'blue', alpha = 0.5)
        #             else:
        #                 ax.hist(param.detach().view(-1).cpu(), bins=bins, density=True,
        #                         color = 'blue', alpha = 0.5)
        #             ax.set_xlabel(name)
        #             ax.set_ylabel('density')
        #             plot_index += 1
        #     fig.suptitle('Histogram of Weights')
        #     fig.tight_layout()
        #     fig.subplots_adjust(top=0.925)
        #     plt.show()
        # plot_weight_distribution(self.generator, count_nonzero_only=True)
        #pruner = FineGrainedPruner(self.generator, sparsity_dict)
        
        # model_int8 = torch.quantization.quantize_dynamic(
        #                     self.generator,  # the original model
        #                     {torch.nn.Linear, StyledConv, EqualLinear, ModulatedConv2d, ToRGB, ConvLayer, ResBlock, EqualConv2d, torch.nn.functional.conv2d, torch.nn.LSTM, torch.nn.GRU},  # a set of layers to dynamically quantize
        #                     dtype=torch.qint8)
        # self.generator = model_int8
        # finetune_prune(5, callbacks=[lambda: pruner.apply(self.generator)])
        # for name, param in self.generator.named_parameters():
        #     print(name)
        #self.generator = channel_prune(self.generator, prune_ratio=0.1)
        #KMeansQuantizer(self.generator, 4)
        print(get_model_size(self.generator, count_nonzero_only=True))
        # plot_weight_distribution(self.generator, count_nonzero_only=True)
        self.generator.eval()
        self.mean_latent = self.generator.mean_style(10000)

        # select only a subset of the directions to use
        '''
        possible keys:
        ['00_5_o_Clock_Shadow', '01_Arched_Eyebrows', '02_Attractive', '03_Bags_Under_Eyes', '04_Bald', '05_Bangs',
         '06_Big_Lips', '07_Big_Nose', '08_Black_Hair', '09_Blond_Hair', '10_Blurry', '11_Brown_Hair', '12_Bushy_Eyebrows',
         '13_Chubby', '14_Double_Chin', '15_Eyeglasses', '16_Goatee', '17_Gray_Hair', '18_Heavy_Makeup', '19_High_Cheekbones',
         '20_Male', '21_Mouth_Slightly_Open', '22_Mustache', '23_Narrow_Eyes', '24_No_Beard', '25_Oval_Face', '26_Pale_Skin',
         '27_Pointy_Nose', '28_Receding_Hairline', '29_Rosy_Cheeks', '30_Sideburns', '31_Smiling', '32_Straight_Hair',
         '33_Wavy_Hair', '34_Wearing_Earrings', '35_Wearing_Hat', '36_Wearing_Lipstick', '37_Wearing_Necklace',
         '38_Wearing_Necktie', '39_Young']
        '''

        direction_map = {
            'smiling': '31_Smiling',
            'young': '39_Young',
            'wavy hair': '33_Wavy_Hair',
            'gray hair': '17_Gray_Hair',
            'blonde hair': '09_Blond_Hair',
            'eyeglass': '15_Eyeglasses',
            'mustache': '22_Mustache',
        }

        boundaries = models.get_pretrained('boundary', config)
        self.direction_dict = dict()
        for k, v in direction_map.items():
            self.direction_dict[k] = boundaries[v].view(1, 1, -1)

        # 3. prepare the latent code and original images
        file_names = sorted(os.listdir(os.path.join(assets_dir, 'input_images')))
        self.file_names = [f for f in file_names if f.endswith('.png') or f.endswith('.jpg')]
        self.latent_code_list = []
        self.org_image_list = []

        for fname in self.file_names:
            org_image = np.asarray(Image.open(os.path.join(assets_dir, 'input_images', fname)).convert('RGB'))
            latent_code = torch.from_numpy(
                np.load(os.path.join(assets_dir, 'projected_latents',
                                     fname.replace('.jpg', '.npy').replace('.png', '.npy'))))
            self.org_image_list.append(org_image)
            self.latent_code_list.append(latent_code.view(1, -1, 512))

        # set up the initial display
        self.sample_idx = 0
        self.org_latent_code = self.latent_code_list[self.sample_idx]

        # input kwargs for the generator
        self.input_kwargs = {'styles': self.org_latent_code, 'noise': None, 'randomize_noise': False,
                             'input_is_style': True}

    @staticmethod
    def np2pixmap(np_arr):
        height, width, channel = np_arr.shape
        q_image = QImage(np_arr.data, width, height, 3 * width, QImage.Format_RGB888)
        return QPixmap(q_image)

    @staticmethod
    def set_img_location(img_op, x, y, w, h):
        img_op.setScaledContents(True)
        img_op.setFixedSize(w, h)  # w, h
        img_op.move(x, y)  # x, y

    @staticmethod
    def set_text_format(text_op, align='center', font_size=18):
        if align == 'center':
            align = Qt.AlignCenter
        elif align == 'left':
            align = Qt.AlignLeft
        elif align == 'right':
            align = Qt.AlignRight
        else:
            raise NotImplementedError
        text_op.setAlignment(align)
        text_op.setFont(QFont('Arial', font_size))

    def select_image(self, idx):
        self.sample_idx = idx
        self.org_latent_code = self.latent_code_list[self.sample_idx]
        pixmap = self.np2pixmap(self.org_image_list[self.sample_idx])
        self.original_image.setPixmap(pixmap)
        self.input_kwargs['styles'] = self.org_latent_code
        self.projected_image = self.generate_image()
        self.edited_image.setPixmap(self.projected_image)
        self.reset_sliders()

    def reset_sliders(self):
        for slider in self.attr_sliders.values():
            slider.setValue(0)
        self.edited_image_label.setText('projected')
        self.statusBar().showMessage('Ready.')
        self.time_label.setText('')

    def generate_image(self):
        def image_to_np(x):
            assert x.shape[0] == 1
            x = x.squeeze(0).permute(1, 2, 0)
            x = (x + 1) * 0.5  # 0-1
            x = (x * 255).cpu().numpy().astype('uint8')
            return x

        with torch.no_grad():
            out = self.generator(**self.input_kwargs)[0].clamp(-1, 1)
            out = image_to_np(out)
            out = np.ascontiguousarray(out)
            return self.np2pixmap(out)

    def set_sliders_status(self, active):
        for slider in self.attr_sliders.values():
            slider.setEnabled(active)

    def slider_update(self, force_full_g=False):
        self.set_sliders_status(False)
        self.statusBar().showMessage('Running...')
        self.time_label.setText('')
        self.loading_label.setVisible(True)
        max_value = 0.6
        edited_code = self.org_latent_code.clone()
        for direction_name in self.attr_sliders.keys():
            edited_code[:, :n_style_to_change] = \
                edited_code[:, :n_style_to_change] \
                + self.attr_sliders[direction_name].value() * self.direction_dict[direction_name] / 100 * max_value
        self.input_kwargs['styles'] = edited_code
        if not force_full_g:
            set_uniform_channel_ratio(self.generator, self.anycost_channel)
            self.generator.target_res = self.anycost_resolution
        # generate the images in a separate thread
        worker = Worker(self.generate_image)
        worker.signals.result.connect(self.after_slider_update)
        self.thread_pool.start(worker)

    def after_slider_update(self, ret):
        edited, used_time = ret
        self.edited_image.setPixmap(edited)

        reset_generator(self.generator)
        self.edited_image_label.setText('edited')
        self.statusBar().showMessage('Done in {:.2f}s'.format(used_time))
        self.time_label.setText('{:.2f}s'.format(used_time))
        self.set_sliders_status(True)
        self.loading_label.setVisible(False)

    def model_update(self):
        self.anycost_channel = [0.25, 0.5, 0.75, 1.0][self.channel_slider.value()]
        self.anycost_resolution = [128, 256, 512, 1024][self.resolution_slider.value()]

    def reset_clicked(self):
        self.reset_sliders()
        self.edited_image.setPixmap(self.projected_image)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = FaceEditor()
    sys.exit(app.exec_())
