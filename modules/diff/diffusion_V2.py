from collections import deque
from functools import partial

import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Conv1d
from modules.commons.common_layers import Mish
from modules.encoder import SvcEncoder
from utils.hparams import hparams


def exists(x):
    return x is not None


def extract(a, t):
    return a[t].reshape((1, 1, 1, 1))


def linear_beta_schedule(timesteps, max_beta=hparams.get('max_beta', 0.01)):
    betas = np.linspace(1e-4, max_beta, timesteps)
    return betas


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)


beta_schedule = {
    "cosine": cosine_beta_schedule,
    "linear": linear_beta_schedule,
}


def extract_1(a, t):
    return a[t].reshape((1, 1, 1, 1))


def predict_stage0(noise_pred, noise_pred_prev):
    return (noise_pred + noise_pred_prev) / 2


def predict_stage1(noise_pred, noise_list):
    return (noise_pred * 3
            - noise_list[-1]) / 2


def predict_stage2(noise_pred, noise_list):
    return (noise_pred * 23
            - noise_list[-1] * 16
            + noise_list[-2] * 5) / 12


def predict_stage3(noise_pred, noise_list):
    return (noise_pred * 55
            - noise_list[-1] * 59
            + noise_list[-2] * 37
            - noise_list[-3] * 9) / 24


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.half_dim = dim // 2
        self.emb = 9.21034037 / (self.half_dim - 1)
        self.emb = torch.exp(torch.arange(self.half_dim) * torch.tensor(-self.emb)).unsqueeze(0)
        self.emb = self.emb.cpu()

    def forward(self, x):
        emb = self.emb * x
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ResidualBlock(nn.Module):
    def __init__(self, encoder_hidden, residual_channels, dilation):
        super().__init__()
        self.residual_channels = residual_channels
        self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
        self.diffusion_projection = nn.Linear(residual_channels, residual_channels)
        self.conditioner_projection = Conv1d(encoder_hidden, 2 * residual_channels, 1)
        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, conditioner, diffusion_step):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        conditioner = self.conditioner_projection(conditioner)
        y = x + diffusion_step
        y = self.dilated_conv(y) + conditioner

        gate, filter_1 = torch.split(y, [self.residual_channels, self.residual_channels], dim=1)

        y = torch.sigmoid(gate) * torch.tanh(filter_1)
        y = self.output_projection(y)

        residual, skip = torch.split(y, [self.residual_channels, self.residual_channels], dim=1)

        return (x + residual) / 1.41421356, skip


class DiffNet(nn.Module):
    def __init__(self, in_dims=80):
        super().__init__()
        self.encoder_hidden = hparams['hidden_size']
        self.residual_layers = hparams['residual_layers']
        self.residual_channels = hparams['residual_channels']
        self.dilation_cycle_length = hparams['dilation_cycle_length']
        self.input_projection = Conv1d(in_dims, self.residual_channels, 1)
        self.diffusion_embedding = SinusoidalPosEmb(self.residual_channels)
        dim = self.residual_channels
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            Mish(),
            nn.Linear(dim * 4, dim)
        )
        self.residual_layers = nn.ModuleList([
            ResidualBlock(self.encoder_hidden, self.residual_channels, 2 ** (i % self.dilation_cycle_length))
            for i in range(self.residual_layers)
        ])
        self.skip_projection = Conv1d(self.residual_channels, self.residual_channels, 1)
        self.output_projection = Conv1d(self.residual_channels, in_dims, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, spec, diffusion_step, cond):
        x = spec.squeeze(0)
        x = self.input_projection(x)  # x [B, residual_channel, T]
        x = F.relu(x)
        # skip = torch.randn_like(x)
        diffusion_step = diffusion_step.float()
        diffusion_step = self.diffusion_embedding(diffusion_step)
        diffusion_step = self.mlp(diffusion_step)

        x, skip = self.residual_layers[0](x, cond, diffusion_step)
        # noinspection PyTypeChecker
        for layer in self.residual_layers[1:]:
            x, skip_connection = layer.forward(x, cond, diffusion_step)
            skip = skip + skip_connection
        x = skip / math.sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)  # [B, 80, T]
        return x.unsqueeze(1)


class AfterDiffusion(nn.Module):
    def __init__(self, spec_max, spec_min):
        super().__init__()
        self.spec_max = spec_max
        self.spec_min = spec_min

    def forward(self, x):
        x = x.squeeze(1).permute(0, 2, 1).cpu()
        d = (self.spec_max - self.spec_min) / 2
        m = (self.spec_max + self.spec_min) / 2
        mel_out = x * d.cpu() + m.cpu()
        mel_out = mel_out * 2.30259
        return mel_out.transpose(2, 1)


class Pred(nn.Module):
    def __init__(self, alphas_cumprod):
        super().__init__()
        self.alphas_cumprod = alphas_cumprod

    def forward(self, x_1, noise_t, t_1, t_prev):
        a_t = extract(self.alphas_cumprod, t_1).cpu()
        a_prev = extract(self.alphas_cumprod, t_prev).cpu()
        a_t_sq, a_prev_sq = a_t.sqrt().cpu(), a_prev.sqrt().cpu()
        x_delta = (a_prev - a_t) * ((1 / (a_t_sq * (a_t_sq + a_prev_sq))) * x_1 - 1 / (
                a_t_sq * (((1 - a_prev) * a_t).sqrt() + ((1 - a_t) * a_prev).sqrt())) * noise_t)
        x_pred = x_1 + x_delta.cpu()
        return x_pred


class GaussianDiffusionOnnx(nn.Module):
    def __init__(self, phone_encoder, out_dims, denoise_fn,
                 timesteps=1000, K_step=1000, loss_type=hparams.get('diff_loss_type', 'l1'), betas=None, spec_min=None,
                 spec_max=None):
        super().__init__()
        self.denoise_fn = DiffNet(out_dims)
        self.fs2 = SvcEncoder(phone_encoder, out_dims)
        self.mel_bins = out_dims

        if exists(betas):
            betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        else:
            if 'schedule_type' in hparams.keys():
                betas = beta_schedule[hparams['schedule_type']](timesteps)
            else:
                betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.K_step = K_step
        self.loss_type = loss_type

        self.noise_list = deque(maxlen=4)

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        self.register_buffer('spec_min', torch.FloatTensor(spec_min)[None, None, :hparams['keep_bins']])
        self.register_buffer('spec_max', torch.FloatTensor(spec_max)[None, None, :hparams['keep_bins']])
        self.mel_vmin = hparams['mel_vmin']
        self.mel_vmax = hparams['mel_vmax']

        self.ad = AfterDiffusion(self.spec_max, self.spec_min)
        self.xp = Pred(self.alphas_cumprod)

    def get_x_pred(self, x_1, noise_t, t_1, t_prev):
        a_t = extract(self.alphas_cumprod, t_1)
        a_prev = extract(self.alphas_cumprod, t_prev)
        a_t_sq, a_prev_sq = a_t.sqrt(), a_prev.sqrt()
        x_delta = (a_prev - a_t) * ((1 / (a_t_sq * (a_t_sq + a_prev_sq))) * x_1 - 1 / (
                a_t_sq * (((1 - a_prev) * a_t).sqrt() + ((1 - a_t) * a_prev).sqrt())) * noise_t)
        x_pred = x_1 + x_delta
        return x_pred

    def OnnxExport(self, project_name=None):
        Onnx=True

        hubert = torch.rand(1, 10, 256)
        f0 = torch.rand(1, 10)
        mel2ph = torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).unsqueeze(0)
        spk_embed = torch.LongTensor([0])

        torch.onnx.export(
            self.fs2,
            (hubert, mel2ph, spk_embed, f0),
            f"{project_name}_encoder.onnx",
            input_names=["hubert", "mel2ph", "spk_embed", "f0"],
            output_names=["mel_pred", "f0_pred"],
            dynamic_axes={
                "hubert": [1],
                "f0": [1],
                "mel2ph": [1]
            },
            opset_version=16
        )

        cond = torch.randn([1, 256, 10]).cpu()
        x = torch.randn((1, 1, self.mel_bins, cond.shape[2]), dtype=torch.float32).cpu()
        pndms = 100

        device = cond.device
        n_frames = cond.shape[2]
        step_range = torch.arange(0, self.K_step, pndms, dtype=torch.long, device=device).flip(0)
        plms_noise_stage = torch.tensor(0, dtype=torch.long, device=device)
        noise_list = torch.zeros((0, 1, 1, self.mel_bins, n_frames), device=device)

        ot = step_range[0]
        ot_1 = torch.full((1,), ot, device=device, dtype=torch.long)
        torch.onnx.export(
            self.denoise_fn,
            (x.cpu(), ot_1.cpu(), cond.cpu()),
            f"{project_name}_denoise.onnx",
            input_names=["noise", "time", "condition"],
            output_names=["noise_pred"],
            dynamic_axes={
                "noise": [3],
                "condition": [2]
            },
            opset_version=16
        )

        for t in step_range:
            t_1 = torch.full((1,), t, device=device, dtype=torch.long)
            noise_pred = self.denoise_fn(x, t_1, cond)
            t_prev = t_1 - pndms
            t_prev = t_prev * (t_prev > 0)
            if plms_noise_stage == 0:
                torch.onnx.export(
                    self.xp,
                    (x.cpu(), noise_pred.cpu(), t_1.cpu(), t_prev.cpu()),
                    f"{project_name}_pred.onnx",
                    input_names=["noise", "noise_pred", "time", "time_prev"],
                    output_names=["noise_pred_o"],
                    dynamic_axes={
                        "noise": [3],
                        "noise_pred": [3]
                    },
                    opset_version=16
                )

                x_pred = self.get_x_pred(x, noise_pred, t_1, t_prev)
                noise_pred_prev = self.denoise_fn(x_pred, t_prev, cond=cond)
                noise_pred_prime = predict_stage0(noise_pred, noise_pred_prev)

            elif plms_noise_stage == 1:
                noise_pred_prime = predict_stage1(noise_pred, noise_list)

            elif plms_noise_stage == 2:
                noise_pred_prime = predict_stage2(noise_pred, noise_list)

            else:
                noise_pred_prime = predict_stage3(noise_pred, noise_list)

            noise_pred = noise_pred.unsqueeze(0)

            if plms_noise_stage < 3:
                noise_list = torch.cat((noise_list, noise_pred), dim=0)
                plms_noise_stage = plms_noise_stage + 1

            else:
                noise_list = torch.cat((noise_list[-2:], noise_pred), dim=0)

            x = self.get_x_pred(x, noise_pred_prime, t_1, t_prev)

        torch.onnx.export(
            self.ad,
            x.cpu(),
            f"{project_name}_after.onnx",
            input_names=["x"],
            output_names=["mel_out"],
            dynamic_axes={
                "x": [3]
            },
            opset_version=16
        )
