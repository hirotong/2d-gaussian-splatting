import torch
import torch.nn as nn
from arguments import OptimizationParams
import os
from utils.sh_utils import eval_sh_coef
from utils.general_utils import build_rotation


class DirectLightEnv:
    def __init__(self, num_shs, sh_degree):
        self.num_shs = num_shs
        self.sh_degree = sh_degree

        env_shs = torch.rand((num_shs, 3, (self.sh_degree + 1) ** 2)).float().cuda()
        self.env_shs_dc = nn.Parameter(env_shs[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self.env_shs_rest = nn.Parameter(env_shs[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))

        rots = torch.rand((num_shs, 4), device="cuda")
        # rots[:, 0] = 1
        self._rotation = nn.Parameter(rots.requires_grad_(True))

        self.rotation_activation = torch.nn.functional.normalize

    @property
    def get_env_shs(self):
        shs_dc = self.env_shs_dc
        shs_rest = self.env_shs_rest
        return torch.cat((shs_dc, shs_rest), dim=1)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    def training_setup(self, training_args: OptimizationParams):
        if training_args.env_rest_lr < 0:
            training_args.env_rest_lr = training_args.env_lr / 20.0
        l = [
            {"params": [self.env_shs_dc], "lr": training_args.env_lr, "name": "env_dc"},
            {"params": [self.env_shs_rest], "lr": training_args.env_rest_lr, "name": "env_rest"},
            {"params": [self._rotation], "lr": training_args.env_rotation_lr, "name": "env_rotations"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    def capture(self):
        captured_list = [
            self.num_shs,
            self.sh_degree,
            self._rotation,
            self.env_shs_dc,
            self.env_shs_rest,
            self.optimizer.state_dict(),
        ]

        return captured_list

    def restore(self, model_args, training_args, is_training=False, restore_optimizer=True):
        pass

    def create_from_ckpt(self, checkpoint_path, restore_optimizer=False):
        (model_args, first_iter) = torch.load(checkpoint_path)
        (self.num_shs, self.sh_degree, self._rotation, self.env_shs_dc, self.env_shs_rest, opt_dict) = model_args[:6]

        if restore_optimizer:
            try:
                self.optimizer.load_state_dict(opt_dict)
            except:
                print("Not loading optimizer state_dict!")

        return first_iter

    def save_ckpt(self, model_path, iteration):
        print("\n[ITER {}] Saving Global Lighting Checkpoint".format(iteration))
        torch.save((self.capture(), iteration), os.path.join(model_path, f"env_light_ckpt_{iteration}.pth"))

    # TODO: Saving environment light image
    def save_lighting(self):
        pass

    # @torch.no_grad()
    def render_envmap(self, H, W, upper_hemi=False):
        if upper_hemi:
            theta, phi = torch.meshgrid(
                torch.linspace(0.0 + 0.5 / H, 0.5 - 0.5 / H, H),
                torch.linspace(-1.0 + 1.0 / W, 1.0 - 1.0 / W, W),
                indexing="ij",
            )  # H, W
        else:
            theta, phi = torch.meshgrid(
                torch.linspace(0.0 + 1.0 / H, 1.0 - 1.0 / H, H),
                torch.linspace(-1.0 + 1.0 / W, 1.0 - 1.0 / W, W),
                indexing="ij",
            )

        sintheta, costheta = torch.sin(theta * torch.pi), torch.cos(theta * torch.pi)
        sinphi, cosphi = torch.sin(phi * torch.pi), torch.cos(phi * torch.pi)

        viewdirs = torch.stack([sintheta * cosphi, sintheta * sinphi, costheta], dim=-1)  # H, W, 3
        viewdirs = viewdirs.view(-1, 3).cuda()

        rotated_viewdirs = viewdirs.unsqueeze(0) @ build_rotation(self.get_rotation).transpose(
            1, 2
        )  # (n_shs, H * W, 3)

        env_shs_coef = eval_sh_coef(self.sh_degree, rotated_viewdirs)  # (n_shs, H * W, (sh_degree + 1) ** 2)
        # env_shs_coef = eval_sh_coef(self.sh_degree, viewdirs).unsqueeze(1)    # (H * W, (sh_degree + 1) ** 2)
        env_shs = self.get_env_shs.transpose(1, 2).contiguous()  # (n_shs, 3, (sh_degree + 1) ** 2)

        env_light = torch.clamp_min((env_shs_coef.unsqueeze(-2) * env_shs.unsqueeze(1)).sum(-1).sum(0) + 0.5, 0.0)
        # env_light = torch.clamp_min((env_shs_coef * env_shs).sum(-1) + 0.5, 0.0)
        env_light = env_light.view(H, W, 3)

        return env_light
