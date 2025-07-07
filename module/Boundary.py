import torch
import torch.nn as nn
import torch.nn.functional as F
from .Attention import NeighborhoodCrossAttention, unfold_neighborhood
from .MLPMixer import NeighborhoodMLPMixer

PI2 = torch.pi * 2

class BoundaryAttention(nn.Module):
    def __init__(self, C=3, dim_gamma=64, dim_pi=8, neighbor_size=11, ps=[3,9,17], device='cuda'):
        super().__init__()
        self.C = C
        self.dim_gamma = dim_gamma
        self.dim_pi = dim_pi
        out_dim = dim_gamma + dim_pi
        self.out_dim = out_dim
        self.neighbor_size = neighbor_size
        self.device = device
        self.ps = ps

        self.mlp1 = nn.Linear(out_dim, out_dim, device=device)
        self.mlp2 = nn.Linear(dim_gamma+C, out_dim, device=device)
        self.attn1 = NeighborhoodCrossAttention(out_dim, out_dim, neighborhood_size=neighbor_size, device=device)
        self.attn2 = NeighborhoodCrossAttention(out_dim, out_dim, neighborhood_size=neighbor_size, device=device)
        
        self.mlp_g = nn.Linear(dim_gamma, 7, device=device)
        self.mlp_p = nn.Linear(dim_pi, 3, device=device)

        self.u_cache = None
        self.theta_cache = None
        self.omega_cache = None

    def forward(self, f_t, gamma_0, gamma_t, pi_t):
        # print(torch.cuda.memory_allocated() / 1024 / 1024)
        gamma = gamma_0 + gamma_t
        
        q = self.mlp1(torch.cat([gamma, pi_t], dim=-1))
        kv = self.mlp2(torch.cat([gamma, f_t], dim=-1))

        q =F.gelu(q)
        kv = F.gelu(kv)

        kv = self.attn1(q, kv)
        gp = self.attn2(q, kv)
        
        gamma_next = gp[...,:self.dim_gamma]
        pi_next = gp[...,self.dim_gamma:]

        g = self.mlp_g(gamma_next)
        g[...,:4] = F.sigmoid(g[...,:4])
        g[...,4:] = F.softmax(g[...,4:], dim=-1)


        p = self.mlp_p(pi_next)
        p = F.softmax(p, dim=-1)
        # print(torch.cuda.memory_allocated() / 1024 / 1024)

        return g, p, gamma_next, pi_next
    
    def gather(self, g, p, x):
        B, H, W, C = x.shape

        u = g[..., :2]
        theta = g[..., 2:4]
        theta = torch.atan2(theta[..., 1], theta[..., 0])
        theta = theta.reshape(B, H, W, 1, 1)
        omega = g[..., 4:] * PI2
        omega = omega.unsqueeze(-1)
        omega_1 = omega[..., 0:1, :]
        omega_2 = omega[..., 1:2, :]

        self.u_cache = u
        self.theta_cache = theta
        self.omega_cache = omega

        w = torch.ones((B, H, W, self.neighbor_size, self.neighbor_size), device=self.device)

        D_max = max(self.ps)
        with torch.no_grad():
            row_clo = torch.arange(D_max, device=self.device, dtype=int) - (D_max - 1) // 2
            coordinates = torch.stack(list(torch.meshgrid(row_clo, row_clo, indexing='ij')), dim=-1)
            coordinates = coordinates.to(torch.float32)
            coordinates[..., 0] = coordinates[..., 0] / H
            coordinates[..., 1] = coordinates[..., 1] / W
            coordinates = coordinates.view(1, 1, 1, D_max, D_max, 2)
            coordinates = coordinates.to(self.device)
            coordinates = coordinates.expand(1, H, W, D_max, D_max, 2)
        
        cross_dir = coordinates - u.unsqueeze(-2).unsqueeze(-2)
        
        angles = torch.atan2(cross_dir[..., 0], cross_dir[..., 1])
        angles_diff = angles - theta
        angles_diff = torch.where(angles_diff < 0, angles_diff + PI2, angles_diff)

        thresh1 = omega_1
        thresh2 = omega_1 + omega_2
        mask1 = angles_diff <= thresh1
        mask2 = ~mask1 & (angles_diff <= thresh2)
        mask3 = ~(mask1 | mask2)

        s = torch.stack([mask1, mask2, mask3], dim=-1) # [B, H, W, D, D, S]
        
        windows = []
        p = p.reshape(B, H, W, 1, 1, 3)
        for i in range(len(self.ps)):
            with torch.no_grad():
                w0 = torch.ones((self.ps[i], self.ps[i]))
                D_diff_half = (D_max - self.ps[i]) // 2
                w_padded = F.pad(w0, (D_diff_half, D_diff_half, D_diff_half, D_diff_half))
                w = w_padded.view(1,1,1,D_max,D_max).expand(B, H, W, D_max,D_max).to(self.device)
            w = w * p[...,i]
            windows.append(w)

        w = sum(windows) # [B, H, W, D, D]
        we = w.reshape(B, H, W, D_max, D_max, 1, 1)
        se = s.reshape(B, H, W, D_max, D_max, 3, 1)
        
        # f_denominator [B, H, W, S]，S表示楔形支撑分区
        f_denominator = torch.sum(we * se, dim=(-4,-3))
        # f_numerator [B, H, W, S, C]，S表示楔形支撑分区
        f_numerator = torch.sum((x.reshape(B, H, W, 1, 1, 1, C) * we) * se, dim=(-4,-3))
        # f [B, H, W, S, C]，S表示楔形支撑分区
        f = f_numerator / (f_denominator + 1e-6)
        f[(f_denominator==0).squeeze(-1)] = 0
        return f, w, s

    def slice(self, f, w, s, requires_distance=False, requires_variance=False):
        B, H, W, S, C = f.shape
        D_max = max(self.ps)

        with torch.no_grad():
            row_clo = torch.arange(D_max, device=self.device, dtype=int) - (D_max - 1) // 2
            row_clo = -row_clo
            coordinates = torch.stack(list(torch.meshgrid(row_clo, row_clo, indexing='ij')), dim=-1)
            coordinates = coordinates.to(torch.float32)
            coordinates[..., 0] = coordinates[..., 0] / H
            coordinates[..., 1] = coordinates[..., 1] / W
            coordinates = coordinates.view(1, 1, 1, D_max, D_max, 2)
            coordinates = coordinates.to(self.device)
            coordinates = coordinates.expand(1, H, W, D_max, D_max, 2)

            mask = torch.ones((1, H, W, 1), device=self.device)
            mask_neighbor = unfold_neighborhood(mask, D_max).squeeze(-1)
            
        w_masked = w * mask_neighbor
        denominator = torch.sum(w_masked, dim=(-2,-1))
        
        if requires_distance:
            cross_dir = coordinates - self.u_cache.unsqueeze(-2).unsqueeze(-2)
            
            angles = torch.atan2(cross_dir[..., 0], cross_dir[..., 1])
            theta_1 = self.theta_cache
            theta_2 = (self.theta_cache + self.omega_cache[..., 0:1, :])
            theta_3 = (theta_2 + self.omega_cache[..., 1:2, :])
            
            theta_2 = torch.where(theta_2 > PI2, theta_2 - PI2, theta_2)
            theta_3 = torch.where(theta_3 > PI2, theta_3 - PI2, theta_3)
            
            diff_1 = torch.abs(angles - theta_1)
            diff_2 = torch.abs(angles - theta_2)
            diff_3 = torch.abs(angles - theta_3)

            min_diff = torch.min(torch.stack([diff_1, diff_2, diff_3], dim=-1), dim=-1).values
            dist = cross_dir.norm(dim=-1)

            d_k = dist * min_diff.sin()
            mask = d_k < 0
            d_k[mask] = dist[mask]
            d_predict = (d_k * w_masked).sum(dim=(-2,-1)) / denominator
        else:
            d_predict = None

        f_neighbor = unfold_neighborhood(f.reshape(B, H, W, -1), D_max).view(B, H, W, D_max, D_max, 3, C).sum(-2)

        f = f_neighbor * w_masked.unsqueeze(-1)
        f = f.sum(dim=(-3,-2)) / denominator.unsqueeze(-1)

        if requires_variance and requires_distance:
            var_f = f.var(dim=(1,2))
            b = 1 / (1 + d_predict)
            var_b = b.var(dim=(1,2))
        else:
            var_f = None
            var_b = None
            
        return f, d_predict, var_f, var_b


class BoundaryAttentionModule(nn.Module):
    REPEAT = 4

    def __init__(self, dim_hidden, dim_gamma, dim_pi, neighbor_size=11, device='cuda'):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.dim_gamma = dim_gamma
        self.dim_pi = dim_pi
        out_dim = dim_gamma + dim_pi
        self.out_dim = out_dim
        self.neighbor_size = neighbor_size
        self.device = device

        self.mlp_mixer = NeighborhoodMLPMixer(embed_dim=dim_gamma, device=device)
        
        self.annention1 = BoundaryAttention(dim_gamma=dim_gamma, dim_pi=dim_pi, neighbor_size=neighbor_size, device=device)
        self.annention2 = BoundaryAttention(dim_gamma=dim_gamma, dim_pi=dim_pi, neighbor_size=neighbor_size, device=device)
        
        self.pi_0 = nn.Parameter(torch.randn((1,1,1,dim_pi), device=device))

        self.d_cache = None
        self.varf_cache = None
        self.varb_cache = None
        self.f_cache = None
        self.b_cache = None

    def forward(self, x):
        B, H, W, C = x.shape

        gamma_0 = self.mlp_mixer(x)
       
        gamma_t = gamma_0
        f_t = x
        pi_t = self.pi_0.expand(B, H, W, self.dim_pi)

        for _ in range(self.REPEAT):
            g, p, gamma_t, pi_t = self.annention1(f_t, gamma_0, gamma_t, pi_t)
            f, w, s = self.annention1.gather(g, p, x)
            f_t, _, _, _ = self.annention1.slice(f, w, s)
        for _ in range(self.REPEAT-1):
            g, p, gamma_t, pi_t = self.annention2(f_t, gamma_0, gamma_t, pi_t)
            f, w, s = self.annention2.gather(g, p, x)
            f_t, _, _, _ = self.annention2.slice(f, w, s)
        g, p, gamma_t, pi_t = self.annention2(f_t, gamma_0, gamma_t, pi_t)
        f, w, s = self.annention2.gather(g, p, x)
        if self.training:
            f_t, d, var_f, var_b = self.annention2.slice(f, w, s, True, True)
            self.d_cache = d
            self.varf_cache = var_f
            self.varb_cache = var_b
        else:
            f_t, d, _, _ = self.annention1.slice(f, w, s, True)
        x = f_t
        return x, d
    
    def loss(self, lable_f, lable_d, pred_f, pred_d):
        B, H, W = lable_d.shape
        beta, delta, C = 0.1, 1, 0.3
        alpha = torch.exp(-beta * (lable_d + delta)) + C
        alpha.requires_grad_(False)
        
        loss_f_spi = alpha.unsqueeze(-1) * (pred_f - lable_f).pow(2)
        loss_d_spi = alpha * (pred_d - lable_d).pow(2)

        loss_f = loss_f_spi.sum()
        loss_d = loss_d_spi.sum()
        
        # loss_var = self.varb_cache + self.varf_cache.sum(dim=-1)
        # loss_var = loss_var.sum()
        loss_var = (alpha.unsqueeze(-1) * pred_f).var(dim=(1,2)).sum() + (alpha * pred_d).var(dim=(1,2)).sum()

        d_lable_neighbor = unfold_neighborhood(lable_d.unsqueeze(-1), self.neighbor_size)
        _1_chi = d_lable_neighbor.mean(dim=(-1,-2,-3))
        _1_chi.requires_grad_(False)

        loss_f_neighbor = unfold_neighborhood(loss_f_spi, self.neighbor_size)
        var_loss_f = loss_f_neighbor.var(dim=(-1,-2,-3)) / _1_chi
        var_loss_f = var_loss_f.sum()
        loss_d_neighbor = unfold_neighborhood(loss_d_spi.unsqueeze(-1), self.neighbor_size)
        var_loss_d = loss_d_neighbor.var(dim=(-1,-2,-3)) / _1_chi
        var_loss_d = var_loss_d.sum()

        total_loss = loss_f + loss_d + loss_var + var_loss_f + var_loss_d
        total_loss = total_loss / (B * H * W)

        return total_loss
    
    def has_nan(self):
        for param in self.parameters():
            if  torch.isnan(param).any():
                return True
        return False

    

# if __name__ == "__main__":
#     model = BoundaryAttention(3, 64, 8, device='cuda').eval()
#     shape = (2, 32, 32, 3)
#     x = torch.randn(shape, device='cuda')
#     gamma = torch.randn((2, 32, 32, 64), device='cuda')
#     pi = torch.randn((2, 32, 32, 8), device='cuda')
#     g, p, gamma_next, pi_next = model(x, gamma, gamma, pi)
#     f, w, s = model.gather(g, p, x)
#     f, d, var_f, var_b = model.slice(f, w, s,requires_variance=True)
#     torch.cuda.empty_cache()

if __name__ == "__main__":
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    model = BoundaryAttentionModule(3, 64, 8, device='cuda')
    shape = (2, 32, 32, 3)
    x = torch.randn(shape, device='cuda')
    pred_x, pred_d = model(x)

    # lable_f = torch.randn((2, 32, 32, 3), device='cuda')
    # lable_d = torch.randn((2, 32, 32), device='cuda')
    # loss_f, loss_d, loss_var = model.loss(lable_f, lable_d, pred_x, pred_d)
    