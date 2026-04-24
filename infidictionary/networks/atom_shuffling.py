import torch
from torch import nn

from .base import TimeEvolvingField, NerfFourierFeatures, _build_mlp
from .time_embedding import SinusoidalTimeEmbedding
from infidictionary.dictionaries import FourierDictionary

# TODO: maybe try channel selectors for multi-channel stuff
class AtomShufflingField(TimeEvolvingField):
    def __init__(
        self,
        coords_dim: int,
        output_dim: int,
        n_truncation: int,
        top_k: int | None,
        time_hidden_dims: tuple = (64, 64),
        spatial_hidden_dims: tuple = (256, 256, 256),   
        n_time_freqs: int = 8,
        activation=nn.SiLU,
        nerf_n_levels: int = 8,
        nerf_n_base: int = 64,
        nerf_freq_min: float = 1.0,
        nerf_freq_max: float = 64.0,
        residual_weight: float = 0.1,
    ):
        super().__init__(input_dim=coords_dim, output_dim=output_dim)

        self.residual_weight = residual_weight
        self.C = output_dim
        self.d = coords_dim
        self.top_k = top_k

        self.fourier_dict = FourierDictionary(
            domain_dim=coords_dim,
            num_channels=output_dim,
            p=0.3,
        )
        indices_of_interest = self.fourier_dict.get_truncated_indices(n_truncation)
        n_atoms = len(indices_of_interest)
        self.register_buffer("indices", indices_of_interest)

        # Time embedding
        self.time_embedding = SinusoidalTimeEmbedding(n_time_freqs)
        emb_dim = self.time_embedding.out_dim

        # Alpha MLP
        self.alpha_feature_selector = _build_mlp(
            emb_dim,
            self.C * n_atoms,
            time_hidden_dims,
            activation,
            use_batchnorm=False,
            use_rmsnorm=True,
            bias=True,
        )

        # Beta MLP
        self.beta_feature_selector = _build_mlp(
            emb_dim,
            self.C * n_atoms,
            time_hidden_dims,
            activation,
            use_batchnorm=False,
            use_rmsnorm=True,
            bias=True,
        )

        # creative residual
        self.nerf_features = NerfFourierFeatures(
            coords_dim, nerf_n_levels, nerf_n_base, nerf_freq_min, nerf_freq_max,
        )
        ff_spatial_dim = coords_dim + self.nerf_features.out_dim  # cat(x, nerf(x))

        self.ff_alpha = _build_mlp(
            emb_dim + ff_spatial_dim,
            output_dim,
            spatial_hidden_dims,
            activation,
            use_batchnorm=False,
            use_rmsnorm=True,
            bias=False,
        )

        self.ff_beta = _build_mlp(
            emb_dim + ff_spatial_dim,
            output_dim,
            spatial_hidden_dims,
            activation,
            use_batchnorm=False,
            use_rmsnorm=True,
            bias=False,
        )


    def forward(self, t: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # t: (N,)   x: (N, d)   Time query: (N, emb_dim)
        N = x.shape[0]

        t_emb = self.time_embedding(t)
        
        # --- Alpha: Attention-weighted sum of initial atoms ---
        with torch.no_grad():
            # Evaluate the base atoms at the given coordinates x
            # Shape: (n_atoms, N, C)
            all_features = self.fourier_dict.get_atoms(x.detach(), self.indices).detach()
            all_features = all_features.permute(1, 0, 2)  # (N, n_atoms, C)

        sel_alpha = self.alpha_feature_selector(t_emb).view(N, -1, self.C)  # (N, n_atoms, C)
        if self.top_k is not None:
            topk_vals, topk_indices = torch.topk(sel_alpha, self.top_k, dim=1)
            sparse_scores = torch.full_like(sel_alpha, float('-inf'))
            sparse_scores.scatter_(dim=1, index=topk_indices, src=topk_vals)
            mask_alpha = torch.softmax(sparse_scores, dim=1)  # (N, n_atoms, C), softmax over the n_atoms dimension
            alpha = torch.sum(mask_alpha * all_features, dim=1)  # (N, C), weighted sum of atom features
        else:
            alpha = torch.sum(sel_alpha * all_features, dim=1)  # (N, C), weighted sum of atom features
        alpha_res = self.ff_alpha(torch.cat([t_emb, x, self.nerf_features(x)], dim=-1))
        if self.C > 1:
            alpha_res = alpha_res / (0.1 + alpha_res.norm(dim=-1, keepdim=True))  # normalize residual to prevent explosion
        alpha = alpha + self.residual_weight * alpha_res

        sel_beta = self.beta_feature_selector(t_emb).view(N, -1, self.C)  # (N, n_atoms, C)
        if self.top_k is not None:
            topk_vals_beta, topk_indices_beta = torch.topk(sel_beta, self.top_k, dim=1)
            sparse_scores_beta = torch.full_like(sel_beta, float('-inf'))
            sparse_scores_beta.scatter_(dim=1, index=topk_indices_beta, src=topk_vals_beta)
            mask_beta = torch.softmax(sparse_scores_beta, dim=1)  # (N, n_atoms, C), softmax over the n_atoms dimension
            beta = torch.sum(mask_beta * all_features, dim=1)  # (N, C), weighted sum of atom features
        else:
            beta = torch.sum(sel_beta * all_features, dim=1)  # (N, C), weighted sum of atom features
        beta_res = self.ff_beta(torch.cat([t_emb, x, self.nerf_features(x)], dim=-1))
        if self.C > 1:
            beta_res = beta_res / (0.1 + beta_res.norm(dim=-1, keepdim=True))  # normalize residual to prevent explosion
        beta = beta + self.residual_weight * beta_res
        
        return alpha, beta
