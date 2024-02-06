import torch
import torch.nn.functional as F
import numpy as np
import math
from torch_scatter import scatter_add
from src import utils
from src.egnn import Dynamics
from src.noise import GammaNetwork, PredefinedNoiseSchedule
from typing import Union



class EDM(torch.nn.Module):
    def __init__(
            self,
            dynamics: Union[Dynamics],
            in_node_nf: int,
            n_dims: int,
            timesteps: int = 1000,
            noise_schedule='learned',
            noise_precision=1e-4,
            loss_type='vlb',
            norm_values=(1., 1., 1.),
            norm_biases=(None, 0., 0.),
    ):
        super().__init__()
        if noise_schedule == 'learned':
            assert loss_type == 'vlb', 'A noise schedule can only be learned with a vlb objective'
            self.gamma = GammaNetwork()
        else:
            self.gamma = PredefinedNoiseSchedule(noise_schedule, timesteps=timesteps, precision=noise_precision)

        self.dynamics = dynamics
        self.in_node_nf = in_node_nf
        self.n_dims = n_dims
        self.T = timesteps
        self.norm_values = norm_values
        self.norm_biases = norm_biases

    def noised_representation(self,xh,ligand_diff,context,batch_seg,gamma_t):
        alpha_t = self.alpha(gamma_t)
        sigma_t = self.sigma(gamma_t)
        eps_t = self.sample_combined_position_feature_noise(xh,ligand_diff)
        z_t = alpha_t[batch_seg] * xh + sigma_t[batch_seg] * eps_t
        z_t = xh * context + z_t * ligand_diff
        return z_t,eps_t
    
    def forward(self, x, h,  context, ligand_diff, batch_seg,batch_size, ligand_group=None):
        # Normalization and concatenation
        x, h = self.normalize(x, h)
        xh = torch.cat([x, h], dim=1)
        delta_log_px=(self.n_dims*self.inflate_batch_array(ligand_diff,batch_seg)*np.log(self.norm_values[0]))
        # Sample t
        lowest_t=0 if self.training else 1
        t_int = torch.randint(lowest_t, self.T + 1, size=(batch_size, 1), device=x.device).float()
        s_int = t_int - 1
        t = t_int / self.T
        s = s_int / self.T

        # Masks for t=0 and t>0
        t_is_zero = (t_int == 0).float()
        t_is_not_zero = 1 - t_is_zero
        

        # Compute gamma_t and gamma_s according to the noise schedule
        gamma_t = self.gamma(t)
        gamma_s = self.gamma(s)
        z_t,eps_t=self.noised_representation(xh,ligand_diff,context,batch_seg,gamma_t)
        # Neural net prediction
        eps_t_hat = self.dynamics.forward(
            xh=z_t,
            t=t,
            ligand_diff=ligand_diff,
            ligand_group=ligand_group,
            batch_seg=batch_seg,
        )

        eps_t_hat = eps_t_hat * ligand_diff
        # Computing basic error (further used for computing NLL and L2-loss)
        squared_error=(eps_t-eps_t_hat)**2
        error_t=self.inflate_batch_array(squared_error,batch_seg)
        SNR_weight = (self.SNR(gamma_s - gamma_t) - 1).squeeze(1)
        assert error_t.size() == SNR_weight.size()
        # The _constants_ depending on sigma_0 from the
        # cross entropy term E_q(z0 | x) [log p(x | z0)]
        neg_log_constants = -self.log_constant_of_p_x_given_z0(ligand_diff,batch_seg,batch_size)
        # The KL between q(z_T | x) and p(z_T) = Normal(0, 1) (should be close to zero)
        kl_prior = self.kl_prior(xh, ligand_diff,batch_seg)
        if self.training:
            # Computes the L_0 term (even if gamma_t is not actually gamma_0)
            # and selected only relevant via masking
            log_p_x_given_z0_without_constants,log_ph_given_z0 = self.log_p_xh_given_z0_without_constants(h, z_t, gamma_t, eps_t, eps_t_hat, ligand_diff,batch_seg)
            loss_0_x = -log_p_x_given_z0_without_constants * t_is_zero.squeeze()               
            loss_0_h = -log_ph_given_z0 * t_is_zero.squeeze()
            #apply t_is_zero mask
            error_t=error_t*t_is_not_zero.squeeze()  
        else:
            # Computes noise values for t=0
            t_zeros=torch.zeros_like(s)
            gamma_0=self.gamma(t_zeros)
            z_0,eps_0=self.noised_representation(xh,ligand_diff,context,batch_seg,gamma_0)
            eps_0_hat = self.dynamics.forward(z_0,t_zeros,  ligand_diff, ligand_group,batch_seg )
            eps_0_hat = eps_0_hat * ligand_diff
            log_p_x_given_z0_without_constants, log_ph_given_z0 = \
                self.log_p_xh_given_z0_without_constants(h, z_0, gamma_0, eps_0, eps_0_hat, ligand_diff,batch_seg)
            loss_0_x = -log_p_x_given_z0_without_constants
            loss_0_h = -log_ph_given_z0

        loss_terms = (
            delta_log_px, error_t, SNR_weight,
            loss_0_x, loss_0_h, neg_log_constants,
            kl_prior
        )
        return loss_terms
    
    
    def sample_normal(self,mu_xh,ligand_diff,sigma,batch_seg):
        eps=self.sample_combined_position_feature_noise(mu_xh,ligand_diff)
        out_xh=mu_xh+sigma[batch_seg]*eps
        return out_xh




    @torch.no_grad()
    def sample_chain(self, x, h, context, ligand_diff, batch_seg,batch_size, ligand_group, keep_frames=None,timesteps=None):
        timesteps = self.T if timesteps is None else timesteps
        assert 0 < keep_frames <= timesteps
        assert timesteps % keep_frames == 0
        
        x, h, = self.normalize(x, h)
        xh = torch.cat([x, h], dim=1)
        mu_x=scatter_add(x*context, batch_seg, dim=0)/scatter_add(context, batch_seg, dim=0)
        mu_h=torch.zeros((batch_size,self.in_node_nf),device=x.device)
        mu_xh=torch.cat([mu_x,mu_h],dim=1)[batch_seg]
        sigma=torch.ones((batch_size,1),device=x.device)
        z=self.sample_normal(mu_xh,ligand_diff,sigma,batch_seg)
        z=xh*context+z*ligand_diff
    
        chain = torch.zeros((keep_frames,) + z.size(), device=z.device)

        # Sample p(z_s | z_t)
        
        for s in reversed(range(0, timesteps)):
            s_array = torch.full((batch_size, 1), fill_value=s, device=z.device)
            t_array = s_array + 1
            s_array = s_array / timesteps
            t_array = t_array / timesteps
            z = self.sample_p_zs_given_zt_only_ligandDiff(
                s=s_array,
                t=t_array,
                z_t=z,
                context=context,
                ligand_diff=ligand_diff,
                batch_seg=batch_seg,
                ligand_group=ligand_group,
            )
            if (s*keep_frames) % self.T==0:
                write_index = (s * keep_frames) // self.T
                chain[write_index] = self.unnormalize_z(z)
            
        # Finally sample p(x, h | z_0)
        x, h = self.sample_p_xh_given_z0_only_ligandDiff(
            z_0=z,
            context=context,
            ligand_diff=ligand_diff,
            batch_size=batch_size,
            batch_seg=batch_seg,
            ligand_group=ligand_group
        )
        
        # Correct CoM drift for examples without intermediate states
        if keep_frames==1:
            max_cog = scatter_add(x, batch_seg, dim=0).abs().max().item()
            if max_cog > 5e-2:
                print(f'Warning CoG drift with error {max_cog:.3f}. Projecting '
                      f'the positions down.')
                x = utils.remove_partial_mean_with_mask(x, ligand_diff,batch_seg)
                    
        chain[0] = torch.cat([x, h], dim=1)
        
        return chain
    
        
    def sample_p_zs_given_zt_only_ligandDiff(self, s, t, z_t, context, ligand_diff, batch_seg, ligand_group):
        """Samples from zs ~ p(zs | zt). Only used during sampling. Samples only ligandDiff features and coords"""
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = self.sigma_and_alpha_t_given_s(gamma_t, gamma_s)
        sigma_s = self.sigma(gamma_s)
        sigma_t = self.sigma(gamma_t)
        # Neural net prediction.
        
        eps_hat = self.dynamics.forward(
            xh=z_t,
            t=t,
            ligand_diff=ligand_diff,
            ligand_group=ligand_group,
            batch_seg=batch_seg,
        )
        eps_hat = eps_hat * ligand_diff

        # Compute mu for p(z_s | z_t)
        mu = z_t / alpha_t_given_s[batch_seg] - (sigma2_t_given_s / alpha_t_given_s / sigma_t)[batch_seg] * eps_hat
        # Compute sigma for p(z_s | z_t)
        sigma = sigma_t_given_s * sigma_s / sigma_t

        # Sample z_s given the parameters derived from zt
        z_s = self.sample_normal(mu,ligand_diff,sigma,batch_seg)
        z_s=z_t*context+z_s*ligand_diff
        return z_s

    def sample_p_xh_given_z0_only_ligandDiff(self, z_0, context, ligand_diff, batch_size, batch_seg,ligand_group):
        """Samples x ~ p(x|z0). Samples only ligandDiff features and coords"""
        zeros = torch.zeros(size=(batch_size, 1), device=z_0.device)
        gamma_0 = self.gamma(zeros)

        # Computes sqrt(sigma_0^2 / alpha_0^2)
        sigma_x = self.SNR(-0.5 * gamma_0)
        eps_hat = self.dynamics.forward(
            xh=z_0,
            t=zeros,
            ligand_diff=ligand_diff,
            ligand_group=ligand_group,
            batch_seg=batch_seg,
        )
        eps_hat = eps_hat * ligand_diff

        mu_x = self.compute_x_pred(eps_t=eps_hat, z_t=z_0, gamma_t=gamma_0,batch_seg=batch_seg)
        xh = self.sample_normal(mu_x,ligand_diff,sigma_x,batch_seg)
        xh=z_0*context+xh*ligand_diff
        x, h = xh[:, :self.n_dims], xh[:, self.n_dims:]
        x, h = self.unnormalize(x, h)
        h = F.one_hot(torch.argmax(h, dim=1), self.in_node_nf) 

        return x, h

    def compute_x_pred(self, eps_t, z_t, gamma_t,batch_seg):
        """Computes x_pred, i.e. the most likely prediction of x."""
        sigma_t = self.sigma(gamma_t)
        alpha_t = self.alpha(gamma_t)
        x_pred = 1. / alpha_t[batch_seg] * (z_t - sigma_t[batch_seg] * eps_t)
        return x_pred

    def kl_prior(self, xh,mask,batch_seg):
        """
        Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).
        This is essentially a lot of work for something that is in practice negligible in the loss.
        However, you compute it so that you see it when you've made a mistake in your noise schedule.
        """
        # Compute the last alpha value, alpha_T
        batch_size=torch.max(batch_seg)+1
        ones = torch.ones((batch_size, 1), device=xh.device)
        gamma_T = self.gamma(ones)
        alpha_T = self.alpha(gamma_T)
        # Compute means
        mu_T = alpha_T[batch_seg].view(-1,1)*xh
        mu_T_x, mu_T_h = mu_T[ :, :self.n_dims], mu_T[:, self.n_dims:]
        # Compute standard deviations (only batch axis for x-part, inflated for h-part)
        sigma_T_x = self.sigma(gamma_T).squeeze(1)
        sigma_T_h = self.sigma(gamma_T).squeeze(1)

        # Compute KL for h-part
        zeros, ones = torch.zeros_like(mu_T_h), torch.ones_like(sigma_T_h)
        mu_norm2 = self.inflate_batch_array((mu_T_h - zeros) ** 2*mask, batch_seg)
        kl_distance_h = self.gaussian_kl(mu_norm2, sigma_T_h, ones, d=1)
       
        # Compute KL for x-part
        zeros, ones = torch.zeros_like(mu_T_x), torch.ones_like(sigma_T_x)
        mu_norm2 = self.inflate_batch_array((mu_T_x - zeros) ** 2*mask, batch_seg)
        d = self.n_dims*(self.inflate_batch_array(mask,batch_seg)-1)
        kl_distance_x = self.gaussian_kl(mu_norm2, sigma_T_x, ones, d)
        return kl_distance_x + kl_distance_h

    def log_constant_of_p_x_given_z0(self, mask,batch_seg,batch_size):
        degrees_of_freedom_x = self.n_dims*(self.inflate_batch_array(mask,batch_seg)-1)
        zeros = torch.zeros((batch_size, 1), device=mask.device)
        gamma_0 = self.gamma(zeros)

        # Recall that sigma_x = sqrt(sigma_0^2 / alpha_0^2) = SNR(-0.5 gamma_0)
        log_sigma_x = 0.5 * gamma_0.view(batch_size)

        return degrees_of_freedom_x * (- log_sigma_x - 0.5 * np.log(2 * np.pi))

    def log_p_xh_given_z0_without_constants(self, h, z_0, gamma_0, eps, eps_hat, mask, batch_seg,epsilon=1e-10):
        # Discrete properties are predicted directly from z_0
        z_h = z_0[ :, self.n_dims:]

        # Take only part over x
        eps_x = eps[:, :self.n_dims]
        eps_hat_x = eps_hat[:, :self.n_dims]

        # Compute sigma_0 and rescale to the integer scale of the data
        sigma_0 = self.sigma(gamma_0) * self.norm_values[1]

        # Computes the error for the distribution N(x | 1 / alpha_0 z_0 + sigma_0/alpha_0 eps_0, sigma_0 / alpha_0),
        # the weighting in the epsilon parametrization is exactly '1'
        squared_error=(eps_x - eps_hat_x)**2
        log_p_x_given_z_without_constants = -0.5 * self.inflate_batch_array(squared_error, batch_seg)
        

        # Categorical features
        # Compute delta indicator masks
        h = h * self.norm_values[1] + self.norm_biases[1]
        estimated_h = z_h * self.norm_values[1] + self.norm_biases[1]

        # Centered h_cat around 1, since onehot encoded
        centered_h = estimated_h - 1

        # Compute integrals from 0.5 to 1.5 of the normal distribution
        # N(mean=centered_h_cat, stdev=sigma_0_cat)
        log_p_h_proportional = torch.log(
            self.cdf_standard_gaussian((centered_h + 0.5) / sigma_0[batch_seg]) -
            self.cdf_standard_gaussian((centered_h - 0.5) / sigma_0[batch_seg]) +
            epsilon
        )

        # Normalize the distribution over the categories
        log_Z = torch.logsumexp(log_p_h_proportional, dim=1, keepdim=True)
        log_probabilities = log_p_h_proportional - log_Z

        # Select the log_prob of the current category using the onehot representation
        log_p_h_given_z=self.inflate_batch_array(log_probabilities * h * mask,batch_seg)
        # Combine log probabilities for x and h
        #log_p_xh_given_z = log_p_x_given_z_without_constants + log_p_h_given_z

        return log_p_x_given_z_without_constants,log_p_h_given_z

    def sample_combined_position_feature_noise(self, x,ligand_diff):
        z_x = torch.randn(x.shape[0],self.n_dims,device=x.device)*ligand_diff
        z_h = torch.randn(x.shape[0],self.in_node_nf,device=x.device)*ligand_diff
        z = torch.cat([z_x, z_h], dim=1)
        
        return z


    def normalize(self, x, h):
        new_x = x / self.norm_values[0]
        new_h = (h.float() - self.norm_biases[1]) / self.norm_values[1]
        return new_x, new_h

    def unnormalize(self, x, h):
        new_x = x * self.norm_values[0]
        new_h = h * self.norm_values[1] + self.norm_biases[1]
        return new_x, new_h

    def unnormalize_z(self, z):
        assert z.size(1) == self.n_dims + self.in_node_nf
        x, h = z[:, :self.n_dims], z[:, self.n_dims:]
        x, h = self.unnormalize(x, h)
        return torch.cat([x, h], dim=1)

    def delta_log_px(self, mask):
        return -self.dimensionality(mask) * np.log(self.norm_values[0])
        

    def sigma(self, gamma):
        """Computes sigma given gamma."""
        return torch.sqrt(torch.sigmoid(gamma))

    def alpha(self, gamma):
        """Computes alpha given gamma."""
        return torch.sqrt(torch.sigmoid(-gamma))

    def SNR(self, gamma):
        """Computes signal to noise ratio (alpha^2/sigma^2) given gamma."""
        return torch.exp(-gamma)

    def sigma_and_alpha_t_given_s(self, gamma_t: torch.Tensor, gamma_s: torch.Tensor):
        """
        Computes sigma t given s, using gamma_t and gamma_s. Used during sampling.

        These are defined as:
            alpha t given s = alpha t / alpha s,
            sigma t given s = sqrt(1 - (alpha t given s) ^2 ).
        """
        sigma2_t_given_s = -torch.expm1(F.softplus(gamma_s) - F.softplus(gamma_t))

        # alpha_t_given_s = alpha_t / alpha_s
        log_alpha2_t = F.logsigmoid(-gamma_t)
        log_alpha2_s = F.logsigmoid(-gamma_s)
        log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s
        alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)
        sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

        return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s


    @staticmethod
    def inflate_batch_array(x,batch_seg):
        """
        Inflates the batch array (array) with only a single axis (i.e. shape = (batch_size,),
        or possibly more empty axes (i.e. shape (batch_size, 1, ..., 1)) to match the target shape.
        """
        
        return scatter_add(x.sum(-1), batch_seg, dim=0)


    @staticmethod
    def expm1(x: torch.Tensor) -> torch.Tensor:
        return torch.expm1(x)

    @staticmethod
    def softplus(x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x)

    @staticmethod
    def cdf_standard_gaussian(x):
        return 0.5 * (1. + torch.erf(x / math.sqrt(2)))

    @staticmethod
    def gaussian_kl(q_mu_minus_p_mu_squared, q_sigma, p_sigma, d):
        """Computes the KL distance between two normal distributions.
            Args:
                q_mu_minus_p_mu_squared: Squared difference between mean of
                    distribution q and distribution p: ||mu_q - mu_p||^2
                q_sigma: Standard deviation of distribution q.
                p_sigma: Standard deviation of distribution p.
                d: dimension
            Returns:
                The KL distance
            """
        return d * torch.log(p_sigma / q_sigma) + \
               0.5 * (d * q_sigma ** 2 + q_mu_minus_p_mu_squared) / \
               (p_sigma ** 2) - 0.5 * d
        


    

