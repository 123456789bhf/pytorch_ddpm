import torch
import torch.nn as nn
import torch.nn.functional as F

#我们首先看一下GaussianDiffusionTrainer的初始化init部分，model指的就是Unet, T就是T步的forward diffusion step, beta_1, beta_T，而beta_1,和beta_T 参数指的是方差的最小值和最大值，通过这2个参数产生一个linear schecule,值从1e-4到0.02的一个增长。 初始化这部分都是在初始化参数。

def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    '''
    假设 v 是 [0.1, 0.2, 0.3, 0.4]，t 是 [1, 3]，x_shape 是 [2, 3, 32, 32]，我们来看看这个函数的执行过程：
    torch.gather(v, index=t, dim=0) 提取出 [0.2, 0.4]。
    t.shape[0] 是 2。
    len(x_shape) - 1 是 3，所以 [1] * (len(x_shape) - 1) 是 [1, 1, 1]。
    out.view([t.shape[0]] + [1] * (len(x_shape) - 1)) 将 [0.2, 0.4] 重塑为 [2, 1, 1, 1]。
    '''
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        #参考链接：https://zhuanlan.zhihu.com/p/557715177
        super().__init__()

        self.model = model
        self.T = T
        #register_buffer 方法用于将参数注册为模型的“缓冲区”。这些参数不会随着优化步骤而更新，但它们在模型转移到 GPU 或保存模型时会被包含在内。
        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())       #得到每个时间步的条件概率的方差
        alphas = 1. - self.betas  #得到论文中的α
        alphas_bar = torch.cumprod(alphas, dim=0)  #计算α的累积，表示时间步1到t的累积衰减因子

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0):
        """
        Algorithm 1.
        """
        #在【0，T-1】范围内随机采样时间步t,T是总的时间步
        #device=x_0.device 确保生成的张量与输入数据 x_0 位于同一个设备上（如 CPU 或 GPU）。
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)  
        noise = torch.randn_like(x_0)
        #实现的是X_t的计算过程
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        loss = F.mse_loss(self.model(x_t, t), noise, reduction='none')
        return loss


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, img_size=32,
                 mean_type='eps', var_type='fixedlarge'):
        assert mean_type in ['xprev' 'xstart', 'epsilon']  #均值类型
        assert var_type in ['fixedlarge', 'fixedsmall']  #方差类型
        super().__init__()

        self.model = model
        self.T = T
        self.img_size = img_size
        self.mean_type = mean_type
        self.var_type = var_type

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)  #是alphas的累积成绩
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]  

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_recip_alphas_bar', torch.sqrt(1. / alphas_bar))  
        self.register_buffer(
            'sqrt_recipm1_alphas_bar', torch.sqrt(1. / alphas_bar - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer(
            'posterior_var',
            self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
        # below: log calculation clipped because the posterior variance is 0 at
        # the beginning of the diffusion chain
        self.register_buffer(
            'posterior_log_var_clipped',
            torch.log(
                torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])))
        self.register_buffer(
            'posterior_mean_coef1',
            torch.sqrt(alphas_bar_prev) * self.betas / (1. - alphas_bar))
        self.register_buffer(
            'posterior_mean_coef2',
            torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def q_mean_variance(self, x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior
        q(x_{t-1} | x_t, x_0)
        """
        assert x_0.shape == x_t.shape
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_var_clipped = extract(
            self.posterior_log_var_clipped, t, x_t.shape)
        return posterior_mean, posterior_log_var_clipped

    def predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
        )

    def predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            extract(
                1. / self.posterior_mean_coef1, t, x_t.shape) * xprev -
            extract(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t,
                x_t.shape) * x_t
        )

    def p_mean_variance(self, x_t, t):
        # below: only log_variance is used in the KL computations
        model_log_var = {
            # for fixedlarge, we set the initial (log-)variance like so to
            # get a better decoder log likelihood
            'fixedlarge': torch.log(torch.cat([self.posterior_var[1:2],
                                               self.betas[1:]])),
            'fixedsmall': self.posterior_log_var_clipped,
        }[self.var_type]
        model_log_var = extract(model_log_var, t, x_t.shape)

        # Mean parameterization
        #模型预测的方法：第一个是预测x_{t-1},第二个是预测x_0,第三个是预测epsilon
        if self.mean_type == 'xprev':       # the model predicts x_{t-1}
            x_prev = self.model(x_t, t)
            x_0 = self.predict_xstart_from_xprev(x_t, t, xprev=x_prev)
            model_mean = x_prev
        elif self.mean_type == 'xstart':    # the model predicts x_0
            x_0 = self.model(x_t, t)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        elif self.mean_type == 'epsilon':   # the model predicts epsilon
            eps = self.model(x_t, t)
            x_0 = self.predict_xstart_from_eps(x_t, t, eps=eps)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        else:
            raise NotImplementedError(self.mean_type)
        x_0 = torch.clip(x_0, -1., 1.)

        return model_mean, model_log_var

    def forward(self, x_T):
        """
        Algorithm 2.
        这段代码定义了一个名为 forward 的方法，用于在扩散模型的逆向扩散过程中生成数据。该方法实现了类似于扩散模型论文中的算法2。以下是对这段代码的详细解释：
        类似于去噪过程
        """
        x_t = x_T
        for time_step in reversed(range(self.T)):
            #生成一个形状为 [batch_size,] 的张量 t，每个元素的值都是当前的 time_step。这个张量用于在调用模型时指定当前时间步。
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, log_var = self.p_mean_variance(x_t=x_t, t=t)
            # no noise when t == 0
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            
            x_t = mean + torch.exp(0.5 * log_var) * noise
        x_0 = x_t
        #使用 torch.clip 将 x_0 的值裁剪到 [-1, 1] 范围内，以确保数值稳定性。
        return torch.clip(x_0, -1, 1)
