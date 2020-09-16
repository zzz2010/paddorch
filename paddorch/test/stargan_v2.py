import argparse
import copy


from munch import Munch

import numpy as np
import math

from paddorch.vision.models.wing import FAN

def eval_pytorch_model(args):
    import   torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import models

    class ResBlk(nn.Module):
        def __init__(self, dim_in, dim_out, actv=None,
                     normalize=False, downsample=False):
            super().__init__()
            if actv is None:
                actv=nn.LeakyReLU(0.2)
            self.actv = actv
            self.normalize = normalize
            self.downsample = downsample
            self.learned_sc = dim_in != dim_out
            self._build_weights(dim_in, dim_out)

        def _build_weights(self, dim_in, dim_out):
            self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
            self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
            if self.normalize:
                self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
                self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
            if self.learned_sc:
                self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

        def _shortcut(self, x):
            if self.learned_sc:
                x = self.conv1x1(x)
            if self.downsample:
                x = F.avg_pool2d(x, 2)
            return x

        def _residual(self, x):
            if self.normalize:
                x = self.norm1(x)
            x = self.actv(x)
            x = self.conv1(x)
            if self.downsample:
                x = F.avg_pool2d(x, 2)
            if self.normalize:
                x = self.norm2(x)
            x = self.actv(x)
            x = self.conv2(x)
            return x

        def forward(self, x):
            x = self._shortcut(x) + self._residual(x)
            return x / math.sqrt(2)  # unit variance


    class AdaIN(nn.Module):
        def __init__(self, style_dim, num_features):
            super().__init__()
            self.norm = nn.InstanceNorm2d(num_features, affine=False)
            self.fc = nn.Linear(style_dim, num_features*2)

        def forward(self, x, s):
            h = self.fc(s)
            h = h.view(h.shape[0], h.shape[1], 1, 1) #fluid.layers.reshape(h, shape=[h.shape[0], h.shape[1], 1, 1])
            gamma, beta = torch.chunk(h, chunks=2, dim=1)
            return (1 + gamma) * self.norm(x) + beta


    class AdainResBlk(nn.Module):
        def __init__(self, dim_in, dim_out, style_dim=64, w_hpf=0,
                     actv=nn.LeakyReLU(0.2), upsample=False):
            super().__init__()
            self.w_hpf = w_hpf
            self.actv = actv
            self.upsample = upsample
            self.learned_sc = dim_in != dim_out
            self._build_weights(dim_in, dim_out, style_dim)

        def _build_weights(self, dim_in, dim_out, style_dim=64):
            self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
            self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
            self.norm1 = AdaIN(style_dim, dim_in)
            self.norm2 = AdaIN(style_dim, dim_out)
            if self.learned_sc:
                self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

        def _shortcut(self, x):
            if self.upsample:
                x = F.interpolate(x, scale_factor=2, mode='nearest')
            if self.learned_sc:
                x = self.conv1x1(x)
            return x

        def _residual(self, x, s):
            x = self.norm1(x, s)
            x = self.actv(x)
            if self.upsample:
                x = F.interpolate(x, scale_factor=2, mode='nearest')
            x = self.conv1(x)
            x = self.norm2(x, s)
            x = self.actv(x)
            x = self.conv2(x)
            return x

        def forward(self, x, s):
            out = self._residual(x, s)
            if self.w_hpf == 0:
                out = (out + self._shortcut(x)) / math.sqrt(2)
            return out


    class HighPass(nn.Module):
        def __init__(self, w_hpf, device):
            super(HighPass, self).__init__()
            self.filter = torch.tensor([[-1, -1, -1],
                                        [-1, 8., -1],
                                        [-1, -1, -1]]) / w_hpf

        def forward(self, x):
            filter = self.filter.unsqueeze(0).unsqueeze(1).repeat(x.size(1), 1, 1, 1)
            return F.conv2d(x, filter, padding=1, groups=x.size(1))



    class Generator(nn.Module):
        def __init__(self, img_size=256, style_dim=64, max_conv_dim=512, w_hpf=1):
            super().__init__()
            dim_in = 2**14 // img_size
            self.img_size = img_size
            self.from_rgb = nn.Conv2d(3, dim_in, 3, 1, 1)
            self.encode = nn.ModuleList()
            self.decode = nn.ModuleList()
            self.to_rgb = nn.Sequential(
                nn.InstanceNorm2d(dim_in, affine=True),
                nn.LeakyReLU(0.2),
                nn.Conv2d(dim_in, 3, 1, 1, 0))

            # down/up-sampling blocks
            repeat_num = int(np.log2(img_size)) - 4
            if w_hpf > 0:
                repeat_num += 1
            for _ in range(repeat_num):
                dim_out = min(dim_in*2, max_conv_dim)
                self.encode.append(
                    ResBlk(dim_in, dim_out, normalize=True, downsample=True))
                self.decode.insert(
                    0, AdainResBlk(dim_out, dim_in, style_dim,
                                   w_hpf=w_hpf, upsample=True))  # stack-like
                dim_in = dim_out

            # bottleneck blocks
            for _ in range(2):
                self.encode.append(
                    ResBlk(dim_out, dim_out, normalize=True))
                self.decode.insert(
                    0, AdainResBlk(dim_out, dim_out, style_dim, w_hpf=w_hpf))

            if w_hpf > 0:
                device = torch.device(
                    'cuda' if torch.cuda.is_available() else 'cpu')
                self.hpf = HighPass(w_hpf, device)

        def forward(self, x, s, masks=None):
            print("Generator",x.shape,s.shape)
            x = self.from_rgb(x)
            print("from_rgb", x.shape )
            cache = {}
            for block in self.encode :
                if (masks is not None) and (x.size(2) in [32, 64, 128]):
                    cache[x.size(2)] = x
                x = block(x)

            for block in self.decode :
                x = block(x, s)

                if (masks is not None) and (x.size(2) in [32, 64, 128]):
                    mask = masks[0] if x.size(2) in [32] else masks[1]
                    mask = F.interpolate(mask, size=x.size(2), mode='bilinear')
                    x = x + self.hpf(mask * cache[x.size(2)])

            return self.to_rgb(x)





    class MappingNetwork(nn.Module):
        def __init__(self, latent_dim=16, style_dim=64, num_domains=2):
            super().__init__()
            layers = []
            layers += [nn.Linear(latent_dim, 512)]
            layers += [nn.ReLU()]
            for _ in range(3):
                layers += [nn.Linear(512, 512)]
                layers += [nn.ReLU()]
            self.shared = nn.Sequential(*layers)

            self.unshared = nn.ModuleList()
            for _ in range(num_domains):
                self.unshared.append(nn.Sequential(nn.Linear(512, 512),
                                                nn.ReLU(),
                                                nn.Linear(512, 512),
                                                nn.ReLU(),
                                                nn.Linear(512, 512),
                                                nn.ReLU(),
                                                nn.Linear(512, style_dim)) )

        def forward(self, z, y):
            print("MappingNetwork", z.shape, y.shape)
            print("torch z:", z.detach().numpy().mean())
            uu=self.shared[1]( self.shared[0](z))
            print("paddle uu:", uu.detach().numpy().mean())
            h0 = self.shared[2](uu)
            print("paddle h0:", h0.detach().numpy().mean())
            h = self.shared(z)

            print("torch h:", h.detach().numpy().mean())
            out = []
            for layer in self.unshared:
                out += [layer(h)]
            out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
            print("torch out:", out.detach().numpy().mean())
            idx = torch.LongTensor(range(y.size(0))).to(y.device)
            s = out[idx, y]  # (batch, style_dim)
            return s

        def finetune(self, z, y):
            h = self.shared(z)
            out = []
            for layer in self.unshared:
                out += [layer(h)]
            out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)

            idx = torch.LongTensor(range(y.size(0))).to(y.device)
            s = out[idx, y]  # (batch, style_dim)
            return s,h,out
    class StyleEncoder(nn.Module):
        def __init__(self, img_size=256, style_dim=64, num_domains=2, max_conv_dim=512):
            super().__init__()
            dim_in = 2**14 // img_size
            blocks = []
            blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]

            repeat_num = int(np.log2(img_size)) - 2
            for _ in range(repeat_num):
                dim_out = min(dim_in*2, max_conv_dim)
                blocks += [ResBlk(dim_in, dim_out, downsample=True)]
                dim_in = dim_out

            blocks += [nn.LeakyReLU(0.2)]
            blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
            blocks += [nn.LeakyReLU(0.2)]
            self.shared = nn.Sequential(*blocks)

            self.unshared = nn.ModuleList()
            for _ in range(num_domains):
                self.unshared.append(nn.Linear(dim_out, style_dim) )

        def forward(self, x, y):
            print("StyleEncoder", x.shape, y.shape)
            h = self.shared(x)
            h = h.view(h.size(0), -1)
            out = []
            for layer in self.unshared:
                out += [layer(h)]
            out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
            idx = torch.LongTensor(range(y.size(0))).to(y.device)
            s = out[idx, y]  # (batch, style_dim)
            return s



    generator = Generator(args.img_size, args.style_dim, w_hpf=args.w_hpf)
    mapping_network = MappingNetwork(args.latent_dim, args.style_dim, args.num_domains)
    style_encoder = StyleEncoder(args.img_size, args.style_dim, args.num_domains)
    return  generator,mapping_network,style_encoder

def build_model(args):
    import torch as pytorch
    from paddorch.convert_pretrain_model import load_pytorch_pretrain_model
    pytorch_state_dict = pytorch.load("../../../starganv2_paddle/expr/checkpoints/celeba_hq/100000_nets_ema.pt",
                                      map_location=pytorch.device('cpu'))

    generator, mapping_network, style_encoder = eval_pytorch_model(args)
    x=np.ones( (1,3,256,256)).astype("float32")
    batch_size=2000

    batch_size2=3
    x= np.random.randn(batch_size2,3,256,256).astype("float32")
    y = np.random.randint(0,2, batch_size2).astype("int32")
    s = np.random.randn(args.style_dim*batch_size2).astype("float32").reshape(batch_size2,-1)
    z = np.random.randn(16*batch_size2).astype("float32").reshape(batch_size2, -1)
    generator.load_state_dict(pytorch_state_dict['generator'])
    g_out=generator(pytorch.FloatTensor(x), pytorch.FloatTensor(s))
    mapping_network.load_state_dict(pytorch_state_dict['mapping_network'])
    m_out=mapping_network(pytorch.FloatTensor(z), pytorch.LongTensor(y))
    y_train = np.random.randint(0,2, batch_size).astype("int32")
    z_train = np.random.randn(16*batch_size).astype("float32").reshape(batch_size, -1)
    m_out_train,m_out_train_1,m_out_train_2 = mapping_network.finetune(pytorch.FloatTensor(z_train), pytorch.LongTensor(y_train))
    style_encoder.load_state_dict(pytorch_state_dict['style_encoder'])
    s_out=style_encoder(pytorch.FloatTensor(x), pytorch.LongTensor(y) )

    import paddle.fluid as fluid
    import paddorch as torch
    import paddorch.nn.functional as F
    # place = fluid.CPUPlace()
    place = fluid.CUDAPlace(0)


    with fluid.dygraph.guard(place=place):
        import sys
        sys.path.append("../../../starganv2_paddle")
        import core.model



        generator_ema = core.model.Generator(args.img_size, args.style_dim, w_hpf=args.w_hpf) #copy.deepcopy(generator)
        mapping_network_ema =core.model.MappingNetwork(args.latent_dim, args.style_dim, args.num_domains)  # copy.deepcopy(mapping_network)
        style_encoder_ema =core.model.StyleEncoder(args.img_size, args.style_dim, args.num_domains) # copy.deepcopy(style_encoder)

        nets = Munch(generator=generator,
                     mapping_network=mapping_network,
                     style_encoder=style_encoder,
                     )


        print("load generator")
        load_pytorch_pretrain_model(generator_ema, pytorch_state_dict['generator'])
        print("load mapping_network")
        load_pytorch_pretrain_model(mapping_network_ema, pytorch_state_dict['mapping_network'])
        print("load style_encoder")
        load_pytorch_pretrain_model(style_encoder_ema, pytorch_state_dict['style_encoder'])

        nets_ema = Munch(generator=generator_ema,
                         mapping_network=mapping_network_ema,
                         style_encoder=style_encoder_ema)
        nets_ema_state_dict=dict()

        # nets_ema['mapping_network'].load_state_dict(
        #     torch.load("../../expr/checkpoints/celeba_hq/100000_nets_ema.ckpt/mapping_network.pdparams"))

        nets_ema['mapping_network'].train()
        d_optimizer = fluid.optimizer.AdamOptimizer(learning_rate=1e-4,
                                                    parameter_list=nets_ema['mapping_network'].parameters())
        from tqdm import tqdm
        z_train_p = torch.Tensor(z_train)
        y_train_p = torch.LongTensor(y_train)
        m_out_train_p = torch.Tensor(m_out_train.detach().numpy())
        m_out_train_1p = torch.Tensor(m_out_train_1.detach().numpy())
        m_out_train_2p = torch.Tensor(m_out_train_2.detach().numpy())
        # dummy_net=fluid.dygraph.Linear(16,1)
        # d_optimizer = fluid.optimizer.AdamOptimizer(learning_rate=1e-4,
        #                                             parameter_list=dummy_net.parameters())
        for ii in range(10000):
            # predictions=dummy_net(z_train_p )
            # predictions2=torch.varbase_to_tensor(predictions)
            # d_avg_cost = fluid.layers.mse_loss(predictions2, m_out_train_p)
            # fluid.layers.assign(predictions,predictions2)
            out, out1, out2 = nets_ema['mapping_network'].finetune(z_train_p, y_train_p)
            d_avg_cost = fluid.layers.mse_loss(out, m_out_train_p) + fluid.layers.mse_loss(out1,
                                                                                           m_out_train_1p) + fluid.layers.mse_loss(
                out2, m_out_train_2p)

            d_avg_cost.backward()
            d_optimizer.minimize(d_avg_cost)
            nets_ema['mapping_network'].clear_gradients()
            if ii % 99 == 0:
                print("d_avg_cost", d_avg_cost.numpy())
                nets_ema['mapping_network'].eval()
                m_out_t = nets_ema['mapping_network'](torch.Tensor(z), torch.LongTensor(y)).numpy()
                print(ii, "torch m result:", m_out.detach().numpy().mean(), m_out_t.mean())
                nets_ema['mapping_network'].train()

                y_train = np.random.randint(0, 2, batch_size).astype("int32")
                z_train = np.random.randn(16 * batch_size).astype("float32").reshape(batch_size, -1)
                m_out_train, m_out_train_1, m_out_train_2 = mapping_network.finetune(pytorch.FloatTensor(z_train),
                                                                                     pytorch.LongTensor(y_train))
                z_train_p = torch.Tensor(z_train)
                y_train_p = torch.LongTensor(y_train)
                m_out_train_p = torch.Tensor(m_out_train.detach().numpy())
                if np.abs(m_out.detach().numpy().mean() - m_out_t.mean()) < 1e-5:
                    break

        nets_ema['mapping_network'].eval()
        g_out_t = nets_ema['generator'](torch.Tensor(x), torch.Tensor(s)).numpy()
        m_out_t = nets_ema['mapping_network'](torch.Tensor(z), torch.LongTensor(y)).numpy()
        s_out_t = nets_ema['style_encoder'](torch.Tensor(x), torch.LongTensor(y)).numpy()

        print("torch g result:", g_out.mean().detach().numpy(), g_out_t.mean())
        print("torch s result:", s_out.mean().detach().numpy(), s_out_t.mean())
        print("torch m result:", m_out.mean().detach().numpy(), m_out_t.mean())
        nets_ema_state_dict['generator'] = generator_ema.state_dict()
        nets_ema_state_dict['mapping_network'] = mapping_network_ema.state_dict()
        nets_ema_state_dict['style_encoder'] = style_encoder_ema.state_dict()

        torch.save(nets_ema_state_dict, "../../../starganv2_paddle/expr/checkpoints/celeba_hq/100000_nets_ema.ckpt")
        if args.w_hpf > 0:
            fan = FAN(fname_pretrained="../../../starganv2_paddle/expr/checkpoints/wing.pdparams").eval()
            nets.fan = fan
            nets_ema.fan = fan

        return nets, nets_ema

if __name__ == '__main__':

    ##this is for celeba_hq
    args=argparse.Namespace(batch_size=8, beta1=0.0, beta2=0.99,
              checkpoint_dir='expr/checkpoints/celeba_hq',
              ds_iter=100000, eval_dir='expr/eval/celeba_hq',
              eval_every=50000, f_lr=1e-06, hidden_dim=512,
              img_size=256, inp_dir='assets/representative/custom/female',
              lambda_cyc=1, lambda_ds=1, lambda_reg=1, lambda_sty=1, latent_dim=16,
              lm_path='expr/checkpoints/celeba_lm_mean.npz', lr=0.0001, mode='eval',
              num_domains=2, num_outs_per_domain=10, num_workers=4, out_dir='assets/representative/celeba_hq/src/female',
              print_every=10, randcrop_prob=0.5, ref_dir='assets/representative/celeba_hq/ref', result_dir='expr/results',
              resume_iter=100000, sample_dir='expr/samples', sample_every=5000, save_every=10000, seed=777, src_dir='assets/representative/celeba_hq/src',
              style_dim=64, total_iters=100000, train_img_dir='data/celeba_hq/train', val_batch_size=32, val_img_dir='data/celeba_hq/val',
              w_hpf=1, weight_decay=0.0001, wing_path='expr/checkpoints/wing')
    build_model(args)