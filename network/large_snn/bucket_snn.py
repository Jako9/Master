from ..base_network import Large_SNN
import torch
from snntorch import spikegen
import wandb

class Bucket_SNN(Large_SNN):
    def __init__(self, env, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.num_buckets = 101
        self.buckets = torch.nn.Parameter(torch.linspace(0, 5, self.num_buckets))
        self.head = torch.nn.Linear(512, self.action_space * self.num_buckets)

    def every_init(self):
        super().every_init()
        self.buckets = self.buckets.to(next(self.parameters()).device)


    def _forward(self, x, global_step):
        size = x.size(0)
        """x = x.unsqueeze(0).expand(10, -1, -1, -1, -1)
        random_values = torch.rand_like(x).to(x.device)
        spike_train = (random_values < x).float().to(x.device)"""

        spike_train = spikegen.rate(x, num_steps=self.num_steps)

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem_fc = self.lif_fc.init_leaky()
        #mem_head = self.lif_head.init_leaky()

        output = torch.zeros(size, self.action_space * self.num_buckets).to(x.device)

        spk1_average = 0
        spk2_average = 0
        spk3_average = 0
        spk_fc_average = 0
        
        for step in range(self.num_steps):
            out = self.conv2d_1(spike_train[step])
            out = self.pop1(out)
            spk1, mem1 = self.lif1(out, mem1)

            out = self.conv2d_2(spk1)
            out = self.pop2(out)
            spk2, mem2 = self.lif2(out, mem2)

            out = self.conv2d_3(spk2)
            out = self.pop3(out)
            spk3, mem3 = self.lif3(out, mem3)

            out = self.flatten(spk3)
            out = self.linear(out)
            out = self.pop_fc(out)

            spk_fc, mem_fc = self.lif_fc(out, mem_fc)

            output += self.head(spk_fc)

            #_, mem_head = self.lif_head(out, mem_head)

            #Only for logging purposes
            if self.track and global_step is not None and global_step % 100 == 0:
                spk1_average += spk1.mean().item()
                spk2_average += spk2.mean().item()
                spk3_average += spk3.mean().item()
                spk_fc_average += spk_fc.mean().item()

            #TODO: Be able to add other pooling methods (mean, last, etc.)
            #mem_out = torch.max(mem_head, mem_out)
        
        #out = self.head(spk_fc)
        if self.track and global_step is not None and global_step % 100 == 0:
            wandb.log({
                                "spikes/layer1": spk1_average / self.num_steps,
                                "spikes/layer2": spk2_average / self.num_steps,
                                "spikes/layer3": spk3_average / self.num_steps,
                                "spikes/fc": spk_fc_average / self.num_steps,
                                "buckets/max": self.buckets.max().item(),
                                "buckets/min": self.buckets.min().item(),
                                "buckets/mean": self.buckets.mean().item(),
                                "buckets/std": self.buckets.std().item()
                            })
        output = output.view(size, self.action_space, self.num_buckets)
        output = torch.softmax(output, dim=2) * self.buckets
        output = output.sum(dim=2)
        return output