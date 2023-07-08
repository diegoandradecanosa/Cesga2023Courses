import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import os

class SimpleModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

class MyDataset(Dataset):
    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        return torch.randn(10), torch.tensor([1.0])


def main():
    # Inicialización del backend para las comunicaciones distribuidas (nccl)
    dist.init_process_group(backend='nccl')
    
    # Global rank del proceso actual
    global_rank = dist.get_rank()

    # El dispositivo usado será igual al global_rank
    torch.cuda.set_device(global_rank)

    
    model = SimpleModel(10, 1).cuda()

    # En esta ocasión pasamos el modelo por el Wrapper DDP
    ddp_model = DDP(model, device_ids=[global_rank], output_device=global_rank)

    # El dataLoader ahora usa un sampler distribuido
    dataset = MyDataset()
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)


    criterion = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data[0].cuda(), data[1].cuda()

            optimizer.zero_grad()
            outputs = ddp_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print('[%d] loss: %.3f' % (epoch + 1, running_loss / i))

    print('Finished Training')


if __name__ == '__main__':
    main()
