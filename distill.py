import argparse
from collections import defaultdict
import os
from sklearn.metrics import f1_score, accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim


# load soft target file, T = 30
def load_soft_target(dataset):
    file_path = 'output/soft_target/' + dataset + '.txt'
    tensors = []
    with open(file_path, 'r') as file:
        for line in file:
            numbers = line.strip().split()
            numbers = [float(num) for num in numbers]
            tensor = torch.tensor(numbers).cuda()
            tensors.append(tensor)
    return tensors


# load images
def get_transformer(train_flag):
    MEAN = [0.48145466, 0.4578275, 0.40821073]
    STD = [0.26862954, 0.26130258, 0.27577711]

    if train_flag:
        transformer = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)])
    else:
        transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)])

    return transformer


def get_dataloader(train_flag, args):
    transformer = get_transformer(train_flag)
    dataset = torchvision.datasets.__dict__[args.data](root=args.data_path, train=train_flag,
                                                       download=True, transform=transformer)

    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=args.batch_size, shuffle=train_flag==True,
                                             num_workers=args.num_workers, pin_memory=True)
    return dataloader


# small model
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_class):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_class)


def ResNet34(num_class):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_class)


def ResNet50(num_class):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_class)


def ResNet101(num_class):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_class)


def ResNet152(num_class):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_class)


def load_network(model, num_class):
    if model == 'resnet18':
        network = ResNet18(num_class)
    elif model == 'resnet34':
        network = ResNet34(num_class)
    elif model == 'resnet50':
        network = ResNet50(num_class)
    elif model == 'resnet101':
        network = ResNet101(num_class)
    elif model == 'resnet152':
        network = ResNet152(num_class)

    return network.cuda()


# compute loss
class LossCalulcator(nn.Module):
    def __init__(self, temperature, distillation_weight):
        super().__init__()

        self.temperature = temperature
        self.distillation_weight = distillation_weight
        self.loss_log = defaultdict(list)
        self.kldiv = nn.KLDivLoss(reduction='batchmean')

    def forward(self, outputs, labels, teacher_outputs=None):
        # Distillation Loss
        soft_target_loss = 0.0
        if teacher_outputs is not None and self.distillation_weight > 0.0:
            soft_target_loss = self.kldiv(F.log_softmax(outputs/self.temperature, dim=1),
                                          F.softmax(teacher_outputs / self.temperature, dim=1)) * (self.temperature ** 2)

        # Ground Truth Loss
        hard_target_loss = F.cross_entropy(outputs, labels, reduction='mean')

        total_loss = (soft_target_loss * self.distillation_weight) + hard_target_loss

        # Logging
        if self.distillation_weight > 0:
            self.loss_log['soft-target_loss'].append(soft_target_loss.item())

        if self.distillation_weight < 1:
            self.loss_log['hard-target_loss'].append(hard_target_loss.item())

        self.loss_log['total_loss'].append(total_loss.item())

        return total_loss

    def get_log(self, length=100):
        log = []
        # calucate the average value from lastest N losses
        for key in self.loss_log.keys():
            if len(self.loss_log[key]) < length:
                length = len(self.loss_log[key])
            log.append("%s: %2.3f" % (key, sum(self.loss_log[key][-length:]) / length))
        return ", ".join(log)


def train(student, dataloader, optimizer, scheduler, loss, device, args, val_dataloader=None):
    best_accuracy = 0
    best_epoch = 0

    for epoch in range(1, args.epoch + 1):
        # train one epoch
        train_step(student, dataloader, optimizer, loss, device, args, epoch)

        # validate the network
        if (val_dataloader is not None) and (epoch % args.valid_interval == 0):
            _, accuracy = measurement(student, val_dataloader, device)
            if accuracy >= best_accuracy:
                best_accuracy = accuracy
                best_epoch = epoch

        # learning rate schenduling
        scheduler.step()

        # save check point
        if (epoch % args.save_epoch == 0) or (epoch == args.epoch):
            torch.save({'argument': args,
                        'epoch': epoch,
                        'state_dict': student.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'loss_log': loss.loss_log},
                       os.path.join(args.save_path, 'check_point_%d.pth' % epoch))

    print("Finished Training, Best Accuracy: %f (at %d epochs)" % (best_accuracy, best_epoch))
    return student

def train_step(student, dataloader, optimizer, loss_calculator, device, args, epoch, teacher=None):
    student.train()

    for i, (inputs, labels) in enumerate(dataloader, 1):
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = student(inputs.to(device))

        teacher_outputs = None
        if teacher is not None and args.distillation_weight > 0.0:
            with torch.no_grad():
                teacher_outputs = teacher(inputs.to(device))

        loss = loss_calculator(outputs=outputs,
                               labels=labels.to(device),
                               teacher_outputs=teacher_outputs)
        loss.backward()
        optimizer.step()

        # print log
        if i % args.print_interval == 0:
            print("Epoch [%3d/%3d], Iteration [%5d/%5d], Loss [%s]" % (epoch,
                                                                         args.epoch,
                                                                         i,
                                                                         len(dataloader),
                                                                         loss_calculator.get_log()))
    return None


def measurement(model, dataloader, device):
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs.data, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    f1 = f1_score(all_labels, all_predictions, average='macro')
    acc = accuracy_score(all_labels, all_predictions)

    return f1, acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=777)
    parser.add_argument('--train_flag', action='store_true', default=False)
    # Data
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='dtd', help="training dataset")
    parser.add_argument('--num_class', type=int, default=47, help="dependent argument with 'data'")
    # Train Validate
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--print_interval', type=int, default=100, help="print log per every N iterations")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--save_path', type=str, default='./distill_models')
    parser.add_argument('--save_epoch', type=int, default=100, help="save model per every N epochs")
    parser.add_argument('--valid_interval', type=int, default=10, help="validate per every N epochs")
    # Network
    parser.add_argument('--model', type=str, default='resnet18', help="type of ResNet")
    parser.add_argument('--distillation_weight', type=float, default=0.3,
                        help="0: no distillation, 1: use only soft-target")
    parser.add_argument('--temperature', type=int, default=30, help="temperature")
    # Optimization
    parser.add_argument('--optim', type=str, default='Adam', help="SGD | Adam")
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--sgd_momentum', type=float, default=0.9)
    parser.add_argument('--adam_betas', type=float, nargs='+', default=(0.9, 0.999))
    # optimization
    parser.add_argument('--scheduler', type=str, default=None, help="StepLR | MStepLR")
    parser.add_argument('--lr_milestones', type=int, nargs='+', default=[150, 225])
    parser.add_argument('--lr_stepsize', type=int, default=150)
    parser.add_argument('--lr_gamma', type=float, default=0.1)

    args = parser.parse_args()

    soft_target = load_soft_target(args.dataset)
    train_loader = get_dataloader(train_flag=True, args=args)
    val_loader = get_dataloader(train_flag=False, args=args)
    loss = LossCalulcator(args.temperature, args.distillation_weight).cuda()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = load_network(args.model, args.num_class)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=args.beta, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=args.lr_stepsize, gamma=args.lr_gamma)

    # train the model
    model = train(student=model, dataloader=train_loader, optimizer=optimizer, scheduler=scheduler, loss=loss,
                  device=device, args=args, val_dataloader=val_loader)

    # evaluate
    f1, acc = measurement(model, val_loader, device)
    print(f"F1 Score: {f1}")
    print(f"Accuracy: {acc * 100}%")
