# -*- coding:utf-8 -*-
# from __future__ import print_function
import torch
import matplotlib as plt
from torch.nn import functional as F
import torch.optim as optim
import pickle
from torch.autograd import Variable
from torchvision.utils import save_image
from util import log_sum_exp
from torch.utils.data import DataLoader, TensorDataset
import sys
import argparse
from networks import Generator, Discriminator
from dataset import *
import pdb
# import tensorboardX
import os


#


class ImprovedGAN(object):
    def __init__(self, G, D, labeled, unlabeled, test, args):
        if os.path.exists(args.savedir):
            print('Loading model from ' + args.savedir)
            self.G = torch.load(os.path.join(args.savedir, 'G.pkl'))
            self.D = torch.load(os.path.join(args.savedir, 'D.pkl'))
        else:
            os.makedirs(args.savedir)
            self.G = G
            self.D = D
            torch.save(self.G, os.path.join(args.savedir, 'G.pkl'))
            torch.save(self.D, os.path.join(args.savedir, 'D.pkl'))
        # self.writer = tensorboardX.SummaryWriter(log_dir=args.logdir)
        if args.cuda:
            self.G.cuda()
            self.D.cuda()
        self.labeled = labeled
        self.unlabeled = unlabeled
        self.test = test
        self.Doptim = optim.Adam(self.D.parameters(), lr=args.lr, betas=(args.momentum, 0.999))
        self.Goptim = optim.Adam(self.G.parameters(), lr=args.lr, betas=(args.momentum, 0.999))
        self.args = args

    def trainD(self, x_label, y, x_unlabel):
        x_label, x_unlabel, y = Variable(x_label), Variable(x_unlabel), Variable(y, requires_grad=False)
        if self.args.cuda:
            x_label, x_unlabel, y = x_label.cuda(), x_unlabel.cuda(), y.cuda()
        output_label, output_unlabel, output_fake = self.D(x_label, cuda=self.args.cuda), self.D(x_unlabel,
                                                                                                 cuda=self.args.cuda), self.D(
            self.G(x_unlabel.size()[0], cuda=self.args.cuda).view(x_unlabel.size()).detach(), cuda=self.args.cuda)
        logz_label, logz_unlabel, logz_fake = log_sum_exp(output_label), log_sum_exp(output_unlabel), log_sum_exp(
            output_fake)  # log ∑e^x_i
        prob_label = torch.gather(output_label, 1, y.unsqueeze(1))  # log e^x_label = x_label
        loss_supervised = -torch.mean(prob_label) + torch.mean(logz_label)
        loss_unsupervised = 0.5 * (
            -torch.mean(logz_unlabel) + torch.mean(F.softplus(logz_unlabel)) +  # real_data: log Z/(1+Z)
            torch.mean(F.softplus(logz_fake)))  # fake_data: log 1/(1+Z)
        loss = loss_supervised + self.args.unlabel_weight * loss_unsupervised
        acc = torch.mean((output_label.max(1)[1] == y).float())
        self.Doptim.zero_grad()
        loss.backward()
        self.Doptim.step()
        return loss_supervised.data.cpu().numpy(), loss_unsupervised.data.cpu().numpy(), acc

    def trainG(self, x_unlabel):
        fake = self.G(x_unlabel.size()[0], cuda=self.args.cuda).view(x_unlabel.size())
        #        fake.retain_grad()
        mom_gen, output_fake = self.D(fake, feature=True, cuda=self.args.cuda)
        mom_unlabel, _ = self.D(Variable(x_unlabel), feature=True, cuda=self.args.cuda)
        mom_gen = torch.mean(mom_gen, dim=0)
        mom_unlabel = torch.mean(mom_unlabel, dim=0)
        loss_fm = torch.mean((mom_gen - mom_unlabel) ** 2)
        # loss_adv = -torch.mean(F.softplus(log_sum_exp(output_fake)))
        loss = loss_fm  # + 1. * loss_adv
        self.Goptim.zero_grad()
        self.Doptim.zero_grad()
        loss.backward()
        self.Goptim.step()
        return loss.data.cpu().numpy(), fake

    def train(self):
        assert type(self.labeled) == TensorDataset
        gn = 0
        losses_gen = []
        accuracy_track = []
        train_loss_u = []
        train_loss_s = []
        for epoch in range(self.args.epochs):
            self.G.train()
            self.D.train()
            unlabel_loader1 = DataLoader(self.unlabeled, batch_size=self.args.batch_size, shuffle=True, drop_last=True,
                                         num_workers=0)
            unlabel_loader2 = DataLoader(self.unlabeled, batch_size=self.args.batch_size, shuffle=True, drop_last=True,
                                         num_workers=0).__iter__()
            label_loader = DataLoader(self.labeled, batch_size=self.args.batch_size, shuffle=True, drop_last=True,
                                      num_workers=0).__iter__()
            loss_supervised = loss_unsupervised = loss_gen = accuracy = 0.
            batch_num = 0
            for (unlabel1, _label1) in unlabel_loader1:
                #                pdb.set_trace()
                batch_num += 1
                unlabel2, _label2 = unlabel_loader2.next()
                x, y = label_loader.next()
                if args.cuda:
                    x, y, unlabel1, unlabel2 = x.cuda(), y.cuda(), unlabel1.cuda(), unlabel2.cuda()
                ll, lu, acc = self.trainD(x, y, unlabel1)
                loss_supervised += ll
                loss_unsupervised += lu
                accuracy += acc
                lg, gen_imgs = self.trainG(unlabel2)
                if epoch > 1 and lg > 1:
                    #                    pdb.set_trace()
                    lg, gen_imgs = self.trainG(unlabel2)

                loss_gen += lg
                if (batch_num + 1) % self.args.log_interval == 0:
                    print('Training: %d / %d' % (batch_num + 1, len(unlabel_loader1)))
                    gn += 1
                    # self.writer.add_scalars('loss', {'loss_supervised': ll, 'loss_unsupervised': lu, 'loss_gen': lg},
                    #                         gn)
                    # self.writer.add_histogram('real_feature',
                    #                           self.D(Variable(x, volatile=True), cuda=self.args.cuda, feature=True)[0],
                    #                           gn)
                    # self.writer.add_histogram('fake_feature', self.D(self.G(self.args.batch_size, cuda=self.args.cuda),
                    #                                                  cuda=self.args.cuda, feature=True)[0], gn)
                    # self.writer.add_histogram('fc3_bias', self.G.fc3.bias, gn)
                    # self.writer.add_histogram('D_feature_weight', self.D.layers[-1].weight, gn)
                    #                    self.writer.add_histogram('D_feature_bias', self.D.layers[-1].bias, gn)
                    # print('Eval: correct %d/%d, %.4f' % (self.eval(), self.test.__len__(), acc))
                    self.D.train()
                    self.G.train()
            loss_supervised /= batch_num
            loss_unsupervised /= batch_num
            loss_gen /= batch_num
            accuracy /= batch_num
            accuracy_track.append((epoch, accuracy))
            losses_gen.append((epoch, loss_gen))
            train_loss_s.append((epoch, loss_supervised))
            train_loss_u.append((epoch, loss_unsupervised))
            print("Iteration %d, loss_supervised = %.4f, loss_unsupervised = %.4f, loss_gen = %.4f train acc = %.4f" % (
                epoch, loss_supervised, loss_unsupervised, loss_gen, accuracy))
            sys.stdout.flush()
            if (epoch + 1) % self.args.eval_interval == 0:
                print("Eval: correct %d / %d" % (self.eval(), self.test.__len__()))
                torch.save(self.G, os.path.join(args.savedir, 'G.pkl'))
                torch.save(self.D, os.path.join(args.savedir, 'D.pkl'))
                save_image(gen_imgs.data[:25], 'images/%d.png' % epoch, nrow=5)
        accuracy_track = np.asarray(accuracy_track)
        filename = 'accuracy'
        outfile = open(filename, 'wb')
        pickle.dump(accuracy_track, outfile)
        outfile.close()

        train_loss_s = np.asarray(train_loss_s)
        filename = 'train_loss_s'
        outfile = open(filename, 'wb')
        pickle.dump(train_loss_s, outfile)
        outfile.close()

        train_loss_u = np.asarray(train_loss_u)
        filename = 'train_loss_u'
        outfile = open(filename, 'wb')
        pickle.dump(train_loss_u, outfile)
        outfile.close()

        losses_gen = np.asarray(losses_gen)
        filename = 'losses_gen'
        outfile = open(filename, 'wb')
        pickle.dump(losses_gen, outfile)
        outfile.close()

        plt.plot(losses_gen[:, 0], losses_gen[:, 1])
        plt.suptitle('Gen Loss')
        plt.xlabel('epoch')
        plt.ylabel('Gen loss')
        plt.show()

        plt.plot(accuracy_track[:, 0], accuracy_track[:, 1])
        plt.suptitle('Accuracy')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.show()

        plt.plot(train_loss_s[:, 0], train_loss_s[:, 1])
        plt.suptitle('Supervised loss')
        plt.xlabel('epoch')
        plt.ylabel('Train Loss Supervised')
        plt.show()

        plt.plot(train_loss_u[:, 0], train_loss_u[:, 1])
        plt.suptitle('UnSupervised loss')
        plt.xlabel('epoch')
        plt.ylabel('Train Loss UnSupervised')
        plt.show()

    def predict(self, x):
        output = torch.max(self.D(Variable(x), cuda=self.args.cuda), 1)[1].data
        return output

    def eval(self):
        self.G.eval()
        self.D.eval()
        d, l = [], []
        for (datum, label) in self.test:
            d.append(datum)
            l.append(label)
        x, y = torch.stack(d), torch.LongTensor(l)
        if self.args.cuda:
            x, y = x.cuda(), y.cuda()
        pred = self.predict(x)
        return torch.sum(pred == y)

    def draw(self, batch_size):
        self.G.eval()
        return self.G(batch_size, cuda=self.args.cuda)


if __name__ == '__main__':
    # torch.multiprocessing.set_start_method("spawn")
    parser = argparse.ArgumentParser(description='PyTorch Improved GAN')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.003, metavar='LR',
                        help='learning rate (default: 0.003)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--eval-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before evaling training status')
    parser.add_argument('--unlabel-weight', type=float, default=1, metavar='N',
                        help='scale factor between labeled and unlabeled data')
    parser.add_argument('--logdir', type=str, default='./logfile', metavar='LOG_PATH',
                        help='logfile path, tensorboard format')
    parser.add_argument('--savedir', type=str, default='./models', metavar='SAVE_PATH',
                        help='saving path, pickle format')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    np.random.seed(args.seed)
    gan = ImprovedGAN(Generator(100), Discriminator(), MnistLabel(10), MnistUnlabel(), MnistTest(), args)
    gan.train()
