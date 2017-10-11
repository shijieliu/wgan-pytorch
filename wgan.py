#!/usr/bin/env python
#encoding=utf-8

from torch.autograd import Variable
import torch
from argparse import ArgumentParser
import os
import json
import sys
from model import datest
from torch.utils.data import DataLoader
from model import vocab
from model import networkG,networkD
from torch import optim
from torch.autograd import grad

def criterion_G(D_fake):
    D_fake = D_fake.type(torch.FloatTensor)
    return -torch.mean(D_fake)

def criterion_D(true_input,fake_input):
    return -(torch.mean(true_input)-torch.mean(fake_input))

def gen_random():
    return torch.randn((1,3,32,32)).cuda()

def set_fixed_parameter(module):
    for i in module.parameters():
        i.requires_grad = False

def set_unfixed_parameter(module):
    for i in module.parameters():
        i.requires_grad = True
    #print module

def changeonehot(tensor,vocab_size):
    list_ = []
    #print tensor,tensor.shape
    for i in tensor[0,:]:
        index = long(i)
        new_tensor = torch.zeros(vocab_size)
        new_tensor[index] = 1
        list_.append(new_tensor)
    return torch.unsqueeze(torch.stack(list_),0)

def cal_penalty(net,true_input,fake_input):
    alpha = torch.rand(1,1,1)
    alpha = Variable(alpha.expand(true_input.size()).cuda())
    interpolates = alpha*true_input + (1-alpha)*fake_input
    disc_interpolates = net(interpolates)

    gradients = grad(outputs = disc_interpolates,inputs = interpolates,grad_outputs = torch.ones(disc_interpolates.size()).cuda(),create_graph = True,retain_graph = True,only_inputs = True)[0]
    gradient_penalty = ((gradients.norm(2,dim = 1)-1)**2).mean()*10
    return gradient_penalty

def train(FLAGS):
    poem_datest = datest(FLAGS)
    poem_data = DataLoader(
            poem_datest,
            batch_size = FLAGS.batchsize,
            #batch_size = 1,
            shuffle = True,
            num_workers = 1)

    netG = networkG(FLAGS)
    netG.cuda()
    netD = networkD(FLAGS)
    netD.cuda()

    if FLAGS.type == "test":
        if FLAGS.netG_param:
            netG.load_state_dict(torch.load(FLAGS.netG_param))
        if FLAGS.netD_param:
            netD.load_state_dict(torch.load(FLAGS.netD_param))

    loss_G = criterion_G
    loss_D = criterion_D
    optimizer_G = optim.Adam(netG.parameters(),lr = FLAGS.learning_rate_G)
    optimizer_D = optim.Adam(netD.parameters(),lr = FLAGS.learning_rate_D)
    for epoch in range(1,FLAGS.epochs+1):
        for idx,data in enumerate(poem_data):
            ####################################
            #1.get true data and fake data
            ####################################
            netD.zero_grad()
            step = data.shape[1]
            true_input = Variable(changeonehot(data,FLAGS.vocab_size).cuda())
            #print true_input
            #true_input = Variable(data.cuda())
            #print data.shape
            noise = Variable(gen_random())
            fake_input = netG(noise,step)
            #print fake_input.data.shape
            #print true_output.data.shape,fake_output.data.shape
        #TODO
        #1.without penalty train


            #################################
            #2.train D
            #################################
            set_fixed_parameter(netG)
            set_unfixed_parameter(netD)
            #print true_input.data.shape,fake_input.data.shape
            true_output = netD(true_input)
            fake_output = netD(fake_input)
            loss_d = loss_D(true_output,fake_output)
            gradient_penalty = cal_penalty(netD,true_input,fake_input)
            loss_total_d = loss_d + gradient_penalty
            loss_total_d = loss_d
            loss_total_d.backward()
            optimizer_D.step()

            #########################
            #3.train_G
            #########################
            set_fixed_parameter(netD)
            set_unfixed_parameter(netG)
            netG.zero_grad()
            noise_input = Variable(gen_random(),requires_grad=True)
            step = 24
            fake_output = netD(netG(noise_input,step))
            loss_g = loss_G(fake_output)
            loss_g.backward()
            optimizer_G.step()
            if idx % 200 == 199:
                print "111"
                print "%d epoch %d run: lossd = %.3f,lossg = %.3f" % (epoch,idx+1,loss_d.data[0],loss_g.data[0])
                tmp_output = netG(Variable(gen_random()),24)
                _,softmax = torch.max(tmp_output,2)
                softmax = softmax.data
                sentence = []
                for i in softmax[0,:]:
                    index = int(i)
                    sentence.append(FLAGS.vocab[index])
                print "".join(sentence).encode('utf-8')
                netGsavename = os.path.join(FLAGS.save_dir,"netG%d-%d.pth" %(epoch,idx+1))
                netDsavename = os.path.join(FLAGS.save_dir,"netD%d-%d.pth" %(epoch,idx+1))
                torch.save(netG.state_dict(),netGsavename)
                torch.save(netD.state_dict(),netDsavename)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--type",type = str)
    parser.add_argument("--batchsize",type = int)
    parser.add_argument("--input",type = str)
    parser.add_argument("--vocab_dir",type = str)
    parser.add_argument("--epochs",type = int)
    parser.add_argument("--learning_rate_G",type = float)
    parser.add_argument("--learning_rate_D", type = float)
    parser.add_argument('--save_dir',type = str)
    parser.add_argument('--netG_param',type = str,default = None)
    parser.add_argument('--netD_param',type = str,default = None)
    FLAGS = parser.parse_args()

    FLAGS.vocab = vocab(FLAGS.vocab_dir)
    FLAGS.vocab_size = len(FLAGS.vocab)
    print FLAGS

    if FLAGS.type == "train":
        train(FLAGS)
    if FLAGS.type == "predict":
       predict(FLAGS)
