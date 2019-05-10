#Custom Example (bash script):
import os
from os.path import join as osj
import requests, json

USERNAME = "v-xuj"
CLUSTER = "rr2"
VCID = "msrlabs"

def main_command(name, file_name):
    CMD = "https://philly/api/submit?"
    CMD += "clusterId=%s&" % CLUSTER
    CMD += "registry=phillyregistry.azurecr.io&"
    CMD += "repository=philly%2Fjobs%2Fcustom%2Fpytorch&"
    CMD += "tag=pytorch0.4.0-py36-reid&"
    CMD += "buildId=0000&"
    CMD += "toolType=cust&"
    CMD += "vcId=%s&" % VCID
    CMD += "rackid=anyConnected&"
    CMD += "configFile=%2Fphilly%2F" + CLUSTER + "%2F" + VCID + "%2F" + USERNAME
    CMD += "%2Fswa%2F" + file_name + "&"
    CMD += "minGPUs=1&"
    CMD += "name=%s&" % name
    CMD += "isdebug=false&"
    CMD += "ismemcheck=false&"
    CMD += "isperftrace=false&"
    CMD += "iscrossrack=false&"
    CMD += "oneProcessPerContainer=true&"
    CMD += "dynamicContainerSize=false&"
    CMD += "numOfContainers=1&"
    CMD += "inputDir=%2Fhdfs%2F" + VCID + "%2F" + USERNAME +"%2Fdata%2F&"
    CMD += "extraParams=--philly%20rr2%20--batch_size%20128%20"
    return CMD

def send_request(CMD):
    s = "curl " + "\"" + CMD + "\" " + "-k --ntlm --negotiate -u " + open("scripts/passwd.txt").read().strip()

    print(s)
    os.system(s)

def feat_diverge_bn_command(diverge, g_bn, d_bn, dataset="CIFAR10"):
    name = "cifargan"
    name += "_" + diverge + "_g" + g_bn + "_d" + d_bn

    CMD = main_command(name, "gan.py")

    CMD += "--diverge%20" + diverge + "%20"
    CMD += "--g_bn%20" + g_bn + "%20"
    CMD += "--d_bn%20" + d_bn + "%20"
    CMD += "--dataset%20" + dataset + "%20"
    CMD += "&"
    CMD += "userName=" + USERNAME + "&"
    CMD += "submitCode=p"
    send_request(CMD)

def feat_diverge_dataset_command(diverge, dataset):
    name = "gan"
    name += "_" + dataset + "_" + diverge + "_gbn" + "_dbn"

    CMD = main_command(name, "gan.py")

    CMD += "--diverge%20" + diverge + "%20"
    CMD += "--g_bn%20bn%20"
    CMD += "--d_bn%20bn%20"
    CMD += "--dataset%20" + dataset + "%20"
    CMD += "&"
    CMD += "userName=" + USERNAME + "&"
    CMD += "submitCode=p"
    send_request(CMD)


def feat_diverge_bn_study():
    for g_bn in ['none', 'bn', 'in']:
        for d_bn in ['none', 'bn']:
            for diverge in ['none', 'feat']:
                feat_diverge_bn_command(diverge, g_bn, d_bn)

def feat_diverge_study():
    for diverge in ['none', 'feat']:
        for dataset in ['celebA']:
            feat_diverge_bn_command(diverge, g_bn='bn', d_bn='bn', dataset=dataset)

#feat_diverge_study()
feat_diverge_dataset_command("none", dataset="celeba")
#feat_diverge_dataset_command("feat", dataset="celeba")
#feat_diverge_dataset_command("none", dataset="cifar")
#feat_diverge_dataset_command("feat", dataset="cifar")