"""

"""

import sys
sys.path.insert(0, ".")
import torch, argparse, os
import numpy as np

import lib

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, default="4")
parser.add_argument("--mode", type=str, default="path", help="caffe: first call caffe evaluation; path: a single path; paths: a number of paths")
parser.add_argument("--ref_path", type=str, default="", help="reference dataset")
parser.add_argument("--data_path", type=str, default="", help="target dataset")
parser.add_argument("--output", type=str, default="fid.txt", help="FID output file")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
evaluator = lib.evaluator.FIDEvaluator(args.ref_path)

if args.mode == "cifar":
    gpu_id = 2
    caffe_dir = "../caffe"
    name_list = []
    log_list = []
    proto_list = []
    weight_list = []
    output_list = []
    for path,expr in zip(["examples/cifar_gan"], ["cifar"]):
        expr_path = caffe_dir + "/" + path
        for g_model in ["deconv", "pxs", "upsample"]:
            expr_name = "%s_pool_%s" % (expr, g_model)
            log_path = caffe_dir + "/log/" + expr_name
            name_list.append(expr_name)
            proto_list.append(expr_path + "/" + g_model + "_g.prototxt")
            weight_list.append(log_path + "/pool_" + g_model + "_g_solver_iter_%d.caffemodel")
            output_list.append(caffe_dir + "/log/eval/" + expr_name)
    
    ref_mu, ref_sigma = evaluator.calculate_statistics_given_path("/home/atlantix/data/cifar10_image")

    cmd = "CUDA_VISIBLE_DEVICES=%d %s/build/tools/caffe_gan test --gpu 0 --iterations 100 --g_model %s --g_weights %s --output %s"
    fids = []
    for i in range(len(name_list)):
        for j in range(1, 11):
            pcmd = cmd % (gpu_id, caffe_dir, proto_list[i], weight_list[i], output_list[i])
            pcmd = pcmd % (j * 2000)
            print(pcmd)
            os.system(pcmd)
            tar_mu, tar_sigma = evaluator.calculate_statistics_given_path(output_list[i], save_npy=False)
            fid = lib.fid.fid_score.calculate_frechet_distance(ref_mu, ref_sigma, tar_mu, tar_sigma)
            print("%s %d %.3f\n" % (name_list[i], j * 2000, fid))
            fids.append(fid)
            with open(args.output, "w") as f:
                for fid in fids:
                    f.write("%s %d %.3f\n" % (name_list[i], j * 2000, fid))
            np.save(args.output.replace(".txt", ".npy"), fids)

if args.mode == "mnist":
    gpu_id = 2
    caffe_dir = "../caffe"
    name_list = []
    log_list = []
    proto_list = []
    weight_list = []
    output_list = []
    for path,expr in zip(["examples/cifar_gan", "examples/mnist_gan"], ["cifar", "mnist"]):
        expr_path = caffe_dir + "/" + path
        for g_model in ["deconv", "pxs", "upsample"]:
            expr_name = "%s_pool_%s" % (expr, g_model)
            log_path = caffe_dir + "/log/" + expr_name
            name_list.append(expr_name)
            proto_list.append(expr_path + "/" + g_model + "_g.prototxt")
            weight_list.append(log_path + "/pool_" + g_model + "_g_solver_iter_%d.caffemodel")
            output_list.append(caffe_dir + "/log/eval/" + expr_name)
    
    ref_mu, ref_sigma = evaluator.calculate_statistics_given_path('/home/atlantix/data/mnist/testSet/')

    cmd = "CUDA_VISIBLE_DEVICES=%d %s/build/tools/caffe_gan test --gpu 0 --iterations 100 --g_model %s --g_weights %s --output %s"
    fids = []
    for i in range(len(name_list)):
        for j in range(1, 11):
            pcmd = cmd % (gpu_id, caffe_dir, proto_list[i], weight_list[i], output_list[i])
            pcmd = pcmd % (j * 2000)
            print(pcmd)
            os.system(pcmd)
            tar_mu, tar_sigma = evaluator.calculate_statistics_given_path(output_list[i], save_npy=False)
            fid = lib.fid.fid_score.calculate_frechet_distance(ref_mu, ref_sigma, tar_mu, tar_sigma)
            print("%s %d %.3f\n" % (name_list[i], j * 2000, fid))
            fids.append(fid)
            with open(args.output, "w") as f:
                for fid in fids:
                    f.write("%s %d %.3f\n" % (name_list[i], j * 2000, fid))
            np.save(args.output.replace(".txt", ".npy"), fids)

elif args.mode == "path":
    ref_mu, ref_sigma = evaluator.calculate_statistics_given_path(args.ref_path)
    tar_mu, tar_sigma = evaluator.calculate_statistics_given_path(args.data_path)
    fid = lib.fid.fid_score.calculate_frechet_distance(ref_mu, ref_sigma, tar_mu, tar_sigma)
    with open(args.output) as f:
        f.write("%.3f\n" % fid)

