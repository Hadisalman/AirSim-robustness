import os
import sys
import shutil

class PedDetectionMetrics(object):
    def __init__(self):
        self.False_Negative = AverageMeter()
        self.False_Positive = AverageMeter()        
        self.True_Negative = AverageMeter()
        self.True_Positive = AverageMeter()        
        self.reset()

    def reset(self):
        self.count = 0
        self.False_Negative.reset()
        self.False_Positive.reset()
        self.True_Negative.reset()
        self.True_Positive.reset()
        self.Precision = 0
        self.Recall = 0
        self.Accuracy = 0

    def update(self, pred, ground_truth, n=1):
        self.count += n
        self.False_Negative.update(pred==False and ground_truth==True, n)
        self.False_Positive.update(pred==True and ground_truth==False, n)
        self.True_Negative.update(pred==False and ground_truth==False, n)
        self.True_Positive.update(pred==True and ground_truth==True, n)
        
        self.num_positive = self.True_Positive.sum + self.False_Positive.sum
        self.num_True = self.True_Positive.sum + self.False_Negative.sum
        
        self.Precision = self.True_Positive.sum/self.num_positive if self.num_positive != 0 else 'NA'
        self.Recall = self.True_Positive.sum/self.num_True if self.num_True != 0 else 'NA'
        self.Accuracy = (self.True_Positive.sum + self.True_Negative.sum)/self.count

    def get(self):
        return {
            'Number of measurements': self.count,
            'Accuracy': self.Accuracy,
            'Precision': self.Precision,
            'Recall': self.Recall,
            'FP':int(self.False_Positive.sum),
            'FN':int(self.False_Negative.sum),
            'TP':int(self.True_Positive.sum),
            'TN':int(self.True_Negative.sum),
        }

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def init_logfile(filename: str, text: str):
    f = open(filename, 'w')
    f.write(text+"\n")
    f.close()

def log(filename: str, text: str):
    f = open(filename, 'a')
    f.write(text+"\n")
    f.close()

def copy_code(outdir):
    """Copies files to the outdir to store complete script with each experiment"""
    code = []
    exclude = set([])
    for root, _, files in os.walk("./code", topdown=True):
        for f in files:
            if not f.endswith('.py'):
                continue
            code += [(root,f)]

    for r, f in code:
        codedir = os.path.join(outdir,r)
        if not os.path.exists(codedir):
            os.mkdir(codedir)
        shutil.copy2(os.path.join(r,f), os.path.join(codedir,f))
    print("Code copied to '{}'".format(outdir))
