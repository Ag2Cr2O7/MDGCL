from myparser import parse_args
import os
import sys
import numpy as np
import random
import torch
from time import time
import logging
from pynvml import *
class Logger(object):
    """`Logger` is a simple encapsulation of python logger.

    This class can show a message on standard output and write it into the
    file named `filename` simultaneously. This is convenient for observing
    and saving training results.
    """

    def __init__(self, filename):
        """Initializes a new `Logger` instance.

        Args:
            filename (str): File name to create. The directory component of this
                file will be created automatically if it is not existing.
        """
        self.filename=filename
        dir_name = os.path.dirname(filename)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        self.logger = logging.getLogger(filename)
        self.logger.setLevel(logging.DEBUG)


    def set_log(self,format=logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')):
        if self.logger.hasHandlers():
            self.logger.removeHandler(self.logger.handlers[0])
        formatter = format
        # formatter=_Formatter(datefmt='%Y-%m-%d %H:%M:%S')

        # write into file
        fh = logging.FileHandler(self.filename,encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)

        # show on console
        # ch = logging.StreamHandler(sys.stdout)
        # ch.setLevel(logging.DEBUG)
        # ch.setFormatter(formatter)

        # add to Handler
        self.logger.addHandler(fh)
        # self.logger.addHandler(ch)

    def _flush(self):
        for handler in self.logger.handlers:
            handler.flush()

    def debug(self, message):
        self.logger.debug(message)
        self._flush()

    def info(self, message):
        self.logger.info(message)
        self._flush()

    def warning(self, message):
        self.logger.warning(message)
        self._flush()

    def error(self, message):
        self.logger.error(message)
        self._flush()

    def critical(self, message):
        self.logger.critical(message)
        self._flush()



def model_info(log,args,runname='MDGCL',use_nvidia=True):
    log.set_log(format=logging.Formatter())
    log.info('-'*50+runname+'-'*50)
    log.set_log()
    log.info('Model parameters and Equipment information')
    log.set_log(format=logging.Formatter())
    log.info('Dataset:'+str(args.data))
    log.info('Lgraph:' + str(args.Lgraph))
    log.info('t:' + str(args.t))
    log.info('Lhyper:' + str(args.Lhyper))
    log.info('t\':' + str(args.temp))
    log.info('learning rate:' + str(args.lr))
    log.info('embedding dim:' + str(args.d))
    log.info('GNN layers:' + str(args.gnn_layer))
    log.info('Drop rou:' + str(args.dropout))
    log.info('l2 reg:' + str(args.lambda2))
    if torch.cuda.is_available():
        log.info('cuda version of pytorch:'+str(torch.version.cuda))
        log.info('device:' + 'cuda:' + str(args.cuda))
        if use_nvidia:
            nvmlInit()
            index=int(args.cuda)
            handle = nvmlDeviceGetHandleByIndex(index)
            gpu_name = nvmlDeviceGetName(handle)
            log.info('GPU name:'+gpu_name)
    else:
        log.info('use:' + 'cpu')




# if __name__=="__main__":
#     mydata='amazon'
#     runname='MDGCL-amazon'
#     args=parse_args(dataset=mydata)
#     log_path = 'save_log/' + mydata + '/' + runname +'_'+ str(time())[6:10] + '.log'
#     print('log:', log_path)
#     log = Logger(log_path)
#     model_info(log, args, runname)
#     log.set_log()
#     log.debug('debug')

