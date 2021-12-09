#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import codecs
import os
import sys
reload(sys)
sys.setdefaultencoding("utf-8")


__all__ = ["Logger"]


class Logger(object):
    LOGFILE = None
    LOGTEMPLATE = "{0}----{1}\n"


    @classmethod
    def init(cls, logdir):
        if Logger.LOGFILE is None:
            makedirs(logdir + "/logs/")
            Logger.LOGFILE = logdir + "/logs/log.txt"
        else:
            Logger.warning("Logger已经初始化，忽略此次初始化")


    @classmethod
    def info(cls, info):
        if Logger.LOGFILE is None:
            print "Logger未初始化"
            return

        info = "Info: " + info
        print info
        with codecs.open(Logger.LOGFILE, "a", "utf-8") as out:
            out.write(Logger.LOGTEMPLATE.format(time.strftime("%Y%m%d-%X", time.localtime()), info))


    @classmethod
    def error(cls, error):
        if Logger.LOGFILE is None:
            print "Logger未初始化"
            return

        error = "Error: " + error
        print error
        with codecs.open(Logger.LOGFILE, "a", "utf-8") as out:
            out.write(Logger.LOGTEMPLATE.format(time.strftime("%Y%m%d-%X", time.localtime()), error))


    @classmethod
    def warning(cls, warning):
        if Logger.LOGFILE is None:
            print "Logger未初始化"
            return

        warning = "Warning: " + warning
        print warning
        with codecs.open(Logger.LOGFILE, "a", "utf-8") as out:
            out.write(Logger.LOGTEMPLATE.format(time.strftime("%Y%m%d-%X", time.localtime()), warning))


# Logger中重写makedirs，避免与utils互相依赖
def makedirs(abs_dir):
    if os.path.exists(abs_dir) == False:
        os.makedirs(abs_dir)