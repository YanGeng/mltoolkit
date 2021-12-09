#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import os
import time
import configparser
from logger import Logger

a = sys.getdefaultencoding()
# if sys.getdefaultencoding() != 'utf-8':
#     reload(sys)
#     sys.setdefaultencoding('utf-8')

class ConfigParser(object):
    def __init__(self, config_file = None):
        if config_file is not None:
            self.config_file = config_file
        else:
            self.config_file = os.getcwd() + "/config/config.ini"
        
        self.template_file = os.getcwd() + "/config/template"
        self.output_dir = os.getcwd() + "/outputs/export/"

        path_with_time = time.strftime("%Y%m%d-%X", time.localtime()) + "/"
        self.model_dir = os.getcwd() + "/outputs/models/" + path_with_time
        self.log_dir_ = os.getcwd() + "/outputs/summary/" + path_with_time
        Logger.init(self.log_dir_)

        try:
            self.config = configparser.ConfigParser()
            self.config.read(self.config_file, encoding = 'utf-8')

            self.run_type = self.get("general", "run_type")
            # 如果是相对路径，则指基于当前根目录的相对路径
            if self.getboolean("general", "is_relative_path"):
                self.train_data = os.getcwd() + "/" + self.get("data", "train_data")
                self.test_data = os.getcwd() + "/" + self.get("data", "test_data")
                self.valid_data = os.getcwd() + "/" + self.get("data", "valid_data")
                self.embedding_file = os.getcwd() + "/" + self.get("data", "embedding_file")
                self.restore_model = os.getcwd() + "/" + self.get("model_path", "restore_model")
                self.dics_path = os.getcwd() + "/" + self.get("data", "dics_path")
            else:
                self.train_data = self.get("data", "train_data")
                self.test_data = self.get("data", "test_data")
                self.valid_data = self.get("data", "valid_data")
                self.embedding_file = self.get("data", "embedding_file")
                self.restore_model = self.get("model_path", "restore_model")
                self.dics_path = self.get("data", "dics_path")
        except Exception:
            Logger.error("读取配置文件失败")
        else:
            Logger.info("读取配置文件成功")

    def get(self, section, option):
        if self.config.has_option(section, option):
            return self.config.get(section, option).encode("utf-8")
        else:
            return None

    def getint(self, section, option):
        if self.config.has_option(section, option):
            return self.config.getint(section, option)
        else:
            return None

    def getboolean(self, section, option):
        if self.config.has_option(section, option):
            return self.config.getboolean(section, option)
        else:
            return None
    
    def getfloat(self, section, option):
        if self.config.has_option(section, option):
            return self.config.getfloat(section, option)
        else:
            return None