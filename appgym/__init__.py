from uiautomator import Device
import numpy as np
import timeit
import urllib3

from scipy import misc

import pandas as pd

import sys
import os

from subprocess import call

FNULL = open(os.devnull, 'w')
PROJECT_ROOT = "/Users/vini/Dev/uni/dissertation/code/"
APP_UNDER_TEST_ROOT = "/Users/vini/Dev/uni/dissertation/code/sample_app/"

http_client = urllib3.PoolManager()

class Action:

    def __init__(self, gui_object, action_type='click'):
        self.gui_object = gui_object
        self.action_type = action_type

    def execute(self):
        if self.action_type == 'click':
            self.gui_object.click()

class AndroidEnv:

    def __init__(self, app_package, screen_size, resize_scale=0.1, coverage_target=0.8):
        self.app_package = app_package
        self.device = Device()
        self.screen_size = screen_size
        self._exec(f"ng ng-cp {PROJECT_ROOT}lib/org.jacoco.ant-0.7.9-nodeps.jar")
        self._exec(f"ng ng-cp {PROJECT_ROOT}")
        self._exec("adb forward tcp:8981 tcp:8981")
        self.resize_scale = resize_scale
        self.coverage_target = coverage_target

    def reset(self):
        self._exec(f"adb shell am force-stop {self.app_package}")
        self._exec(f"adb shell pm clear {self.app_package}")
        self._exec(f"adb shell monkey -p {self.app_package} 1")
        return self._get_screen(), self._get_actions()

    def step(self, action):
        action.execute()
        coverage = self._get_current_coverage()
        done = coverage > self.coverage_target
        return self._get_screen(), self._get_actions(), coverage, done

    def _exec(self, command):
        call(command, shell=True, stdout=FNULL)

    def _get_actions(self):
        actions = []
        for gui_obj in self.device():
            if gui_obj.clickable:
                actions.append(Action(gui_obj))
        return actions

    def _get_screen(self):
        self.device.screenshot("state.png")
        img = misc.imread("state.png")
        return self._image_to_torch(img)

    def _image_to_torch(self, image):
        img_resized = misc.imresize(image, size=self.resize_scale)
        return np.ascontiguousarray(img_resized, dtype=np.float32) / 255

    def _get_current_coverage(self):
        start_time = timeit.default_timer()
        with http_client.request("GET", "http://localhost:8981", preload_content=False) as r, open("coverage/coverage.exec", "wb") as coverage_file:
            coverage_file.write(r.read())
        generate_report_cmd = f"ng ReportGenerator {APP_UNDER_TEST_ROOT}"
        self._exec(generate_report_cmd)
        df = pd.read_csv("coverage/report.csv")
        missed, covered = df[['LINE_MISSED', 'LINE_COVERED']].sum()
        print(f"Complete in {timeit.default_timer() - start_time} seconds")
        return covered / (missed + covered)
