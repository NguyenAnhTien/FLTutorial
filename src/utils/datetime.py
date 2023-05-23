"""
@author : Tien Nguyen
@date   : 2023-05-21
"""

import datetime

def get_time_now():
    now = datetime.datetime.now()
    return (now.year, now.month, now.day, now.hour, now.minute, now.second)

def create_report_dir():
    now = get_time_now()
    report_dir = ""
    for item in now[:-1]:
        report_dir += str(item)
    report_dir += str(now[-1])
    return report_dir
