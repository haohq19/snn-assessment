import datetime


def get_timestamp():
    now = datetime.datetime.now()  # 获取当前时间
    year = now.strftime("%Y")  # 获取年份，四位数
    month = now.strftime("%m")  # 获取月份，两位数
    day = now.strftime("%d")  # 获取日期，两位数
    second_of_day = (now - now.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()  # 获取今天的第几秒
    number = "{}{}{}{}".format(year, month, day, int(second_of_day))
    return number

if __name__ == '__main__':
    ts = get_timestamp()
    print(ts)