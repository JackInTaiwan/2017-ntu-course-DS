import urllib.request
import ssl
import matplotlib.pyplot as plt
import sys



def parse_to_list(data_str) :
    data_list = data_str.split('\n')
    for i, item in enumerate(data_list) :
        item = item.split(',')
        data_list[i] = item
    return data_list


def data_parse(data_list) :
    data_dic = dict()
    categ = ''
    while [''] in data_list :
        data_list.remove([''])
    for data in data_list[1:] :
        if '' in data :
            categ = data[0][0].upper()
            data_dic[categ] = list()
            data_dic[categ].append(data[0])
        else :
            row_data = []
            for i, item in enumerate(data) :
                if i==0 :
                    row_data.append(item)
                else :
                    row_data.append(float(item))
            row_data.append((row_data[2]*row_data[1]+row_data[4]*row_data[3])/(row_data[1]+row_data[3]))
            del row_data[3]
            del row_data[1]
            data_dic[categ].append(row_data)

    return data_dic


def group_barchart(data_dic, type) :
    '''
        :param data_dic: all data in a dic
        :param type: one of ('E', 'A', 'W')
    '''
    ## Produce the appropriate data format based on the type
    data_chosen = data_dic[type]
    data_type = data_chosen[0]
    data_stat = data_chosen[1:]
    vals = [[data_stat[j][i+1] for j in range(len(data_stat))] for i in range(3)]     # 取分群資料而不是row資料

    ## Setting the attribution of space and colors
    width = 2
    space = width * 2
    poses = [ [width * i + (3 * width+space) * j for j in range(len(data_stat))] for i in range(3)]
    colors = ['r', 'b', 'g']
    labels = ['Male', 'Female', 'Total']

    ## Draw bars
    for i, member_poses in enumerate(poses) :
        plt.bar(member_poses, vals[i], width=width, alpha=0.5, color=colors[i], label=labels[i])
    xticks = [rowdata[0] for rowdata in data_stat]
    plt.xticks(poses[1], xticks)

    ## Put tags above each bar
    for i in range(3) :
        for j in range(len(data_stat)) :
            plt.text(poses[i][j], vals[i][j]+3, '%.1f'%vals[i][j], ha='center', va='bottom', fontsize=7)

    ## Draw chart info
    plt.ylim(0, max([max(item) for item in vals]) * 1.2)
    plt.xlabel(data_type)
    plt.ylabel('Smoking Percentage (%)')
    plt.title('Smoking Percentage vs {}'.format(data_type))
    plt.legend()
    plt.show()


def group_lineplot(data_dic, type) :
    ## Produce the appropriate data format based on the type
    data_chosen = data_dic[type]
    data_type = data_chosen[0]
    data_stat = data_chosen[1:]
    vals = [[data_stat[j][i + 1] for j in range(len(data_stat))] for i in range(3)]  # 取分群資料而不是row資料
    x = [i*10 for i in range(len(data_stat))]

    ## Draw line plot
    markers = ['o', 'x', '^']
    colors = ['b', 'r', 'g']
    labels = ['Male', 'Female', 'Total']
    for i, data in enumerate(vals) :
        plt.plot(x, data, marker=markers[i], linestyle='-', label=labels[i], color=colors[i], alpha=0.5)

    ## Put tags above line spots
    for i in range(3) :
        for pos in zip(x,vals[i]) :
            plt.text(pos[0], pos[1]+1, '%.1f'%pos[1], ha='center', fontsize=10)

    ## Draw chart info
    xticks = [rowdata[0] for rowdata in data_stat]
    plt.xticks(x, xticks)
    plt.xlabel(data_type)
    plt.ylabel('Smoking Percentage (%)')
    plt.ylim(0, max([max(item) for item in vals]) * 1.2)
    plt.title('Smoking Percentage vs {}'.format(data_type))
    plt.legend()
    plt.show()


def piechart(data_dic, type) :
    ## Produce the appropriate data format based on the type
    data_chosen = data_dic[type]
    data_type = data_chosen[0]
    data_stat = data_chosen[1:]
    labels = [rowdata[0] for rowdata in data_stat]
    vals = [data[3] for data in data_stat]

    ## Draw piechart
    plt.figure(figsize=(8,8))
    plt.axis('equal')   #確保都是正圓
    plt.pie(vals, labels=labels, autopct='%1.1f%%')
    plt.title('Propotion of Different {} in Smoking Population'.format(data_type))
    plt.show()



if __name__ == '__main__' :
    url = 'https://ceiba.ntu.edu.tw/course/481ea4/hw1_data.csv'
    context = ssl._create_unverified_context()
    res = urllib.request.urlopen(url, context=context)
    data = res.read()   # type: bytes
    data_str = str(data, 'utf-8')   #transform bytes to str
    data_list = parse_to_list(data_str)
    data_dic = data_parse(data_list)

    argv = sys.argv
    if len(argv)>1 :
        for mode in argv[1:] :
            type, chart = mode[1].upper(), mode[2].lower()
            if chart == 'b' :
                group_barchart(data_dic, type)
            elif chart == 'l' :
                group_lineplot(data_dic, type)
            else :
                piechart(data_dic, type)