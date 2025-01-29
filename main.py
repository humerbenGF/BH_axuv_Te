import matplotlib.pyplot as plt


import load_axuv_Te

def make_shot_list_edges(min_shot, max_shot):
    L = []
    while min_shot <= max_shot:
        L.append(min_shot)
        min_shot += 1

    return L


if __name__ == '__main__':
    
    shots_list = make_shot_list_edges(19718, 23016)
    
    for shot in shots_list:
        try:
            data = load_axuv_Te.load_axuv_Te_data(shot)
        except:
            print(f"No Data for Shot {shot}")
    
    # shot=23016
    # load_axuv_Te.load_axuv_Te_data(shot)
    
    # data = load_axuv_Te.load_axuv_Te_data(21426)
    
    # print(data.keys())
    
    # plt.plot(data['axuv_Te_time (ms)'], data['Mylar12.4&6.2_axuv_Te (eV)'])
    # plt.plot(data['axuv_Te_time (ms)'], data['Mylar21&6.2_axuv_Te (eV)'])
    # plt.plot(data['axuv_Te_time (ms)'], data['Mylar21&12.4_axuv_Te (eV)'])
    # plt.show()