import os
import numpy as np
import nibabel as nib
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import ITHscore
import glob
import  csv
import warnings
from scipy import ndimage
import cc3d
import SimpleITK as sitk
import six
from radiomics import featureextractor
warnings.filterwarnings("ignore")
######################################################################################################
# ITH 高级方式计算方法代码，改进原有代码只基于2D mask的缺陷。拓展了3D mask，3D 纹理的计算
######################################################################################################
# 所有超参数设置
main_path = './dataset/'   # 数据所在大文件夹名称
best_n_clusters = 6
save_str =  './radiomics_ITHscore_advance.csv'  # 把结果保存成CSV文件
create_image_list_shuffle = 1   # 判断是否创建打乱的image list
data_list_txt_str = './data_list_txt_str/'  # 所有数据List和打乱的真题数据存放的位置
image_dir_list = 'shuffle_data_ITH_advance_texture.txt'
num_class = 3
##############################################################################
def create_image_list(base_path, image_dir_list,num_class):
    print('生成数据清单txt')
    image_path=[]
    for i in range(num_class):
        image_path.append(base_path+'/'+str(i)+'/')
    sum=0
    img_path=[]
    #遍历上面的路径，依次把信息追加到img_path列表中
    for label,p in enumerate(image_path):
        image_dir = glob.glob(p + "/" + "*")  # 返回路径下的所有图片详细的路径
        sum+=len(image_dir)
        print(len(image_dir))
        for image in image_dir:
            img_path.append((image,str(label)))
    print("%d 个图像信息已经加载到txt文本!!!"%(sum))
    np.random.seed(123)  # 固定随机种子
    np.random.shuffle(img_path)
    file=open(image_dir_list,"w",encoding="utf-8")
    for img  in img_path:
        file.write(img[0]+','+img[1]+'\n')
    file.close()   # 写入后的文件内容：图片路径+对应label
########################################################################
if not os.path.exists(data_list_txt_str):
    os.makedirs(data_list_txt_str)

if create_image_list_shuffle==1:
    create_image_list(main_path, data_list_txt_str+image_dir_list,num_class)
########################################################################
train_images_path = []  # 存储训练集的所有图片路径
train_images_label = []  # 存储训练集图片对应索引信息
file = open(data_list_txt_str+image_dir_list, 'r', encoding='utf-8',newline="")
reader = csv.reader(file)
imgs_ls = []
for line in reader:
    imgs_ls.append(line)
print('Total image num=',len(imgs_ls))
file.close()
for i, row in enumerate(imgs_ls):
    train_images_path.append(row[0])  # 存储验证集的所有图片路径
    train_images_label.append(int(row[1]))  # 存储验证集图片对应索引信息
print('train num=',np.array(train_images_path).shape)
num_train = len(train_images_path)
########################################################################
feature_all = ['id','label','ITH','ITH 3D','shape_Sphericity','glcm_Correlation','glcm_MCC','glszm_ZonePercentage','glrlm_RunPercentage','ngtdm_Coarseness']
for i in range(num_train):
    result_list = []
    feature_name = []
    feat_name_tmp = []

    str_split = train_images_path[i].split('/')
    img_path = train_images_path[i] + '/' + str_split[4] + '.nii.gz'
    mask_path = train_images_path[i] + '/' + str_split[4] + '-label.nii.gz'
    print("image path: " + img_path)
    print("mask path: " + mask_path)

    try:
        # 1 Load image and segmentation
        image = nib.load(img_path).get_fdata()
        seg = nib.load(mask_path).get_fdata() # 读取对应的mask文件
        image = image.transpose(2, 0, 1)  # 调换数据的1/3维度
        seg = seg.transpose(2, 0, 1)
        print('image.shape = ',image.shape)

        # 2 Get the slice with maximum tumor size
        img1, mask1 = ITHscore.get_largest_slice(image, seg)
        sub_img1, sub_mask1 = ITHscore.locate_tumor(img1, mask1)
        features1 = ITHscore.extract_radiomic_features(sub_img1, sub_mask1)  # 单线程
        label_map1 = ITHscore.pixel_clustering(sub_mask1, features1, cluster=best_n_clusters)
        ithscore = ITHscore.calITHscore(label_map1)
        print('ITHscore = ', ithscore)

        # 3 Locate and extract tumor
        def locate_tumor_3D(img, mask, padding=2):
            """
            Locate and extract tumor from CT image using mask
            Args:
                img: Numpy array. The whole image
                mask: Numpy array. Same size as img, binary mask with tumor area set as 1, background as 0
                padding: Int. Number of pixels padded on each side after extracting tumor
            Returns:
                sub_img: Numpy array. The tumor area defined by mask
                sub_mask: Numpy array. The subset of mask in the same position of sub_img
            """
            top_margin = min(np.where(mask == 1)[0])
            bottom_margin = max(np.where(mask == 1)[0])
            left_margin = min(np.where(mask == 1)[1])
            right_margin = max(np.where(mask == 1)[1])
            up_margin = min(np.where(mask == 1)[2])
            down_margin = max(np.where(mask == 1)[2])
            # padding two pixels at each edges for further computation
            sub_img = img[top_margin:bottom_margin,
                      left_margin - padding:right_margin + padding + 1,up_margin - padding:down_margin + padding + 1]
            sub_mask = mask[top_margin:bottom_margin,
                       left_margin - padding:right_margin + padding + 1,up_margin - padding:down_margin + padding + 1]

            return sub_img, sub_mask

        sub_img, sub_mask = locate_tumor_3D(image, seg)
        print('sub_img.shape = ', sub_img.shape)

        label_map = np.zeros_like(sub_img)
        for i in range(sub_img.shape[0]):
            features = ITHscore.extract_radiomic_features(sub_img[i,:,:], sub_mask[i,:,:])  # 单线程
            label_map_tmp = ITHscore.pixel_clustering(sub_mask[i,:,:], features, cluster=best_n_clusters)
            label_map[i,:,:] = label_map_tmp

        print('label_map shape = ', label_map.shape)
        ##############################################################################################################
        # 7 Calculate ITHscore
        # calculate 3D ITHscore from clustering label map.
        def calITHscore_3D(label_map, min_area=200, thresh=1):
            """
            Calculate ITHscore from clustering label map
            Args:
                label_map: Numpy array. Clustering label map
                min_area: Int. For tumor area (pixels) smaller than "min_area", we don't consider connected-region smaller than "thresh"
                thresh: Int. The threshold for connected-region's area, only valid when tumor area < min_area
            Returns:
                ith_score: Float. The level of ITH, between 0 and 1
            """
            size = np.sum(label_map > 0)  # Record the number of total pixels
            num_regions_list = []
            max_area_list = []
            for i in np.unique(label_map)[1:]:  # For each gray level except 0 (background)
                labeled = cc3d.connected_components(label_map == i, connectivity=26)  # 3D 连通区域计算
                num_regions = np.max(labeled)
                flag = 1  # Flag to count this gray level, in case this gray level has only one pixel
                max_area = 0
                for j in np.unique(labeled)[1:]:  # 0 is background (here is all the other regions)
                    # Ignore the region with only 1 or "thresh" px
                    if size <= min_area:
                        if np.sum(labeled == j) <= thresh:
                            num_regions -= 1
                            if num_regions == 0:  # In case there is only one region
                                flag = 0
                        else:
                            temp_area = np.sum(labeled == j)
                            if temp_area > max_area:
                                max_area = temp_area
                    else:
                        if np.sum(labeled == j) <= 1:
                            num_regions -= 1
                            if num_regions == 0:  # In case there is only one region
                                flag = 0
                        else:
                            temp_area = np.sum(labeled == j)
                            if temp_area > max_area:
                                max_area = temp_area
                if flag == 1:
                    num_regions_list.append(num_regions)
                    max_area_list.append(max_area)
            # Calculate the ITH score
            ith_score = 0
            for k in range(len(num_regions_list)):
                ith_score += float(max_area_list[k]) / num_regions_list[k]
            # Normalize each area with total size
            ith_score = ith_score / size
            ith_score = 1 - ith_score

            return ith_score


        ithscore_3D = calITHscore_3D(label_map)
        print('ITHscore_3D = ', ithscore_3D)
        ##############################################################################################################
        # calculate texture ITH, average of sub-region
        settings = {}
        settings['binWidth'] = 25  # 5
        extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
        extractor.disableAllFeatures()  # 禁用所有特征
        extractor.enableFeaturesByName( shape=['Sphericity'])
        extractor.enableFeaturesByName(glcm=['Correlation', 'MCC'])
        extractor.enableFeaturesByName( glszm=['ZonePercentage'])
        extractor.enableFeaturesByName( glrlm=['RunPercentage'])
        extractor.enableFeaturesByName( ngtdm=['Coarseness'])
        # 具体按子区间提取并取平均
        def calITHscore_3Dtexture(sub_img, label_map):
            result_list = 0
            img_ex = sitk.GetImageFromArray(sub_img)
            list_for = np.unique(label_map)[1:]
            for i in list_for:  # For each gray level except 0 (background)
                result_list_mask = []
                mask_tmp =  label_map.copy()
                mask_tmp[mask_tmp != i ] = 0 # 使用布尔索引将非i的元素置零
                mask_tmp[mask_tmp == i] = 1
                mask_ex = sitk.GetImageFromArray(mask_tmp)
                ITH_texture = extractor.execute(img_ex, mask_ex)  # 抽取特征
                # 将特征取出并转换为数组矩阵
                i_order = 0  # 舍去计算出的一些头文件
                for key, value in ITH_texture.items():  # 输出特征
                    i_order = i_order + 1
                    if i_order > 22:  # 把计算得到的特征值堆叠
                        result_list_mask.append([value])

                result_list = result_list+ np.array(result_list_mask)
            return result_list/best_n_clusters

        ITH_texture = calITHscore_3Dtexture(sub_img, label_map)
        print('ITH_texture',ITH_texture)
        ##############################################################################################################
        # save result
        # 遍历列表，对每个字符串去除方括号
        result_list_add = [np.array(str_split[4]), np.array(str_split[3]), ithscore, ithscore_3D] + ITH_texture.flatten().tolist()
        print('result_list_add',result_list_add)

        #  把结果保存成CSV文件
        feature_all = np.vstack((feature_all, result_list_add))  # 默认情况下，axis=0可以不写

        #  把结果保存成CSV文件
        print('feature shape=', np.array(feature_all).shape)
        np.savetxt(save_str, feature_all, delimiter=',', fmt='%s')
        print('Saved, run over!')

    # LASSO不成功,打印报错信息并跳过到下一个seed
    except Exception as e:
        print(e)
    pass