import numpy as np
import pandas as pd
from sklearn import decomposition
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import copy
from tqdm import tqdm

# import image feature data
data_path = './data/dataframe/feature_whole.csv'
pca_output_dir = './data/dataframe/pca_whole.csv'

def save_data(df_new1,df_new2,df_new3,df_new4,df_new5,df_new6,df_new7,df_new8,df_new9,df_new10,df_new11,df_new12,df_new13,df_new14,df_new15,df_new16,df_new17,df_new18,df_new19,df_new20,pca_num):
    output_dir = './data/img_feature/pca_whole/'
    df_new1.reset_index().drop(['index'],axis=1)
    df_new1.to_csv(output_dir + 'feature_' + str(pca_num) + '_1.csv', index=False)

    df_new2.reset_index().drop(['index'],axis=1)
    df_new2.to_csv(output_dir + 'feature_' + str(pca_num) + '_2.csv', index=False)

    df_new3.reset_index().drop(['index'],axis=1)
    df_new3.to_csv(output_dir + 'feature_' + str(pca_num) + '_3.csv', index=False)

    df_new4.reset_index().drop(['index'],axis=1)
    df_new4.to_csv(output_dir + 'feature_' + str(pca_num) + '_4.csv', index=False)

    df_new5.reset_index().drop(['index'],axis=1)
    df_new5.to_csv(output_dir + 'feature_' + str(pca_num) + '_5.csv', index=False)

    df_new6.reset_index().drop(['index'],axis=1)
    df_new6.to_csv(output_dir + 'feature_' + str(pca_num) + '_6.csv', index=False)

    df_new7.reset_index().drop(['index'],axis=1)
    df_new7.to_csv(output_dir + 'feature_' + str(pca_num) + '_7.csv', index=False)

    df_new8.reset_index().drop(['index'],axis=1)
    df_new8.to_csv(output_dir + 'feature_' + str(pca_num) + '_8.csv', index=False)

    df_new9.reset_index().drop(['index'],axis=1)
    df_new9.to_csv(output_dir + 'feature_' + str(pca_num) + '_9.csv', index=False)

    df_new10.reset_index().drop(['index'],axis=1)
    df_new10.to_csv(output_dir + 'feature_' + str(pca_num) + '_10.csv', index=False)

    df_new11.reset_index().drop(['index'],axis=1)
    df_new11.to_csv(output_dir + 'feature_' + str(pca_num) + '_11.csv', index=False)

    df_new12.reset_index().drop(['index'],axis=1)
    df_new12.to_csv(output_dir + 'feature_' + str(pca_num) + '_12.csv', index=False)

    df_new13.reset_index().drop(['index'],axis=1)
    df_new13.to_csv(output_dir + 'feature_' + str(pca_num) + '_13.csv', index=False)

    df_new14.reset_index().drop(['index'],axis=1)
    df_new14.to_csv(output_dir + 'feature_' + str(pca_num) + '_14.csv', index=False)

    df_new15.reset_index().drop(['index'],axis=1)
    df_new15.to_csv(output_dir + 'feature_' + str(pca_num) + '_15.csv', index=False)

    df_new16.reset_index().drop(['index'],axis=1)
    df_new16.to_csv(output_dir + 'feature_' + str(pca_num) + '_16.csv', index=False)

    df_new17.reset_index().drop(['index'],axis=1)
    df_new17.to_csv(output_dir + 'feature_' + str(pca_num) + '_17.csv', index=False)

    df_new18.reset_index().drop(['index'],axis=1)
    df_new18.to_csv(output_dir + 'feature_' + str(pca_num) + '_18.csv', index=False)

    df_new19.reset_index().drop(['index'],axis=1)
    df_new19.to_csv(output_dir + 'feature_' + str(pca_num) + '_19.csv', index=False)

    df_new20.reset_index().drop(['index'],axis=1)
    df_new20.to_csv(output_dir + 'feature_' + str(pca_num) + '_20.csv', index=False)



# set new dataframe
colum_name = []
for i in range(1,15):
            for j in range(1,513):
                colum_name.append('img'+str(i) +'-'+str(j))
colum_name_new = colum_name

    
df_new1 = pd.DataFrame(columns=colum_name_new)
df_new2 = pd.DataFrame(columns=colum_name_new)
df_new3 = pd.DataFrame(columns=colum_name_new)
df_new4 = pd.DataFrame(columns=colum_name_new)
df_new5 = pd.DataFrame(columns=colum_name_new)
df_new6 = pd.DataFrame(columns=colum_name_new)
df_new7 = pd.DataFrame(columns=colum_name_new)
df_new8 = pd.DataFrame(columns=colum_name_new)
df_new9 = pd.DataFrame(columns=colum_name_new)
df_new10 = pd.DataFrame(columns=colum_name_new)
df_new11 = pd.DataFrame(columns=colum_name_new)
df_new12 = pd.DataFrame(columns=colum_name_new)
df_new13 = pd.DataFrame(columns=colum_name_new)
df_new14 = pd.DataFrame(columns=colum_name_new)
df_new15 = pd.DataFrame(columns=colum_name_new)
df_new16 = pd.DataFrame(columns=colum_name_new)
df_new17 = pd.DataFrame(columns=colum_name_new)
df_new18 = pd.DataFrame(columns=colum_name_new)
df_new19 = pd.DataFrame(columns=colum_name_new)
df_new20 = pd.DataFrame(columns=colum_name_new)

data_num = 512
interval = 6


df_feature = pd.DataFrame()
df = pd.read_csv(data_path)
data = df.to_numpy()
scaler = StandardScaler()


# pca process
pca = decomposition.PCA(27)

for i in tqdm(range(1)):
    if i==0:
        star_layer = 1
        end_layer = 14
    elif i==1:
        star_layer = 10
        end_layer = 14
    data_pca = data[:,((star_layer-1)*data_num):(end_layer*data_num)]
    # data_pca = data[:,((i-1)*data_num):(i*data_num)]
    scaler.fit(data_pca)
    scaled_data = scaler.transform(data_pca)
    scaled_mean = np.mean(data_pca,axis=0)
    scaled_std = np.std(data_pca,axis=0)
    
    # print(scaled_data*np.std(data_pca,axis=0)+np.mean(data_pca,axis=0))
    pca.fit(scaled_data)
    pca_result = pca.transform(scaled_data)
    pca_result_dataframe = pd.DataFrame(pca_result)
    df_feature = pd.concat([df_feature, pca_result_dataframe],axis=1)
    feature_std = np.std(pca_result,axis=0)
    whole_std = interval*feature_std/2
    

    # pca_min = np.amin(pca_result,axis=0)
    # pca_max = np.amax(pca_result,axis=0)
    # alpha1 = (pca_max[0] - pca_min[0])/interval
    # alpha2 = (pca_max[1] - pca_min[1])/interval
    b = np.matrix(pca.components_)
    
    # df_feature.reset_index().drop(['index'],axis=1)
    # df_feature.to_csv(pca_output_dir, index=False)
    # exit()
    # pca_weight = np.zeros((6,2))
    # for j in range(6):
    #     for i in range(2):
    #         each_pca_matrix = b[j,i*512*9:(i+1)*512*9]
    #         pca_weight[j,i] = each_pca_matrix.mean()
    # print(pca_weight)
    # exit()
   
    for j in range(interval+1):
        pca_result_new1_pre = np.zeros(27) + feature_std[0]*j-whole_std[0]
        pca_result_new2_pre = np.zeros(27) + feature_std[1]*j-whole_std[1]
        pca_result_new3_pre = np.zeros(27) + feature_std[2]*j-whole_std[2]
        pca_result_new4_pre = np.zeros(27) + feature_std[3]*j-whole_std[3]
        pca_result_new5_pre = np.zeros(27) + feature_std[4]*j-whole_std[4]
        pca_result_new6_pre = np.zeros(27) + feature_std[5]*j-whole_std[5]
        pca_result_new7_pre = np.zeros(27) + feature_std[6]*j-whole_std[6]
        pca_result_new8_pre = np.zeros(27) + feature_std[7]*j-whole_std[7]
        pca_result_new9_pre = np.zeros(27) + feature_std[8]*j-whole_std[8]
        pca_result_new10_pre = np.zeros(27) + feature_std[9]*j-whole_std[9]
        
        
        # pca_result_new16_pre = pca_result[:,15] + 0.5*feature_std[15]*j-whole_std[15]
        
        
        '''pca_result_new1_result = np.concatenate((pca_result_new1_pre.reshape(27,1),pca_result[:,1].reshape(27,1),pca_result[:,2].reshape(27,1)),axis=1)
        pca_result_new2_result = np.concatenate((pca_result[:,0].reshape(27,1), pca_result_new2_pre.reshape(27,1),pca_result[:,2].reshape(27,1)),axis=1)
        pca_result_new3_result = np.concatenate((pca_result[:,0].reshape(27,1),pca_result[:,1].reshape(27,1), pca_result_new3_pre.reshape(27,1)),axis=1)'''
        pca_result_new1_result = np.concatenate((pca_result_new1_pre.reshape(27,1),pca_result[3675:3702,1].reshape(27,1),pca_result[3675:3702,2].reshape(27,1),pca_result[3675:3702,3].reshape(27,1),pca_result[3675:3702,4].reshape(27,1),pca_result[3675:3702,5].reshape(27,1),pca_result[3675:3702,6:].reshape(27,21)),axis=1)
        pca_result_new2_result = np.concatenate((pca_result[3675:3702,0].reshape(27,1), pca_result_new2_pre.reshape(27,1),pca_result[3675:3702,2].reshape(27,1),pca_result[3675:3702,3].reshape(27,1),pca_result[3675:3702,4].reshape(27,1),pca_result[3675:3702,5].reshape(27,1),pca_result[3675:3702,6:].reshape(27,21)),axis=1)
        pca_result_new3_result = np.concatenate((pca_result[3675:3702,0].reshape(27,1),pca_result[3675:3702,1].reshape(27,1), pca_result_new3_pre.reshape(27,1),pca_result[3675:3702,3].reshape(27,1),pca_result[3675:3702,4].reshape(27,1),pca_result[3675:3702,5].reshape(27,1),pca_result[3675:3702,6:].reshape(27,21)),axis=1)
        pca_result_new4_result = np.concatenate((pca_result[3675:3702,0].reshape(27,1),pca_result[3675:3702,1].reshape(27,1),pca_result[3675:3702,2].reshape(27,1),pca_result_new4_pre.reshape(27,1),pca_result[3675:3702,4].reshape(27,1),pca_result[3675:3702,5].reshape(27,1),pca_result[3675:3702,6:].reshape(27,21)),axis=1)
        pca_result_new5_result = np.concatenate((pca_result[3675:3702,0].reshape(27,1),pca_result[3675:3702,1].reshape(27,1),pca_result[3675:3702,2].reshape(27,1),pca_result[3675:3702,3].reshape(27,1),pca_result_new5_pre.reshape(27,1),pca_result[3675:3702,5].reshape(27,1),pca_result[3675:3702,6:].reshape(27,21)),axis=1)
        pca_result_new6_result = np.concatenate((pca_result[3675:3702,0].reshape(27,1),pca_result[3675:3702,1].reshape(27,1),pca_result[3675:3702,2].reshape(27,1),pca_result[3675:3702,3].reshape(27,1),pca_result[3675:3702,4].reshape(27,1),pca_result_new6_pre.reshape(27,1),pca_result[3675:3702,6:].reshape(27,21)),axis=1)
        pca_result_new7_result = np.concatenate((pca_result[3675:3702,0].reshape(27,1),pca_result[3675:3702,1].reshape(27,1),pca_result[3675:3702,2].reshape(27,1),pca_result[3675:3702,3].reshape(27,1),pca_result[3675:3702,4].reshape(27,1),pca_result[3675:3702,5].reshape(27,1),pca_result_new7_pre.reshape(27,1),pca_result[3675:3702,7:].reshape(27,20)),axis=1)
        pca_result_new8_result = np.concatenate((pca_result[3675:3702,0].reshape(27,1),pca_result[3675:3702,1].reshape(27,1),pca_result[3675:3702,2].reshape(27,1),pca_result[3675:3702,3].reshape(27,1),pca_result[3675:3702,4].reshape(27,1),pca_result[3675:3702,5].reshape(27,1),pca_result[3675:3702,6].reshape(27,1),pca_result_new8_pre.reshape(27,1),pca_result[3675:3702,8:].reshape(27,19)),axis=1)
        pca_result_new9_result = np.concatenate((pca_result[3675:3702,0].reshape(27,1),pca_result[3675:3702,1].reshape(27,1),pca_result[3675:3702,2].reshape(27,1),pca_result[3675:3702,3].reshape(27,1),pca_result[3675:3702,4].reshape(27,1),pca_result[3675:3702,5].reshape(27,1),pca_result[3675:3702,6].reshape(27,1),pca_result[3675:3702,7].reshape(27,1),pca_result_new9_pre.reshape(27,1),pca_result[3675:3702,9:].reshape(27,18)),axis=1)
        pca_result_new10_result = np.concatenate((pca_result[3675:3702,0].reshape(27,1),pca_result[3675:3702,1].reshape(27,1),pca_result[3675:3702,2].reshape(27,1),pca_result[3675:3702,3].reshape(27,1),pca_result[3675:3702,4].reshape(27,1),pca_result[3675:3702,5].reshape(27,1),pca_result[3675:3702,6].reshape(27,1),pca_result[3675:3702,7].reshape(27,1),pca_result[3675:3702,8].reshape(27,1),pca_result_new10_pre.reshape(27,1),pca_result[3675:3702,10:].reshape(27,17)),axis=1)
        # pca_result_new16_result = np.concatenate((pca_result[:,0].reshape(27,1),pca_result[:,1].reshape(27,1),pca_result[:,2].reshape(27,1),pca_result[:,3].reshape(27,1),pca_result[:,4].reshape(27,1),pca_result[:,5].reshape(27,1),pca_result[:,6].reshape(27,1),pca_result[:,7].reshape(27,1),pca_result[:,8].reshape(27,1),pca_result[:,9].reshape(27,1),pca_result[:,10].reshape(27,1),pca_result[:,11].reshape(27,1),pca_result[:,12].reshape(27,1),pca_result[:,13].reshape(27,1),pca_result[:,14].reshape(27,1),pca_result_new16_pre.reshape(27,1),pca_result[:,16:].reshape(27,11)),axis=1)
        
        
        
        a1 = np.matrix(pca_result_new1_result)
        a2 = np.matrix(pca_result_new2_result)
        a3 = np.matrix(pca_result_new3_result)
        a4 = np.matrix(pca_result_new4_result)
        a5 = np.matrix(pca_result_new5_result)
        a6 = np.matrix(pca_result_new6_result)
        a7 = np.matrix(pca_result_new7_result)
        a8 = np.matrix(pca_result_new8_result)
        a9 = np.matrix(pca_result_new9_result)
        a10 = np.matrix(pca_result_new10_result)
        
        c1 = a1 * b
        c2 = a2 * b
        c3 = a3 * b
        c4 = a4 * b
        c5 = a5 * b
        c6 = a6 * b
        c7 = a7 * b
        c8 = a8 * b
        c9 = a9 * b
        c10 = a10 * b

        pca_new1 = np.asarray(c1)*scaled_std+scaled_mean
        pca_new2 = np.asarray(c2)*scaled_std+scaled_mean
        pca_new3 = np.asarray(c3)*scaled_std+scaled_mean
        pca_new4 = np.asarray(c4)*scaled_std+scaled_mean
        pca_new5 = np.asarray(c5)*scaled_std+scaled_mean
        pca_new6 = np.asarray(c6)*scaled_std+scaled_mean
        pca_new7 = np.asarray(c7)*scaled_std+scaled_mean
        pca_new8 = np.asarray(c8)*scaled_std+scaled_mean
        pca_new9 = np.asarray(c9)*scaled_std+scaled_mean
        pca_new10 = np.asarray(c10)*scaled_std+scaled_mean
        
        data_pca1 = copy.deepcopy(data[3675:3702,:])
        data_pca2= copy.deepcopy(data[3675:3702,:])
        data_pca3= copy.deepcopy(data[3675:3702,:])
        data_pca4 = copy.deepcopy(data[3675:3702,:])
        data_pca5= copy.deepcopy(data[3675:3702,:])
        data_pca6= copy.deepcopy(data[3675:3702,:])
        data_pca7= copy.deepcopy(data[3675:3702,:])
        data_pca8= copy.deepcopy(data[3675:3702,:])
        data_pca9= copy.deepcopy(data[3675:3702,:])
        data_pca10= copy.deepcopy(data[3675:3702,:])

        data_pca1[:,((star_layer-1)*data_num):(end_layer*data_num)] = pca_new1
        data_pca2[:,((star_layer-1)*data_num):(end_layer*data_num)] = pca_new2
        data_pca3[:,((star_layer-1)*data_num):(end_layer*data_num)] = pca_new3
        data_pca4[:,((star_layer-1)*data_num):(end_layer*data_num)] = pca_new4
        data_pca5[:,((star_layer-1)*data_num):(end_layer*data_num)] = pca_new5
        data_pca6[:,((star_layer-1)*data_num):(end_layer*data_num)] = pca_new6
        data_pca7[:,((star_layer-1)*data_num):(end_layer*data_num)] = pca_new7
        data_pca8[:,((star_layer-1)*data_num):(end_layer*data_num)] = pca_new8
        data_pca9[:,((star_layer-1)*data_num):(end_layer*data_num)] = pca_new9
        data_pca10[:,((star_layer-1)*data_num):(end_layer*data_num)] = pca_new10
        
        data_pca_new = data_pca10
        df_pca1 = pd.DataFrame(data_pca_new[0,:].reshape(1,-1), columns = colum_name_new)
        df_pca2 = pd.DataFrame(data_pca_new[1,:].reshape(1,-1), columns = colum_name_new)
        df_pca3 = pd.DataFrame(data_pca_new[2,:].reshape(1,-1), columns = colum_name_new)
        df_pca4 = pd.DataFrame(data_pca_new[3,:].reshape(1,-1), columns = colum_name_new)
        df_pca5 = pd.DataFrame(data_pca_new[4,:].reshape(1,-1), columns = colum_name_new)
        df_pca6 = pd.DataFrame(data_pca_new[5,:].reshape(1,-1), columns = colum_name_new)
        df_pca7 = pd.DataFrame(data_pca_new[6,:].reshape(1,-1), columns = colum_name_new)
        df_pca8 = pd.DataFrame(data_pca_new[7,:].reshape(1,-1), columns = colum_name_new)
        df_pca9 = pd.DataFrame(data_pca_new[8,:].reshape(1,-1), columns = colum_name_new)
        df_pca10 = pd.DataFrame(data_pca_new[9,:].reshape(1,-1), columns = colum_name_new)
        df_pca11 = pd.DataFrame(data_pca_new[10,:].reshape(1,-1), columns = colum_name_new)
        df_pca12 = pd.DataFrame(data_pca_new[11,:].reshape(1,-1), columns = colum_name_new)
        df_pca13 = pd.DataFrame(data_pca_new[12,:].reshape(1,-1), columns = colum_name_new)
        df_pca14 = pd.DataFrame(data_pca_new[13,:].reshape(1,-1), columns = colum_name_new)
        df_pca15 = pd.DataFrame(data_pca_new[14,:].reshape(1,-1), columns = colum_name_new)
        df_pca16 = pd.DataFrame(data_pca_new[15,:].reshape(1,-1), columns = colum_name_new)
        df_pca17 = pd.DataFrame(data_pca_new[16,:].reshape(1,-1), columns = colum_name_new)
        df_pca18 = pd.DataFrame(data_pca_new[17,:].reshape(1,-1), columns = colum_name_new)
        df_pca19 = pd.DataFrame(data_pca_new[18,:].reshape(1,-1), columns = colum_name_new)
        df_pca20 = pd.DataFrame(data_pca_new[19,:].reshape(1,-1), columns = colum_name_new)

        df_new1 = pd.concat([df_new1, df_pca1])
        df_new2 = pd.concat([df_new2, df_pca2])
        df_new3 = pd.concat([df_new3, df_pca3])
        df_new4 = pd.concat([df_new4, df_pca4])
        df_new5 = pd.concat([df_new5, df_pca5])
        df_new6 = pd.concat([df_new6, df_pca6])
        df_new7 = pd.concat([df_new7, df_pca7])
        df_new8 = pd.concat([df_new8, df_pca8])
        df_new9 = pd.concat([df_new9, df_pca9])
        df_new10 = pd.concat([df_new10, df_pca10])
        df_new11 = pd.concat([df_new11, df_pca11])
        df_new12 = pd.concat([df_new12, df_pca12])
        df_new13 = pd.concat([df_new13, df_pca13])
        df_new14 = pd.concat([df_new14, df_pca14])
        df_new15 = pd.concat([df_new15, df_pca15])
        df_new16 = pd.concat([df_new16, df_pca16])
        df_new17 = pd.concat([df_new17, df_pca17])
        df_new18 = pd.concat([df_new18, df_pca18])
        df_new19 = pd.concat([df_new19, df_pca19])
        df_new20 = pd.concat([df_new20, df_pca20])
save_data(df_new1,df_new2,df_new3,df_new4,df_new5,df_new6,df_new7,df_new8,df_new9,df_new10,df_new11,df_new12,df_new13,df_new14,df_new15,df_new16,df_new17,df_new18,df_new19,df_new20,10)

