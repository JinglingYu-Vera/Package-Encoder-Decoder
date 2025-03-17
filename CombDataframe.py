from typing import List
import numpy as np
import pandas as pd
import xlrd

data_path = './data/img_feature/feature_pca.xlsx'
data_dir = './data/dataset/ResNetl5_time_Q2_new.xlsx'
output_dir = './data/dataset/psp_time_Q2.xlsx'

class LossCalculate():
    def __init__(self):
        super(LossCalculate, self).__init__()

    def read_excel(self, file_name):
        xls = pd.ExcelFile(file_name)
        self.df = pd.read_excel(xls)

    def forward(self):
        colum_name_new = ['ID', 'Sub_ID', 'Task_ID', 'Target', 'Question', 'T', 'Choice', 'Distractor Similarity','Similarity 1', 'Similarity 2', 'Similarity 3', 'Similarity 4', 'Similarity 5', 'Similarity 6', 'Similarity 7', 'Similarity 8', 'Similarity 9', 'Similarity 10', 'Similarity 11', 'Similarity 12', 'Similarity 13', 'Similarity 14', 'Similarity 15', 'Similarity 16', 'Similarity 17', 'Similarity 18', 'Similarity 19', 'Similarity 20', 'Similarity 21', 'Similarity 22', 'Similarity 23', 'Similarity 24', 'Similarity 25', 'Similarity 26', 'Similarity 27', 'Refix 1', 'Refix 2', 'Refix 3', 'Refix 4', 'Refix 5', 'Refix 6', 'Refix 7', 'Refix 8', 'Refix 9', 'Refix 10', 'Refix 11', 'Refix 12', 'Refix 13', 'Refix 14', 'Refix 15', 'Refix 16', 'Refix 17', 'Refix 18', 'Refix 19', 'Refix 20', 'Refix 21', 'Refix 22','Refix 23', 'Refix 24', 'Refix 25', 'Refix 26', 'Refix 27', 'Revisit 1', 'Revisit 2', 'Revisit 3', 'Revisit 4', 'Revisit 5', 'Revisit 6', 'Revisit 7', 'Revisit 8', 'Revisit 9', 'Revisit 10', 'Revisit 11', 'Revisit 12', 'Revisit 13', 'Revisit 14', 'Revisit 15', 'Revisit 16', 'Revisit 17', 'Revisit 18', 'Revisit 19', 'Revisit 20', 'Revisit 21', 'Revisit 22','Revisit 23', 'Revisit 24', 'Revisit 25', 'Revisit 26', 'Revisit 27','Saliency 1', 'Saliency 2', 'Saliency 3', 'Saliency 4', 'Saliency 5', 'Saliency 6', 'Saliency 7', 'Saliency 8', 'Saliency 9', 'Saliency 10', 'Saliency 11', 'Saliency 12', 'Saliency 13', 'Saliency 14', 'Saliency 15', 'Saliency 16', 'Saliency 17', 'Saliency 18', 'Saliency 19', 'Saliency 20', 'Saliency 21', 'Saliency 22', 'Saliency 23', 'Saliency 24', 'Saliency 25', 'Saliency 26', 'Saliency 27','img1-1','img1-2','img1-3','img2-1','img2-2','img2-3']
        df_new = pd.DataFrame(columns=colum_name_new)

        df1 = self.df[['ID', 'Sub_ID', 'Task_ID', 'Target', 'Question', 'T', 'Choice', 'Distractor Similarity','Similarity 1', 'Similarity 2', 'Similarity 3', 'Similarity 4', 'Similarity 5', 'Similarity 6', 'Similarity 7', 'Similarity 8', 'Similarity 9', 'Similarity 10', 'Similarity 11', 'Similarity 12', 'Similarity 13', 'Similarity 14', 'Similarity 15', 'Similarity 16', 'Similarity 17', 'Similarity 18', 'Similarity 19', 'Similarity 20', 'Similarity 21', 'Similarity 22', 'Similarity 23', 'Similarity 24', 'Similarity 25', 'Similarity 26', 'Similarity 27', 'Refix 1', 'Refix 2', 'Refix 3', 'Refix 4', 'Refix 5', 'Refix 6', 'Refix 7', 'Refix 8', 'Refix 9', 'Refix 10', 'Refix 11', 'Refix 12', 'Refix 13', 'Refix 14', 'Refix 15', 'Refix 16', 'Refix 17', 'Refix 18', 'Refix 19', 'Refix 20', 'Refix 21', 'Refix 22','Refix 23', 'Refix 24', 'Refix 25', 'Refix 26', 'Refix 27', 'Revisit 1', 'Revisit 2', 'Revisit 3', 'Revisit 4', 'Revisit 5', 'Revisit 6', 'Revisit 7', 'Revisit 8', 'Revisit 9', 'Revisit 10', 'Revisit 11', 'Revisit 12', 'Revisit 13', 'Revisit 14', 'Revisit 15', 'Revisit 16', 'Revisit 17', 'Revisit 18', 'Revisit 19', 'Revisit 20', 'Revisit 21', 'Revisit 22','Revisit 23', 'Revisit 24', 'Revisit 25', 'Revisit 26', 'Revisit 27','Saliency 1', 'Saliency 2', 'Saliency 3', 'Saliency 4', 'Saliency 5', 'Saliency 6', 'Saliency 7', 'Saliency 8', 'Saliency 9', 'Saliency 10', 'Saliency 11', 'Saliency 12', 'Saliency 13', 'Saliency 14', 'Saliency 15', 'Saliency 16', 'Saliency 17', 'Saliency 18', 'Saliency 19', 'Saliency 20', 'Saliency 21', 'Saliency 22', 'Saliency 23', 'Saliency 24', 'Saliency 25', 'Saliency 26', 'Saliency 27']]
        
        img_feature_dataframe = pd.read_excel(data_path)
       
        lines = df1.shape[0]
        for i in range(lines):
            print(i)
            df2 = df1[i:i+1]
            df2_new = df2.reset_index(drop=True)
            target_name = list(df2['Target'])
            target_name = "".join(target_name)
            
            
            img_feature_name = target_name[3:]
            
            img_feature = img_feature_dataframe[(int(img_feature_name)-1):int(img_feature_name)]
            img_feature_new = img_feature.reset_index(drop=True)
            # img_feature_name = "".join(img_feature_name)
            
            
            df3 = pd.concat([df2_new, img_feature_new], axis=1)
            df_new = pd.concat([df_new, df3])
        df_new.reset_index().drop(['index'],axis=1)
        df_new.to_excel(output_dir, index=False)
        print('finish')

if __name__ == '__main__':
    LossCalculate = LossCalculate()
    LossCalculate.read_excel(data_dir)
    LossCalculate.forward() 