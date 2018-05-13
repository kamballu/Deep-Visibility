# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import cv2

from data_utils import *
from collections import defaultdict

class db_TID2013():
    def __init__(self,data_source, batch_size, patch, patch_size_w,patch_size_h):
        self.data_source = data_source
        self.patch = patch
        self.patch_size = [patch_size_w, patch_size_h]
        if not os.path.isdir(self.data_source):
            raise(Exception('No folder at source location'))
        self.poplulate_file_queue()
        self.counter = 0
        self.batch_size = batch_size
        
    def poplulate_file_queue(self):
        location = self.data_source
        image_list = location + "mos_with_names.txt"
        
        with open(image_list) as f:
            bigdata = f.read()
        
        Alldata = bigdata.split('\n')        
        refnum = 0;
        olref = 'a'
        refid = []
        
        tmp_reference_image = []
        tmp_distorted_image= []
        tmp_scores = []
        dct = defaultdict(list)

        for i in range(len(Alldata)):    
            fields = Alldata[i].split('\r')[0]
            fields = fields.split()            
            sub_fields = fields[1].split('_')
            
            dfilename = location+'\\distorted_images\\'+fields[1]
            rfilename = location+'reference_images\\'+sub_fields[0].upper()+'.BMP'
            score =  np.float32(fields[0])
            dct[rfilename].append(i)
            tmp_reference_image.append(rfilename)
            tmp_distorted_image.append(dfilename)
            tmp_scores.append(score)
        # Seperate images based on reference image content
        self.dist =tmp_distorted_image 
        self.ref = tmp_reference_image
        self.mos = tmp_scores
        tmp_scores = 1 - tmp_scores/np.max(tmp_scores)
        index_of_contents = []
        c_length = len(dct.keys() )
        for count, key in enumerate(dct.keys()):
            index_of_contents.append( dct[key] )
        [train_content, test_content] = train_test_split(c_length)
        file_id_list_train = []
        for indexs in train_content:
            file_id_list_train += index_of_contents[indexs]
            
        file_id_list_test = []
        for indexs in test_content:
            file_id_list_test += index_of_contents[indexs]
        # Global storate of file list
        self.file_reference_image_train = []
        self.file_distorted_image_train= []
        self.MOS_scores_train = []
        self.file_reference_image_test = []
        self.file_distorted_image_test= []
        self.MOS_scores_test = []
        
        for file_idxs in file_id_list_train:
            self.file_distorted_image_train.append(tmp_distorted_image[file_idxs])
            self.file_reference_image_train.append(tmp_reference_image[file_idxs])
            self.MOS_scores_train.append(tmp_scores[file_idxs])
        
        for file_idxs in file_id_list_test:
            self.file_distorted_image_test.append(tmp_distorted_image[file_idxs])
            self.file_reference_image_test.append(tmp_reference_image[file_idxs])
            self.MOS_scores_test.append(tmp_scores[file_idxs])
        self.refill_file_queues()
    def get_test_sets(self):
        return [self.file_distorted_image_test,self.file_reference_image_test,self.MOS_scores_test ]
    def refill_file_queues(self):
        # Queue for read
        self.tmp_queue_train_ref = self.file_reference_image_train[:]
        self.tmp_queue_train = self.file_distorted_image_train[:]
        self.tmp_queue_train_mos = self.MOS_scores_train[:]
        
        self.tmp_queue_test_ref = self.file_reference_image_test[:]
        self.tmp_queue_test = self.file_distorted_image_test[:]
        self.tmp_queue_test_mos = self.MOS_scores_test[:]
        
    def get_count(self):
        return(len (self.file_reference_image_train))
        
    def get_next_batch_2(self,n1,n2,show=0,subsample =1):
#        if self.patch:
#            if self.patch_size <= 0:
#                raise(ValueError('Need patch size more than 0 in patch mode.'))
        x_ref = []        
        x_dis = []
        y_mos = []
        if self.batch_size >= self.get_count():
            self.batch_size = len(self.tmp_queue_train)-1
        if len(self.tmp_queue_train) > self.batch_size:
            for ids in range(n1,n2):
                train_ids = ids 
                
                fname_dis_img = self.dist[train_ids]
                fname_ref_img = self.ref[train_ids]
                mos_train = self.mos[train_ids]
#                print(fname_dis_img)
                im_ref = get_luminance( np.float32( cv2.imread(fname_ref_img)) )
                im_dis =  get_luminance( np.float32( cv2.imread(fname_dis_img) ) )
                if im_ref is None:
                    print ((('No image at '+fname_ref_img)))                    
                    continue
                if im_dis is None:
                    print ((('No image at '+fname_dis_img)))                    
                    continue
                
                
                if len(im_dis.shape)==3:
                    c = im_dis.shape[2]
                else:
                    c = 1
                    im_dis = im_dis[:,:,np.newaxis]
                    im_ref = im_ref[:,:,np.newaxis]
                    
                if self.patch:
                    im_ref,im_dis,mos_train, _ =  extractpatchwithLabel( im_ref, 
                                    im_dis ,self.patch_size[0],self.patch_size[1],
                                    score=mos_train)
                x_dis+=[im_dis]
                x_ref+=[im_ref]
                y_mos+=[mos_train]
            x_dis = np.concatenate(x_dis)
            x_ref = np.concatenate(x_ref)
            y_mos = np.concatenate(y_mos).reshape(-1)
#            print(x_dis.shape)
            
            if len(np.shape(x_dis)) >4:
                [bz, npatch, pw,ph,c] = np.shape(x_dis)
                x_dis = x_dis.reshape(bz*npatch, pw,ph,c)
                x_ref = x_ref.reshape(bz*npatch, pw,ph,c)
                y_mos = y_mos.reshape(bz*npatch, 1)
            self.counter += 1
        else:
            self.refill_file_queues()
            self.counter = 0
            [x_dis,x_ref,y_mos] = self.get_next_batch(self.patch,show)
        return( [x_dis, x_ref,y_mos] )
        
    def get_next_batch(self,show=0,subsample =1):
#        if self.patch:
#            if self.patch_size <= 0:
#                raise(ValueError('Need patch size more than 0 in patch mode.'))
        x_ref = []        
        x_dis = []
        y_mos = []
        
        if len(self.tmp_queue_train) > self.batch_size:
            for ids in range(self.batch_size-1):
                train_ids = np.random.randint(0,len(self.tmp_queue_train) )
                
                fname_dis_img = self.tmp_queue_train.pop(train_ids)
                fname_ref_img = self.tmp_queue_train_ref.pop(train_ids)
                mos_train = self.tmp_queue_train_mos.pop(train_ids)
                im_ref = get_luminance( np.float32( cv2.imread(fname_ref_img)) )
                im_dis =  get_luminance( np.float32( cv2.imread(fname_dis_img) ) )
                if im_ref is None:
                    print ((('No image at '+fname_ref_img)))                    
                    continue
                if im_dis is None:
                    print ((('No image at '+fname_dis_img)))                    
                    continue
#                flip_var = np.random.rand()
#                if  flip_var> 0.8:
#                    im_ref = cv2.flip( im_ref, 0 )
#                    im_dis = cv2.flip( im_dis, 0 )
#                elif flip_var>0.6:
#                    im_ref = cv2.flip( im_ref, 1 )
#                    im_dis = cv2.flip( im_dis, 1 )
#                elif flip_var>0.4:
#                    im_ref = cv2.flip( im_ref, -1 )
#                    im_dis = cv2.flip( im_dis, -1 )
                
                if len(im_dis.shape)==3:
                    c = im_dis.shape[2]
                else:
                    c = 1
                    im_dis = im_dis[:,:,np.newaxis]
                    im_ref = im_ref[:,:,np.newaxis]
                    
                if self.patch:
                    im_ref,im_dis,mos_train, _ =  extract_patch_random( im_ref, 
                                    im_dis ,self.patch_size[0],self.patch_size[1],
                                    score=mos_train)
                x_dis+=[im_dis]
                x_ref+=[im_ref]
                y_mos+=[mos_train]
            rid = np.random.permutation(len(y_mos))
            x_dis = np.array(x_dis)[rid]
            x_ref = np.array(x_ref)[rid]
            y_mos = np.array(y_mos)[rid]
            
            if len(np.shape(x_dis)) >4:
                [bz, npatch, pw,ph,c] = np.shape(x_dis)
                x_dis = x_dis.reshape(bz*npatch, pw,ph,c)
                x_ref = x_ref.reshape(bz*npatch, pw,ph,c)
                y_mos = y_mos.reshape(bz*npatch, 1)
            self.counter += 1
        else:
            self.refill_file_queues()
            self.counter = 0
            [x_dis,x_ref,y_mos] = self.get_next_batch(self.patch,show)
        return( [x_dis, x_ref,y_mos] )


class db_CSIQ():
    def __init__(self,data_source, batch_size, patch, patch_size_w,patch_size_h):
        self.data_source = data_source
        self.patch = patch
        self.patch_size = [patch_size_w, patch_size_h]
        if not os.path.isdir(self.data_source):
            raise(Exception('No folder at source location'))
        self.poplulate_file_queue()
        self.counter = 0
        if len(self.file_distorted_image_train) > batch_size:
            self.batch_size = batch_size
        else:
            self.batch_size = len(self.file_distorted_image_train)-1
        
    def poplulate_file_queue(self):
        location = self.data_source
        image_list = location + "data.txt"
        dct = defaultdict(list)
        tmp_reference_image = []
        tmp_distorted_image= []
        tmp_scores = []
        with open(location+"data.txt") as f:
            file_contents = f.read()
            lines = file_contents.split('\n')
            lines = [item for item in lines if item != '']
            old_content = '-1'
            for i in range(len(lines)):# for the full dataset
                if i == 0:
                    continue
                line_data = lines[i].split('\t')
                line_data
                
                dfilename = location+ str(line_data[3])+'\\'+str(line_data[0])+'.'+str(line_data[3])+'.'+str(line_data[4])+'.png';
                rfilename = location + str(line_data[0]) + '.png'
                
                score = np.float16(line_data[6])
                content = int(line_data[1])
                ntype = int(line_data[2])
                nlevel = line_data[4]
                dct[rfilename].append(i-1)
                tmp_reference_image.append(rfilename)
                tmp_distorted_image.append(dfilename)
                tmp_scores.append(score)
#                print(dfilename)
                
        # Seperate images based on reference image content
        self.dist =tmp_distorted_image 
        self.ref = tmp_reference_image
        self.mos = tmp_scores
        index_of_contents = []
        c_length = len(dct.keys() )
        for count, key in enumerate(dct.keys()):
            index_of_contents.append( dct[key] )
        [train_content, test_content] = train_test_split(c_length, ratio=0.99)
        file_id_list_train = []
        for indexs in train_content:
            file_id_list_train += index_of_contents[indexs]
            
        file_id_list_test = []
        for indexs in test_content:
            file_id_list_test += index_of_contents[indexs]
        # Global storate of file list
        self.file_reference_image_train = []
        self.file_distorted_image_train= []
        self.MOS_scores_train = []
        self.file_reference_image_test = []
        self.file_distorted_image_test= []
        self.MOS_scores_test = []
        
        for file_idxs in file_id_list_train:
            
            self.file_distorted_image_train.append(tmp_distorted_image[file_idxs])
            self.file_reference_image_train.append(tmp_reference_image[file_idxs])
            self.MOS_scores_train.append(tmp_scores[file_idxs])
        
        for file_idxs in file_id_list_test:
            
            self.file_distorted_image_test.append(tmp_distorted_image[file_idxs])
            self.file_reference_image_test.append(tmp_reference_image[file_idxs])
            self.MOS_scores_test.append(tmp_scores[file_idxs])
        self.refill_file_queues()
    def get_all(self):
        D = self.file_distorted_image_train+self.file_distorted_image_test
        R = self.file_reference_image_train+self.file_reference_image_test
        S = self.MOS_scores_train+self.MOS_scores_test
        return [D,R,S]
    def get_test_sets(self,num=0):
        if num==0:
            return [self.file_distorted_image_test,self.file_reference_image_test,self.MOS_scores_test ]
        else:
            return [self.file_distorted_image_test[0:1],self.file_reference_image_test[0:1],self.MOS_scores_test[0:1] ]
    def refill_file_queues(self):
        # Queue for read
        self.tmp_queue_train_ref = self.file_reference_image_train[:]
        self.tmp_queue_train = self.file_distorted_image_train[:]
        self.tmp_queue_train_mos = self.MOS_scores_train[:]
        
        self.tmp_queue_test_ref = self.file_reference_image_test[:]
        self.tmp_queue_test = self.file_distorted_image_test[:]
        self.tmp_queue_test_mos = self.MOS_scores_test[:]
        
    def get_count(self):
        return(len (self.file_reference_image_train))
        
    def get_next_batch(self,show=0,subsample =1):
#        if self.patch:
#            if self.patch_size <= 0:
#                raise(ValueError('Need patch size more than 0 in patch mode.'))
        x_ref = []        
        x_dis = []
        y_mos = []
        if self.batch_size >= self.get_count():
            self.batch_size = len(self.tmp_queue_train)-1
        if len(self.tmp_queue_train) > self.batch_size:
            for ids in range(self.batch_size-1):
                train_ids = np.random.randint(0,len(self.tmp_queue_train) )
                
                fname_dis_img = self.tmp_queue_train.pop(train_ids)
                fname_ref_img = self.tmp_queue_train_ref.pop(train_ids)
                mos_train = self.tmp_queue_train_mos.pop(train_ids)
                
                im_ref = get_luminance( np.float32( cv2.imread(fname_ref_img)) )
                im_dis =  get_luminance( np.float32( cv2.imread(fname_dis_img) ) )
                if im_ref is None:
                    print ((('No image at '+fname_ref_img)))                    
                    continue
                if im_dis is None:
                    print ((('No image at '+fname_dis_img)))                    
                    continue
                
                
                if len(im_dis.shape)==3:
                    c = im_dis.shape[2]
                else:
                    c = 1
                    im_dis = im_dis[:,:,np.newaxis]
                    im_ref = im_ref[:,:,np.newaxis]
                    
                if self.patch:
                    im_ref,im_dis,mos_train, _ =  extract_patch_random( im_ref, 
                                    im_dis ,self.patch_size[0],self.patch_size[1],
                                    score=mos_train)
                x_dis+=[im_dis]
                x_ref+=[im_ref]
                y_mos+=[mos_train]
            rid = np.random.permutation(len(y_mos)*len(x_dis[0]))
#            print(len(x_dis))
            x_dis = np.concatenate(x_dis)[rid]
            x_ref = np.concatenate(x_ref)[rid]
            y_mos = np.concatenate(y_mos)[rid].reshape(-1)
#            print(x_dis.shape)
            
            if len(np.shape(x_dis)) >4:
                [bz, npatch, pw,ph,c] = np.shape(x_dis)
                x_dis = x_dis.reshape(bz*npatch, pw,ph,c)
                x_ref = x_ref.reshape(bz*npatch, pw,ph,c)
                y_mos = y_mos.reshape(bz*npatch, 1)
            self.counter += 1
        else:
            self.refill_file_queues()
            self.counter = 0
            [x_dis,x_ref,y_mos] = self.get_next_batch(self.patch,show)
        return( [x_dis, x_ref,y_mos] )
        
    def get_next_batch_2(self,n1,n2,show=0,subsample =1):
#        if self.patch:
#            if self.patch_size <= 0:
#                raise(ValueError('Need patch size more than 0 in patch mode.'))
        x_ref = []        
        x_dis = []
        y_mos = []
        if self.batch_size >= self.get_count():
            self.batch_size = len(self.tmp_queue_train)-1
        if len(self.tmp_queue_train) > self.batch_size:
            for ids in range(n1,n2):
                train_ids = ids 
                
                fname_dis_img = self.dist[train_ids]
                fname_ref_img = self.ref[train_ids]
                mos_train = self.mos[train_ids]
#                print(fname_dis_img)
                im_ref = get_luminance( np.float32( cv2.imread(fname_ref_img)) )
                im_dis =  get_luminance( np.float32( cv2.imread(fname_dis_img) ) )
                if im_ref is None:
                    print ((('No image at '+fname_ref_img)))                    
                    continue
                if im_dis is None:
                    print ((('No image at '+fname_dis_img)))                    
                    continue
                
                
                if len(im_dis.shape)==3:
                    c = im_dis.shape[2]
                else:
                    c = 1
                    im_dis = im_dis[:,:,np.newaxis]
                    im_ref = im_ref[:,:,np.newaxis]
                    
                if self.patch:
                    im_ref,im_dis,mos_train, _ =  extractpatchwithLabel( im_ref, 
                                    im_dis ,self.patch_size[0],self.patch_size[1],
                                    score=mos_train)
                x_dis+=[im_dis]
                x_ref+=[im_ref]
                y_mos+=[mos_train]
            x_dis = np.concatenate(x_dis)
            x_ref = np.concatenate(x_ref)
            y_mos = np.concatenate(y_mos).reshape(-1)
#            print(x_dis.shape)
            
            if len(np.shape(x_dis)) >4:
                [bz, npatch, pw,ph,c] = np.shape(x_dis)
                x_dis = x_dis.reshape(bz*npatch, pw,ph,c)
                x_ref = x_ref.reshape(bz*npatch, pw,ph,c)
                y_mos = y_mos.reshape(bz*npatch, 1)
            self.counter += 1
        else:
            self.refill_file_queues()
            self.counter = 0
            [x_dis,x_ref,y_mos] = self.get_next_batch(self.patch,show)
        return( [x_dis, x_ref,y_mos] )
        

class db_LIVE():
    def __init__(self,data_source, batch_size, patch, patch_size_w,patch_size_h):
        self.data_source = data_source
        self.patch = patch
        self.patch_size = [patch_size_w, patch_size_h]
        if not os.path.isdir(self.data_source):
            raise(Exception('No folder at source location'))
        self.poplulate_file_queue()
        self.counter = 0
        self.batch_size = batch_size
        
    def poplulate_file_queue(self):
        location = self.data_source
        image_list = location + "exp_table.txt"
        
        with open(image_list) as f:
            bigdata = f.read()
        
        Alldata = bigdata.split('\n')        
        refnum = 0;
        olref = 'a'
        refid = []
        
        tmp_reference_image = []
        tmp_distorted_image= []
        tmp_scores = []
        dct = defaultdict(list)

        for i in range(len(Alldata)):    
            fields = Alldata[i].split('\r')[0]
            fields = fields.split()            
            if fields==[]:
                continue
            
            dfilename = location+fields[0]
            rfilename = location+fields[1]
            score =  np.float32(fields[2])
            dct[rfilename].append(i)
            tmp_reference_image.append(rfilename)
            tmp_distorted_image.append(dfilename)
            tmp_scores.append(score)
        # Seperate images based on reference image content
        self.dist =tmp_distorted_image 
        self.ref = tmp_reference_image
        self.mos = tmp_scores
        tmp_scores = tmp_scores/np.max(tmp_scores)
        index_of_contents = []
        c_length = len(dct.keys() )
        for count, key in enumerate(dct.keys()):
            index_of_contents.append( dct[key] )
        [train_content, test_content] = train_test_split(c_length,0.99)
        file_id_list_train = []
        for indexs in train_content:
            file_id_list_train += index_of_contents[indexs]
            
        file_id_list_test = []
        for indexs in test_content:
            file_id_list_test += index_of_contents[indexs]
        # Global storate of file list
        self.file_reference_image_train = []
        self.file_distorted_image_train= []
        self.MOS_scores_train = []
        self.file_reference_image_test = []
        self.file_distorted_image_test= []
        self.MOS_scores_test = []
        
        for file_idxs in file_id_list_train:
            self.file_distorted_image_train.append(tmp_distorted_image[file_idxs])
            self.file_reference_image_train.append(tmp_reference_image[file_idxs])
            self.MOS_scores_train.append(tmp_scores[file_idxs])
        
        for file_idxs in file_id_list_test:
            self.file_distorted_image_test.append(tmp_distorted_image[file_idxs])
            self.file_reference_image_test.append(tmp_reference_image[file_idxs])
            self.MOS_scores_test.append(tmp_scores[file_idxs])
        self.refill_file_queues()
    def get_test_sets(self):
        return [self.file_distorted_image_test,self.file_reference_image_test,self.MOS_scores_test ]
    def refill_file_queues(self):
        # Queue for read
        self.tmp_queue_train_ref = self.file_reference_image_train[:]
        self.tmp_queue_train = self.file_distorted_image_train[:]
        self.tmp_queue_train_mos = self.MOS_scores_train[:]
        
        self.tmp_queue_test_ref = self.file_reference_image_test[:]
        self.tmp_queue_test = self.file_distorted_image_test[:]
        self.tmp_queue_test_mos = self.MOS_scores_test[:]
        
    def get_count(self):
        return(len (self.file_reference_image_train))
        
    def get_next_batch_2(self,n1,n2,show=0,subsample =1):
#        if self.patch:
#            if self.patch_size <= 0:
#                raise(ValueError('Need patch size more than 0 in patch mode.'))
        x_ref = []        
        x_dis = []
        y_mos = []
        if self.batch_size >= self.get_count():
            self.batch_size = len(self.tmp_queue_train)-1
        if len(self.tmp_queue_train) > self.batch_size:
            for ids in range(n1,n2):
                train_ids = ids 
                
                fname_dis_img = self.dist[train_ids]
                fname_ref_img = self.ref[train_ids]
                mos_train = self.mos[train_ids]
#                print(fname_dis_img)
                im_ref = get_luminance( np.float32( cv2.imread(fname_ref_img)) )
                im_dis =  get_luminance( np.float32( cv2.imread(fname_dis_img) ) )
                if im_ref is None:
                    print ((('No image at '+fname_ref_img)))                    
                    continue
                if im_dis is None:
                    print ((('No image at '+fname_dis_img)))                    
                    continue
                
                
                if len(im_dis.shape)==3:
                    c = im_dis.shape[2]
                else:
                    c = 1
                    im_dis = im_dis[:,:,np.newaxis]
                    im_ref = im_ref[:,:,np.newaxis]
                    
                if self.patch:
                    im_ref,im_dis,mos_train, _ =  extractpatchwithLabel( im_ref, 
                                    im_dis ,self.patch_size[0],self.patch_size[1],
                                    score=mos_train)
                x_dis+=[im_dis]
                x_ref+=[im_ref]
                y_mos+=[mos_train]
            x_dis = np.concatenate(x_dis)
            x_ref = np.concatenate(x_ref)
            y_mos = np.concatenate(y_mos).reshape(-1)
#            print(x_dis.shape)
            
            if len(np.shape(x_dis)) >4:
                [bz, npatch, pw,ph,c] = np.shape(x_dis)
                x_dis = x_dis.reshape(bz*npatch, pw,ph,c)
                x_ref = x_ref.reshape(bz*npatch, pw,ph,c)
                y_mos = y_mos.reshape(bz*npatch, 1)
            self.counter += 1
        else:
            self.refill_file_queues()
            self.counter = 0
            [x_dis,x_ref,y_mos] = self.get_next_batch(self.patch,show)
        return( [x_dis, x_ref,y_mos] )
        
    def get_next_batch(self,show=0,subsample =1):
#        if self.patch:
#            if self.patch_size <= 0:
#                raise(ValueError('Need patch size more than 0 in patch mode.'))
        x_ref = []        
        x_dis = []
        y_mos = []
        if self.batch_size >= self.get_count():
            self.batch_size = len(self.tmp_queue_train)-1
        if len(self.tmp_queue_train) > self.batch_size:
            for ids in range(self.batch_size-1):
                train_ids = np.random.randint(0,len(self.tmp_queue_train) )
                
                fname_dis_img = self.tmp_queue_train.pop(train_ids)
                fname_ref_img = self.tmp_queue_train_ref.pop(train_ids)
                mos_train = self.tmp_queue_train_mos.pop(train_ids)
#                if mos_train>0.3:
#                    continue
                im_ref = get_luminance( np.float32( cv2.imread(fname_ref_img)) )
                im_dis =  get_luminance( np.float32( cv2.imread(fname_dis_img) ) )
                flip_var = np.random.rand()
                if  flip_var> 0.8:
                    im_ref = cv2.flip( im_ref, 0 )
                    im_dis = cv2.flip( im_dis, 0 )
                elif flip_var>0.6:
                    im_ref = cv2.flip( im_ref, 1 )
                    im_dis = cv2.flip( im_dis, 1 )
                elif flip_var>0.4:
                    im_ref = cv2.flip( im_ref, -1 )
                    im_dis = cv2.flip( im_dis, -1 )
                    
                if im_ref is None:
                    print ((('No image at '+fname_ref_img)))                    
                    continue
                if im_dis is None:
                    print ((('No image at '+fname_dis_img)))                    
                    continue
                
                
                if len(im_dis.shape)==3:
                    c = im_dis.shape[2]
                else:
                    c = 1
                    im_dis = im_dis[:,:,np.newaxis]
                    im_ref = im_ref[:,:,np.newaxis]
                    
                if self.patch:
                    im_ref,im_dis,mos_train, _ =  extract_patch_random( im_ref, 
                                    im_dis ,self.patch_size[0],self.patch_size[1],
                                    score=mos_train)
#                print(im_dis.shape)
                x_dis+=[im_dis]
                x_ref+=[im_ref]
                y_mos+=[mos_train]
#                print(len(x_dis))
            rid = np.random.permutation(len(y_mos)*len(x_dis[0]))
#            print(len(x_dis))
            x_dis = np.concatenate(x_dis)[rid]
            x_ref = np.concatenate(x_ref)[rid]
            y_mos = np.concatenate(y_mos)[rid].reshape(-1)
#            print(x_dis.shape)
            
            if len(np.shape(x_dis)) >4:
                [bz, npatch, pw,ph,c] = np.shape(x_dis)
                x_dis = x_dis.reshape(bz*npatch, pw,ph,c)
                x_ref = x_ref.reshape(bz*npatch, pw,ph,c)
                y_mos = y_mos.reshape(bz*npatch, 1)
            self.counter += 1
        else:
            self.refill_file_queues()
            self.counter = 0
            [x_dis,x_ref,y_mos] = self.get_next_batch(self.patch,show)
        return( [x_dis, x_ref,y_mos] )
        

        
if __name__ == '__main__':
    test2 = db_CSIQ("D:\Documents\PhD\LDR datasets\\CSIQ\\",855, 1,32,32)
    for i in range(3):
        [a,b,c] = test2.get_test_sets()
        print(len(a))
        
    for i in range(1000):
        [x1,x2,y] = test2.get_next_batch()
        plt.imshow(x1[0,:,:,0])
        plt.colorbar()
        plt.show()
        
        plt.imshow(x2[0,:,:,0])
        plt.colorbar()
        plt.show()
        
        print (np.shape(x1), np.shape(x2), np.shape(y))