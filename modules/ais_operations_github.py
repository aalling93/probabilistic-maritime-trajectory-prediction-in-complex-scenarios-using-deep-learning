import glob
import math
import os
import random

import cartopy.crs as ccrs
import cartopy.crs as crs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from haversine import Unit, haversine
from loadbar import LoadBar
from termcolor import colored


class ais_df_process():
    '''
    This is for a dataset
    
    '''
    def __init__(self):
        '''
        '''
        self.training_df = None
        self.testing_df = None
        self.validation_df = None
        
        self.statistics_training = None
        self.statistics_testing = None
        self.statistics_validation = None
        
        self.statistics_training_zscore = None
        self.statistics_testing_zscore = None
        self.statistics_validation_zscore = None
        
        self.statistics_training_minmax = None
        self.statistics_testing_minmax = None
        self.statistics_validation_minmax = None
        
        self.training_normalized_df = None
        self.testing_normalized_df = None
        self.validation_normalized_df = None
        
        self.training_df_dynamic = None
        self.testing_df_dynamic = None
        self.validation_df_dynamic = None
        
        self.samples_training = []
        self.samples_validation = []
        self.samples_testing = []

        self.samples_description = None
        self.targets_description = None

        self.targets_training = []
        self.targets_validation = []
        self.targets_testing = []
        
        
        self.feature_index = None #which feature you wanna use..
        self.lookback = 5 #timestamp. Time lookback is = lookback*sampling_frq
        self.lookback_offset = 1 #if starting the count self.lookback_offset steps after first location
        self.dynamic_data_type = None # Normalized or not. infor
        self.target_observations = None
        self.sampling_rate = None #minutsglobal x
        self.verbose = 1
        
        
        self.allowed_stop = None
        self.Ids_time = None
        self.min_messages = None
        self.max_speed = None
        self.min_speed = None
        self.max_cog = None
        self.min_cog = None
        self.lat_min = None
        self.lat_max = None
        self.long_min = None
        self.long_max = None
        
    def load_datasets(self,
                      train,
                      val,
                      test):
        
        
        
        self.training_df = pd.read_pickle(train)
        self.testing_df = pd.read_pickle(test)
        self.validation_df = pd.read_pickle(val)
        

    def load_datafull(self,
                      df):
        
        try:
            df = pd.read_pickle(df)
            df = df.dropna()
            
        except:
            df = df
            df = df.dropna()
        
        try:
            df = df.rename(columns={"datatime": "time"})
        except:
            pass
        
        try:
            df = df.rename(columns={"lon": "long"})
        except:
            pass
        
        try:
            df.cog = df.cog.astype(np.float32)
            df.sog = df.sog.astype(np.float32)
        except:
            pass     
    
        try:
            which_mmsi = df.mmsi.unique()
        except:
            which_mmsi = df.MMSI.unique()
            
        random.Random(42).shuffle(which_mmsi)
        
        #dividing into training, testing and validation data
        train_val = which_mmsi[0:int(len(which_mmsi)*0.8)]
        test = which_mmsi[int(len(which_mmsi)*0.8):]
        train = train_val[0:int(len(train_val)*0.8)]
        val = train_val[int(len(train_val)*0.8):]
        
        try:
            self.training_df = df[df.mmsi.isin(train)]
            self.testing_df = df[df.mmsi.isin(test)]
            self.validation_df = df[df.mmsi.isin(val)]
        except:
            self.training_df = df[df.MMSI.isin(train)]
            self.testing_df = df[df.MMSI.isin(test)]
            self.validation_df = df[df.MMSI.isin(val)]
            
            
        try:
            self.training_df = self.training_df.rename(columns={"id": "Ids"})
            self.testing_df = self.testing_df.rename(columns={"id": "Ids"})
            self.validation_df = self.validation_df.rename(columns={"id": "Ids"})
        except:
            pass
        
        
        del df, which_mmsi, val, test, train
        
    def check_data(self,
                   speed_range=[None,None],
                   cog_range=[None,None],
                   lat_range=[None,None],
                   long_range=[None,None]):
        '''
        Filter data if wanted... 
        '''
        
        self.max_speed = speed_range[1]
        self.min_speed = speed_range[0]
        self.max_cog = cog_range[1]
        self.min_cog = cog_range[0]
        self.lat_min = lat_range[0]
        self.lat_max = lat_range[1]
        self.long_min = long_range[0]
        self.long_max = long_range[1]
        
        
        try:
            self.training_df = self.training_df.rename(columns={"id": "Ids"})
            self.testing_df = self.testing_df.rename(columns={"id": "Ids"})
            self.validation_df = self.validation_df.rename(columns={"id": "Ids"})
        except:
            pass
        
        try:
            self.training_df = self.training_df.drop(['level_0', 'index'], axis=1)
            self.testing_df = self.testing_df.drop(['level_0', 'index'], axis=1)
            self.validation_df = self.validation_df.drop(['level_0', 'index'], axis=1)
        except:
            pass
        
        
        
        try:
            self.training_df['Time'] = self.training_df.time
            self.testing_df['Time'] = self.testing_df.time
            self.validation_df['Time'] = self.validation_df.time
        except:
            pass
        
        try:            
            #removing NaN...
            delete_train = np.array([])
            delete_train = np.append(delete_train,(self.validation_df.groupby('Ids').cog.count()<5)[(self.validation_df.groupby('Ids').cog.count()<5)].index.values)
            delete_train = np.append(delete_train,(self.validation_df.groupby('Ids').sog.max()<0.5)[(self.validation_df.groupby('Ids').sog.max()<0.5)].index.values)
            self.training_df = self.training_df[~self.training_df.Ids.isin(delete_train)]
            
            delete_test =(self.testing_df.groupby('Ids').cog.count()<10)
            delete_test = delete_test[delete_test].index
            self.testing_df = self.testing_df[~self.testing_df.Ids.isin(delete_test)]
            
            delete_val =(self.validation_df.groupby('Ids').cog.count()<10)
            delete_val = delete_val[delete_val].index
            self.validation_df = self.validation_df[~self.validation_df.Ids.isin(delete_val)]
            
            del delete_train, delete_test, delete_val
        except:
            pass

        
        
        
        if self.max_speed!=None:  
            try:
                self.training_df = self.training_df[self.training_df.sog<self.max_speed]
                self.testing_df = self.testing_df[self.testing_df.sog<self.max_speed]
                self.validation_df = self.validation_df[self.validation_df.sog<self.max_speed]
            except Exception as e:
                print('\ncant remove outliers',e)
                
        if self.min_speed!=None:  
            try:
                self.training_df = self.training_df[self.training_df.sog>self.min_speed]
                self.testing_df = self.testing_df[self.testing_df.sog>self.min_speed]
                self.validation_df = self.validation_df[self.validation_df.sog>self.min_speed]
            except Exception as e:
                print('\ncant remove outliers',e)
                
        if self.max_cog!=None:     
            try:
                self.training_df = self.training_df[self.training_df.cog<self.max_cog]
                self.testing_df = self.testing_df[self.testing_df.cog<self.max_cog]
                self.validation_df = self.validation_df[self.validation_df.cog<self.max_cog]
            except Exception as e:
                print('\ncant remove outliers',e)
            
        if self.min_cog!=None:    
            try:
                self.training_df = self.training_df[self.training_df.cog>self.min_cog]
                self.testing_df = self.testing_df[self.testing_df.cog>self.min_cog]
                self.validation_df = self.validation_df[self.validation_df.cog>self.min_cog]
            except Exception as e:
                print('\ncant remove outliers',e)
                
                
        if lat_range[0]!=None:    
            try:
                self.training_df = self.training_df[self.training_df.lat>lat_range[0]]
                self.testing_df = self.testing_df[self.testing_df.lat>lat_range[0]]
                self.validation_df = self.validation_df[self.validation_df.lat>lat_range[0]]
            except Exception as e:
                print('\ncant remove outliers',e)
                
        if lat_range[1]!=None:    
            try:
                self.training_df = self.training_df[self.training_df.lat<lat_range[1]]
                self.testing_df = self.testing_df[self.testing_df.lat<lat_range[1]]
                self.validation_df = self.validation_df[self.validation_df.lat<lat_range[1]]
            except Exception as e:
                print('\ncant remove outliers',e)
                
                
        if long_range[0]!=None:    
            try:
                self.training_df = self.training_df[self.training_df.long>long_range[0]]
                self.testing_df = self.testing_df[self.testing_df.long>long_range[0]]
                self.validation_df = self.validation_df[self.validation_df.long>long_range[0]]
            except Exception as e:
                pass
                
                
            try:
                self.training_df = self.training_df[self.training_df.lon>long_range[0]]
                self.testing_df = self.testing_df[self.testing_df.lon>long_range[0]]
                self.validation_df = self.validation_df[self.validation_df.lon>long_range[0]]
            except Exception as e:
                pass
                
        if long_range[1]!=None:    
            try:
                self.training_df = self.training_df[self.training_df.long<long_range[1]]
                self.testing_df = self.testing_df[self.testing_df.long<long_range[1]]
                self.validation_df = self.validation_df[self.validation_df.long<long_range[1]]
            except Exception as e:
                print('\ncant remove outliers',e)
                
            try:
                self.training_df = self.training_df[self.training_df.lon<long_range[1]]
                self.testing_df = self.testing_df[self.testing_df.lon<long_range[1]]
                self.validation_df = self.validation_df[self.validation_df.lon<long_range[1]]
            except Exception as e:
                pass
            
            
        try:
            self.training_df['Lat'] = self.training_df.lat
            self.testing_df['Lat'] = self.testing_df.lat
            self.validation_df['Lat'] = self.validation_df.lat
        except:
            pass
        
        
                
        
        
    def add_derived_values_dist(self,
                                which=['pct','running']):
        '''
        adding various derived values such as change in lat or lon. Calculated speed, calculared bering etc..
        can add derived valued with:
            - distance
            - pct
            - running
        '''
        if 'distance' in(which):
            try:
                self.training_df = ais_df_process.add_distance(self.training_df)
                self.testing_df = ais_df_process.add_distance(self.testing_df)
                self.validation_df = ais_df_process.add_distance(self.validation_df)
            except:
                pass
            
          
        
        if 'pct' in(which):
            try:
                self.training_df = ais_df_process.add_dist_pct(self.training_df)
                self.testing_df = ais_df_process.add_dist_pct(self.testing_df)
                self.validation_df = ais_df_process.add_dist_pct(self.validation_df)
            except:
                pass
            
        if 'running' in(which):
            try:
                self.training_df = ais_df_process.add_running_distance(self.training_df)
                self.testing_df = ais_df_process.add_running_distance(self.testing_df)
                self.validation_df = ais_df_process.add_running_distance(self.validation_df)
            except:
                pass
            
        
          
        
    def add_derived_values_time(self):   
        self.training_df = ais_df_process.add_time(self.training_df)
        self.testing_df = ais_df_process.add_time(self.testing_df)
        self.validation_df = ais_df_process.add_time(self.validation_df)

        self.training_df = ais_df_process.add_time_spent(self.training_df)
        self.testing_df = ais_df_process.add_time_spent(self.testing_df)
        self.validation_df = ais_df_process.add_time_spent(self.validation_df)
        
        self.training_df = ais_df_process.add_total_time_spent(self.training_df)
        self.testing_df = ais_df_process.add_total_time_spent(self.testing_df)
        self.validation_df = ais_df_process.add_total_time_spent(self.validation_df)
        
        
        
    def add_derived_values_bearing(self):   
        
        self.training_df = ais_df_process.add_bearing(self.training_df)
        self.testing_df = ais_df_process.add_bearing(self.testing_df)
        self.validation_df = ais_df_process.add_bearing(self.validation_df)
    def add_derived_values_speed(self):    
        self.training_df = ais_df_process.add_speed(self.training_df)
        self.testing_df = ais_df_process.add_speed(self.testing_df)
        self.validation_df = ais_df_process.add_speed(self.validation_df)
        
    def add_derived_values_delta_coordinated(self):
        self.training_df = ais_df_process.add_delta_coordinates(self.training_df)
        self.testing_df = ais_df_process.add_delta_coordinates(self.testing_df)
        self.validation_df = ais_df_process.add_delta_coordinates(self.validation_df)
        
        
     
        
    def add_resampling(self,Resampling_frq=5):
        global resampling_frq
        self.sampling_rate = Resampling_frq
        try:
            self.training_df_dynamic = ais_df_process.get_resampling(self.training_df,resampling_frq=Resampling_frq)
            self.testing_df_dynamic = ais_df_process.get_resampling(self.testing_df,resampling_frq=Resampling_frq)
            self.validation_df_dynamic = ais_df_process.get_resampling(self.validation_df,resampling_frq=Resampling_frq)       
        except:
            pass
        try:
            delete_id = []
            for Id in self.training_df_dynamic.Ids.unique():
                if len(self.training_df_dynamic[self.training_df_dynamic.Ids==Id])<self.min_messages:
                    delete_id.append(Id)
                    
            self.training_df_dynamic = self.training_df_dynamic[~self.training_df_dynamic.Ids.isin(delete_id)]
        except:
            pass
            
        try:
            delete_id = []
            for Id in self.testing_df_dynamic.Ids.unique():
                if len(self.testing_df_dynamic[self.testing_df_dynamic.Ids==Id])<self.min_messages:
                    delete_id.append(Id)
                    
            self.testing_df_dynamic = self.testing_df_dynamic[~self.testing_df_dynamic.Ids.isin(delete_id)]
        except:
            pass
            
        try:
            delete_id = []
            for Id in self.validation_df_dynamic.Ids.unique():
                if len(self.validation_df_dynamic[self.validation_df_dynamic.Ids==Id])<self.min_messages:
                    delete_id.append(Id)
                    
            self.validation_df_dynamic = self.validation_df_dynamic[~self.validation_df_dynamic.Ids.isin(delete_id)]
        except:
            pass
        
        
        try:
            self.training_normalized_df = self.training_normalized_df.drop(['index'], axis=1)
            self.validation_normalized_df = self.validation_normalized_df.drop(['index'], axis=1)
            self.testing_normalized_df = self.testing_normalized_df.drop(['index'], axis=1)
        except:
            pass
            
        try:
            self.training_df = self.training_df.drop(['index'], axis=1)
            self.validation_df = self.validation_df.drop(['index'], axis=1)
            self.testing_df = self.testing_df.drop(['index'], axis=1)
        except:
            pass



        
    def clean_data(self):
        self.training_df.lat = self.training_df.lat.round(3).astype(np.float32)
        self.training_df.long = self.training_df.long.round(3).astype(np.float32)
        self.training_df.sog = self.training_df.sog.round(1).astype(np.float32)
        self.training_df.cog = self.training_df.cog.round(1).astype(np.float32)
        
        self.testing_df.lat = self.testing_df.lat.round(3).astype(np.float32)
        self.testing_df.long = self.testing_df.long.round(3).astype(np.float32)
        self.testing_df.sog = self.testing_df.sog.round(1).astype(np.float32)
        self.testing_df.cog = self.testing_df.cog.round(1).astype(np.float32)
        
        self.validation_df.lat = self.validation_df.lat.round(3).astype(np.float32)
        self.validation_df.long = self.validation_df.long.round(3).astype(np.float32)
        self.validation_df.sog = self.validation_df.sog.round(1).astype(np.float32)
        self.validation_df.cog = self.validation_df.cog.round(1).astype(np.float32)
        
        
    def get_stats(self):
        '''
        '''
        try:
            self.statistics_training = self.training_df_dynamic.describe()
            self.statistics_testing = self.testing_df_dynamic.describe()
            self.statistics_validation = self.validation_df_dynamic.describe()
        except:
            self.statistics_training = self.training_df.describe()
            self.statistics_testing = self.testing_df.describe()
            self.statistics_validation = self.validation_df.describe()
            if self.verbose>1:
                print('Can not get statistics of dynamic set. Making it from the full set.')
        
        
    def get_sub_sequences(self,
                          allowed_stop=180,
                          min_messages = 50,
                          Ids_time='2016'):
        
        self.allowed_stop = allowed_stop
        self.min_messages = min_messages
        self.Ids_time = Ids_time
        
        try:
            self.training_df = ais_df_process.get_split_trajecotries(self.training_df,
                                                                     allowed_stop=allowed_stop,
                                                                     min_messages = min_messages,
                                                                     time_id=Ids_time)
            self.testing_df = ais_df_process.get_split_trajecotries(self.testing_df,
                                                                     allowed_stop=allowed_stop,
                                                                    min_messages = min_messages,
                                                                     time_id=Ids_time)
            self.validation_df = ais_df_process.get_split_trajecotries(self.validation_df,
                                                                     allowed_stop=allowed_stop,
                                                                       min_messages = min_messages,
                                                                     time_id=Ids_time)
        except Exception as e:
            print('cant split trajectories\n',e)
            pass          
            
        
       
       
    ######################################################################     
    ######### Derived features ########################################### 
    ###################################################################### 
    
    
    ######################################################################
    ######################### TIME ######################### 
    ######################################################################

    
    def time(df):
        try:
            df['time_left']=np.round((df.to_time-df.Time).astype('timedelta64[s]')/60,2)
            df = df.reset_index(drop=True)
        except:
            df['time_left']=np.round((df.Time.iloc[-1]-df.Time).astype('timedelta64[s]')/60,2)
            df = df.reset_index(drop=True)
        return df
    
    
    def add_time(df):
        try:
            df = df.groupby('Ids').apply(ais_df_process.time)
            df = df.reset_index(drop=True)
        except:
            df = df.groupby('mmsi').apply(ais_df_process.time)
            df = df.reset_index(drop=True)
        return df
    
    
    
    
    def time_spent(df):
        try:
            temp = df.Time.diff()
            temp.iloc[0] = pd.Timedelta(np.timedelta64(0, "ms"))
            df['Running_time'] = temp
            df.Running_time =df.Running_time / np.timedelta64(1, 's')
            df = df.reset_index(drop=True)
            del temp
        except:
            temp = df.Time.diff()
            temp.iloc[0] = pd.Timedelta(np.timedelta64(0, "ms"))
            df['Running_time'] = temp
            
            df.Running_time =df.Running_time / np.timedelta64(1, 's')
            df = df.reset_index(drop=True)
            del temp
        return df

    def add_time_spent(df):
        try:
            df = df.groupby('Ids').apply(ais_df_process.time_spent)
            df = df.reset_index(drop=True)
        except:
            df = df.groupby('mmsi').apply(ais_df_process.time_spent)
            df = df.reset_index(drop=True)
        return df
    
    
    
    def total_time_spent(df):
        df['Total_time_spent'] = df['Running_time'].cumsum().astype(np.float32)
        df = df.reset_index(drop=True)
        return df
    
    def add_total_time_spent(df):
        try:
            df = df.groupby('Ids').apply(ais_df_process.total_time_spent)
            df = df.reset_index(drop=True)
        except:
            df = df.groupby('mmsi').apply(ais_df_process.total_time_spent)
            df = df.reset_index(drop=True)
        return df
    
    
    
    
    ######################################################################
    ######################################################################
                                ## DISTANCE ###  
    ######################################################################
    def dist_pct(df):
        try:
            df['Distance_percentage']=(df['Total_distance'] / df['Total_distance'].max()).astype(np.float32)
        except:
            pass
        return df
    def add_dist_pct(df):
        try:
            df = df.groupby('Ids').apply(ais_df_process.dist_pct)
            df = df.reset_index(drop=True)
        except:
            pass
        return df
        
    
    def haversine_distance(df):
        dist = haversine((df.index[0],df.iloc[0]), (df.index[1],df.iloc[1]), unit=Unit.NAUTICAL_MILES)
        return dist
    
    def total_distance(df):
        
        try:
            df = df.set_index('Lat')
            df['Distance2'] = df['long'].rolling(2).apply(ais_df_process.haversine_distance, raw=False)
            df['Total_distance'] = df['Distance2'].cumsum().astype(np.float32)
            df['Total_distance'].iloc[1] = df['Distance2'].iloc[1].astype(np.float32)
            #df=df.iloc[1:,:]
            del df['Distance2']
            df = df.reset_index()
        except:
            pass
        return df

    def add_distance(df):
        try:
            df = df.groupby('Ids').apply(ais_df_process.total_distance)
            df = df.reset_index(drop=True)
        except:
            df = df.groupby('Ids').apply(ais_df_process.total_distance)
            df = df.reset_index(drop=True)
            
            
        try:
            del df['Distance2']
        except:
            pass
        return df
    
    
    def running_distance(df):
        df = df.set_index('Lat')
        df['Running_distance'] = df['long'].rolling(2).apply(ais_df_process.haversine_distance, raw=False).astype(np.float32)
        df=df.iloc[1:,:]
        df = df.reset_index()
        
        return df
    
    def add_running_distance(df):
        try:
            df = df.groupby('Ids').apply(ais_df_process.running_distance)
            df = df.reset_index(drop=True)
        except:
            df = df.groupby('Ids').apply(ais_df_process.running_distance)
            df = df.reset_index(drop=True)
        return df
    
    
    ### Speed ####
    def speed(df):
        #runnin distances (nautilus miles) to meters
        # m/s to knots (by multiplying 1.943844)
        df['speed_calculated'] = ((df.Running_distance*1852)/(df.Running_time )*1.943844).astype(np.float32)
        df['speed_calculated'].iloc[0] = 0
        df = df.reset_index(drop=True)
        return df

    def add_speed(df):
        try:
            df = df.groupby('Ids').apply(ais_df_process.speed)
            df = df.reset_index(drop=True)
        except:
            df = df.groupby('Ids').apply(ais_df_process.speed)
    
        return df
    
    ## bearing ###  
    def calculate_bearing(df):
        pointA = df.index[0],df.iloc[0]
        pointB = df.index[1],df.iloc[1]
        lat1 = math.radians(pointA[0])
        lat2 = math.radians(pointB[0])
        
        diffLong = math.radians(pointB[1] - pointA[1])
        x = math.sin(diffLong) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(diffLong))
        initial_bearing = math.atan2(x, y)
        initial_bearing = math.degrees(initial_bearing)
        compass_bearing = (initial_bearing + 360) % 360
        #print(initial_bearing,compass_bearing)
        return compass_bearing
        
    def bearing(df):
        df = df.set_index('Lat')
        df['bearing_calculated'] = df['long'].rolling(2).apply(ais_df_process.calculate_bearing, raw=False).astype(np.float32)
        df['bearing_calculated'].iloc[0] = df['bearing_calculated'].iloc[1]
        #df = df.reset_index(drop=True)
        df = df.reset_index(drop=True)
        return df
    
    def add_bearing(df):
        df = df.groupby('Ids').apply(ais_df_process.bearing)
        df = df.reset_index(drop=True)
        return df
    
    ######################3
    def co_ordinates(df):
        df['delta_lat'] = df.lat.diff().astype(np.float32)
        df['delta_lat'].iloc[0] = 0
        
        try:
            df['delta_lon'] = df.lon.diff().astype(np.float32)
            df['delta_lon'].iloc[0] = 0
        except:
            df['delta_long'] = df.long.diff().astype(np.float32)
            df['delta_long'].iloc[0] = 0
        else:
            pass
        return df
    
    def add_delta_coordinates(df):
        df = df.groupby('Ids').apply(ais_df_process.co_ordinates)
        df = df.reset_index(drop=True)
        return df
        
    ######################################################################  
    ######################################################################  
    def add_normalize(self,which = 'MinMax'):
        '''
        '''
        assert which.lower() in ['minmax','zscore','max_abs'],'wrong normalization'
        
        try:
            self.training_normalized_df, self.validation_normalized_df , self.testing_normalized_df = ais_df_process.normalize(self.training_df_dynamic,
                                                                                                                               self.validation_df_dynamic,
                                                                                                                               self.testing_df_dynamic,
                                                                                                                               which=which)
            self.dynamic_data_type = 'Normalized and resampled'

        except Exception as e:
            if self.verbose>0:
                print('error in normalising the resampled values.\n',e)
            self.training_normalized_df, self.validation_normalized_df , self.testing_normalized_df = ais_df_process.normalize(self.training_df,
                                                                                                                               self.validation_df,
                                                                                                                               self.testing_df,
                                                                                                                               which=which)
            self.dynamic_data_type = 'Normalized and not resampled'
            
        try:
            self.training_normalized_df = self.training_normalized_df.drop(['index'], axis=1)
            self.validation_normalized_df = self.validation_normalized_df.drop(['index'], axis=1)
            self.testing_normalized_df = self.testing_normalized_df.drop(['index'], axis=1)
        except:
            pass
            
        try:
            self.training_df = self.training_df.drop(['index'], axis=1)
            self.validation_df = self.validation_df.drop(['index'], axis=1)
            self.testing_df = self.testing_df.drop(['index'], axis=1)
        except:
            pass
        
        
        
        try:
            delete_id = []
            for Id in self.training_normalized_df.Ids.unique():
                if len(self.training_normalized_df[self.training_normalized_df.Ids==Id])<self.min_messages:
                    delete_id.append(Id)
                    
            self.training_normalized_df = self.training_normalized_df[~self.training_normalized_df.Ids.isin(delete_id)]
        except:
            pass
          
        try:
            delete_id = []
            for Id in self.validation_normalized_df.Ids.unique():
                if len(self.validation_normalized_df[self.validation_normalized_df.Ids==Id])<self.min_messages:
                    delete_id.append(Id)
                    
            self.validation_normalized_df = self.validation_normalized_df[~self.validation_normalized_df.Ids.isin(delete_id)]
        except:
            pass
        
        try:
            delete_id = []
            for Id in self.testing_normalized_df.Ids.unique():
                if len(self.testing_normalized_df[self.testing_normalized_df.Ids==Id])<self.min_messages:
                    delete_id.append(Id)
                    
            self.testing_normalized_df = self.testing_normalized_df[~self.testing_normalized_df.Ids.isin(delete_id)]
        except:
            pass
    
    def normalize(training_df,validation_df,testing_df,which = 'minMax'):
        '''
        
        '''
        training_normalized_df = training_df.copy()
        testing_normalized_df = testing_df.copy()
        validation_normalized_df = validation_df.copy()
        
        no_normalize = ['Ids','mmsi','time','from_locode','to_locode','combi']
        normalize = training_df.columns[~training_df.columns.isin(no_normalize)]
        if which.lower()=='minmax':
            for column in training_df[normalize].columns:
                try:
                    training_normalized_df[column] = (training_df[column] - training_df[column].min()) / (training_df[column].max() - training_df[column].min())
                    testing_normalized_df[column] = (testing_df[column] - training_df[column].min()) / (training_df[column].max() - training_df[column].min())
                    validation_normalized_df[column] = (validation_df[column] - training_df[column].min()) / (training_df[column].max() - training_df[column].min())
                except:
                    pass
            
            try:
                training_normalized_df.to_lat = (training_df.to_lat - training_df.lat.min()) / (training_df.lat.max() - training_df.lat.min())
                training_normalized_df.to_long = (training_df.to_long - training_df.long.min()) / (training_df.long.max() - training_df.long.min())
            
                testing_normalized_df.to_lat = (testing_df.to_lat - training_df.lat.min()) / (training_df.lat.max() - training_df.lat.min())
                testing_normalized_df.to_long = (testing_df.to_long - training_df.long.min()) / (training_df.long.max() - training_df.long.min())
            
                validation_normalized_df.to_lat = (validation_df.to_lat - training_df.lat.min()) / (training_df.lat.max() - training_df.lat.min())
                validation_normalized_df.to_long = (validation_df.to_long - training_df.long.min()) / (training_df.long.max() - training_df.long.min())
            
                training_normalized_df.from_lat = (training_df.from_lat - training_df.lat.min()) / (training_df.lat.max() - training_df.lat.min())
                training_normalized_df.from_long = (training_df.from_long - training_df.long.min()) / (training_df.long.max() - training_df.long.min())
            
                testing_normalized_df.from_lat = (testing_df.from_lat - training_df.lat.min()) / (training_df.lat.max() - training_df.lat.min())
                testing_normalized_df.from_long = (testing_df.from_long - training_df.long.min()) / (training_df.long.max() - training_df.long.min())
                
                validation_normalized_df.from_lat = (validation_df.from_lat - training_df.lat.min()) / (training_df.lat.max() - training_df.lat.min())
                validation_normalized_df.from_long = (validation_df.from_long - training_df.long.min()) / (training_df.long.max() - training_df.long.min())
            except:
                pass
        
        if which.lower()=='zscore':
            for column in training_df[normalize].columns:
                try:
                    training_normalized_df[column] =(training_df[column] -training_df[column].mean()) / training_df[column].std()
                    testing_normalized_df[column] = (testing_df[column] - training_df[column].mean()) / training_df[column].std()
                    validation_normalized_df[column] =(validation_df[column] - training_df[column].mean()) / training_df[column].std()
                except:
                    pass
                
               
        if which.lower()=='max_abs':
            for column in training_df[normalize].columns:
                try:
                    training_normalized_df[column] =training_df[column]  / training_df[column].abs().max()
                    testing_normalized_df[column] = testing_df[column]  / training_df[column].abs().max()
                    validation_normalized_df[column] =validation_df[column]  / training_df[column].abs().max()
                except:
                    pass
            try:    
                training_normalized_df.to_lat =training_df.to_lat / training_df.lat.abs().max()
                training_normalized_df.to_long =training_df.to_long / training_df.long.abs().max()
            
                testing_normalized_df.to_lat =testing_df.to_lat / training_df.lat.abs().max()
                testing_normalized_df.to_long =testing_df.to_long / training_df.long.abs().max()
            
                validation_normalized_df.to_lat =validation_df.to_lat / training_df.lat.abs().max()
                validation_normalized_df.to_long =validation_df.to_long / training_df.long.abs().max()
            except:
                pass
            
        return training_normalized_df, validation_normalized_df , testing_normalized_df
    
    
    

            
    def resampling(df,resampling_frq=5):
        '''
        First, we generate the underlying data grid by using mean(). 
        This generates the grid with NaNs as values. Afterwards, we fill the NaNs with interpolated values by calling the interpolate() method on the read value column
        '''
        df.index = df.time
        
        try:
            try:
                df = df.resample(f'{resampling_frq}min').mean().interpolate(method='cubicspline')
                
            except:
                try:
                    df = df.resample(f'{resampling_frq}min').mean().interpolate(method='spline', order=3)
                except:
                    df = df.resample(f'{resampling_frq}min').mean().interpolate(method='spline', order=3, s=0.)
                    
                    #print('error in finding sampling frq. 5 min is used.')
                    pass
                pass
        except Exception as e:
            print(f'Error in resampling id: {df.Ids.iloc[0]}: {e}')
            pass


        return df
    
    def get_resampling(df,resampling_frq):
        try:
            df = df.groupby('Ids').apply(ais_df_process.resampling,resampling_frq=resampling_frq)
            df = df.reset_index(drop=True)
        except:
            no_normalize = ['from_locode','to_locode','combi']
            df = df.columns[~df.columns.isin(no_normalize)]
            #df = df[['time','Ids','long','lat','cog','sog']]
            df = df.groupby('Ids').apply(ais_df_process.resampling,resampling_frq=resampling_frq)
            df = df.reset_index(drop=True)
            
        
        #df_dynamic = df_dynamic.reset_index()
        try:
            df = df.drop(['index'], axis=1)
        except:
            pass
        try:
            df = df.drop(['index'], axis=1)
            df = df.drop(['index'], axis=1)
        except:
            pass

        return df
    
    
        
    
    

    
    def datasets_training(df):
        for rows in range(lookback_offset,df.shape[0]-lookback-target_observations+1):
            samples_training.append(df.iloc[rows:rows+lookback,:].to_numpy())
            targets_training.append(df.iloc[rows+lookback:rows+lookback+target_observations,:].to_numpy())
        return None
    
    def datasets_validation(df):
        for rows in range(lookback_offset,df.shape[0]-lookback-target_observations+1):
            samples_validation.append(df.iloc[rows:rows+lookback,:].to_numpy())
            targets_validation.append(df.iloc[rows+lookback:rows+lookback+target_observations,:].to_numpy())
        return None
    
    def datasets_testing(df):
        for rows in range(lookback_offset,df.shape[0]-lookback-target_observations+1):
            samples_testing.append(df.iloc[rows:rows+lookback,:].to_numpy())
            targets_testing.append(df.iloc[rows+lookback:rows+lookback+target_observations,:].to_numpy())
        return None
    
    
    def add_datasets(df_train,df_val,df_test):
        try:
            df_train.groupby('Ids').apply(ais_df_process.datasets_training)
            df_train.reset_index(drop=True)
        except:
            df_train.groupby('id').apply(ais_df_process.datasets_training)
            df_train.reset_index(drop=True)
        
        try:
            df_val.groupby('Ids').apply(ais_df_process.datasets_validation)
            df_val.reset_index(drop=True)
        except:
            df_val.groupby('id').apply(ais_df_process.datasets_validation)
            df_val.reset_index(drop=True)
        
        try:
            df_test.groupby('Ids').apply(ais_df_process.datasets_testing)
            df_test.reset_index(drop=True)
        except:
            df_test.groupby('id').apply(ais_df_process.datasets_testing)
            df_test.reset_index(drop=True)
            
        return None
    
    def get_split_trajecotries(df,
                               allowed_stop=100,
                               min_messages = 120,
                               time_id='2016'):
        '''
    
        '''
        
        try:
            df_temp = df.groupby('mmsi').apply(ais_df_process.split_trajectory,allowed_stop=allowed_stop,time_id=time_id)
            df_temp = df_temp.reset_index(drop=True)
        except:
            df_temp = df.groupby('Ids').apply(ais_df_process.split_trajectory,allowed_stop=allowed_stop,time_id=time_id)
            df_temp = df_temp.reset_index(drop=True)
        
        
        
        try:
            delete_id = []
            for Id in df_temp.Ids.unique():
                if len(df_temp[df_temp.Ids==Id])<min_messages:
                    delete_id.append(Id)
                
            df_temp = df_temp[~df_temp.Ids.isin(delete_id)] 
        except:
            pass
            
        return df_temp
    
    def split_trajectory(df,allowed_stop=100,time_id = '2016'):
        '''
        '''
        df = df.reset_index(drop=True)
        df_combined = pd.DataFrame()
    
        allowed_stop = allowed_stop*60 #sek to min.
        try:
            position_split = df.query(f'elapsed > {allowed_stop} or elapsed < 0').index.values.tolist()
        except Exception as e:
            position_split = df.query(f'Running_time > {allowed_stop} or Running_time < 0').index.values.tolist()
            

        try:
            if len(position_split)>0:
                position_split.insert(0,0)
                position_split.insert(len(position_split),len(df))
                for split in range(len(position_split)-1):
                    df_temp = df.iloc[position_split[split]:position_split[split+1],:]
                    df_temp['Ids'] = f'{df.mmsi.iloc[0]}{time_id}{split+1}'
                    try:
                        df_temp.elapsed.iloc[0] = 0
                    except:
                        pass
                    df_combined = df_combined.append(df_temp)       
            else:
                df_combined = df
                df_combined['Ids'] = f'{df.mmsi.iloc[0]}{time_id}{1}'
                
        except Exception as e:
            print('error in split_trajectory: ',e)
            pass
    
        df_combined.Ids = df_combined.Ids.astype(int)
        df_combined = df_combined.reset_index(drop=True)
        del df
        return df_combined
                    
    
    
    def get_datasets(self,
                     Lookback_offset,
                     Target_observations,
                     Lookback=5):
        global lookback_offset, lookback, samples_training, targets_training, samples_validation, targets_validation, samples_testing, targets_testing, target_observations
        try:
            lookback_offset = Lookback_offset
            lookback = Lookback
            target_observations = Target_observations
            self.lookback_offset = lookback_offset
            self.lookback = lookback
            self.target_observations = target_observations
        except:
            lookback_offset = self.lookback_offset 
            lookback = self.lookback 
            target_observations = self.target_observations 
            
            
        
        
 
        
        samples_training = []
        targets_training = []
        
        samples_validation = []
        targets_validation= []
        
        samples_testing = []
        targets_testing = []
        
        
        try:
            ais_df_process.add_datasets(self.training_normalized_df,
                                    self.validation_normalized_df,
                                    self.testing_normalized_df)
            
            self.samples_description = self.training_normalized_df.columns
            self.targets_description = self.training_normalized_df.columns
        
        except:
            if self.verbose>0:
                print('error in making dataset from normalized values. making them from dynamic')
            ais_df_process.add_datasets(self.training_df_dynamic,
                                    self.validation_df_dynamic,
                                    self.testing_df_dynamic)
            self.samples_description = self.training_df_dynamic.columns
            self.targets_description = self.training_df_dynamic.columns
        


        
        self.samples_training = np.array(samples_training)
        self.targets_training = np.array(targets_training)
        self.samples_validation = np.array(samples_validation)
        self.targets_validation = np.array(targets_validation)
        self.samples_testing = np.array(samples_testing)
        self.targets_testing = np.array(targets_testing)
        del samples_training, targets_training, targets_testing, samples_testing, targets_validation, samples_validation
        
        return None
    
    def get_info(self):
        '''
        '''
        
        info_data = {'Paramter':  ['samples_shape', 
                                   'targets_shape',
                                   'samples_description',
                                   'features_used',
                                   'lookback',
                                   'lookback_offset',
                                   'number_target_observations',
                                   'sampling_rate'],
                     'Value': [self.samples_training.shape, 
                               self.targets_training.shape,
                               self.samples_description,
                               self.feature_index,
                               self.lookback,
                               self.lookback_offset,
                               self.target_observations,
                               self.sampling_rate]
                    }

        df = pd.DataFrame (info_data, columns = ['Paramter','Value'])
        

        

    
    def get_index(self,parms = ['lat',
                       'long',
                       'sog',
                       'cog',
                       'Total_distance',
                       'Running_Distance',
                       'delta_lat',
                       'delta_long']
                 ):
        features = self.samples_description.isin(parms)
        features = pd.Series(features)
        features = features[features].index
        self.feature_index = features
    
        return features
    
    
    def plot_tracks(df):
        
        mrc = ccrs.Mercator()
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(1,1,1, projection=crs.PlateCarree())

        ax.stock_img()
        ax.gridlines(draw_labels=True)
        try:
            ax.scatter(df.long.to_numpy(),df.lat.to_numpy(),c='red', s=10)
        except:
            ax.scatter(df.lon.to_numpy(),df.lat.to_numpy(),c='red', s=10)
        ax.coastlines()
        try:
            ax.set_extent([ df.long.min()-1, df.long.max()+1,df.lat.min()-1, df.lat.max()+1])
        except:
            ax.set_extent([ df.lon.min()-1, df.lon.max()+1,df.lat.min()-1, df.lat.max()+1])
        plt.show()

        
        
        
    def plot(self):
        (self.training_df.speed_calculated-self.training_df.sog).plot()
        (self.training_df.bearing_calculated-self.training_df.cog).plot()
        
        (self.training_df.bearing_calculated).plot()
        (self.training_df.cog).plot()
        
                

            
        
def save_ais_df(df,filename='ais'):
    '''
    
    '''
    df.reset_index().to_feather(f'{filename}.feather')
    df.reset_index().to_pickle(f"{filename}.pkl")
    
    return None   



