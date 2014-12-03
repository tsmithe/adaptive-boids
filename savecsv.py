import csv
import numpy

class savecsv:
    
    def __init__(self,file_name):
        self.file_name = file_name
        self.file_handle = open(self.file_name,'a')
        self.writer = csv.writer(self.file_handle)
       
    def write_row(self, array):
        try:
            self.writer.writerow(array.tolist())
        except:
            self.writer.writerow(array)
        
    def close_writer():
        self.file_handle.close()
