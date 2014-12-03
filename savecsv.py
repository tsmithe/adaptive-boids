import csv
import numpy

class savecsv:
    
    def __init__(self,file_name,append_to_file):
        self.file_name = file_name
        if append_to_file:
            self.file_handle = open(self.file_name,'a')
        else:
            self.file_handle = open(self.file_name,'w')
        self.writer = csv.writer(self.file_handle)
       
    def write_row(self, array):
        try:
            self.writer.writerow(array.tolist())
        except:
            self.writer.writerow(array)
        
    def close_writer():
        self.file_handle.close()
