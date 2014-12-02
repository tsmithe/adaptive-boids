import csv

class loadcsv:
    def __init__(self):
        pass
    
    def load_file(self, file_path, dimensions):
        data = []
        with open(file_path, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                row_data = list(self.partition(row,dimensions))
                data.append(row_data)
        
        return data
        
    def partition(self, values, n):
        for i in range(0, len(values), n):
            yield values[i:i+n]