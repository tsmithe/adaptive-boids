import random

class weight:
    '''
    This class represents a number between zero and one.
    The class provides a utility function to perform
    creep mutation on the number.
    '''
    def __init__(self, creep_range, mutation_probability, nr_of_decimals):
        self.creep_range = creep_range
        self.mutation_probability = mutation_probability
        self.nr_of_decimals = nr_of_decimals
        self.digits = [random.randint(0,9) for i in range(nr_of_decimals)]
    
    def mutate(self):
        new_digits = []
        for d in self.digits:
            if random.random() < self.mutation_probability:
                new_digit = d + random.randint(-self.creep_range,self.creep_range)
                if new_digit < 0:
                    new_digit = 0
                elif new_digit > 9:
                    new_digit = 9
                new_digits+=[new_digit]
            else:
                new_digits+=[d]
        self.digits = new_digits
        
    @property
    def value(self):
        val = 0.0
        for i, d in enumerate(self.digits):
            val += d*10**(-i-1)
        return val