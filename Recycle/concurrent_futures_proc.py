from  concurrent import futures
import math
import numpy as  np
class Te():
    def __init__(self):
        self.PRIMES = (
            (112272535095293, 1),
            (112582705942171, 2),
            (112272535095293, 3),
            (115280095190773, 4),
            (115797848077099, 5),
            (1099726899285419,6))

    def is_prime(self, n, m):
        if n < 2:
            return (False, m)
        if n == 2:
            return (True, m)
        if n % 2 == 0:
            return (False, m)

        sqrt_n = int(math.floor(math.sqrt(n)))
        for i in range(3, sqrt_n + 1, 2):
            if n % i == 0:
                return (False, m)
        return (True, m)

    def main(self):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            A = ( (i)  for i in self.PRIMES )

            #K = executor.map(self.is_prime, A)
            K = executor.map(lambda x:self.is_prime(*x), A)
            print(list(K))
            print( tuple(A), 1111111)

            for i in self.PRIMES: 
                print( self.is_prime(*i))
            #for number, prime in zip(PRIMES, executor.map(is_prime, PRIMES)):
            #    print('%d is prime: %s' % (number, prime))
#main()


PRIMES = (
        (112272535095293, 1),
        (112582705942171, 2),
        (112272535095293, 3),
        (115280095190773, 4),
        (115797848077099, 5),
        (1099726899285419,6))

def is_prime(args):
    print(args)
    (n,m,) = args
    if n < 2:
        return (False, m)
    if n == 2:
        return (True, m)
    if n % 2 == 0:
        return (False, m)

    sqrt_n = int(math.floor(math.sqrt(n)))
    for i in range(3, sqrt_n + 1, 2):
        if n % i == 0:
            return (False, m)
    return (True, m)

def main():
    A = ( (i[0], i[1]) for i in PRIMES )
    B = np.array(A)
    #print(A)
    print(PRIMES)
    a= range(1,20)
    b= range(1,20)
    #A=zip(a,b)

    with futures.ProcessPoolExecutor() as executor:

        #K = executor.map(is_prime, PRIMES)
        #K = executor.map(lambda x:self.is_prime(*x), A)
        for K in executor.map(is_prime, A):
            print(list(K))

if __name__ == '__main__':
    main()
