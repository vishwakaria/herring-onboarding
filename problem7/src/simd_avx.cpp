#include <iostream>
#include <immintrin.h>

void initPrintArrayRand(int *arr, int numel, int range) {
        for (int i = 0; i < numel; i++) {
                arr[i] = rand() % range;
                std::cout<<arr[i]<<"\t";
        }
        std::cout<<std::endl;
}

void println_avx(__m256i vec) {
        int* arr = (int*)&vec;
        for (int i = 0; i<8; i++) {
                std::cout<<arr[i]<<"\t";
        }
        std::cout<<std::endl;
}

void task1() {
        int numArrays = 5, numel = 8;
        int arr[numArrays][numel];

        //Initialize arrays with random integers
        std::cout<<"Arrays are:\n";
        for (int i=0; i<numArrays; i++) {
                initPrintArrayRand(arr[i],numel,10);
        }

        //Vertically sum arrays using AVX
        __m256i result = _mm256_loadu_si256((const __m256i*)arr[0]);
        for (int i=1; i<5; i++) {
                __m256i array2 = _mm256_loadu_si256((const __m256i*)arr[i]);
                result = _mm256_add_epi64(result, array2);
        }

        std::cout<<"Result is: \n";
        println_avx(result);
}

void task2() {
        int numel = 14;
        int array1[numel], array2[numel];
        std::cout<<"Arrays are:\n";
        initPrintArrayRand(array1, numel, 10);
        initPrintArrayRand(array2, numel, 10);

        int resultArray[numel], numInts = 8;
        for (int i=0; i < numel; i += numInts) {
                __m256i vec1 = _mm256_loadu_si256((const __m256i*)(array1 + i));
                __m256i vec2 = _mm256_loadu_si256((const __m256i*)(array2 + i));
                __m256i result = _mm256_add_epi64(vec1, vec2);
                _mm256_storeu_si256((__m256i*)(resultArray + i), result);
        }
        std::cout<<"Sum calculated is: \n";
        for (int i=0; i<numel; i++) {
                std::cout<<resultArray[i]<<"\t";
        }
        std::cout<<std::endl;
}

int main() {
        std::cout<<"PART 1: Vertically sum 5 arrays of 8 integers each using AVX"<<std::endl;
        task1();

        std::cout<<"PART 2: Vertically sum 2 arrays with more than 8 elements using AVX\n";
        task2();

        return 0;
        }

