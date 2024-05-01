#include "mmap.h"

void* mmap_alloc(npy_uint64 size) {
    static unsigned mmap_file_num = 0;
    static char tmp[128];
    snprintf(tmp,128,"/datos/data/mmap/mmap_file_%05d.mmap",mmap_file_num++);
    int fd = open(tmp,O_RDWR | O_CREAT, S_IRWXU);
    printf("requested size=%luB\t",size);
    const npy_uint64 psize = getpagesize();
    //const npy_uint64 psize = (1UL<<21); // 2MB
    npy_int64 npages = (size + psize)/psize;
    printf("page size=%luB\tnum. pages=%lu\t",psize,npages);
    size =  npages * psize;
    printf("paged size=%luB\n",size);
    const unsigned long t = 0;
    const npy_int64 nwrites = size / sizeof(unsigned long);
    FILE* FD = fdopen(fd,"w");
    npy_int64 nw = 0;
    for (npy_int64 i = 0; i < nwrites; i++) {
        //write(fd,&t,sizeof(unsigned long));
        nw += fwrite(&t,sizeof(unsigned long),1,FD);
    }
    printf("written %lu bytes.\n",nw*sizeof(unsigned long));
    //close(fd);
    fclose(FD);
    fd = open(tmp,O_RDWR);
    void* data = mmap(NULL,size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    //void* data = mmap(NULL,size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_HUGETLB | (21 << MAP_HUGE_SHIFT), fd, 0);
    if (data == MAP_FAILED) {
        printf("mmap failed with errno %d!!\n",errno);
        exit(errno);
    }
    return data;
}
