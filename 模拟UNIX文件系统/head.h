/* head.h */

#include <stdio.h>
#include <stdlib.h>
#include <conio.h>
#include <string.h>
#include <time.h> // 添加时间库头文件

#define BLKSIZE    512        // 数据块的大小
#define BLKNUM     512        // 数据块的块数
#define INODESIZE  32         // i节点的大小
#define INODENUM   32         // i节点的数目
#define FILENUM    8          // 打开文件表的数目

// 用户(20B)
typedef struct {
    char user_name[10];       // 用户名
    char password[10];        // 密码
} User;

// i节点(32B)
typedef struct {
    short inum;              // 文件i节点号
    char  file_name[10];     // 文件名
    char  type;              // 文件类型
    char  user_name[10];     // 文件所有者
    short iparent;           // 父目录的i节点号
    short length;            // 文件长度
    short address[2];        // 存放文件的地址
    int   create_time;       // 创建时间
    int   modify_time;       // 最后修改时间
    char  times[26];         // 存储时间信息，例如 "Wed Jun 30 21:49:08 1993\n"
} Inode;

// 打开文件表(16B)
typedef struct {
    short inum;             // i节点号
    char  file_name[10];    // 文件名
    short mode;             // 读写模式(1:read, 2:write, 3:read and write)
    short offset;           // 偏移量
} File_table;

// 声明函数
void login(void);
void init(void);
int  analyse(char *str);
void save_inode(int);
int  get_blknum(void); // 确保返回类型正确
void read_blk(int);
void write_blk(int);
void release_blk(int);
void pathset();
void delet(int inum);
int  check(int i);
void copy(void);
void find(void);
// 用户命令处理函数
void help(void);
void cd(void);
void dir(void);
void mkdir(void);
void creat(void);
void open(void);
void read(void);
void write(void);
void close(void);
void del(void);
void logout(void);
void command(void);
void rd(void);
void quit(void);
