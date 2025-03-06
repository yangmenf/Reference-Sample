/* head.h */

#include <stdio.h>
#include <stdlib.h>
#include <conio.h>
#include <string.h>
#include <time.h> // ���ʱ���ͷ�ļ�

#define BLKSIZE    512        // ���ݿ�Ĵ�С
#define BLKNUM     512        // ���ݿ�Ŀ���
#define INODESIZE  32         // i�ڵ�Ĵ�С
#define INODENUM   32         // i�ڵ����Ŀ
#define FILENUM    8          // ���ļ������Ŀ

// �û�(20B)
typedef struct {
    char user_name[10];       // �û���
    char password[10];        // ����
} User;

// i�ڵ�(32B)
typedef struct {
    short inum;              // �ļ�i�ڵ��
    char  file_name[10];     // �ļ���
    char  type;              // �ļ�����
    char  user_name[10];     // �ļ�������
    short iparent;           // ��Ŀ¼��i�ڵ��
    short length;            // �ļ�����
    short address[2];        // ����ļ��ĵ�ַ
    int   create_time;       // ����ʱ��
    int   modify_time;       // ����޸�ʱ��
    char  times[26];         // �洢ʱ����Ϣ������ "Wed Jun 30 21:49:08 1993\n"
} Inode;

// ���ļ���(16B)
typedef struct {
    short inum;             // i�ڵ��
    char  file_name[10];    // �ļ���
    short mode;             // ��дģʽ(1:read, 2:write, 3:read and write)
    short offset;           // ƫ����
} File_table;

// ��������
void login(void);
void init(void);
int  analyse(char *str);
void save_inode(int);
int  get_blknum(void); // ȷ������������ȷ
void read_blk(int);
void write_blk(int);
void release_blk(int);
void pathset();
void delet(int inum);
int  check(int i);
void copy(void);
void find(void);
// �û��������
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
