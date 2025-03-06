/*main.c*/

#include "E:/3_����ϵͳ/ʵ���� ģ��UNIX�ļ�ϵͳ/head.h"
char		choice;
int		argc;		// �û�����Ĳ�������
char		*argv[5];		// �û�����Ĳ���
int		inum_cur;		// ��ǰĿ¼
char		temp[2*BLKSIZE];	// ������
User		user;		// ��ǰ���û�
char		bitmap[BLKNUM];	// λͼ����
Inode	inode_array[INODENUM];	// i�ڵ�����
File_table file_array[FILENUM];	// ���ļ�������
char	image_name[10] = "hd.dat";	// �ļ�ϵͳ����
FILE		*fp;					// ���ļ�ָ��

//����ӳ��hd�����������û����ļ����
void format(void)
{
    FILE *fp;
    int i;
    Inode inode;
    printf("Will be to format filesystem...\n");
    printf("WARNING:ALL DATA ON THIS FILESYSTEM WILL BE LOST!\n");
    printf("Proceed with Format(Y/N)?");
    scanf("%c", &choice);
    gets(temp);
    if ((choice == 'y') || (choice == 'Y'))
    {
        if ((fp = fopen(image_name, "w+b")) == NULL)
        {
            printf("Can't create file %s\n", image_name);
            exit(-1);
        }
        for (i = 0; i < BLKSIZE; i++)
            fputc('0', fp);
        inode.inum = 0;
        strcpy(inode.file_name, "/");
        inode.type = 'd';
        strcpy(inode.user_name, "all");
        inode.iparent = 0;
        inode.length = 0;
        inode.address[0] = -1;
        inode.address[1] = -1;
        fwrite(&inode, sizeof(Inode), 1, fp);
        for (i = 1; i < INODENUM; i++)
        {
            inode.inum = -1;
            fwrite(&inode, sizeof(Inode), 1, fp);
        }
        for (i = 0; i < BLKNUM * BLKSIZE; i++)
            fputc('\0', fp);
        fclose(fp);
        // ���ļ�user.txt
        if ((fp = fopen("user.txt", "w+")) == NULL)
        {
            printf("Can't create file %s\n", "user.txt");
            exit(-1);
        }
        fclose(fp);
        printf("Filesystem created successful.Please first login!\n");
    }
    return;
}
// ����: �û���½����������û��򴴽��û�
void login(void)
{
    char *p;
    int  flag;
    char user_name[10];
    char password[10];
    char file_name[10] = "user.txt";
    do
    {
        printf("login:");
        gets(user_name);
        if (strcmp(user_name, "") == 0) {
            printf("User name cannot be empty.\n");
            continue;
        }
        printf("password:");
        printf("password:");
        p = password;
        while (1) {
            int ch = getch(); // ��ȡһ���ַ�
            if (ch == 0x0d) { // ����Ƿ��ǻس���
                *p = '\0';
                break;
            } else if (ch == 0x08) { // ����Ƿ���ɾ������ASCII��0x08��
                if (p > password) { // ������ǵ�һ���ַ�����ɾ��ǰһ���ַ�
                    p--;
                    printf("\b \b"); // ����ɾ������
                }
            } else {
                if (p - password < sizeof(password) - 1) { // ȷ�����ᳬ����������Ľ���
                    *p++ = ch; // �洢�ַ�
                    printf("*"); // ��ʾ*
                }
            }
        }
        flag = 0;
        if ((fp = fopen(file_name, "r+")) == NULL)
        {
            printf("\nCan't open file %s.\n", file_name);
            printf("This filesystem not exist, it will be create!\n");
            format();
            login();
        }
        while (!feof(fp))
        {
            fread(&user, sizeof(User), 1, fp);
            // �Ѿ����ڵ��û�, ��������ȷ
            if (!strcmp(user.user_name, user_name) &&
                !strcmp(user.password, password))
            {
                fclose(fp);
                printf("\n");
                return;
            }
            // �Ѿ����ڵ��û�, ���������
            else if (!strcmp(user.user_name, user_name))
            {
                printf("\nThis user is exist, but password is incorrect.\n");
                flag = 1;
                fclose(fp);
                break;
            }
        }
        if (flag == 0) break;
    } while (flag);
    // �������û�
    if (flag == 0)
    {
        printf("\nDo you want to create a new user?(y/n):");
        scanf("%c", &choice);
        gets(temp);
        if ((choice == 'y') || (choice == 'Y'))
        {
            if (strcmp(user_name, "") == 0) {
                printf("User name cannot be empty.\n");
                login();
            } else {
                strcpy(user.user_name, user_name);
                strcpy(user.password, password);
                fwrite(&user, sizeof(User), 1, fp);
                fclose(fp);
                return;
            }
        }
        if ((choice == 'n') || (choice == 'N'))
            login();
    }
}
// ����: ������i�ڵ�����ڴ�
void init(void)
{
    int i;
    if((fp = fopen(image_name, "r+b")) == NULL)
    {
        printf("Can't open file %s.\n", image_name);
        exit(-1);
    }
    // ����λͼ
    for(i = 0; i < BLKNUM; i++)
        bitmap[i] = fgetc(fp);
    // ����i�ڵ���Ϣ
    for (i = 0; i < INODENUM; i++) {
        inode_array[i].inum = -1; // ȷ������Inode��ʼΪδʹ��
    }
    fread(&inode_array[i], sizeof(Inode), 1, fp); // ����Ӧ����ѭ����ȡ����inode
    // ��ǰĿ¼Ϊ��Ŀ¼
    inum_cur = 0;
    // ��ʼ�����ļ���
    for(i = 0; i < FILENUM; i++)
        file_array[i].inum = -1;
}

//��������
// ���ܣ������ļ�
void copy() {
    char path[100]; // ԭ��
    char now[100]; // ���Ƶ�
    time_t rawtime;
    struct tm *timeinfo;
    time(&rawtime);
    timeinfo = localtime(&rawtime); // ����ʱ��

    int cnt = 0;
    int jug = 0;
    int temp_length;

    printf("��������Ҫ���Ƶ��ļ�����");
    gets(path);

    int k;
    for (k = 0; k < INODENUM; k++) { // �����������Ƶ�ѭ����
        if (strcmp(inode_array[k].file_name, path) == 0) {
            jug++;
            temp_length = inode_array[k].length;
            read_blk(inode_array[k].inum); // ��ȡ�ļ����ݵ�������

            printf("������Ҫ���Ƶ����ƣ�");
            gets(now);

            if (strcmp(inode_array[k].file_name, now) != 0) {
                cnt = cnt + 1;
                inode_array[cnt].length = temp_length;
                inode_array[cnt].address[0] = get_blknum(); // ����һ���µ����ݿ�
                strcpy(inode_array[cnt].times, asctime(timeinfo)); // ����ʱ����Ϣ
                strcpy(inode_array[cnt].file_name, now); // �������ļ���
                inode_array[cnt].iparent = inum_cur; // ���ø�Ŀ¼
                strcpy(inode_array[cnt].user_name, user.user_name); // �����û�����
                inode_array[cnt].type = '-'; // �����ļ�����
                inode_array[cnt].inum = cnt; // ������Inode���
                save_inode(inode_array[cnt].inum); // ����Inode��Ϣ
                write_blk(inode_array[cnt].inum); // д�����ݿ�
            } else {
                printf("Ҫ���Ƶ��ļ�ͬ��\n");
            }
        }
    }

    if (jug == 0) {
        printf("Ҫ���Ƶ��ļ�������\n");
    }
}

// ���ܣ��鿴�ļ���ϸ��Ϣ
void list_directory(void) {
    int i;
    int dcount = 0, fcount = 0;
    short bcount = 0;
    if (argc != 1) {
        printf("Command list_directory must have one args.\n");
        return;
    }
    // ����i�ڵ����飬��ʾ��ǰĿ¼�µ���Ŀ¼���ļ���
    for (i = 0; i < INODENUM; i++) {
        if ((inode_array[i].inum > 0) &&
            (inode_array[i].iparent == inum_cur)) {
            if (inode_array[i].type == 'd' && check(i)) {
                dcount++;
                printf("%-20s<DIR>\n", inode_array[i].file_name);
            }
            if (inode_array[i].type == '-' && check(i)) {
                fcount++;
                bcount += inode_array[i].length;
                printf("%-20s%12d bytes %s\n", inode_array[i].file_name, inode_array[i].length, inode_array[i].times);
            }
        }
    }
    printf("\n                    %d file(s)%11d bytes\n", fcount, bcount);
    printf("                    %d dir(s) %11d bytes FreeSpace\n", dcount, 1024 * 1024 - bcount);
}

// ���ܣ������ļ�
void find() {
    int i;
    if (argc != 2) {
        printf("Usage: find <keyword>\n");
        return;
    }
    printf("Searching for files containing '%s':\n", argv[1]);
    for (i = 0; i < INODENUM; i++) {
        if (inode_array[i].inum > 0 && strstr(inode_array[i].file_name, argv[1])) {
            printf("%-6d %-20s\n", i, inode_array[i].file_name);
        }
    }
}

// ����: �����û�����, ������������argc��argv
// ���: 0-18Ϊϵͳ����, 19Ϊ�������
int analyse(char *str)
{
    int i;
    char temp[20];
    char *ptr_char;
    char *syscmd[] = {"help", "cd", "dir", "mkdir", "creat", "open", "read", "write", "close", "delete", "logout", "clear", "format", "quit", "rd", "copy", "list_directory", "ls_detail", "find"};
    argc = 0;
    for (i = 0, ptr_char = str; *ptr_char != '\0'; ptr_char++) {
        if (*ptr_char != ' ') {
            while (*ptr_char != ' ' && (*ptr_char != '\0'))
                temp[i++] = *ptr_char++;
            argv[argc] = (char *)malloc(i + 1);
            strncpy(argv[argc], temp, i);
            argv[argc][i] = '\0';
            argc++;
            i = 0;
            if (*ptr_char == '\0') break;
        }
    }
    if (argc != 0) {
        for (i = 0; (i < 19) && strcmp(argv[0], syscmd[i]); i++);
        return i;
    } else return 19; // �����������
}
// ����: ��num��i�ڵ㱣�浽hd.dat
void save_inode(int num)
{
    if ((fp = fopen(image_name, "r+b")) == NULL)
    {
        printf("Can't open file %s\n", image_name);
        exit(-1);
    }
    fseek(fp, 512 + num * sizeof(Inode), SEEK_SET); // ��λ����ȷ��inodeλ��
    fwrite(&inode_array[num], sizeof(Inode), 1, fp);
    fclose(fp);
}
// ����: ����һ�����ݿ�
int get_blknum(void)
{
int i;
for(i = 0; i < BLKNUM; i++)
if(bitmap[i] == '0') break;
// δ�ҵ��������ݿ�
if(i == BLKNUM)
{
printf("Data area is full.\n");
exit(-1);
}
bitmap[i] = '1';
if((fp=fopen(image_name, "r+b")) == NULL)
{
printf("Can't open file %s\n", image_name);
exit(-1);
}
fseek(fp, i, SEEK_SET);
fputc('1', fp);
fclose(fp);
return i;
}
// ����: ��i�ڵ��Ϊnum���ļ�����temp
void read_blk(int num)
{
int  i, len;
char ch;
int  add0, add1;
len = inode_array[num].length;
add0 = inode_array[num].address[0];
if(len > 512)
add1 = inode_array[num].address[1];
if((fp = fopen(image_name, "r+b")) == NULL)
{
printf("Can't open file %s.\n", image_name);
exit(-1);
}
fseek(fp, 1536+add0*BLKSIZE, SEEK_SET);
ch = fgetc(fp);
for(i=0; (i < len) && (ch != '\0') && (i < 512); i++)
{
temp[i] = ch;
ch = fgetc(fp);
}
if(i >= 512)
{
fseek(fp, 1536+add1*BLKSIZE, SEEK_SET);
ch = fgetc(fp);
for(; (i < len) && (ch != '\0'); i++)
{
temp[i] = ch;
ch = fgetc(fp);
}
}
temp[i] = '\0';
fclose(fp);
}
// ����: ��temp����������hd��������
void write_blk(int num)
{
int  i, len;
int  add0, add1;
add0 = inode_array[num].address[0];
len  = inode_array[num].length;
if((fp = fopen(image_name, "r+b")) == NULL)
{
printf("Can't open file %s.\n", image_name);
exit(-1);
}
fseek(fp, 1536+add0*BLKSIZE, SEEK_SET);
for(i=0; (i<len)&&(temp[i]!='\0')&&(i < 512); i++)
fputc(temp[i], fp);
if(i == 512)
{
add1 = inode_array[num].address[1];
fseek(fp, 1536+add1*BLKSIZE, SEEK_SET);
for(; (i < len) && (temp[i] != '\0'); i++)
fputc(temp[i], fp);
}
fputc('\0', fp);
fclose(fp);
}
// ����: �ͷ��ļ����Ϊnum���ļ�ռ�õĿռ�
void release_blk(int num)
{
FILE *fp;
if((fp=fopen(image_name, "r+b")) == NULL)
{
printf("Can't open file %s\n", image_name);
exit(-1);
}
bitmap[num] = '0';
fseek(fp, num, SEEK_SET);
fputc('0', fp);
fclose(fp);
}
// ����: ��ʾ��������
void help(void)
{
printf("command: \n\
help   ---  show help menu \n\
clear  ---  clear the screen \n\
cd     ---  change directory \n\
mkdir  ---  make directory   \n\
creat  ---  create a new file \n\
open   ---  open a exist file \n\
read   ---  read a file \n\
write  ---  write something to a file \n\
close  ---  close a file \n\
delete ---  delete a exist file \n\
format ---  format a exist filesystem \n\
logout ---  exit user \n\
rd     ---  delete a directory \n\
quit   ---  exit this system\n\
dir    ---  show directory and file\n");
}
//�����ļ�·��
void pathset()
{
char path[50];
int m,n;
if(inode_array[inum_cur].inum == 0)
strcpy(path,user.user_name);
else
{
strcpy(path,user.user_name);
m=0;
n=inum_cur;
while(m != inum_cur)
{
while(inode_array[n].iparent != m)
{
n = inode_array[n].iparent;
}
strcat(path,"/");
strcat(path,inode_array[n].file_name);
m = n;
n = inum_cur;
}
}
printf("[%s]@",path);
}
// ����: �л�Ŀ¼(cd .. ���� cd dir1)
void cd(void)
{
int i;
if(argc != 2)
{
printf("Command cd must have two args. \n");
return ;
}
if(!strcmp(argv[1], ".."))
inum_cur = inode_array[inum_cur].iparent;
else
{
// ����i�ڵ�����
for(i = 0; i < INODENUM; i++)
if((inode_array[i].inum>0)&&
(inode_array[i].type=='d')&&
(inode_array[i].iparent==inum_cur)&&
!strcmp(inode_array[i].file_name,argv[1])&&
check(i))
break;
if(i == INODENUM)
printf("This directory isn't exsited.\n");
else inum_cur = i;
}
}
// ����: ��ʾ��ǰĿ¼�µ���Ŀ¼���ļ�(dir)
void dir(void)
{
int i;
int dcount=0,fcount=0;
short bcount=0;
if(argc != 1)
{
printf("Command dir must have one args. \n");
return ;
}
// ����i�ڵ�����, ��ʾ��ǰĿ¼�µ���Ŀ¼���ļ���
for(i = 0; i < INODENUM; i++)
if((inode_array[i].inum> 0) &&
(inode_array[i].iparent == inum_cur))
{
if(inode_array[i].type == 'd' && check(i))
{
dcount++;
printf("%-20s<DIR>\n", inode_array[i].file_name);
}
if(inode_array[i].type == '-' && check(i))
{
fcount++;
bcount+=inode_array[i].length;
printf("%-20s%12d bytes\n", inode_array[i].file_name,inode_array[i].length);
}
}
printf("\n                    %d file(s)%11d bytes\n",fcount,bcount);
printf("                    %d dir(s) %11d bytes FreeSpace\n",dcount,1024*1024-bcount);
}
// ����: ɾ��Ŀ¼��(rd dir1)
void rd()
{
int i,j,t,flag=0;
int chk;
if(argc != 2)
{
printf("Command delete must have one args. \n");
return ;
}
for(i = 0; i < INODENUM; i++)//���Ҵ�ɾ��Ŀ¼
if((inode_array[i].inum > 0) &&//�Ƿ�Ϊ��
(inode_array[i].iparent == inum_cur) &&
(inode_array[i].type == 'd')&&
(!strcmp(inode_array[i].file_name,argv[1])))
{
chk=check(i);//����û�Ȩ��
if(chk!=1)
{
printf("This directory is not your !\n");
return ;
}
else j=inode_array[i].inum;
for(t=0;t<INODENUM;t++)
{
if((inode_array[t].inum>0)&&
(inode_array[t].iparent==j)&&
(inode_array[i].type == '-'))
delet(t);//Ŀ¼�����ļ���ɾ��
else if((inode_array[t].inum>0)&&
(inode_array[t].iparent==j)&&
(inode_array[i].type == 'd'))
delet(t);//Ŀ¼���п�Ŀ¼��ɾ��
}
if(t == INODENUM)
delet(j);//�²�Ŀ¼Ϊ��ɾ��֮
}
if(i == INODENUM)
delet(i);//��ɾ��Ŀ¼Ϊ��ɾ��֮
return;
}
// ����: �ڵ�ǰĿ¼�´�����Ŀ¼(mkdir dir1)
void mkdir(void) {
    int i;
    if (argc != 2) {
        printf("command mkdir must have two args. \n");
        return;
    }
    // ����Ƿ��Ѵ���ͬ���ļ���Ŀ¼
    for (i = 0; i < INODENUM; i++) {
        if ((inode_array[i].inum > 0) &&
            (inode_array[i].iparent == inum_cur) &&
            !strcmp(inode_array[i].file_name, argv[1])) {
            printf("Directory or file with the same name already exists.\n");
            return;
        }
    }
    // ����i�ڵ�����, ����δ�õ�i�ڵ�
    for (i = 0; i < INODENUM; i++) {
        if (inode_array[i].inum < 0) break;
    }
    if (i == INODENUM) {
        printf("Inode is full.\n");
        exit(-1);
    }
    inode_array[i].inum = i;
    strcpy(inode_array[i].file_name, argv[1]);
    inode_array[i].type = 'd';
    strcpy(inode_array[i].user_name, user.user_name);
    inode_array[i].iparent = inum_cur;
    inode_array[i].length = 0;
    save_inode(i);
}
// ����: �ڵ�ǰĿ¼�´����ļ�(creat file1)
void creat(void)
{
    int i;
    if (argc != 2)
    {
        printf("command creat must have one args. \n");
        return;
    }
    for (i = 0; i < INODENUM; i++)
    {
        if ((inode_array[i].inum > 0) &&
            (inode_array[i].type == '-') &&
            !strcmp(inode_array[i].file_name, argv[1]))
        {
            printf("This file is exist.\n");
            return;
        }
    }
    for (i = 0; i < INODENUM; i++)
    {
        if (inode_array[i].inum == -1) break; // �ҵ���һ��δʹ�õ�inode
    }
    if (i == INODENUM)
    {
        printf("Inode is full.\n");
        exit(-1);
    }
    inode_array[i].inum = i; // ����inum
    strcpy(inode_array[i].file_name, argv[1]);
    inode_array[i].type = '-';
    strcpy(inode_array[i].user_name, user.user_name);
    inode_array[i].iparent = inum_cur;
    inode_array[i].length = 0;
    save_inode(i); // �����µ�inode��Ϣ
}
// ����: �򿪵�ǰĿ¼�µ��ļ�(open file1)
void open()
{
    int i, inum, mode, filenum, chk;
    if (argc != 2)
    {
        printf("command open must have one args. \n");
        return;
    }
    for (i = 0; i < INODENUM; i++)
    {
        if ((inode_array[i].inum > 0) &&
            (inode_array[i].type == '-') &&
            !strcmp(inode_array[i].file_name, argv[1]) &&
            (inode_array[i].iparent == inum_cur))
        {
            break;
        }
    }
    if (i == INODENUM)
    {
        printf("The file you want to open doesn't exist.\n");
        return;
    }
    inum = i;
    chk = check(i);
    if (chk != 1)
    {
        printf("This file is not your !\n");
        return;
    }
    printf("Please input open mode:(1: read, 2: write, 3: read and write):");
    scanf("%d", &mode);
    gets(temp);
    if ((mode < 1) || (mode > 3))
    {
        printf("Open mode is wrong.\n");
        return;
    }
    for (i = 0; i < FILENUM; i++)
    {
        if (file_array[i].inum < 0) break;
    }
    if (i == FILENUM)
    {
        printf("The file table is full, please close some file.\n");
        return;
    }
    filenum = i;
    file_array[filenum].inum = inum;
    strcpy(file_array[filenum].file_name, inode_array[inum].file_name);
    file_array[filenum].mode = mode;
    file_array[filenum].offset = 0;
    printf("Open file %s by ", file_array[filenum].file_name);
    if (mode == 1) printf("read only.\n");
    else if (mode == 2) printf("write only.\n");
    else printf("read and write.\n");
}
// ���ܣ����ļ��ж����ַ�(read file1)
void read()
{
    int i, inum, start, num, length;
    if (argc != 2)
    {
        printf("command read must have one args. \n");
        return;
    }
    for (i = 0; i < FILENUM; i++)
        if ((file_array[i].inum > 0) &&
            !strcmp(file_array[i].file_name, argv[1]))
            break;
    if (i == FILENUM)
    {
        printf("Open %s first.\n", argv[1]);
        return;
    }
    else if (file_array[i].mode == 2)
    {
        printf("Can't read %s.\n", argv[1]);
        return;
    }
    inum = file_array[i].inum;
    length = inode_array[inum].length;
    printf("The length of %s:%d.\n", argv[1], length);
    if (length > 0)
    {
        printf("The start position:");
        scanf("%d", &start);
        gets(temp);
        if ((start < 0) || (start >= length))
        {
            printf("Start position is wrong.\n");
            return;
        }
        printf("The bytes you want to read:");
        scanf("%d", &num);
        gets(temp);
        if (num <= 0 || num + start > length)
        {
            printf("The num you want to read is wrong.\n");
            return;
        }
        read_blk(inum);
        for (i = start; (i < start + num) && (temp[i] != '\0'); i++)
            printf("%c", temp[i]);
        printf("\n");
    }
}
// ����: ���ļ���д���ַ�(write file1)
void write()
{
    int i, inum, length2;
    char temp2[300];
    int orignum, j;

    // �����ļ�����ȡ�ļ���i�ڵ��
    for(i = 0; i < FILENUM; i++)
    {
        if((file_array[i].inum > 0) && !strcmp(file_array[i].file_name, argv[1]))
            break;
    }
    if(i == FILENUM)
    {
        printf("File not found.\n");
        return;
    }
    inum = file_array[i].inum;

    // ��ȡ�ļ���ǰ����
    read_blk(inum);

    // ��ȡ�û���Ҫд��ĳ���
    printf("The length you want to write (0-1024): ");
    scanf("%d", &length2);
    gets(temp2); // ע�⣺gets��������ȫ������ʹ��fgets

    // ������볤���Ƿ����
    if((length2 < 0) || (length2 > 1024))
    {
        printf("Input wrong.\n");
        return;
    }

    // ��ȡԭʼ�ļ�����
    orignum = inode_array[inum].length;

    // �����ļ�����
    inode_array[inum].length += length2;

    // �����µ����ݿ�
    inode_array[inum].address[0] = get_blknum();
    if(length2 > 512)
    {
        inode_array[inum].address[1] = get_blknum();
    }

    // ����i�ڵ���Ϣ
    save_inode(inum);

    // ���û������ȡҪд�������
    printf("Input the data (Enter to end):\n");
    gets(temp2); // ע�⣺gets��������ȫ������ʹ��fgets

    // ��������׷�ӵ��ļ�ĩβ
    for(j = 0; j < length2; j++)
    {
        temp[orignum + j] = temp2[j];
    }

    // д�����ݿ�
    write_blk(inum);
}
// ����: �ر��Ѿ��򿪵��ļ�(close file1)
void close(void)
{
int i;
if(argc != 2)
{
printf("Command close must have one args. \n");
return ;
}
for(i = 0; i < FILENUM; i++)
if((file_array[i].inum > 0) &&
!strcmp(file_array[i].file_name, argv[1])) break;
if(i == FILENUM)
{
printf("This file doesn't be opened.\n");
return ;
}
else
{
file_array[i].inum = -1;
printf("Close %s successful!\n", argv[1]);
}
}
// ����: ɾ���ļ�(delete file1)
void del(void)
{
    int i, j, inum, chk;
    if (argc != 2)
    {
        printf("Command delete must have one args. \n");
        return;
    }

    // ����ļ��Ƿ񱻴�
    for (j = 0; j < FILENUM; j++)
    {
        if (file_array[j].inum > 0 && !strcmp(file_array[j].file_name, argv[1]))
        {
            printf("File is open. Please close it before deleting.\n");
            return;
        }
    }

    for (i = 0; i < INODENUM; i++)
    {
        if ((inode_array[i].inum > 0) &&
            (inode_array[i].type == '-') &&
            !strcmp(inode_array[i].file_name, argv[1]))
        {
            break;
        }
    }

    if (i == INODENUM)
    {
        printf("This file doesn't exist.\n");
        return;
    }

    inum = i;
    chk = check(inum);
    if (chk != 1)
    {
        printf("This file is not your !\n");
        return;
    }

    // ɾ���ļ�
    delet(inum);
}

// ɾ��Ŀ¼��
void delet(int innum)
{
    // ����Ƿ�����Ŀ¼���ļ�
    int i;
    for (i = 0; i < INODENUM; i++)
    {
        if ((inode_array[i].inum > 0) &&
            (inode_array[i].iparent == innum))
        {
            printf("Directory is not empty. Please delete all files and subdirectories first.\n");
            return;
        }
    }

    // �ͷ�i�ڵ�
    inode_array[innum].inum = -1;
    if (inode_array[innum].length > 0)
    {
        release_blk(inode_array[innum].address[0]);
        if (inode_array[innum].length > 512)
        {
            release_blk(inode_array[innum].address[1]);
        }
    }
    save_inode(innum);
}
// ����: �˳���ǰ�û�(logout)
void logout()
{
char choice;
printf("Do you want to exit this user(y/n)?");
scanf("%c", &choice);
gets(temp);
if((choice == 'y') || (choice == 'Y'))
{
printf("\nCurrent user exited!\nPlease to login by other user!\n");
login();
}
return ;
}
//��鵱ǰI�ڵ���ļ��Ƿ����ڵ�ǰ�û�
int check(int i)
{
int j;
char *uuser,*fuser;
uuser=user.user_name;
fuser=inode_array[i].user_name;
  j=strcmp(fuser,uuser);
  if(j==0)  return 1;
else      return 0;
}
// ����: �˳��ļ�ϵͳ(quit)
void quit()
{
char choice;
printf("Do you want to exist(y/n):");
scanf("%c", &choice);
gets(temp);
if((choice == 'y') || (choice == 'Y'))
exit(0);
}
// ����: ��ʾ����
void errcmd()
{
printf("Command Error!!!\n");
}
//����ڴ��д��ڵ��û���
free_user()
{
int i;
for(i=0;i<10;i++)
user.user_name[i]='\0';
}
// ����: ѭ��ִ���û����������, ֱ��logout
void command(void)
{
    char cmd[100];
    system("cls");
    do
    {
        pathset();
        gets(cmd);
        switch (analyse(cmd))
        {
            case 0: help(); break;
            case 1: cd(); break;
            case 2: dir(); break;
            case 3: mkdir(); break;
            case 4: creat(); break;
            case 5: open(); break;
            case 6: read(); break;
            case 7: write(); break;
            case 8: close(); break;
            case 9: del(); break;
            case 10: logout(); break;
            case 11: system("cls"); break;
            case 12: format(); init(); free_user(); login(); break;
            case 13: quit(); break;
            case 14: rd(); break;
            case 15: copy(); break;
            case 16: list_directory(); break;
            case 17: find(); break;
            default: errcmd(); break;
        }
    } while (1);
}
// ������
int main(void)
{
login();
init();
command();
return 0;
}

