/*���㣺��ʾҽԺ���ֿ⡢�������ĵ����������еĽڵ㡣
      ÿ���������һ���ص㣬��һ��ҽԺ��ֿ⡣
Դ�㣺��ʾҽҩ���������е���ʼ�������㣬ͨ����һ���ֿ���������ġ�
      Դ���Ǵ��俪ʼ���㵽������������·����
�ڽӾ�����������ҽҩ���������и�������֮������ӹ�ϵ�;���ɱ���
          �ڽӾ�����һ����ά�������е�Ԫ�ر�ʾ��������֮��ľ����ɱ��������е�ֵ��ӳ����������֮���ֱ����������;��롣
��ҽҩ�������������Ż������У�ͨ�����붥�㡢Դ����ڽӾ����ֵ����������Dijkstra�㷨�������Դ�㵽������������·�����Ӷ������Ż�ҽҩ�������������·���滮�ͳɱ����ơ�
�ó������·���������ŵ�ҩƷ���ͷ���*/
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits.h>
#include <string.h>

#define MAX_V 10 // ��󶥵�����
#define INF INT_MAX // ���������

// ����ڽӾ����Ƿ���Ч
bool validateMatrix(int graph[MAX_V][MAX_V], int num_vertices) {
    int i, j;
    for (i = 0; i < num_vertices; i++) {
        if (graph[i][i] != 0) {
            return false; // �Խ����ϵ�ֵ��Ϊ0����Ч���ڽӾ���
        }

        for (j = 0; j < num_vertices; j++) {
            if (graph[i][j] < 0 && graph[i][j] != INF) {
                return false; // �ڽӾ����д��ڸ�ֵ�����߷�INFֵ
            }
        }
    }
    return true; // �ڽӾ�����Ч
}

// ���ַ���ת��Ϊ����������ַ���Ϊ"inf"��"INF"������INF
int stringToInt(const char *str) {
    if (strcmp(str, "inf") == 0 || strcmp(str, "INF") == 0) {
        return INF;
    }
    return atoi(str);
}

// �����Դ�㵽Ŀ�������·��
void dijkstra(int graph[MAX_V][MAX_V], int num_vertices, int src) {
    int dist[MAX_V]; // �洢��̾���
    bool visited[MAX_V]; // �洢�����Ƿ񱻷���

    // ��ʼ������ͷ�������
    int i;
    for (i = 0; i < num_vertices; i++) {
        dist[i] = INF;
        visited[i] = false;
    }
    dist[src] = 0;
    int count;

    for (count = 0; count < num_vertices - 1; count++) {
        int min_dist = INF, min_index;

        // �ҵ���ǰδ���ʶ����о�����С�Ķ���
        int v;
        for (v = 0; v < num_vertices; v++) {
            if (!visited[v] && dist[v] <= min_dist) {
                min_dist = dist[v];
                min_index = v;
            }
        }
        int u = min_index;
        visited[u] = true;

        // �������·������
        for (v = 0; v < num_vertices; v++) {
            if (!visited[v] && graph[u][v] != INF &&
                dist[u] != INF && dist[u] + graph[u][v] < dist[v]) {
                dist[v] = dist[u] + graph[u][v];
            }
        }
    }

    printf("����(ҽԺ��ֿ���������ĵı��)   ��Դ�����̾���\n");
    for (i = 0; i < num_vertices; i++) {
        printf("%d \t %d\n", i, dist[i]);
    }
}

int main() {
    int num_vertices;
    printf("�����붥��������");
    if (scanf("%d", &num_vertices) != 1 || num_vertices <= 0 || num_vertices > MAX_V) {
        printf("��Ч�Ķ������������������г���������Ч�Ķ���������\n");
        return 1;
    }

    int graph[MAX_V][MAX_V];
    printf("�������ڽӾ����ÿո���з��ָ���inf��INF��������󣩣�\n");

    // �����ڽӾ���
    int i, j;
    char input[10];
    for (i = 0; i < num_vertices; i++) {
        for (j = 0; j < num_vertices; j++) {
            scanf("%s", input);
            graph[i][j] = stringToInt(input);
        }
    }

    // ��֤�ڽӾ����Ƿ���Ч
    if (!validateMatrix(graph, num_vertices)) {
        printf("�ڽӾ�����Ч�����������롣\n");
        return 1;
    }

    int src;
    printf("������Դ��(�ֿ���������ĵı��)��");
    scanf("%d", &src);

    dijkstra(graph, num_vertices, src); // ��ָ��Դ�㿪ʼ�������·��

    return 0;
}

