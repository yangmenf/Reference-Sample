/*顶点：表示医院、仓库、配送中心等物流网络中的节点。
      每个顶点代表一个地点，如一个医院或仓库。
源点：表示医药物流网络中的起始点或出发点，通常是一个仓库或配送中心。
      源点是从其开始计算到其他顶点的最短路径。
邻接矩阵：用于描述医药物流网络中各个顶点之间的连接关系和距离成本。
          邻接矩阵是一个二维矩阵，其中的元素表示两个顶点之间的距离或成本。矩阵中的值反映了两个顶点之间的直接连接情况和距离。
在医药物流配送网络优化问题中，通过输入顶点、源点和邻接矩阵的值，可以利用Dijkstra算法计算出从源点到其他顶点的最短路径，从而帮助优化医药物流配送网络的路径规划和成本控制。
得出的最短路径就是最优的药品配送方法*/
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits.h>
#include <string.h>

#define MAX_V 10 // 最大顶点数量
#define INF INT_MAX // 定义无穷大

// 检查邻接矩阵是否有效
bool validateMatrix(int graph[MAX_V][MAX_V], int num_vertices) {
    int i, j;
    for (i = 0; i < num_vertices; i++) {
        if (graph[i][i] != 0) {
            return false; // 对角线上的值不为0，无效的邻接矩阵
        }

        for (j = 0; j < num_vertices; j++) {
            if (graph[i][j] < 0 && graph[i][j] != INF) {
                return false; // 邻接矩阵中存在负值，或者非INF值
            }
        }
    }
    return true; // 邻接矩阵有效
}

// 将字符串转换为整数，如果字符串为"inf"或"INF"，返回INF
int stringToInt(const char *str) {
    if (strcmp(str, "inf") == 0 || strcmp(str, "INF") == 0) {
        return INF;
    }
    return atoi(str);
}

// 计算从源点到目标点的最短路径
void dijkstra(int graph[MAX_V][MAX_V], int num_vertices, int src) {
    int dist[MAX_V]; // 存储最短距离
    bool visited[MAX_V]; // 存储顶点是否被访问

    // 初始化距离和访问数组
    int i;
    for (i = 0; i < num_vertices; i++) {
        dist[i] = INF;
        visited[i] = false;
    }
    dist[src] = 0;
    int count;

    for (count = 0; count < num_vertices - 1; count++) {
        int min_dist = INF, min_index;

        // 找到当前未访问顶点中距离最小的顶点
        int v;
        for (v = 0; v < num_vertices; v++) {
            if (!visited[v] && dist[v] <= min_dist) {
                min_dist = dist[v];
                min_index = v;
            }
        }
        int u = min_index;
        visited[u] = true;

        // 更新最短路径距离
        for (v = 0; v < num_vertices; v++) {
            if (!visited[v] && graph[u][v] != INF &&
                dist[u] != INF && dist[u] + graph[u][v] < dist[v]) {
                dist[v] = dist[u] + graph[u][v];
            }
        }
    }

    printf("顶点(医院或仓库或配送中心的编号)   到源点的最短距离\n");
    for (i = 0; i < num_vertices; i++) {
        printf("%d \t %d\n", i, dist[i]);
    }
}

int main() {
    int num_vertices;
    printf("请输入顶点数量：");
    if (scanf("%d", &num_vertices) != 1 || num_vertices <= 0 || num_vertices > MAX_V) {
        printf("无效的顶点数量，请重新运行程序并输入有效的顶点数量。\n");
        return 1;
    }

    int graph[MAX_V][MAX_V];
    printf("请输入邻接矩阵（用空格或换行符分隔，inf或INF代表无穷大）：\n");

    // 输入邻接矩阵
    int i, j;
    char input[10];
    for (i = 0; i < num_vertices; i++) {
        for (j = 0; j < num_vertices; j++) {
            scanf("%s", input);
            graph[i][j] = stringToInt(input);
        }
    }

    // 验证邻接矩阵是否有效
    if (!validateMatrix(graph, num_vertices)) {
        printf("邻接矩阵无效，请重新输入。\n");
        return 1;
    }

    int src;
    printf("请输入源点(仓库或配送中心的编号)：");
    scanf("%d", &src);

    dijkstra(graph, num_vertices, src); // 从指定源点开始计算最短路径

    return 0;
}

