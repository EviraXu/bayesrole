#include <iostream>
#include <unordered_map>

int main(){
    
}

int main() {
    const int ASSIGNMENT_TO_VIEWPOINT[60][5] = {
        {  1,    8,   12,    0,    0 },
        {  2,    9,    0,   12,    0 },
        {  3,   10,    0,    0,   12 },
        {  1,   12,    8,    0,    0 },
        {  4,    0,    9,   13,    0 },
        {  5,    0,   10,    0,   13 },
        {  2,   13,    0,    8,    0 },
        {  4,    0,   13,    9,    0 },
        {  6,    0,    0,   10,   14 },
        {  3,   14,    0,    0,    8 },
        {  5,    0,   14,    0,    9 },
        {  6,    0,    0,   14,   10 },
        {  8,    1,   11,    0,    0 },
        {  9,    2,    0,   11,    0 },
        { 10,    3,    0,    0,   11 },
        { 12,    1,    7,    0,    0 },
        {  0,    4,    9,   13,    0 },
        {  0,    5,   10,    0,   13 },
        { 13,    2,    0,    7,    0 },
        {  0,    4,   13,    9,    0 },
        {  0,    6,    0,   10,   14 },
        { 14,    3,    0,    0,    7 },
        {  0,    5,   14,    0,    9 },
        {  0,    6,    0,   14,   10 },
        {  7,   11,    1,    0,    0 },
        {  9,    0,    2,   11,    0 },
        { 10,    0,    3,    0,   11 },
        { 11,    7,    1,    0,    0 },
        {  0,    9,    4,   12,    0 },
        {  0,   10,    5,    0,   12 },
        { 13,    0,    2,    7,    0 },
        {  0,   13,    4,    8,    0 },
        {  0,    0,    6,   10,   14 },
        { 14,    0,    3,    0,    7 },
        {  0,   14,    5,    0,    8 },
        {  0,    0,    6,   14,   10 },
        {  7,   11,    0,    1,    0 },
        {  8,    0,   11,    2,    0 },
        { 10,    0,    0,    3,   11 },
        { 11,    7,    0,    1,    0 },
        {  0,    8,   12,    4,    0 },
        {  0,   10,    0,    5,   12 },
        { 12,    0,    7,    2,    0 },
        {  0,   12,    8,    4,    0 },
        {  0,    0,   10,    6,   13 },
        { 14,    0,    0,    3,    7 },
        {  0,   14,    0,    5,    8 },
        {  0,    0,   14,    6,    9 },
        {  7,   11,    0,    0,    1 },
        {  8,    0,   11,    0,    2 },
        {  9,    0,    0,   11,    3 },
        { 11,    7,    0,    0,    1 },
        {  0,    8,   12,    0,    4 },
        {  0,    9,    0,   12,    5 },
        { 12,    0,    7,    0,    2 },
        {  0,   12,    8,    0,    4 },
        {  0,    0,    9,   13,    6 },
        { 13,    0,    0,    7,    3 },
        {  0,   13,    0,    8,    5 },
        {  0,    0,   13,    9,    6 }
    };

    std::unordered_map<int, int> countMap;

    // 统计第一列中各个数字出现的次数
    for (int i = 0; i < NUM_ASSIGNMENTS; i++) {
        int number = ASSIGNMENT_TO_VIEWPOINT[i][0];
        countMap[number]++;
    }

    // 输出结果
    for (const auto& entry : countMap) {
        std::cout << "数字 " << entry.first << " 出现了 " << entry.second << " 次" << std::endl;
    }

    return 0;
}