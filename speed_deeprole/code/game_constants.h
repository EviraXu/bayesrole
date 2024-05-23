#ifndef GAME_CONSTANTS_H_
#define GAME_CONSTANTS_H_

//游戏中玩家的总数
#define NUM_PLAYERS 5

//游戏中邪恶角色的数量
#define NUM_EVIL 2

//游戏中的视点总数，定义为1 + 6 + 4 + 4，它们代表不同的游戏状态或角色知识的视角
#define NUM_VIEWPOINTS (1 + 6 + 4 + 4)

//代表好角色视点的数量，定义为1 + 6
#define NUM_GOOD_VIEWPOINTS (1 + 6)

//可能的角色分配总数
#define NUM_ASSIGNMENTS 60

//提案选项的数量C(5,2)
#define NUM_PROPOSAL_OPTIONS 10

//用于算法中的一个微小随机扰动，以避免确定性的行为。
#define TREMBLE_VALUE 0.01

//正反两方胜利和失败的收益
#define EVIL_WIN_PAYOFF 1.5
#define EVIL_LOSE_PAYOFF -1.5
#define GOOD_WIN_PAYOFF 1.0
#define GOOD_LOSE_PAYOFF -1.0

#define CFR_PLUS

#endif // GAME_CONSTANTS_H_
