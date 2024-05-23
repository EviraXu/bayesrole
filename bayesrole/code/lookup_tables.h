#ifndef LOOKUP_TABLES_H_
#define LOOKUP_TABLES_H_

#include "game_constants.h"

extern const int PROPOSAL_TO_INDEX_LOOKUP[32];//将32种可能的提案映射到一个索引
extern const int INDEX_TO_PROPOSAL_2[10];//用于从索引映射回到两个玩家参与的提案
extern const int INDEX_TO_PROPOSAL_3[10];//用于从索引映射回到三个玩家参与的提案
extern const int ROUND_TO_PROPOSE_SIZE[5];//定义了游戏中每轮提案的大小。根据游戏轮次，提案可以包含不同数量的玩家。
extern const int VIEWPOINT_TO_BAD[NUM_PLAYERS][NUM_VIEWPOINTS];//表示每个玩家视角下的“坏”角色
extern const int ASSIGNMENT_TO_VIEWPOINT[NUM_ASSIGNMENTS][NUM_PLAYERS];//用于将角色分配映射到玩家视角
extern const int ASSIGNMENT_TO_EVIL[NUM_ASSIGNMENTS];//用于每种角色分配确定坏角色的数量
extern const int ASSIGNMENT_TO_ROLES[NUM_ASSIGNMENTS][3]; // 用于映射每种角色分配到具体角色，这里指定了3个角色，可能是梅林、刺客和爪牙。merlin, assassin, minion

// Only applicable to 5 and 6 player avalon.
extern const int VIEWPOINT_TO_PARTNER_VIEWPOINT[NUM_PLAYERS][NUM_VIEWPOINTS];//用于映射玩家视角到他们合作伙伴的视角。这对于识别游戏中玩家的忠诚度可能非常有用。

#endif // LOOKUP_TABLES_H_
