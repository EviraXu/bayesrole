#ifndef EIGEN_TYPES_H_
#define EIGEN_TYPES_H_

#include <Eigen/Core>
#include "game_constants.h"

//具有15个元素的列向量（由NUM_VIEWPOINTS决定），用于存储与不同视角相关的数据。
typedef Eigen::Array<double, NUM_VIEWPOINTS, 1> ViewpointVector;

//一个15行10列的矩阵（由NUM_VIEWPOINTS和NUM_PROPOSAL_OPTIONS决定），可能用于记录每个视角关于不同提案选项的信息
typedef Eigen::Array<double, NUM_VIEWPOINTS, NUM_PROPOSAL_OPTIONS> ProposeData;

//一个15行2列的矩阵（由NUM_VIEWPOINTS决定），可能用于存储每个视角的两种投票结果的数据。
typedef Eigen::Array<double, NUM_VIEWPOINTS, 2> VoteData;

//一个15行2列的矩阵（同样由NUM_VIEWPOINTS决定），可能用于记录关于不同任务的信息。
//0表示成功，1表示失败
typedef Eigen::Array<double, NUM_VIEWPOINTS, 2> MissionData;

//一个15行5列的矩阵（由NUM_VIEWPOINTS和NUM_PLAYERS决定），可能用于特别表示Merlin角色从每个视角看到的信息。
typedef Eigen::Array<double, NUM_VIEWPOINTS, NUM_PLAYERS> MerlinData;

//具有60个元素的列向量（由NUM_ASSIGNMENTS决定），用于表示不同角色分配的概率。
typedef Eigen::Array<double, NUM_ASSIGNMENTS, 1> AssignmentProbs;

#endif // EIGEN_TYPES_H_
