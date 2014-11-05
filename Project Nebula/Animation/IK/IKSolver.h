#pragma once
#include <Animation/Rig/Skeleton.h>

class IKSolver
{
public:
	IKSolver(Skeleton* skeleton, float tolerance);
	~IKSolver(void);

	void enableIKChain(const QString &rootName, const QString &effectorName);
	void solveIK(const vec3 &targetPos);
	void BoneTransform(QVector<mat4>& Transforms);

private:
	Skeleton* m_skeleton;

	QVector<Bone*> m_boneChain;

	float m_tolerance;

	QVector<float> m_distances;
	float m_totalChainLength;

	Bone *m_effectorBone, *m_rootBone;
};

