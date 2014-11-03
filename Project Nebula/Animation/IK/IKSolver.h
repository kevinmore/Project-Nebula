#pragma once
#include <Animation/Rig/Skeleton.h>

class IKSolver
{
public:
	IKSolver(Skeleton* skeleton);
	~IKSolver(void);

	void solveIK(const QString &effectorName, const QString &rootName, const vec3 &targetPos);

private:
	Skeleton* m_skeleton;

	QVector<Bone*> m_boneChain;
};

