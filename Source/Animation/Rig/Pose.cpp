#include "Pose.h"

Pose::Pose()
{}

Pose::Pose(QVector<Bone*>& boneList)
	: m_boneList(boneList)
{}

Pose::~Pose()
{}

Pose Pose::lerp( Pose from, Pose to, float fraction )
{
	Pose result;

	return result;
}

